"""
Genetics-anchored, target-based drug-repurposing scorer — OpenCure production core.

WHY THIS MODULE IS THE ENGINE
-----------------------------
A multi-agent leak-free research arc established that OpenCure's old "similarity
to known treatments" pillars (knowledge-graph embeddings, chemical-structure
similarity, cell-morphology similarity) do not beat a popularity baseline under
leak-free evaluation. The one approach that WORKS — verified leak-free and
replicated across two runs — is a GENETICS-ANCHORED, TARGET-BASED ranker:

    disease  --human genetics-->  causal gene  <--curated ChEMBL mechanism--  drug

On the genetics-covered subset it beats the popularity baseline ~5x
(Hit@10 ~21%, median rank 64 of ~8000 candidates).

The validated benchmark is `scripts/genetics_anchored_benchmark_v3.py` and its
result `experiments/eval/genetics_anchored_scorecard_v3.json`. This module
extracts the MECH-only (high-precision) configuration from that benchmark into
a clean, reusable scorer.

HONEST SCOPING — NON-NEGOTIABLE
-------------------------------
The old platform's failure was that it ALWAYS produced a confident-looking
ranking, even with no signal. This module does the opposite: when a disease has
no genetic evidence it returns an explicit `not_assessed` status with a
human-readable reason and NEVER emits a ranked list it cannot support.

A disease is `covered` only if BOTH:
  (1) it maps to an Open Targets disease entity via a MeSH/MSH dbXRef, AND
  (2) that entity has >=1 gene with a genetics-datasource association.
Otherwise the verdict is `not_assessed`.

LEAK CONTROL (leak-free by construction)
----------------------------------------
Disease side: ONLY genetics datasources (GENETICS_USE) — GWAS / rare-variant /
clinical-genetics evidence. chembl (known-drug), europepmc / uniprot_literature
(literature), expression / pathway and animal-model datasources are EXCLUDED.
Drug side: ONLY curated ChEMBL `drug_mechanism` targets. The promiscuous
measured-bioactivity supplement is deliberately NOT used — the research showed
it dilutes precision (signal-subset Hit@10 falls 20.8% -> 10.9%).
No `treats` / `indication` edge is read at any point.
"""
from __future__ import annotations

import glob
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from opencure.config import PROJECT_ROOT

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
OT_DIR = PROJECT_ROOT / "data" / "open_targets"
# Score-only cache from the validated benchmark (OT disease id -> {gene: score}).
DS_SCORE_CACHE = OT_DIR / "efo_genetic_datasource_cache.json"
# Richer cache built by this module: OT disease id -> {gene: {score, datasources}}.
DS_DETAIL_CACHE = OT_DIR / "efo_genetic_datasource_detail_cache.json"
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# --------------------------------------------------------------------------
# leak control — genetics datasources only (see module docstring)
# --------------------------------------------------------------------------
GENETICS_DATASOURCES = {
    "ot_genetics_portal",     # legacy L2G genetics portal
    "gwas_credible_sets",     # current L2G credible sets
    "eva",                    # ClinVar
    "eva_somatic",            # ClinVar somatic
    "gene_burden",            # rare-variant burden tests
    "uniprot_variants",       # curated disease variants
    "genomics_england",       # Genomics England PanelApp
    "orphanet",               # Orphanet gene-disease (rare disease)
    "clingen",                # ClinGen gene-disease validity
}
# Datasources that MUST be excluded — using any reintroduces leakage.
_BANNED = {
    "chembl", "europepmc", "uniprot_literature", "expression_atlas",
    "reactome", "slapenrich", "cancer_gene_census", "impc",
    "clinical_precedence", "known_drug",
}
# Final whitelist actually used.
GENETICS_USE = GENETICS_DATASOURCES - _BANNED


# --------------------------------------------------------------------------
# result schema
# --------------------------------------------------------------------------
@dataclass
class DrugCandidate:
    """One mechanism-grounded drug candidate for a genetics-covered disease."""
    drug_id: str                    # ChEMBL id (and DrugBank id when known)
    drug_name: Optional[str]
    drugbank_id: Optional[str]
    score: float                    # OT genetics-datasource association score
    target_gene_id: str             # Ensembl id of the genetic target
    target_gene_symbol: Optional[str]
    target_genetics_score: float    # OT genetics-datasource score for that gene
    evidence_datasources: list[str]  # genetics datasources backing the gene link
    action_type: Optional[str]      # ChEMBL drug_mechanism action_type

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneticsResult:
    """Result of scoring one disease.

    `covered` is True only when the disease maps to an OT entity AND has
    genetics-datasource associations. When False, `candidates` is empty and
    `reason` explains why (honest not-assessed path).
    """
    disease_query: str              # raw input (name or DRKG/MeSH id)
    covered: bool
    status: str                     # "covered" | "not_assessed"
    reason: Optional[str]           # human-readable; set when not covered
    ot_disease_ids: list[str] = field(default_factory=list)
    candidates: list[DrugCandidate] = field(default_factory=list)

    @property
    def not_assessed(self) -> bool:
        return not self.covered

    def to_dict(self) -> dict:
        return {
            "disease_query": self.disease_query,
            "covered": self.covered,
            "status": self.status,
            "reason": self.reason,
            "ot_disease_ids": self.ot_disease_ids,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# --------------------------------------------------------------------------
# crosswalks — built once from local Open Targets parquet, cached in-process
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ot_disease_index() -> tuple[dict, dict, dict]:
    """Build disease crosswalks from the OT diseases parquet.

    Returns (mesh2ot, name2ot, ot2name):
      mesh2ot  : MeSH id (no prefix, upper) -> list of OT disease ids
      name2ot  : lowercased disease name/synonym -> list of OT disease ids
      ot2name  : OT disease id -> canonical name
    """
    mesh2ot: dict[str, set[str]] = defaultdict(set)
    name2ot: dict[str, set[str]] = defaultdict(set)
    ot2name: dict[str, str] = {}
    for f in sorted(glob.glob(str(OT_DIR / "diseases" / "*.parquet"))):
        cols = ["id", "name", "dbXRefs", "synonyms"]
        for r in pq.read_table(f, columns=cols).to_pylist():
            oid = r["id"]
            if r["name"]:
                ot2name[oid] = r["name"]
                name2ot[r["name"].strip().lower()].add(oid)
            # MeSH / MSH dbXRef variants (the benchmark's axis-1 fix).
            for x in r["dbXRefs"] or []:
                xu = str(x).upper()
                if xu.startswith("MESH:") or xu.startswith("MSH:"):
                    mesh2ot[xu.split(":", 1)[1]].add(oid)
            # synonyms can be a dict of {hasExactSynonym: [...], ...} or a list.
            syn = r["synonyms"]
            syn_labels: list[str] = []
            if isinstance(syn, dict):
                for v in syn.values():
                    if v:
                        syn_labels.extend(v)
            elif isinstance(syn, list):
                for s in syn:
                    if isinstance(s, dict):
                        lab = s.get("label") or s.get("name")
                        if lab:
                            syn_labels.append(lab)
                    elif s:
                        syn_labels.append(str(s))
            for lab in syn_labels:
                name2ot[str(lab).strip().lower()].add(oid)
    return (
        {k: sorted(v) for k, v in mesh2ot.items()},
        {k: sorted(v) for k, v in name2ot.items()},
        ot2name,
    )


@lru_cache(maxsize=1)
def _gene_crosswalk() -> dict[str, str]:
    """Ensembl gene id -> approved symbol (for human-readable output)."""
    ens2sym: dict[str, str] = {}
    for f in sorted(glob.glob(str(OT_DIR / "targets" / "*.parquet"))):
        for r in pq.read_table(
            f, columns=["id", "approvedSymbol"]
        ).to_pylist():
            if r["approvedSymbol"]:
                ens2sym[r["id"]] = r["approvedSymbol"]
    return ens2sym


@lru_cache(maxsize=1)
def _chembl_mechanism() -> tuple[dict, dict]:
    """Build ChEMBL drug_mechanism crosswalks.

    Returns (chembl2targets, chembl2action):
      chembl2targets : ChEMBL id -> {Ensembl gene id: action_type}
      chembl2action  : ChEMBL id -> representative action_type (first seen)

    ONLY curated drug_mechanism targets — no measured-bioactivity supplement.
    """
    chembl2targets: dict[str, dict[str, str]] = defaultdict(dict)
    chembl2action: dict[str, str] = {}
    for f in sorted(glob.glob(str(OT_DIR / "mechanismOfAction" / "*.parquet"))):
        for r in pq.read_table(
            f, columns=["chemblIds", "targets", "actionType"]
        ).to_pylist():
            action = r["actionType"]
            for c in r["chemblIds"] or []:
                if action and c not in chembl2action:
                    chembl2action[c] = action
                for t in r["targets"] or []:
                    chembl2targets[c].setdefault(t, action)
    return ({k: dict(v) for k, v in chembl2targets.items()}, chembl2action)


@lru_cache(maxsize=1)
def _molecule_crosswalk() -> tuple[dict, dict, dict]:
    """Build molecule crosswalks from the OT molecule parquet.

    Returns (chembl2name, chembl2drugbank, drugbank2chembl):
      chembl2name     : ChEMBL id -> drug name
      chembl2drugbank : ChEMBL id -> first DrugBank id
      drugbank2chembl : DrugBank id -> list of ChEMBL ids
    """
    chembl2name: dict[str, str] = {}
    chembl2drugbank: dict[str, str] = {}
    drugbank2chembl: dict[str, list[str]] = defaultdict(list)
    for f in sorted(glob.glob(str(OT_DIR / "molecule" / "*.parquet"))):
        for r in pq.read_table(
            f, columns=["id", "name", "crossReferences"]
        ).to_pylist():
            cid = r["id"]
            if r["name"]:
                chembl2name[cid] = r["name"]
            for k, v in r["crossReferences"] or []:
                if str(k).lower() == "drugbank" and v:
                    for dbid in v:
                        drugbank2chembl[dbid].append(cid)
                    chembl2drugbank.setdefault(cid, v[0])
    return (chembl2name, chembl2drugbank, {k: v for k, v in drugbank2chembl.items()})


# --------------------------------------------------------------------------
# genetics evidence — cache-first, OT GraphQL fallback
# --------------------------------------------------------------------------
def _load_detail_cache() -> dict:
    if DS_DETAIL_CACHE.exists():
        return json.loads(DS_DETAIL_CACHE.read_text())
    return {}


def _load_score_cache() -> dict:
    """Legacy score-only cache from the validated benchmark."""
    if DS_SCORE_CACHE.exists():
        return json.loads(DS_SCORE_CACHE.read_text())
    return {}


def _fetch_genetics_from_api(efo_id: str, timeout: float = 30.0) -> dict:
    """Query OT GraphQL for per-datasource genetics scores of one disease.

    Returns {Ensembl gene id: {"score": float, "datasources": [str, ...]}}
    keeping only genetics datasources. Raises on network/HTTP failure.
    """
    import requests

    query = """query {
      disease(efoId: "%s") {
        associatedTargets(page: {size: 500, index: 0}) {
          rows { target { id } datasourceScores { id score } }
        }
      }
    }""" % efo_id
    resp = requests.post(OPENTARGETS_URL, json={"query": query}, timeout=timeout)
    resp.raise_for_status()
    disease = (resp.json().get("data") or {}).get("disease")
    rows = (
        (disease or {}).get("associatedTargets", {}).get("rows", [])
        if disease else []
    )
    out: dict[str, dict] = {}
    for row in rows:
        tid = row.get("target", {}).get("id")
        if not tid:
            continue
        best = 0.0
        sources: list[str] = []
        for ds in row.get("datasourceScores", []):
            if ds["id"] in GENETICS_USE and ds["score"] > 0:
                sources.append(ds["id"])
                if ds["score"] > best:
                    best = ds["score"]
        if best > 0:
            out[tid] = {"score": best, "datasources": sorted(set(sources))}
    return out


def _genetics_for_disease(
    ot_id: str, allow_network: bool = True
) -> Optional[dict]:
    """Resolve genetics-datasource gene scores for one OT disease id.

    Resolution order:
      1. detail cache (this module's richer cache) — full datasource info
      2. legacy score-only benchmark cache — score only, datasources unknown
      3. OT GraphQL API (only if allow_network) — extends the detail cache

    Returns {gene: {"score": float, "datasources": [...]}} or None if the
    disease genuinely has no genetics evidence (or could not be resolved
    offline). An empty dict {} means "resolved, no genetic associations".
    """
    detail = _load_detail_cache()
    if ot_id in detail:
        return detail[ot_id]

    score_cache = _load_score_cache()
    if ot_id in score_cache:
        # Legacy cache: score only, datasources not recorded.
        return {
            g: {"score": s, "datasources": []}
            for g, s in score_cache[ot_id].items()
        }

    if not allow_network:
        return None

    try:
        result = _fetch_genetics_from_api(ot_id)
    except Exception:
        return None
    detail[ot_id] = result
    DS_DETAIL_CACHE.write_text(json.dumps(detail))
    time.sleep(0.2)
    return result


# --------------------------------------------------------------------------
# disease resolution
# --------------------------------------------------------------------------
def _resolve_disease(disease: str) -> tuple[list[str], Optional[str]]:
    """Map a disease query to OT disease ids.

    Accepts: a DRKG entity ("Disease::MESH:D000123"), a bare MeSH id
    ("MESH:D000123" / "D000123"), an OT id ("EFO_0000095"), or a disease name.

    Returns (ot_ids, reason). When ot_ids is empty, reason explains the miss
    ("no EFO mapping" path).
    """
    mesh2ot, name2ot, ot2name = _ot_disease_index()
    q = disease.strip()

    # OT id given directly.
    if q in ot2name:
        return [q], None

    # DRKG entity or MeSH id.
    mesh_id = None
    if q.startswith("Disease::"):
        tail = q.split("::", 1)[1]
        mesh_id = tail.split(":")[-1]
    elif q.upper().startswith("MESH:") or q.upper().startswith("MSH:"):
        mesh_id = q.split(":", 1)[1]
    elif q.upper().startswith("D") and q[1:].isdigit():
        mesh_id = q
    if mesh_id is not None:
        ids = mesh2ot.get(mesh_id.upper(), [])
        if ids:
            return ids, None
        return [], f"no EFO mapping: MeSH id {mesh_id} has no dbXRef in the " \
                   "Open Targets disease ontology snapshot"

    # disease name (exact, case-insensitive, against names + synonyms).
    ids = name2ot.get(q.lower(), [])
    if ids:
        return ids, None
    return [], f"no EFO mapping: '{disease}' did not match any Open Targets " \
               "disease name or synonym (try a MeSH id or an exact EFO name)"


# --------------------------------------------------------------------------
# the public scorer
# --------------------------------------------------------------------------
def score_disease(
    disease: str,
    *,
    top_k: Optional[int] = None,
    allow_network: bool = True,
) -> GeneticsResult:
    """Score one disease with the genetics-anchored, mechanism-target ranker.

    Parameters
    ----------
    disease : str
        Disease name, DRKG entity, MeSH id, or OT disease id.
    top_k : int, optional
        If set, truncate the ranked candidate list to the top K.
    allow_network : bool
        If False, never call the OT API — only the on-disk caches are used
        (the module works fully offline for cached diseases).

    Returns
    -------
    GeneticsResult
        Either `covered` with a ranked list of mechanism-grounded
        DrugCandidate objects, or `not_assessed` with a human-readable reason.
        Never emits a ranked list it cannot support.
    """
    # ---- resolve disease -> OT entity ids --------------------------------
    ot_ids, reason = _resolve_disease(disease)
    if not ot_ids:
        return GeneticsResult(
            disease_query=disease, covered=False, status="not_assessed",
            reason=reason,
        )

    # ---- collect genetics-datasource gene scores -------------------------
    # max score per gene across all matched OT ids; union of datasources.
    gene_score: dict[str, float] = {}
    gene_sources: dict[str, set[str]] = defaultdict(set)
    resolved_any = False
    for oid in ot_ids:
        g = _genetics_for_disease(oid, allow_network=allow_network)
        if g is None:
            continue
        resolved_any = True
        for gene, info in g.items():
            s = info["score"]
            if s > gene_score.get(gene, 0.0):
                gene_score[gene] = s
            gene_sources[gene].update(info.get("datasources", []))

    if not resolved_any:
        return GeneticsResult(
            disease_query=disease, covered=False, status="not_assessed",
            reason="no genetics evidence available offline for this disease "
                   "and network fetch is disabled or failed",
            ot_disease_ids=ot_ids,
        )

    if not gene_score:
        # Disease mapped to an OT entity but has NO genetics-datasource
        # association — the honest not-assessed path.
        return GeneticsResult(
            disease_query=disease, covered=False, status="not_assessed",
            reason="no genetics-datasource associations for this disease "
                   "(it maps to Open Targets but has no GWAS / rare-variant / "
                   "clinical-genetics gene evidence)",
            ot_disease_ids=ot_ids,
        )

    # ---- build candidate list: drugs with a curated mechanism on a gene --
    chembl2targets, _ = _chembl_mechanism()
    chembl2name, chembl2drugbank, _ = _molecule_crosswalk()
    ens2sym = _gene_crosswalk()

    candidates: list[DrugCandidate] = []
    for chembl_id, targets in chembl2targets.items():
        # max-over-targets: pick the drug's best genetically-supported gene.
        best_gene = None
        best_score = 0.0
        best_action = None
        for gene, action in targets.items():
            s = gene_score.get(gene, 0.0)
            if s > best_score:
                best_score = s
                best_gene = gene
                best_action = action
        if best_gene is None:
            continue  # drug has no genetically-supported mechanism target
        candidates.append(DrugCandidate(
            drug_id=chembl_id,
            drug_name=chembl2name.get(chembl_id),
            drugbank_id=chembl2drugbank.get(chembl_id),
            score=round(best_score, 6),
            target_gene_id=best_gene,
            target_gene_symbol=ens2sym.get(best_gene),
            target_genetics_score=round(best_score, 6),
            evidence_datasources=sorted(gene_sources.get(best_gene, set())),
            action_type=best_action,
        ))

    if not candidates:
        # Genetics evidence exists but no curated drug acts on any of those
        # genes — still honest: nothing to rank.
        return GeneticsResult(
            disease_query=disease, covered=False, status="not_assessed",
            reason="genetics evidence exists for this disease but no drug has "
                   "a curated ChEMBL mechanism target on any genetically-"
                   "supported gene",
            ot_disease_ids=ot_ids,
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    if top_k is not None:
        candidates = candidates[:top_k]

    return GeneticsResult(
        disease_query=disease, covered=True, status="covered", reason=None,
        ot_disease_ids=ot_ids, candidates=candidates,
    )
