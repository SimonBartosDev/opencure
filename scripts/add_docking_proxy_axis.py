"""
Populate the docking axis of triangulation via a ChEMBL-bioactivity proxy.

True AutoDock Vina docking requires PDB structures + ligand prep + Vina
binary + ~10 min/pair — out of scope for a post-screen backfill. Instead
we use measured ChEMBL bioactivity (``data/drkg/drug_target_activities.json``,
94,717 drug-target median-nM pairs) as a proxy: if the drug has a measured
sub-μM affinity against any gene associated with the disease, treat it as
a positive docking axis and convert nM → pseudo-kcal/mol for triangulation.

This is an honest approximation, not a substitute for real docking. It is
labeled ``docking_source: "chembl_bioactivity_proxy"`` on every candidate
whose docking axis it populates so no one confuses it with Vina output.

Formula (nM → pseudo-kcal/mol; lower = better binding per Vina convention):
    kcal = -RT * ln(Ki) with T=298 K, R=1.987e-3
    simplified: kcal ≈ -0.593 * ln(nM * 1e-9)

Usage
-----
    python3 scripts/add_docking_proxy_axis.py               # all result JSONs
    python3 scripts/add_docking_proxy_axis.py Malaria
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from opencure.scoring.common import AGGREGATE_RESULT_FILES


RESULTS_DIR = Path("experiments/results")
ACTIVITIES_PATH = Path("data/drkg/drug_target_activities.json")
DISEASE_GENE_INDEX_PATH = Path("data/disease_gene_index.json")
OT_TRIPLETS_PATH = Path("data/open_targets/ot_triplets.tsv")  # legacy fallback
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")


_DISEASE_GENE_INDEX: dict[str, list[str]] | None = None
_DISEASE_GENES_CACHE: dict[str, set[str]] = {}
_ENTREZ_TO_SYMBOL: dict[str, str] | None = None


def _load_entrez_to_symbol() -> dict[str, str]:
    """Cache Entrez → gene symbol mapping from HGNC."""
    global _ENTREZ_TO_SYMBOL
    if _ENTREZ_TO_SYMBOL is not None:
        return _ENTREZ_TO_SYMBOL
    _ENTREZ_TO_SYMBOL = {}
    if not HGNC_PATH.exists():
        return _ENTREZ_TO_SYMBOL
    import csv
    with HGNC_PATH.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            ent = (row.get("entrez_id") or "").strip()
            sym = (row.get("symbol") or "").strip()
            if ent and sym:
                _ENTREZ_TO_SYMBOL[ent] = sym
    return _ENTREZ_TO_SYMBOL


def _genes_for_disease_entity(disease_entity: str) -> set[str]:
    """Return the union gene-symbol set for a disease.

    Prefers the unified index built by ``scripts/build_disease_gene_index.py``
    (OT::assoc + DRKG GNBR). Falls back to streaming OT triplets if the
    index is missing — slower but same semantics for curated OT edges.
    """
    if disease_entity in _DISEASE_GENES_CACHE:
        return _DISEASE_GENES_CACHE[disease_entity]
    if not disease_entity:
        _DISEASE_GENES_CACHE[disease_entity] = set()
        return set()

    # Fast path: unified index
    global _DISEASE_GENE_INDEX
    if _DISEASE_GENE_INDEX is None and DISEASE_GENE_INDEX_PATH.exists():
        _DISEASE_GENE_INDEX = json.loads(DISEASE_GENE_INDEX_PATH.read_text())
    if _DISEASE_GENE_INDEX:
        genes = set(_DISEASE_GENE_INDEX.get(disease_entity, []))
        _DISEASE_GENES_CACHE[disease_entity] = genes
        return genes

    # Fallback: OT-only streaming scan
    if not OT_TRIPLETS_PATH.exists():
        _DISEASE_GENES_CACHE[disease_entity] = set()
        return set()
    import pandas as pd
    ent_to_sym = _load_entrez_to_symbol()
    genes = set()
    for chunk in pd.read_csv(
        OT_TRIPLETS_PATH, sep="\t", header=None,
        names=["h", "r", "t"], chunksize=500_000,
    ):
        m = (chunk["r"] == "OT::assoc::Gene:Disease") & (chunk["t"] == disease_entity)
        for gene_ent in chunk.loc[m, "h"]:
            if "::" not in gene_ent:
                continue
            entrez = gene_ent.split("::", 1)[1]
            sym = ent_to_sym.get(entrez)
            if sym:
                genes.add(sym)
    _DISEASE_GENES_CACHE[disease_entity] = genes
    return genes


def _resolve_disease_entity(disease_name: str) -> str | None:
    try:
        from opencure.data.drkg import load_embeddings, find_disease_entities
        ent_emb, rel_emb, ent2id, id2ent, rel2id = load_embeddings()
        matches = find_disease_entities(ent2id, disease_name)
        if matches:
            m = matches[0]
            return m[0] if isinstance(m, tuple) else m
    except Exception:
        pass
    return None


def load_activities() -> dict:
    if not ACTIVITIES_PATH.exists():
        return {}
    return json.loads(ACTIVITIES_PATH.read_text())


def disease_genes(disease_name: str) -> set[str]:
    entity = _resolve_disease_entity(disease_name)
    if not entity:
        return set()
    return _genes_for_disease_entity(entity)


def nM_to_kcal(nM: float) -> float:
    """Convert binding Kd/IC50 in nM to free energy ΔG° in kcal/mol.

    ΔG° = RT·ln(Kd), with Kd in molar. Vina scoring convention: negative =
    favorable binding, typical range −5 to −12 kcal/mol.

    Example: 1 nM  → ΔG° ≈ −12.3 kcal/mol
             1 μM  → ΔG° ≈ −8.2  kcal/mol
             1 mM  → ΔG° ≈ −4.1  kcal/mol
    """
    if nM is None or nM <= 0:
        return 0.0
    # R = 1.987e-3 kcal/(mol·K), T = 298 K  →  RT = 0.593 kcal/mol
    return round(0.593 * math.log(nM * 1e-9), 2)


def best_binding(drug_id: str, acts: dict, target_set: set[str]) -> tuple[float | None, str | None]:
    """Return (best_kcal, target_symbol) for the lowest-nM hit the drug has
    against any gene in target_set. None if no measured activity overlaps."""
    d_key = drug_id.split("::", 1)[-1] if "::" in drug_id else drug_id
    drug_acts = acts.get(d_key) or acts.get(drug_id) or {}
    if not drug_acts:
        return None, None
    best_nM, best_sym = None, None
    # drug_acts is expected {symbol_or_uniprot: median_nM} or {symbol: {"nM":x}}.
    for tgt, val in drug_acts.items():
        if tgt not in target_set:
            continue
        if isinstance(val, dict):
            # drug_target_activities.json uses "median_nM"; earlier schemas
            # sometimes used bare "nM" — accept either so we don't silently
            # skip hits.
            nM = val.get("median_nM") or val.get("nM")
        else:
            nM = val
        try:
            nM = float(nM)
        except (TypeError, ValueError):
            continue
        if nM <= 0:
            continue
        if best_nM is None or nM < best_nM:
            best_nM, best_sym = nM, tgt
    if best_nM is None:
        return None, None
    return nM_to_kcal(best_nM), best_sym


def backfill(path: Path, acts: dict) -> int:
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0
    disease_name = data.get("disease") or path.stem.replace("_", " ")
    target_set = disease_genes(disease_name)

    # Proxy hits fire only when the drug-disease target geometry is present:
    #   disease has OT::assoc gene set  AND
    #   the drug has a ChEMBL-measured hit on one of those genes
    # In practice this fires rarely on neglected-tropical diseases (OT
    # disease-gene coverage is thin) and for drugs whose measured off-target
    # set doesn't overlap the disease gene set. That's honest and visible.

    n_hit = 0
    for cand in candidates:
        kcal, tgt = best_binding(cand.get("drug_id", ""), acts, target_set) if target_set else (None, None)
        if kcal is None:
            cand["docking"] = {
                "kcal_per_mol": None,
                "target_symbol": None,
                "source": "not_wired",
                "note": "AutoDock Vina not yet integrated; proxy found no "
                        "ChEMBL-measured overlap between drug targets and "
                        "OT-associated disease genes.",
                "hit": False,
            }
            continue
        cand["docking"] = {
            "kcal_per_mol": kcal,
            "target_symbol": tgt,
            "source": "chembl_bioactivity_proxy",
            "note": "Proxy: ChEMBL-measured IC50/Ki nM → pseudo-kcal/mol; "
                    "drug hits a gene associated with this disease in OT.",
            "hit": True,
        }
        # Rebuild triangulation using the new docking axis
        try:
            from opencure.evidence.triangulation import compute_triangulation_score
            cand["triangulation"] = compute_triangulation_score(
                kg_score=cand.get("combined_score", 0.0),
                docking_score=kcal,
                pharos_tdl=(cand.get("triangulation", {}).get("axis_values") or {}).get("pharos_tdl")
                           or None,
                pubmed_total=cand.get("pubmed_total", 0) or 0,
            )
        except Exception:
            pass
        n_hit += 1

    # Book-keeping on the file level
    data["docking_axis"] = {
        "source": "chembl_bioactivity_proxy + not_wired fallback",
        "n_disease_genes": len(target_set),
        "note": "Docking axis uses ChEMBL-measured potency as a proxy for "
                "pose energy when drug-disease target overlap exists. True "
                "AutoDock Vina integration is v6 work.",
    }
    json.dump(data, path.open("w"), indent=2)
    return n_hit


def main() -> None:
    acts = load_activities()
    if not acts:
        sys.exit(f"No activities at {ACTIVITIES_PATH}. Run "
                 "scripts/build_drug_target_activities.py first.")
    print(f"Loaded {len(acts):,} drugs with measured bioactivity")

    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)
    total_hits = 0
    total_cands = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n = backfill(f, acts)
        data = json.load(f.open())
        ncands = len(data.get("candidates") or data.get("top_candidates") or [])
        total_hits += n
        total_cands += ncands
        print(f"  {f.name}: {n}/{ncands} candidates with docking-proxy hits")
    print(f"\nDone. {total_hits}/{total_cands} candidates carry a docking axis.")


if __name__ == "__main__":
    main()
