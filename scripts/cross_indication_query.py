#!/usr/bin/env python3
"""
Four-filter cross-indication repurposing query (OpenCure).

THE QUERY (never run in full before):
  For each genetics-covered disease B, surface a drug that
    (1) hits a gene human genetics says is CAUSAL for B,
    (2) in the CORRECT (disease-opposing) direction,
    (3) is APPROVED (max_phase=4) for a DIFFERENT disease, and
    (4) has NEVER been tried for B (no drug_indication row mapping to B).
  All four together strip out rediscoveries and leave genuine cross-indication
  transfers.

DATA SOURCES
  - disease -> causal gene + Open Targets genetics association score:
      experiments/results/genetics_anchored/*.json  (status == "covered")
  - drug -> target -> mechanism direction, approval phase, indications:
      ChEMBL 34 SQLite.
  - direction of effect (filter 2): Open Targets GraphQL evidence
      directionOnTarget / directionOnTrait, cached locally.

HONEST DIRECTION HANDLING
  In the OT release served by the public API, direction-of-effect is populated
  ONLY on the `clinical_precedence` datasource (which is drug-derived, hence
  leak-prone for a repurposing claim) -- the pure human-genetics datasources
  (gwas_credible_sets, eva, ...) carry NULL direction. We therefore grade
  direction in three honest tiers:
    - "concordant"            : a NON-clinical (genetics/functional) source
                                gives a direction and the drug opposes it.
    - "concordant_leakprone"  : only clinical_precedence gives the direction;
                                concordant but the source is drug-derived.
    - "unverified"            : no usable direction signal -> NOT a concordance
                                claim. These are explicitly weaker.
  Candidates whose drug direction DISCORDS a non-leakprone genetics direction
  are dropped.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import ssl
import time
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GA_DIR = ROOT / "experiments" / "results" / "genetics_anchored"
CHEMBL_DB = ROOT / "data" / "sources_2024" / "chembl_34" / "chembl_34_sqlite" / "chembl_34.db"
DIR_CACHE = ROOT / "data" / "open_targets" / "direction_of_effect_cache.json"
OUT = ROOT / "experiments" / "eval" / "cross_indication_leads.json"

OT_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
# pure-genetics / functional datasources (leak-free); clinical_precedence handled separately
GENETICS_DS = [
    "gene_burden", "eva", "genomics_england", "orphanet", "ot_genetics_portal",
    "gwas_credible_sets", "uniprot_variants", "clingen", "gene2phenotype",
    "uniprot_literature", "eva_somatic",
]

POS = "POSITIVE MODULATOR"
NEG = "NEGATIVE MODULATOR"


# --------------------------------------------------------------------------
# 1. disease -> genetics-causal genes  (from genetics_anchored covered results)
# --------------------------------------------------------------------------
def load_covered_diseases():
    out = []
    for f in sorted(glob.glob(str(GA_DIR / "*.json"))):
        d = json.load(open(f))
        if d.get("status") != "covered":
            continue
        genes = {}
        for c in d.get("candidates", []):
            sym = c.get("target_gene_symbol")
            if not sym:
                continue
            # keep the highest genetics score + an ensembl id per gene
            prev = genes.get(sym)
            sc = c.get("target_genetics_score") or 0.0
            if prev is None or sc > prev["score"]:
                genes[sym] = {
                    "score": sc,
                    "ensembl": c.get("target_gene_id"),
                    "datasources": c.get("evidence_datasources", []),
                }
        if genes:
            out.append({
                "disease": d["disease"],
                "ot_disease_ids": d.get("ot_disease_ids", []),
                "genes": genes,
            })
    return out


# --------------------------------------------------------------------------
# 2. direction of effect via Open Targets (cached)
# --------------------------------------------------------------------------
_DIR_CACHE = None


def _load_dir_cache():
    global _DIR_CACHE
    if _DIR_CACHE is None:
        _DIR_CACHE = json.load(open(DIR_CACHE)) if DIR_CACHE.exists() else {}
    return _DIR_CACHE


def _save_dir_cache():
    DIR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_DIR_CACHE, open(DIR_CACHE, "w"), indent=2)


def _ot_query(ensembl, efo_ids):
    """Return list of (datasource, directionOnTarget, directionOnTrait)."""
    q = (
        "query Q($e:String!,$ds:[String!]!){ target(ensemblId:$e){ "
        "evidences(efoIds:$ds, size:500){ rows { datasourceId directionOnTarget directionOnTrait } } } }"
    )
    body = json.dumps({"query": q, "variables": {"e": ensembl, "ds": efo_ids}}).encode()
    req = urllib.request.Request(OT_GQL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40, context=_SSL_CTX) as r:
        data = json.load(r)
    tgt = (data.get("data") or {}).get("target")
    if not tgt or not tgt.get("evidences"):
        return []
    return [
        (row.get("datasourceId"), row.get("directionOnTarget"), row.get("directionOnTrait"))
        for row in tgt["evidences"]["rows"]
    ]


def disease_gene_direction(ensembl, efo_ids):
    """
    Resolve disease-causing direction of a gene.

    Returns (required_parent_type, tier) where:
      required_parent_type in {POS, NEG, None}
        - NEG : excess gene activity causes disease -> need inhibitor/antagonist
        - POS : deficient gene activity causes disease -> need agonist/activator
        - None: undeterminable
      tier in {"genetics", "leakprone", "none"}
    Logic: each (directionOnTarget, directionOnTrait) row is a variant effect.
      GoF + risk  -> excess causes disease -> NEG
      LoF + protect -> excess causes disease -> NEG   (loss protects => excess harms)
      LoF + risk  -> deficiency causes disease -> POS
      GoF + protect -> deficiency causes disease -> POS
    """
    if not ensembl or not efo_ids:
        return None, "none"
    cache = _load_dir_cache()
    key = f"{ensembl}|{','.join(efo_ids)}"
    if key not in cache or cache[key].get("error"):
        try:
            rows = _ot_query(ensembl, efo_ids)
        except Exception as e:  # network/transient
            rows = []
            cache[key] = {"error": str(e)[:120], "rows": []}
        else:
            cache[key] = {"rows": rows}
        _save_dir_cache()
        time.sleep(0.2)
    rows = cache[key].get("rows", [])

    def vote(rows_subset):
        votes = defaultdict(int)
        for ds, dot, dotr in rows_subset:
            if not dotr:
                continue
            # need a target-direction sense; infer "excess harmful" vs "deficiency harmful"
            harmful = None  # True = excess harmful (NEG), False = deficiency harmful (POS)
            if dot == "GoF" and dotr == "risk":
                harmful = True
            elif dot == "LoF" and dotr == "protect":
                harmful = True
            elif dot == "LoF" and dotr == "risk":
                harmful = False
            elif dot == "GoF" and dotr == "protect":
                harmful = False
            elif dotr == "protect":
                # direction on target unknown but variant protective -> can't sense reliably
                continue
            elif dotr == "risk":
                continue
            if harmful is True:
                votes[NEG] += 1
            elif harmful is False:
                votes[POS] += 1
        if not votes:
            return None
        return max(votes, key=votes.get)

    genetics_rows = [r for r in rows if r[0] != "clinical_precedence"]
    g = vote(genetics_rows)
    if g:
        return g, "genetics"
    leak = vote([r for r in rows if r[0] == "clinical_precedence"])
    if leak:
        return leak, "leakprone"
    return None, "none"


# --------------------------------------------------------------------------
# 3. ChEMBL: gene -> approved drugs w/ mechanism direction + indication set
# --------------------------------------------------------------------------
def chembl_gene_drugs(con, gene_symbol):
    """
    Return approved (max_phase=4) drugs with a curated human-target mechanism
    on `gene_symbol`. Each: chembl_id, name, parent_type (POS/NEG/OTHER), action_type.
    """
    q = """
    SELECT md.chembl_id, md.pref_name, md.molregno,
           dm.action_type, COALESCE(at.parent_type,'OTHER') AS parent
    FROM component_synonyms cs
    JOIN component_sequences cseq ON cseq.component_id = cs.component_id AND cseq.tax_id = 9606
    JOIN target_components tc ON tc.component_id = cs.component_id
    JOIN drug_mechanism dm ON dm.tid = tc.tid
    JOIN molecule_dictionary md ON md.molregno = dm.molregno
    LEFT JOIN action_type at ON at.action_type = dm.action_type
    WHERE cs.syn_type = 'GENE_SYMBOL' AND cs.component_synonym = ?
      AND md.max_phase = 4
    GROUP BY md.molregno, dm.action_type
    """
    out = {}
    for chembl_id, name, molregno, action_type, parent in con.execute(q, (gene_symbol,)):
        rec = out.setdefault(molregno, {
            "chembl_id": chembl_id, "name": name, "molregno": molregno,
            "action_types": set(), "parents": set(),
        })
        rec["action_types"].add(action_type)
        rec["parents"].add(parent)
    return list(out.values())


def drug_indications(con, molregno):
    """Return list of dicts with efo_id, mesh_id, term, max_phase_for_ind."""
    q = """SELECT efo_id, efo_term, mesh_id, mesh_heading, max_phase_for_ind
           FROM drug_indication WHERE molregno = ?"""
    out = []
    for efo, efo_t, mesh, mesh_h, phase in con.execute(q, (molregno,)):
        out.append({
            "efo_id": (efo or "").replace(":", "_") or None,
            "efo_term": efo_t, "mesh_id": mesh or None,
            "mesh_heading": mesh_h, "max_phase_for_ind": float(phase) if phase is not None else None,
        })
    return out


def _norm_ids(ot_ids):
    """Normalise disease ids to a set of canonical forms (EFO_x, MONDO_x, HP_x...)."""
    s = set()
    for i in ot_ids:
        s.add(i.replace(":", "_"))
    return s


def tried_for_B(indications, b_ids, b_name):
    """
    Filter (4): is the drug tried for B? Inclusive / conservative: any efo_id or
    mesh_id matching B's ids counts as tried; also a fuzzy name match on the
    disease term (substring, both directions) -> tried, to stay HONEST about
    novelty (when unsure, treat as tried so we don't over-claim novelty).
    Returns (tried_bool, matching_terms).
    """
    bset = _norm_ids(b_ids)
    bn = b_name.lower().strip().rstrip("s")  # crude singular
    hits = []
    for ind in indications:
        if ind["efo_id"] and ind["efo_id"] in bset:
            hits.append(ind.get("efo_term") or ind["efo_id"])
            continue
        # mesh ids in b? b_ids are EFO/MONDO/HP -> mesh rarely matches directly; skip
        term = (ind.get("efo_term") or ind.get("mesh_heading") or "").lower()
        if not term:
            continue
        # fuzzy: disease name appears in indication term or vice versa
        if bn and (bn in term or term.rstrip("s") in bn):
            hits.append(ind.get("efo_term") or ind.get("mesh_heading"))
    return (len(hits) > 0), sorted(set(h for h in hits if h))


def approved_elsewhere(indications, b_ids, b_name):
    """Filter (3): >=1 indication at phase 4 that is NOT B."""
    bset = _norm_ids(b_ids)
    bn = b_name.lower().strip().rstrip("s")
    others = []
    for ind in indications:
        if (ind["max_phase_for_ind"] or 0) < 4:
            continue
        is_b = ind["efo_id"] in bset if ind["efo_id"] else False
        term = (ind.get("efo_term") or ind.get("mesh_heading") or "")
        if not is_b and bn and term and (bn in term.lower() or term.lower().rstrip("s") in bn):
            is_b = True
        if not is_b and term:
            others.append(term)
    return (len(others) > 0), sorted(set(others))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    diseases = load_covered_diseases()
    con = sqlite3.connect(str(CHEMBL_DB))

    results = []
    total_candidates = 0
    diseases_with_lead = 0

    for D in diseases:
        bname = D["disease"]
        b_ids = D["ot_disease_ids"]
        leads = []
        for gene, ginfo in D["genes"].items():
            req_parent, dir_tier = disease_gene_direction(ginfo["ensembl"], b_ids)
            drugs = chembl_gene_drugs(con, gene)
            for drug in drugs:
                parents = drug["parents"] - {"OTHER"}
                if not parents:
                    continue  # no usable direction sense from the drug side
                # direction grading
                if req_parent is None:
                    dir_status = "unverified"
                else:
                    if req_parent in parents:
                        dir_status = "concordant" if dir_tier == "genetics" else "concordant_leakprone"
                    else:
                        # drug direction contradicts a known disease direction -> drop
                        if dir_tier == "genetics":
                            continue
                        dir_status = "unverified"  # only leakprone said otherwise; don't claim either way
                inds = drug_indications(con, drug["molregno"])
                appr_ok, other_inds = approved_elsewhere(inds, b_ids, bname)
                if not appr_ok:
                    continue  # filter 3 fails
                tried, b_match = tried_for_B(inds, b_ids, bname)
                if tried:
                    continue  # filter 4 fails (already tried for B)
                leads.append({
                    "drug_name": drug["name"] or drug["chembl_id"],
                    "drug_chembl_id": drug["chembl_id"],
                    "causal_gene": gene,
                    "gene_ensembl": ginfo["ensembl"],
                    "genetics_score": round(ginfo["score"], 4),
                    "genetics_datasources": ginfo["datasources"],
                    "drug_action_types": sorted(drug["action_types"]),
                    "drug_direction_class": sorted(parents),
                    "required_direction": req_parent,
                    "direction_status": dir_status,
                    "approved_for": other_inds[:12],
                    "untried_for_B": True,
                })
        # sort leads: concordant > leakprone > unverified, then genetics score
        rank = {"concordant": 0, "concordant_leakprone": 1, "unverified": 2}
        leads.sort(key=lambda x: (rank[x["direction_status"]], -x["genetics_score"]))
        if leads:
            diseases_with_lead += 1
            total_candidates += len(leads)
        results.append({
            "disease": bname,
            "ot_disease_ids": b_ids,
            "n_leads": len(leads),
            "leads": leads,
        })

    summary = {
        "n_covered_diseases": len(diseases),
        "n_diseases_with_lead": diseases_with_lead,
        "total_candidates": total_candidates,
        "n_concordant": sum(1 for r in results for l in r["leads"] if l["direction_status"] == "concordant"),
        "n_concordant_leakprone": sum(1 for r in results for l in r["leads"] if l["direction_status"] == "concordant_leakprone"),
        "n_unverified": sum(1 for r in results for l in r["leads"] if l["direction_status"] == "unverified"),
    }
    out = {"summary": summary, "diseases": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print("=== CROSS-INDICATION FOUR-FILTER QUERY ===")
    print(f"covered diseases scanned : {summary['n_covered_diseases']}")
    print(f"diseases w/ >=1 lead     : {summary['n_diseases_with_lead']}")
    print(f"total candidate leads    : {summary['total_candidates']}")
    print(f"  direction concordant         : {summary['n_concordant']}")
    print(f"  concordant (leakprone src)   : {summary['n_concordant_leakprone']}")
    print(f"  direction unverified         : {summary['n_unverified']}")
    print(f"written -> {OUT}")
    # show concordant + leakprone leads explicitly (the only ones worth verifying)
    print("\n=== DIRECTION-CONCORDANT LEADS (concordant / leakprone) ===")
    any_strong = False
    for r in results:
        strong = [l for l in r["leads"] if l["direction_status"] in ("concordant", "concordant_leakprone")]
        for l in strong:
            any_strong = True
            print(f"  [{l['direction_status']:22}] {r['disease']:28} <- {l['drug_name']} "
                  f"({'/'.join(l['drug_action_types'])} of {l['causal_gene']}, gscore={l['genetics_score']}) "
                  f"approved for: {', '.join(l['approved_for'][:3])}")
    if not any_strong:
        print("  (none -- all leads are direction_unverified)")


if __name__ == "__main__":
    main()
