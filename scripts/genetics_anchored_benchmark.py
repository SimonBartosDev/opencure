"""
Genetics-anchored, target-based drug-repurposing ranker — leak-free benchmark.

WHY THIS EXISTS
---------------
OpenCure's similarity pillars (KG embeddings, chemical structure, cell
morphology) all reduce to "score a drug by similarity to a disease's known
treatments" — which the leak-free benchmark showed collapses to a popularity
baseline. This script prototypes a *fundamentally different paradigm*:

    disease  ->  genetically-implicated gene  ->  drug that modulates that gene

Rationale: genetically-supported drug targets succeed ~2.6x more often in
clinical trials (Minikel et al., Nature 2024). The signal is the disease-gene
GENETIC association, not literature co-mention.

LEAK CONTROL
------------
The ranker's only inputs are:
  (a) disease -> gene genetic-association scores (Open Targets), and
  (b) drug -> mechanism target (ChEMBL drug_mechanism via Open Targets).
Neither is derived from the drug->disease treatment edge being predicted. No
`treats` / `indication` edge is read at any point. The ranker is therefore
leak-free by construction (not merely leak-controlled). Confirmed explicitly.

TIER 1 (built)
--------------
Score(drug, disease) = max over the drug's mechanism target genes of
[ Open Targets genetic_association score of that gene for that disease ].
Drugs with no mechanism target, or whose targets have no genetic association
to the disease, score 0.

TIER 2 (direction concordance) — see DIRECTION note at the bottom; NOT done.

METHODOLOGY
-----------
Identical to scripts/leakfree_benchmark.py: for each held-out (drug, disease)
pair the held-out drug is ranked among the full DRKG compound pool with
tie-aware mid-ranks; Hit@10/30/100, MRR, median rank reported, and compared
to the DRKG node-degree popularity baseline on the SAME pool.

OUTPUT
------
experiments/eval/genetics_anchored_scorecard.json
"""
from __future__ import annotations

import glob
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests

HOLDOUT = Path("data/eval/holdout_test.jsonl")
OT_DIR = Path("data/open_targets")
OUT = Path("experiments/eval/genetics_anchored_scorecard.json")
EFO_GENETIC_CACHE = Path("data/open_targets/efo_genetic_targets_cache.json")
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"


# --------------------------------------------------------------------------
# shared metric helpers (same definitions as leakfree_benchmark.py)
# --------------------------------------------------------------------------
def midrank(scores: np.ndarray, target_score: float) -> float:
    higher = int(np.sum(scores > target_score))
    equal = int(np.sum(scores == target_score))
    return higher + equal / 2.0 + 1.0


def summarise(ranks: list[float], n_pairs: int) -> dict:
    if not ranks:
        return {"evaluable": 0, "n_pairs": n_pairs}
    a = np.array(ranks, dtype=float)
    return {
        "evaluable": len(a),
        "n_pairs": n_pairs,
        "hit_at_10": round(100 * float(np.mean(a <= 10)), 1),
        "hit_at_30": round(100 * float(np.mean(a <= 30)), 1),
        "hit_at_100": round(100 * float(np.mean(a <= 100)), 1),
        "mrr": round(float(np.mean(1.0 / a)), 4),
        "median_rank": int(np.median(a)),
    }


# --------------------------------------------------------------------------
# crosswalks built entirely from local Open Targets parquet
# --------------------------------------------------------------------------
def build_mesh_to_efo() -> dict[str, list[str]]:
    """MeSH id (Dxxxxxx) -> list of EFO/MONDO ids, from diseases parquet dbXRefs."""
    mesh2efo: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(OT_DIR / "diseases" / "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "dbXRefs"]).to_pylist():
            for x in r["dbXRefs"] or []:
                xu = str(x).upper()
                if xu.startswith("MESH:"):
                    mesh2efo.setdefault(xu.split(":", 1)[1], []).append(r["id"])
    return mesh2efo


def build_drugbank_to_chembl() -> dict[str, set[str]]:
    """DrugBank id -> set of ChEMBL ids, from molecule parquet crossReferences."""
    db2chembl: dict[str, set[str]] = {}
    for f in sorted(glob.glob(str(OT_DIR / "molecule" / "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "crossReferences"]).to_pylist():
            for k, v in r["crossReferences"] or []:
                if str(k).lower() == "drugbank":
                    for dbid in v:
                        db2chembl.setdefault(dbid, set()).add(r["id"])
    return db2chembl


def build_chembl_mechanism() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """ChEMBL id -> (set of Ensembl target genes, set of actionTypes).

    Sourced from Open Targets mechanismOfAction parquet, which is ChEMBL's
    drug_mechanism table joined to Ensembl gene ids. Equivalent to
    drug_mechanism -> target_components -> component_sequences.
    """
    chembl2tgt: dict[str, set[str]] = {}
    chembl2action: dict[str, set[str]] = {}
    for f in sorted(glob.glob(str(OT_DIR / "mechanismOfAction" / "*.parquet"))):
        for r in pq.read_table(
            f, columns=["actionType", "chemblIds", "targets"]
        ).to_pylist():
            for c in r["chemblIds"] or []:
                for t in r["targets"] or []:
                    chembl2tgt.setdefault(c, set()).add(t)
                if r["actionType"]:
                    chembl2action.setdefault(c, set()).add(r["actionType"])
    return chembl2tgt, chembl2action


# --------------------------------------------------------------------------
# disease -> gene genetic-association scores
# --------------------------------------------------------------------------
def load_overall_assoc() -> dict[str, dict[str, float]]:
    """efoId -> {ensemblTargetId: overall association score} from local parquet.

    This is the *overall* (mixed-datatype) score — used only as a fallback /
    comparison axis. The genetic-association-specific score (the 2.6x signal)
    is fetched via the API below.
    """
    assoc: dict[str, dict[str, float]] = defaultdict(dict)
    for f in sorted(glob.glob(str(OT_DIR / "associationByOverallDirect" / "*.parquet"))):
        tbl = pq.read_table(
            f, columns=["diseaseId", "targetId", "score"]
        ).to_pylist()
        for r in tbl:
            assoc[r["diseaseId"]][r["targetId"]] = r["score"]
    return assoc


def fetch_genetic_targets(efo_ids: set[str]) -> dict[str, dict[str, float]]:
    """efoId -> {ensemblTargetId: genetic_association datatype score}.

    Queries the Open Targets GraphQL API (free, no auth) for the
    genetic_association datatype score — the GWAS/eQTL/L2G-derived signal.
    Cached to disk so the benchmark is reproducible without re-querying.
    """
    cache: dict[str, dict[str, float]] = {}
    if EFO_GENETIC_CACHE.exists():
        cache = json.loads(EFO_GENETIC_CACHE.read_text())

    todo = sorted(e for e in efo_ids if e not in cache)
    if todo:
        print(f"  fetching genetic_association scores for {len(todo)} EFO ids "
              f"via Open Targets API ...")
    for i, efo in enumerate(todo):
        query = """query {
          disease(efoId: "%s") {
            associatedTargets(page: {size: 500, index: 0}) {
              rows { target { id } datatypeScores { id score } }
            }
          }
        }""" % efo
        scores: dict[str, float] = {}
        try:
            resp = requests.post(
                OPENTARGETS_URL, json={"query": query}, timeout=30
            )
            if resp.status_code == 200:
                disease = (resp.json().get("data") or {}).get("disease")
                rows = (
                    (disease or {}).get("associatedTargets", {}).get("rows", [])
                    if disease else []
                )
                for row in rows:
                    tid = row.get("target", {}).get("id")
                    if not tid:
                        continue
                    for dt in row.get("datatypeScores", []):
                        if dt["id"] == "genetic_association" and dt["score"] > 0:
                            scores[tid] = dt["score"]
        except Exception as exc:  # network hiccup -> empty, recorded as cached
            print(f"    {efo}: {exc}")
        cache[efo] = scores
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(todo)} ...")
            EFO_GENETIC_CACHE.write_text(json.dumps(cache))
        time.sleep(0.25)
    if todo:
        EFO_GENETIC_CACHE.write_text(json.dumps(cache))
    return {e: cache[e] for e in efo_ids if e in cache}


# --------------------------------------------------------------------------
# the ranker
# --------------------------------------------------------------------------
def score_pool(disease_gene_scores: dict[str, float],
               compound_targets: dict[str, set[str]],
               candidates: list[str]) -> np.ndarray:
    """Score every candidate: max genetic-association score over its targets.

    disease_gene_scores : ensembl gene id -> genetic-association score
    compound_targets    : Compound:: entity -> set of ensembl target ids
    """
    out = np.zeros(len(candidates), dtype=float)
    for i, c in enumerate(candidates):
        tgts = compound_targets.get(c)
        if not tgts:
            continue
        vals = [disease_gene_scores[t] for t in tgts if t in disease_gene_scores]
        if vals:
            out[i] = max(vals)
    return out


def main() -> None:
    from opencure.data.drkg import load_triplets

    heldout = [(d["compound"], d["disease"])
               for d in (json.loads(l) for l in HOLDOUT.open())]
    by_disease: dict[str, list[str]] = defaultdict(list)
    for drug, dis in heldout:
        by_disease[dis].append(drug)
    print(f"Held-out pairs: {len(heldout)} across {len(by_disease)} diseases")

    # ---- crosswalks ------------------------------------------------------
    print("Building crosswalks from local Open Targets parquet ...")
    mesh2efo = build_mesh_to_efo()
    db2chembl = build_drugbank_to_chembl()
    chembl2tgt, chembl2action = build_chembl_mechanism()
    print(f"  MeSH->EFO: {len(mesh2efo)}  DrugBank->ChEMBL: {len(db2chembl)}  "
          f"ChEMBL->target: {len(chembl2tgt)}")

    # ---- candidate pool: the full DRKG compound universe -----------------
    triplets = load_triplets()
    all_compounds = sorted(
        set(triplets["head"][triplets["head"].str.startswith("Compound::")])
        | set(triplets["tail"][triplets["tail"].str.startswith("Compound::")])
    )
    print(f"DRKG compound pool: {len(all_compounds)}")

    # popularity baseline = DRKG node degree (same as leakfree_benchmark.py)
    degree: Counter[str] = Counter()
    for h, t in zip(triplets["head"], triplets["tail"]):
        degree[h] += 1
        degree[t] += 1
    cand_deg = np.array([degree.get(c, 0) for c in all_compounds], dtype=float)
    cand_pos = {c: i for i, c in enumerate(all_compounds)}

    # ---- map every DRKG compound -> its mechanism target genes -----------
    # (built once for the whole pool so candidates are scored consistently)
    compound_targets: dict[str, set[str]] = {}
    for comp in all_compounds:
        if not comp.startswith("Compound::DB"):
            continue
        dbid = comp.split("::")[1]
        tgts: set[str] = set()
        for chembl in db2chembl.get(dbid, ()):
            tgts |= chembl2tgt.get(chembl, set())
        if tgts:
            compound_targets[comp] = tgts
    print(f"DRKG compounds with a mechanism target: {len(compound_targets)}")

    # ---- genetic-association scores for the held-out diseases ------------
    needed_efo: set[str] = set()
    for dis in by_disease:
        mesh = dis.split(":")[-1]
        for efo in mesh2efo.get(mesh, []):
            needed_efo.add(efo)
    genetic = fetch_genetic_targets(needed_efo)
    overall = load_overall_assoc()  # local fallback / comparison

    # ---- score ----------------------------------------------------------
    def run(score_source: str) -> tuple[list[float], list[float], dict]:
        ranks: list[float] = []
        pop_ranks: list[float] = []
        un = {"disease_unmapped": 0, "disease_no_gene_scores": 0,
              "drug_not_in_pool": 0, "drug_no_mechanism_target": 0,
              "drug_target_no_genetic_assoc": 0}
        for dis, drugs in by_disease.items():
            mesh = dis.split(":")[-1]
            efos = mesh2efo.get(mesh, [])
            if not efos:
                un["disease_unmapped"] += len(drugs)
                continue
            # union genetic scores across all EFO ids the MeSH term maps to
            gene_scores: dict[str, float] = {}
            for efo in efos:
                src = (genetic.get(efo, {}) if score_source == "genetic"
                       else overall.get(efo, {}))
                for g, s in src.items():
                    if s > gene_scores.get(g, 0.0):
                        gene_scores[g] = s
            if not gene_scores:
                un["disease_no_gene_scores"] += len(drugs)
                continue
            pool_scores = score_pool(gene_scores, compound_targets, all_compounds)
            for drug in drugs:
                if drug not in cand_pos:
                    un["drug_not_in_pool"] += 1
                    continue
                i = cand_pos[drug]
                if drug not in compound_targets:
                    un["drug_no_mechanism_target"] += 1
                elif pool_scores[i] == 0.0:
                    un["drug_target_no_genetic_assoc"] += 1
                # pair is still ranked: a 0-scored true drug gets a fair
                # (poor) mid-rank, exactly as leakfree_benchmark.py does.
                ranks.append(midrank(pool_scores, pool_scores[i]))
                pop_ranks.append(midrank(cand_deg, cand_deg[i]))
        return ranks, pop_ranks, un

    results: dict[str, dict] = {}

    gr, pop, gun = run("genetic")
    results["genetics_anchored (OT genetic_association)"] = summarise(gr, len(heldout))
    results["genetics_anchored (OT genetic_association)"]["unevaluable"] = gun
    results["popularity_baseline (DRKG degree, same pool)"] = summarise(pop, len(heldout))

    # comparison axis: same ranker but using the mixed overall score
    orr, _, oun = run("overall")
    results["genetics_anchored (OT overall score, comparison)"] = summarise(orr, len(heldout))
    results["genetics_anchored (OT overall score, comparison)"]["unevaluable"] = oun

    # evaluable-only subset: pairs where the true drug actually had a target
    # with a genetic association (score > 0) — shows ranker behaviour where
    # it has signal at all, vs. coverage-limited full-set numbers above.
    gr_signal: list[float] = []
    pop_signal: list[float] = []
    for dis, drugs in by_disease.items():
        mesh = dis.split(":")[-1]
        efos = mesh2efo.get(mesh, [])
        gene_scores: dict[str, float] = {}
        for efo in efos:
            for g, s in genetic.get(efo, {}).items():
                if s > gene_scores.get(g, 0.0):
                    gene_scores[g] = s
        if not gene_scores:
            continue
        pool_scores = score_pool(gene_scores, compound_targets, all_compounds)
        for drug in drugs:
            if drug in cand_pos and pool_scores[cand_pos[drug]] > 0.0:
                i = cand_pos[drug]
                gr_signal.append(midrank(pool_scores, pool_scores[i]))
                pop_signal.append(midrank(cand_deg, cand_deg[i]))
    results["genetics_anchored (subset: true drug has genetic signal)"] = \
        summarise(gr_signal, len(heldout))
    results["popularity_baseline (same signal subset)"] = \
        summarise(pop_signal, len(heldout))

    scorecard = {
        "description": "Genetics-anchored target-based ranker. Score(drug, "
                       "disease) = max over the drug's ChEMBL mechanism "
                       "target genes of the Open Targets genetic_association "
                       "score of that gene for that disease. Ranks are "
                       "tie-aware mid-ranks against the full DRKG compound "
                       "pool, identical methodology to leakfree_benchmark.py.",
        "leak_control": "Leak-free by construction: inputs are disease-gene "
                        "genetic associations and drug-target mechanism only; "
                        "no treats/indication edge is read.",
        "tier2_direction": "NOT implemented — see script footer.",
        "n_heldout_pairs": len(heldout),
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2))

    print(f"\n{'=' * 72}\nGENETICS-ANCHORED RANKER — LEAK-FREE SCORECARD\n{'=' * 72}")
    for name, s in results.items():
        if s.get("evaluable"):
            print(f"  {name}")
            print(f"    Hit@10={s['hit_at_10']}%  Hit@30={s['hit_at_30']}%  "
                  f"Hit@100={s['hit_at_100']}%  MRR={s['mrr']}  "
                  f"median={s['median_rank']}  (n={s['evaluable']}/{s['n_pairs']})")
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()

# DIRECTION CONCORDANCE (Tier 2) — NOT IMPLEMENTED, by design.
# ---------------------------------------------------------------------------
# The drug side is cheap: ChEMBL action_type.parent_type gives POSITIVE vs
# NEGATIVE MODULATOR for each mechanism (available here via the OT
# mechanismOfAction `actionType` field, already loaded into chembl2action).
# The disease side is the blocker: a leak-free, disease-specific direction of
# effect (does the disease arise from over- or under-expression of the gene?)
# is not cheaply available. The DRKG GNBR::Ud / GNBR::X edges are
# literature-mined gene-disease expression edges, not gene-direction-of-causal-
# effect, and mixing them in would reintroduce a literature-derived signal —
# the exact contamination this paradigm is meant to escape. Open Targets does
# publish a directionOfEffect field, but only for a subset of genetic evidence
# and not in the local parquet snapshot. Implementing Tier 2 honestly requires
# that data; it is therefore deferred rather than approximated.
