"""
Leak-free per-pillar held-out benchmark — the OpenCure verification instrument.

WHY THIS EXISTS
---------------
Every previous OpenCure "performance" number was contaminated: pillars were
scored from a knowledge graph that still contained the test edges. This script
is the honest replacement. It measures each *timeless* pillar — pillars whose
data is a fixed physical/measured property, not literature-derived — against a
held-out set, with the one possible leak path explicitly closed.

LEAK CONTROL
------------
A timeless similarity pillar scores a drug by similarity to a disease's *known
treatments*. The only way it can leak is if the held-out drug is itself in that
anchor set (it would then be "similar to itself"). `get_known_treatments` reads
anchors from TREATMENT_RELATIONS edges. So we strip, from the triplets the
pillar sees, every held-out (drug, disease) row in those relations. After
stripping, the held-out drug is a rankable candidate scored only against the
disease's *other* known treatments — genuinely leak-free.

PILLARS MEASURED
----------------
- molecular_embedding : ChemBERTa/MoLFormer chemical-structure similarity.
- jump_cell_painting  : JUMP-CP morphological similarity (phenotype space) —
  the genuinely orthogonal pillar; morphology is a measured observable, not
  derived from the literature.
Each is compared against a popularity (graph-degree) baseline computed on the
*same* candidate pool. A pillar earns its place only by beating that baseline.

OUTPUT
------
experiments/eval/leakfree_pillar_scorecard.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOLDOUT = Path("data/eval/holdout_test.jsonl")
JUMP_NPZ = Path("data/jump_cp/drkg_jump_profiles.npz")
OUT = Path("experiments/eval/leakfree_pillar_scorecard.json")


def load_holdout() -> list[tuple[str, str]]:
    return [(d["compound"], d["disease"])
            for d in (json.loads(l) for l in HOLDOUT.open())]


def midrank(scores: np.ndarray, target_score: float) -> float:
    """Standard tie-aware rank (1-indexed): higher score = better rank."""
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


def score_anchored_pillar(embeddings, entities, by_disease, stripped_triplets,
                          get_known_treatments, compute_cosine, degree):
    """Leak-free anchored-similarity scoring of one timeless pillar.

    For each held-out (drug, disease): rank the drug among all candidates by
    max cosine similarity to the disease's leak-controlled known treatments.
    Returns (pillar_ranks, popularity_ranks_same_pool, unevaluable).
    """
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    all_compounds = [e for e in entities if e.startswith("Compound::")]
    pillar_ranks: list[float] = []
    pop_ranks: list[float] = []
    unevaluable = {"no_anchor": 0, "drug_no_embedding": 0}

    for dis, drugs in by_disease.items():
        known = set(get_known_treatments(dis, stripped_triplets))
        known_with_emb = [c for c in known if c in entity_to_idx]
        if not known_with_emb:
            unevaluable["no_anchor"] += len(drugs)
            continue
        known_embs = embeddings[[entity_to_idx[c] for c in known_with_emb]]
        candidates = [c for c in all_compounds if c not in known]
        cand_embs = embeddings[[entity_to_idx[c] for c in candidates]]
        sim = compute_cosine(cand_embs, known_embs).max(axis=1)
        cand_deg = np.array([degree.get(c, 0) for c in candidates], dtype=float)
        cand_pos = {c: i for i, c in enumerate(candidates)}
        for drug in drugs:
            if drug not in cand_pos:
                unevaluable["drug_no_embedding"] += 1
                continue
            i = cand_pos[drug]
            pillar_ranks.append(midrank(sim, sim[i]))
            pop_ranks.append(midrank(cand_deg, cand_deg[i]))
    return pillar_ranks, pop_ranks, unevaluable


def score_absolute_pillar(score_fn, by_disease, all_compounds, degree):
    """Leak-free ranking of a held-out drug by an ABSOLUTE per-(drug,disease)
    score (not anchored similarity).

    Unlike `score_anchored_pillar`, an absolute pillar scores a (drug, disease)
    pair directly — it does NOT read a disease's known-treatment anchor set, so
    there is no anchor to strip. Leak control for these pillars is by
    construction of their inputs (drug->gene and gene->disease only; never a
    treats/indication edge), argued at each call site.

    For each held-out (drug, disease): score EVERY candidate in `all_compounds`
    via `score_fn(disease, all_compounds) -> {entity: float}`, give unscored
    candidates 0.0 (so the pillar pays honestly for its coverage gap), then
    tie-aware mid-rank the held-out drug. Pair each pillar rank with the
    popularity rank on the SAME pool.

    Returns (pillar_ranks, pop_ranks, unevaluable, pool_coverage_pct).
    """
    cand_deg = np.array([degree.get(c, 0) for c in all_compounds], dtype=float)
    pos = {c: i for i, c in enumerate(all_compounds)}
    pillar_ranks: list[float] = []
    pop_ranks: list[float] = []
    unevaluable = {"disease_no_score": 0, "drug_not_in_pool": 0}
    coverage_fracs: list[float] = []

    for dis, drugs in by_disease.items():
        scores_map = score_fn(dis, all_compounds)
        if not scores_map:
            unevaluable["disease_no_score"] += len(drugs)
            continue
        cand_scores = np.array([scores_map.get(c, 0.0) for c in all_compounds],
                               dtype=float)
        coverage_fracs.append(float(np.mean(cand_scores > 0)))
        for drug in drugs:
            if drug not in pos:
                unevaluable["drug_not_in_pool"] += 1
                continue
            i = pos[drug]
            pillar_ranks.append(midrank(cand_scores, cand_scores[i]))
            pop_ranks.append(midrank(cand_deg, cand_deg[i]))
    cov = round(100 * float(np.mean(coverage_fracs)), 2) if coverage_fracs else 0.0
    return pillar_ranks, pop_ranks, unevaluable, cov


def main() -> None:
    from opencure.config import TREATMENT_RELATIONS
    from opencure.data.drkg import load_triplets
    from opencure.scoring.molecular import get_known_treatments
    from opencure.scoring.molecular_embeddings import (
        compute_cosine_similarity,
        load_best_molecular_embeddings,
    )

    heldout = load_holdout()
    by_disease: dict[str, list[str]] = defaultdict(list)
    for drug, dis in heldout:
        by_disease[dis].append(drug)
    print(f"Held-out pairs: {len(heldout)} across {len(by_disease)} diseases")

    # ---- leak control: strip held-out treatment edges -------------------
    triplets = load_triplets()
    heldout_set = set(heldout)
    treat_set = set(TREATMENT_RELATIONS)
    is_heldout_pair = np.array([(h, t) in heldout_set
                                for h, t in zip(triplets["head"], triplets["tail"])])
    is_treat = triplets["relation"].isin(treat_set).to_numpy()
    strip_mask = is_heldout_pair & is_treat
    stripped = triplets[~strip_mask].reset_index(drop=True)
    print(f"Stripped {int(strip_mask.sum())} held-out treatment edges")

    degree: Counter[str] = Counter()
    for h, t in zip(triplets["head"], triplets["tail"]):
        degree[h] += 1
        degree[t] += 1

    pillars: dict[str, dict] = {}

    # ---- pillar 1: molecular-embedding (chemical structure) -------------
    emb, ents, model_name = load_best_molecular_embeddings()
    if emb is not None:
        pr, pop, un = score_anchored_pillar(
            emb, ents, by_disease, stripped, get_known_treatments,
            compute_cosine_similarity, degree)
        pillars[f"molecular_embedding ({model_name})"] = summarise(pr, len(heldout))
        pillars["popularity_baseline (vs molecular pool)"] = summarise(pop, len(heldout))
        pillars[f"molecular_embedding ({model_name})"]["unevaluable"] = un
        print(f"molecular_embedding: {len(pr)} evaluable")

    # ---- pillar 2: JUMP Cell Painting (morphology) ----------------------
    if JUMP_NPZ.exists():
        d = np.load(str(JUMP_NPZ), allow_pickle=True)
        jump_ents = d["entities"].tolist()
        jump_emb = d["embeddings"]
        pr, pop, un = score_anchored_pillar(
            jump_emb, jump_ents, by_disease, stripped, get_known_treatments,
            compute_cosine_similarity, degree)
        pillars["jump_cell_painting"] = summarise(pr, len(heldout))
        pillars["popularity_baseline (vs JUMP pool)"] = summarise(pop, len(heldout))
        pillars["jump_cell_painting"]["unevaluable"] = un
        print(f"jump_cell_painting: {len(jump_ents)} compounds, {len(pr)} evaluable")
    else:
        print(f"jump_cell_painting: SKIPPED — {JUMP_NPZ} not built")

    # ---- pillars 3-4: mechanistic_reversal (non-anchored, ABSOLUTE) -----
    # These pillars score a (drug, disease) pair as the affinity-weighted
    # overlap between the drug's measured ChEMBL bioactivity (drug->gene) and
    # the disease's causal genes (gene->disease). They never read a
    # treats/indication edge, so there is no known-treatment anchor to strip —
    # leak control is by construction of the inputs. Two configs:
    #   * all-datasource : disease genes from OT associationByOverallDirect,
    #     which MIXES chembl/known_drug -> leak-SUSPECT (reported for contrast).
    #   * genetics-filtered : disease genes from OT GENETICS datasources only
    #     (GWAS / rare-variant / clinical-genetics) -> leak-CLEAN. Only this
    #     number is claimable.
    from opencure.scoring.mechanistic_reversal import (
        score_mechanistic_reversal,
        score_mechanistic_reversal_genetics,
    )

    all_compounds_full = sorted(c for c in degree if c.startswith("Compound::"))
    print(f"Full DRKG compound pool: {len(all_compounds_full)}")

    def _mr_adaptor(fn):
        def inner(dis, cands):
            return {c: v[0] for c, v in fn([dis], cands, top_k=len(cands)).items()}
        return inner

    for label, fn in [
        ("mechanistic_reversal (all-datasource, leak-suspect)",
         score_mechanistic_reversal),
        ("mechanistic_reversal (genetics-filtered, leak-clean)",
         score_mechanistic_reversal_genetics),
    ]:
        pr, pop, un, cov = score_absolute_pillar(
            _mr_adaptor(fn), by_disease, all_compounds_full, degree)
        pillars[label] = summarise(pr, len(heldout))
        pillars[label]["unevaluable"] = un
        pillars[label]["pool_coverage_pct"] = cov
        pillars[f"popularity_baseline (vs {label})"] = summarise(pop, len(heldout))
        print(f"{label}: {len(pr)} evaluable, pool_coverage={cov}%")

    scorecard = {
        "description": "Leak-free per-pillar held-out benchmark. Held-out "
                       "treatment edges stripped from anchor sets; ranks are "
                       "tie-aware mid-ranks against the full candidate pool. "
                       "A pillar earns its place only by beating the "
                       "popularity baseline on its own candidate pool.",
        "n_heldout_pairs": len(heldout),
        "pillars": pillars,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2))

    print(f"\n{'='*70}\nLEAK-FREE PILLAR SCORECARD\n{'='*70}")
    for name, s in pillars.items():
        if s.get("evaluable"):
            print(f"  {name}")
            print(f"    Hit@10={s['hit_at_10']}%  Hit@30={s['hit_at_30']}%  "
                  f"Hit@100={s['hit_at_100']}%  MRR={s['mrr']}  "
                  f"median={s['median_rank']}  (n={s['evaluable']}/{s['n_pairs']})")
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
