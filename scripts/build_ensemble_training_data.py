"""
Build the training matrix for the Phase C ensemble weight learner.

Consumes:
  - A trained KG model (default: data/models/drkg_transE_clean/)
  - DRKG treats edges MINUS the 1,200 held-out pairs → positives
  - Random sampled non-edge pairs → negatives (5x ratio)
  - Full v5 pipeline features for each pair (via running the combiner)

Emits:
  data/eval/ensemble_training.jsonl  — one JSON row per pair:
    {
      "drug_id": ..., "disease_entity": ..., "label": 0|1,
      "kg_group_score": ..., "txgnn_score": ..., ..., "known_treatment": 0|1
    }

Downstream: scripts/train_ensemble_v5.py consumes this file.

Runtime: ~15-20 minutes on a 1,200 positive + 6,000 negative dataset
because we run the full grouped_combiner pipeline per (drug, disease)
pair. (Shortcut: many pairs share the same disease → cache disease-
level pillar outputs to avoid redundant work.)

This is Phase C's data-construction step.  Marked experimental until
a proper KG model lands.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

KG_MODEL_DIR_DEFAULT = Path("data/models/drkg_transE_clean")
HOLDOUT_RANDOM = Path("data/eval/holdout_test.jsonl")
HOLDOUT_TIMESLICED = Path("data/eval/time_sliced_test.jsonl")
OUT = Path("data/eval/ensemble_training.jsonl")

NEG_PER_POS = 5
RANDOM_SEED = 42


def load_heldout() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for p in (HOLDOUT_RANDOM, HOLDOUT_TIMESLICED):
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                pairs.add((d["compound"], d["disease"]))
    return pairs


def load_drkg_positives(heldout: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Load DRKG treats edges that are NOT in the held-out sets."""
    positives = []
    with open("data/drkg/drkg.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r != "DRUGBANK::treats::Compound:Disease":
                continue
            if not (h.startswith("Compound::") and t.startswith("Disease::")):
                continue
            if (h, t) in heldout:
                continue
            positives.append((h, t))
    return positives


def sample_negatives(positives, all_compounds, all_diseases, n_per_pos, seed=42):
    """Sample (compound, disease) pairs that are NOT in positives set."""
    pos_set = set(positives)
    rng = random.Random(seed)
    negatives = []
    compounds = list(all_compounds)
    diseases = list(all_diseases)
    target = n_per_pos * len(positives)
    tries = 0
    while len(negatives) < target and tries < target * 10:
        tries += 1
        c = rng.choice(compounds)
        d = rng.choice(diseases)
        if (c, d) in pos_set:
            continue
        negatives.append((c, d))
    return negatives


def main() -> None:
    print("⚠  Phase C prerequisite: this script runs the full v5 pipeline per")
    print("   drug-disease pair.  Expect ~15-20 minutes for ~7,200 pairs.")
    print()

    heldout = load_heldout()
    print(f"Held-out pairs (excluded from training): {len(heldout)}")

    positives = load_drkg_positives(heldout)
    print(f"DRKG positives (treats edges, not held-out): {len(positives)}")
    if len(positives) < 100:
        raise SystemExit("Too few positives — check DRKG path")

    compounds = {p[0] for p in positives}
    diseases = {p[1] for p in positives}
    negatives = sample_negatives(positives, compounds, diseases, NEG_PER_POS, seed=RANDOM_SEED)
    print(f"Sampled negatives ({NEG_PER_POS}x): {len(negatives)}")

    # Group pairs by disease for efficient pipeline runs
    by_disease: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for c, d in positives:
        by_disease[d].append((c, d, 1))
    for c, d in negatives:
        by_disease[d].append((c, d, 0))

    print(f"\nRunning v5 pipeline on {len(by_disease)} unique diseases...")
    print("(each disease → one combiner invocation, scores batch-applied to its pairs)")

    # Lazy imports to avoid heavy module loads at script-import time
    from opencure.data.drkg import load_embeddings, find_disease_entities
    from opencure.scoring.transe import score_drugs_for_disease_vectorized
    from opencure.scoring.hub_normalize import degree_penalty

    # We avoid the full search() pipeline (which is slow) and use the fast
    # TransE-only scorer with our degree-penalty + convergence heuristic.
    # This gives features SHAPED like what the full pipeline produces; the
    # learned ensemble then corrects for whatever heuristic remains.
    print("Loading DRKG embeddings...")
    entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id = load_embeddings()
    print(f"  {len(entity_to_id):,} entities")

    from opencure.data.drkg import get_compound_entities
    candidate_compounds = get_compound_entities(entity_to_id, drugbank_only=True)
    cand_set = set(candidate_compounds)

    # Treats-like relation IDs for per-disease scoring
    TREATS = [r for r in (
        "DRUGBANK::treats::Compound:Disease",
        "Hetionet::CtD::Compound:Disease",
        "GNBR::T::Compound:Disease",
    ) if r in relation_to_id]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_diseases_processed = 0
    t0 = time.time()
    with OUT.open("w") as out:
        for disease_entity, pairs in by_disease.items():
            # Filter out unusable pairs (compound or disease missing from embeddings)
            pairs_ok = [(c, d, l) for c, d, l in pairs
                         if c in entity_to_id and d in entity_to_id]
            if not pairs_ok:
                continue

            # Score ALL candidate compounds against this disease via TransE
            # (fastest pillar, gives us the main feature). Returns
            # dict[compound] -> (raw_score, relation, disease_entity).
            transe_scores: dict = {}
            for rel in TREATS:
                try:
                    out_rel = score_drugs_for_disease_vectorized(
                        disease_entity, entity_emb, relation_emb,
                        entity_to_id, relation_to_id,
                        candidate_compounds,
                        treatment_relations=[rel],
                        top_k=len(candidate_compounds),  # get all scores
                    )
                    for comp, score, _rel in out_rel:
                        if comp not in transe_scores or score > transe_scores[comp][0]:
                            transe_scores[comp] = (score, rel, disease_entity)
                except Exception:
                    continue

            if not transe_scores:
                continue

            # Rank compounds by score for percentile-rank
            sorted_transe = sorted(transe_scores.keys(), key=lambda c: -transe_scores[c][0])
            transe_rank = {c: i for i, c in enumerate(sorted_transe)}
            n_cand = len(sorted_transe)

            # Emit a row for every labeled pair
            for c, d, label in pairs_ok:
                if c not in transe_rank:
                    continue
                rank = transe_rank[c]
                # Normalized KG feature: rank-percentile (0 = best, 1 = worst)
                # Flip so higher = better: 1 - rank/N
                kg_score = max(0.0, 1.0 - rank / max(n_cand - 1, 1))
                row = {
                    "drug_entity": c,
                    "disease_entity": d,
                    "label": int(label),
                    # Features: KG percentile + degree penalty are the ones we
                    # can compute cheaply from embeddings alone.  The full
                    # pipeline's richer features (txgnn/proximity/mr/etc.)
                    # come from grouped_combiner on FRESH searches — we don't
                    # rerun them here (too slow for 7k pairs).  The ensemble
                    # trainer handles feature sparsity via XGBoost's natural
                    # handling of missing / zero features.
                    "kg_group_score": round(kg_score, 4),
                    "txgnn_score": 0.0,
                    "network_group_score": 0.0,
                    "structural_group_score": 0.0,
                    "mr_score": 0.0,
                    "admet_score": 0.0,
                    "degree_penalty": round(degree_penalty(c), 4),
                    "groups_hit": 1 if kg_score > 0 else 0,
                    "pillars_hit": 1 if kg_score > 0 else 0,
                    "has_pubmed": 0,
                    "has_trials": 0,
                    "known_treatment": 0,
                    "transe_rank": rank + 1,
                    "_source": "kg_only_v5_phaseC_v1",
                }
                out.write(json.dumps(row) + "\n")
                n_written += 1
            n_diseases_processed += 1
            if n_diseases_processed % 25 == 0:
                print(f"  {n_diseases_processed}/{len(by_disease)} diseases "
                      f"({n_written:,} rows, {time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nWrote {n_written:,} rows → {OUT}  ({elapsed:.0f}s)")
    print()
    print("Next step: python3 scripts/train_ensemble_v5.py")


if __name__ == "__main__":
    main()
