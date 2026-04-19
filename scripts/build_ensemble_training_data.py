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
    from opencure.search import _get_data
    from opencure.scoring.pillar_groups import (
        group_kg_scores, group_structural_scores, group_network_scores, build_feature_matrix,
    )
    from opencure.scoring.grouped_combiner import combine_grouped_scores

    data = _get_data()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    t0 = time.time()
    with OUT.open("w") as out:
        # TODO: the full pipeline wiring below is a placeholder —
        # for Phase C we need a compact version of search.py that
        # takes (disease_entity, candidate_set) and returns per-pair
        # features.  That's a ~200-line extraction task; for this
        # scaffold we emit the inputs and leave the actual feature
        # computation to a follow-up commit when DRKG-clean training
        # completes and we can verify the loop end-to-end.
        for disease_entity, pairs in list(by_disease.items())[:3]:  # pilot 3 diseases
            # ... would run the grouped pipeline here ...
            # For now: emit a stub row per pair so the file shape is
            # correct and train_ensemble_v5 can smoke-test.
            for c, d, label in pairs:
                stub = {
                    "drug_entity": c, "disease_entity": d, "label": int(label),
                    "kg_group_score": 0.0,  # TODO fill from combiner
                    "txgnn_score": 0.0,
                    "network_group_score": 0.0,
                    "structural_group_score": 0.0,
                    "mr_score": 0.0,
                    "admet_score": 0.0,
                    "degree_penalty": 1.0,
                    "groups_hit": 0,
                    "pillars_hit": 0,
                    "has_pubmed": 0,
                    "has_trials": 0,
                    "known_treatment": 0,
                    "_note": "stub — Phase C feature extraction pending",
                }
                out.write(json.dumps(stub) + "\n")
                n_written += 1

    elapsed = time.time() - t0
    print(f"\nWrote {n_written:,} stub rows → {OUT}  ({elapsed:.0f}s)")
    print()
    print("Next step: replace stub loop with real feature extraction once")
    print("data/models/drkg_transE_clean/trained_model.pkl exists.")


if __name__ == "__main__":
    main()
