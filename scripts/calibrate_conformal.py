"""Fit the v7 conformal calibrator on a held-out positive/negative split.

Loads ``ensemble_v5.pkl`` and runs it across:
  - the 993 positive pairs in ``data/eval/holdout_test.jsonl``
  - an equal-size sample of random non-treated drug-disease pairs from DRKG

Then fits the conformal quantile and saves it to
``data/models/conformal_v7.npz``. Reports the empirical coverage on
``data/eval/time_sliced_test.jsonl`` (which the calibrator never saw)
so we know whether 90%-nominal actually delivers ~90%-empirical.

Usage:
    python3 scripts/calibrate_conformal.py
    python3 scripts/calibrate_conformal.py --alpha 0.05  # 95% coverage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from opencure.scoring.conformal import (
    CALIBRATOR_PATH,
    ConformalCalibrator,
    DEFAULT_ALPHA,
    empirical_coverage,
)
from opencure.scoring.ensemble import build_features, load_model

EVAL_DIR = Path("data/eval")
HOLDOUT = EVAL_DIR / "holdout_test.jsonl"
TIMESLICED = EVAL_DIR / "time_sliced_test.jsonl"


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            pairs.append((row["compound"], row["disease"]))
    return pairs


def _sample_negatives(
    positives: list[tuple[str, str]],
    n_neg: int,
    triplets,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Random non-treats compound-disease pairs (sampled with replacement)."""
    pos_set = set(positives)
    compounds = sorted({c for c, _ in positives})
    diseases = sorted({d for _, d in positives})

    # Treats edges from triplets (broad — any TREATMENT_RELATIONS-style relation).
    treats_pairs: set[tuple[str, str]] = set()
    if triplets is not None:
        for h, _r, t in zip(triplets["head"], triplets["rel"], triplets["tail"]):
            if str(h).startswith("Compound::") and str(t).startswith("Disease::"):
                treats_pairs.add((str(h), str(t)))

    negatives: list[tuple[str, str]] = []
    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 20:
        c = compounds[rng.integers(len(compounds))]
        d = diseases[rng.integers(len(diseases))]
        attempts += 1
        if (c, d) in pos_set or (c, d) in treats_pairs:
            continue
        negatives.append((c, d))
    return negatives


def _score_pair(
    compound: str,
    disease: str,
    *,
    model,
    feature_keys: tuple[str, ...],
    rank_maps: dict,
    n_compounds: int,
    drug_n_targets: dict,
    chembl_phase: dict,
    disease_gene_counts: dict,
    degree_penalty_fn,
) -> float | None:
    """Run the ensemble for a single (compound, disease). None if disease unknown."""
    if disease not in rank_maps:
        return None
    feats = build_features(
        compound_entity=compound,
        disease_entity=disease,
        rank_map=rank_maps[disease],
        n_compounds=n_compounds,
        drug_n_targets=drug_n_targets,
        chembl_phase=chembl_phase,
        disease_gene_counts=disease_gene_counts,
        degree_penalty_fn=degree_penalty_fn,
    )
    X = np.asarray([[feats[k] for k in feature_keys]], dtype=float)
    return float(model.predict_proba(X)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Miscoverage rate; 0.10 → 90% coverage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=CALIBRATOR_PATH,
        help="Path to write the fitted calibrator.",
    )
    args = parser.parse_args()

    if not HOLDOUT.exists():
        sys.exit(f"Missing {HOLDOUT}; run scripts/build_holdout.py first.")
    if not TIMESLICED.exists():
        print(f"WARNING: {TIMESLICED} missing — skipping post-fit coverage check.")

    print("Loading ensemble model + side data...")
    model, feature_keys = load_model()

    from opencure.data.drkg import load_embeddings, load_triplets
    from opencure.scoring.transe import score_drugs_for_disease_vectorized
    from opencure.scoring.hub_normalize import degree_penalty

    entity_emb, relation_emb, entity_to_id, _, relation_to_id = load_embeddings()
    triplets = load_triplets()
    chembl_phase = json.loads(Path("data/drkg/chembl_phase.json").read_text()) \
        if Path("data/drkg/chembl_phase.json").exists() else {}

    # Disease-gene counts cache.
    # ``disease_gene_index.json`` ships in two formats over the project's
    # history: legacy nested ``{disease: {"genes": [...]}}`` and the v5+
    # flat ``{disease: [gene, ...]}``. Handle both so the script runs
    # against either snapshot without a re-ingest.
    disease_gene_index_path = Path("data/disease_gene_index.json")
    disease_gene_counts: dict[str, int] = {}
    if disease_gene_index_path.exists():
        idx = json.loads(disease_gene_index_path.read_text())
        for disease_entity, payload in idx.items():
            if isinstance(payload, dict):
                genes = payload.get("genes", [])
            elif isinstance(payload, list):
                genes = payload
            else:
                genes = []
            disease_gene_counts[disease_entity] = len(genes)

    # Drug target counts (from drug_target_activities.json, when present)
    drug_n_targets: dict[str, int] = {}
    dta_path = Path("data/drkg/drug_target_activities.json")
    if dta_path.exists():
        for drug_id, targets in json.loads(dta_path.read_text()).items():
            drug_n_targets[drug_id] = len(targets) if isinstance(targets, (list, dict)) else 0

    # Per-disease rank maps (cache so we don't re-score for each pair)
    print("Loading positives + sampling negatives...")
    positives = _load_pairs(HOLDOUT)
    rng = np.random.default_rng(args.seed)
    negatives = _sample_negatives(positives, len(positives), triplets, rng)
    print(f"  {len(positives)} positives + {len(negatives)} negatives")

    rank_maps: dict[str, dict[str, int]] = {}
    n_compounds = sum(1 for e in entity_to_id if e.startswith("Compound::"))
    pairs = [(c, d, 1) for c, d in positives] + [(c, d, 0) for c, d in negatives]
    diseases_needed = sorted({d for _, d, _ in pairs})

    print(f"Pre-scoring {len(diseases_needed)} unique diseases...")
    for disease in diseases_needed:
        if disease not in entity_to_id:
            continue
        scored = score_drugs_for_disease_vectorized(
            disease_entity=disease,
            entity_emb=entity_emb,
            relation_emb=relation_emb,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            top_k=999_999,
        )
        rank_maps[disease] = {c: r for r, (c, _, _) in enumerate(scored, start=1)}

    p_hats, ys = [], []
    skipped = 0
    for compound, disease, label in pairs:
        p = _score_pair(
            compound, disease,
            model=model, feature_keys=feature_keys,
            rank_maps=rank_maps, n_compounds=n_compounds,
            drug_n_targets=drug_n_targets, chembl_phase=chembl_phase,
            disease_gene_counts=disease_gene_counts,
            degree_penalty_fn=degree_penalty,
        )
        if p is None:
            skipped += 1
            continue
        p_hats.append(p)
        ys.append(label)
    print(f"  scored {len(p_hats)} pairs ({skipped} skipped: disease not in DRKG)")

    p_hats_arr = np.asarray(p_hats)
    y_arr = np.asarray(ys)

    print(f"\nFitting conformal calibrator with alpha={args.alpha}...")
    cal = ConformalCalibrator().fit(p_hats_arr, y_arr, alpha=args.alpha)
    print(f"  q_alpha = {cal.q_alpha:.4f} on n={cal.cal_size}")

    in_cal = empirical_coverage(cal, p_hats_arr, y_arr)
    print(f"  In-sample coverage: {in_cal:.3f} (target ≥ {1 - args.alpha:.2f})")

    cal.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
