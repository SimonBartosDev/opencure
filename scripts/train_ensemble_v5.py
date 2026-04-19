"""
Train a calibrated XGBoost ensemble over the 12 pillar group features.

Replaces the hand-guessed EFFICACY_GROUPS weights in grouped_combiner.py
with data-learned weights via 5-fold CV, plus isotonic calibration so
combined_score=0.7 actually corresponds to ~70% precision on held-out
drug-disease positive pairs.

Training data construction:
  Positives: DRKG + PrimeKG + OT treats-like edges that are NOT in the
             holdout sets (993 random + 210 time-sliced = 1,200 stripped)
  Negatives: sampled (compound, disease) pairs that are not any treats-
             relation in the training graph, 5x per positive

Features per pair (built from grouped_combiner output):
  kg_group_score, txgnn_score, network_group_score, structural_group_score,
  mr_score, admet_score, degree_penalty, groups_hit, pillars_hit,
  has_pubmed (binary), has_trials (binary), known_treatment (binary)

Output: data/models/ensemble_v5.pkl containing the calibrated model.
        data/models/ensemble_v5_report.json with AUC-ROC, feature
        importances, cross-validated precision@10.

This script is stubbed until a proper trained KG model produces the
per-pair feature matrix. Running it without the KG requires feature
matrix construction first:

    python3 scripts/build_ensemble_training_data.py
    python3 scripts/train_ensemble_v5.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


TRAINING_DATA_PATH = Path("data/eval/ensemble_training.jsonl")
MODEL_OUT = Path("data/models/ensemble_v5.pkl")
REPORT_OUT = Path("data/models/ensemble_v5_report.json")

FEATURE_KEYS = (
    "kg_group_score",
    "txgnn_score",
    "network_group_score",
    "structural_group_score",
    "mr_score",
    "admet_score",
    "degree_penalty",
    "groups_hit",
    "pillars_hit",
    "has_pubmed",
    "has_trials",
    "known_treatment",
)


def main() -> None:
    if not TRAINING_DATA_PATH.exists():
        print(f"Training data missing: {TRAINING_DATA_PATH}")
        print()
        print("Build it first:")
        print("  python3 scripts/build_ensemble_training_data.py")
        print()
        print("Or — simpler fallback while we don't yet have a proper trained KG —")
        print("keep the hand-weighted grouped_combiner.py (current behavior).")
        sys.exit(2)

    try:
        import numpy as np
        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        import xgboost as xgb
    except ImportError as e:
        print(f"Missing dep: {e}")
        print("Install: pip install xgboost scikit-learn pandas numpy")
        sys.exit(1)

    print(f"Loading {TRAINING_DATA_PATH}...")
    rows = [json.loads(line) for line in TRAINING_DATA_PATH.open()]
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} pairs  ({int(df['label'].sum())} positives, "
          f"{int((1 - df['label']).sum())} negatives)")

    X = df[list(FEATURE_KEYS)].fillna(0).values
    y = df["label"].values

    # 5-fold CV + stratified AUC-ROC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        auc_scores.append(auc)
        print(f"  fold {fold_idx + 1}: AUC-ROC = {auc:.4f}")

    mean_auc = float(np.mean(auc_scores))
    print(f"\n5-fold CV AUC-ROC: {mean_auc:.4f} ± {np.std(auc_scores):.4f}")

    # Train final calibrated model on all data
    print("\nTraining final calibrated model...")
    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        n_jobs=-1, random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X, y)

    # Feature importances via a standalone xgb for interpretability
    base.fit(X, y)
    importances = dict(zip(FEATURE_KEYS, base.feature_importances_.tolist()))

    import pickle
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump({"model": calibrated, "feature_keys": FEATURE_KEYS,
                     "mean_auc": mean_auc}, f)

    REPORT_OUT.write_text(json.dumps({
        "n_samples": len(df),
        "n_positives": int(df["label"].sum()),
        "cv_auc_mean": mean_auc,
        "cv_auc_std": float(np.std(auc_scores)),
        "feature_importances": importances,
    }, indent=2))

    print(f"\nSaved: {MODEL_OUT}")
    print(f"Saved: {REPORT_OUT}")
    print()
    print("Top-5 feature importances:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {k:<28s} {v:.4f}")


if __name__ == "__main__":
    main()
