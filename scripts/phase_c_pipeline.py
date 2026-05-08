"""
Phase C end-to-end: feature extraction → XGBoost ensemble → calibrated model.

Pipeline in one script (no state between calls):
  1. Load DRKG treats positives (not held-out)
  2. Sample 5x negatives (compound, disease) pairs
  3. For each pair, compute lightweight features:
       - KG percentile via TransE (DRKG, already-cached)
       - degree_penalty (hub damping)
       - n_drug_targets (ChEMBL-measured activity targets)
       - is_fda_approved (ChEMBL phase >= 2)
       - n_disease_associated_genes (OT count)
  4. Train XGBoost with 5-fold CV + isotonic calibration
  5. Save model to data/models/ensemble_v5.pkl
  6. Emit feature importances + AUC-ROC to data/models/ensemble_v5_report.json

Runtime: ~10-30s for feature extraction on 24k pairs;
         ~1-2 min for XGBoost + calibration.
"""

from __future__ import annotations

import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np


HOLDOUT_PATHS = [Path("data/eval/holdout_test.jsonl"),
                 Path("data/eval/time_sliced_test.jsonl")]
DRKG_PATH = Path("data/drkg/drkg.tsv")
ACTIVITIES_PATH = Path("data/drkg/drug_target_activities.json")
CHEMBL_PHASE_PATH = Path("data/drkg/chembl_phase.json")
OT_TRIPLETS_PATH = Path("data/open_targets/ot_triplets.tsv")

MODEL_OUT = Path("data/models/ensemble_v5.pkl")
REPORT_OUT = Path("data/models/ensemble_v5_report.json")

NEG_PER_POS = 5
SEED = 42

FEATURE_KEYS = (
    "kg_score",
    "degree_penalty",
    "n_drug_targets",
    "is_fda_approved",
    "n_disease_genes",
    "transe_rank_log",
)


def load_heldout() -> set:
    s = set()
    for p in HOLDOUT_PATHS:
        if not p.exists():
            continue
        for line in p.open():
            d = json.loads(line)
            s.add((d["compound"], d["disease"]))
    return s


def load_drug_target_count() -> dict[str, int]:
    if not ACTIVITIES_PATH.exists():
        return {}
    d = json.loads(ACTIVITIES_PATH.read_text())
    return {db_id: len(targets) for db_id, targets in d.items()}


def load_chembl_phase() -> dict[str, float]:
    if not CHEMBL_PHASE_PATH.exists():
        return {}
    return json.loads(CHEMBL_PHASE_PATH.read_text())


def count_disease_associated_genes() -> dict[str, int]:
    """Disease entity → count of OT gene-disease edges (confidence > 0.3)."""
    if not OT_TRIPLETS_PATH.exists():
        return {}
    counts: Counter[str] = Counter()
    for line in OT_TRIPLETS_PATH.open():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        h, r, t = parts
        if r == "OT::assoc::Gene:Disease" and t.startswith("Disease::"):
            counts[t] += 1
    return dict(counts)


def main() -> None:
    from opencure.data.drkg import load_embeddings, get_compound_entities
    from opencure.scoring.transe import score_drugs_for_disease_vectorized
    from opencure.scoring.hub_normalize import degree_penalty

    rng = random.Random(SEED)

    print("Loading DRKG embeddings...")
    ent_emb, rel_emb, ent2id, id2ent, rel2id = load_embeddings()
    compounds = get_compound_entities(ent2id, drugbank_only=True)
    print(f"  {len(ent2id):,} entities, {len(compounds):,} candidate compounds")

    heldout = load_heldout()
    print(f"  Held-out pairs (excluded): {len(heldout)}")

    print("Loading feature helpers...")
    drug_n_targets = load_drug_target_count()
    chembl_phase = load_chembl_phase()
    disease_gene_counts = count_disease_associated_genes()
    print(f"  ChEMBL targets for {len(drug_n_targets):,} drugs, "
          f"phase for {len(chembl_phase):,} drugs, "
          f"OT gene counts for {len(disease_gene_counts):,} diseases")

    # Positives: DRUGBANK::treats edges not in held-out
    positives = []
    for line in DRKG_PATH.open():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        if parts[1] != "DRUGBANK::treats::Compound:Disease":
            continue
        if (parts[0], parts[2]) in heldout:
            continue
        if parts[0] in ent2id and parts[2] in ent2id:
            positives.append((parts[0], parts[2]))
    print(f"  Positives (not held-out): {len(positives):,}")

    # Negatives: 5x random non-positive pairs
    pos_set = set(positives)
    diseases_with_positives = list({p[1] for p in positives})
    negatives = []
    target_n_neg = NEG_PER_POS * len(positives)
    tries = 0
    while len(negatives) < target_n_neg and tries < target_n_neg * 10:
        tries += 1
        c = rng.choice(compounds)
        d = rng.choice(diseases_with_positives)
        if (c, d) in pos_set or (c, d) in heldout:
            continue
        negatives.append((c, d))
    print(f"  Negatives: {len(negatives):,}")

    # Group by disease for efficient TransE scoring (one pass per disease)
    by_dis: dict[str, list] = {}
    for c, d in positives:
        by_dis.setdefault(d, []).append((c, d, 1))
    for c, d in negatives:
        by_dis.setdefault(d, []).append((c, d, 0))
    print(f"  Unique diseases in training: {len(by_dis):,}")

    print("\nExtracting features...")
    t0 = time.time()
    rels = ["DRUGBANK::treats::Compound:Disease"]
    X, y, row_diseases = [], [], []  # row_diseases parallels X/y for per-class slicing
    for i, (dis, pairs) in enumerate(by_dis.items()):
        try:
            ranked = score_drugs_for_disease_vectorized(
                dis, ent_emb, rel_emb, ent2id, rel2id, compounds, rels,
            )
        except Exception:
            continue
        rank_map = {c: rk for rk, (c, _, _) in enumerate(ranked)}
        n = len(ranked)
        n_disease_genes = disease_gene_counts.get(dis, 0)
        for c, _, label in pairs:
            rk = rank_map.get(c, n)
            kg_score = max(0.0, 1.0 - rk / max(n - 1, 1))
            bare = c.split("::", 1)[1] if "::" in c else c
            phase = chembl_phase.get(bare, 0) or 0
            try:
                is_fda = 1 if float(phase) >= 2 else 0
            except (TypeError, ValueError):
                is_fda = 0
            X.append([
                kg_score,
                degree_penalty(c),
                drug_n_targets.get(bare, 0),
                is_fda,
                n_disease_genes,
                np.log1p(rk),
            ])
            y.append(label)
            row_diseases.append(dis)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(by_dis)} diseases processed "
                  f"({len(X):,} rows, {time.time()-t0:.0f}s)")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print(f"  Final: {len(X):,} rows × {X.shape[1]} features in {time.time()-t0:.0f}s")
    print(f"    positives: {int(y.sum()):,} / negatives: {int((1 - y).sum()):,}")

    # Train XGBoost with 5-fold CV for AUC-ROC
    print("\nTraining XGBoost...")
    try:
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:
        print(f"Missing dep: {e}")
        print("Install: pip install xgboost scikit-learn")
        return

    auc_scores, ap_scores = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=SEED,
        )
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        ap = average_precision_score(y[te], p)
        auc_scores.append(auc)
        ap_scores.append(ap)
        print(f"  fold {fi + 1}: AUC={auc:.4f}  AP={ap:.4f}")

    mean_auc = float(np.mean(auc_scores))
    mean_ap = float(np.mean(ap_scores))
    std_auc = float(np.std(auc_scores))
    print(f"\n5-fold CV: AUC-ROC = {mean_auc:.4f} ± {std_auc:.4f}  AP = {mean_ap:.4f}")

    # Final calibrated model on all data
    print("\nTraining final calibrated ensemble...")
    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=SEED,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X, y)

    # Standalone fit for feature importances
    base.fit(X, y)
    importances = dict(zip(FEATURE_KEYS, base.feature_importances_.tolist()))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump({
            "model": calibrated,
            "feature_keys": FEATURE_KEYS,
            "cv_auc_mean": mean_auc,
            "cv_auc_std": std_auc,
            "cv_ap_mean": mean_ap,
            "seed": SEED,
            "n_samples": len(X),
            "n_positives": int(y.sum()),
        }, f)

    REPORT_OUT.write_text(json.dumps({
        "n_samples": len(X),
        "n_positives": int(y.sum()),
        "n_negatives": int((1 - y).sum()),
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "cv_ap_mean": mean_ap,
        "feature_importances": importances,
        "feature_keys": list(FEATURE_KEYS),
    }, indent=2))

    print(f"\nSaved: {MODEL_OUT}")
    print(f"Saved: {REPORT_OUT}")
    print("\nFeature importances:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(v * 30)
        print(f"  {k:<22s} {v:.4f}  {bar}")

    # ---- v7: per-class heads on top of the shared ensemble --------------
    # Each class gets a small logistic head so the routing layer
    # (opencure/scoring/per_class_ensemble.py) can pick a specialised
    # model when scoring a disease. Skips classes with too few positives.
    print("\nTraining per-class heads (v7)...")
    try:
        from opencure.scoring.per_class_train import (
            collect_disease_names, train_per_class_heads,
        )
    except Exception as exc:
        print(f"  [INFO] Per-class training unavailable: {exc}")
    else:
        entity_to_name = collect_disease_names(ent2id)
        if not entity_to_name:
            print("  [INFO] No disease entity → name map; skipping per-class heads")
        else:
            summary = train_per_class_heads(
                X, y, row_diseases,
                entity_to_name=entity_to_name,
                feature_keys=FEATURE_KEYS,
                seed=SEED,
            )
            for cls, info in summary.items():
                print(f"  saved {info['path']} "
                      f"({info['n_pos']} pos / {info['n_neg']} neg)")


if __name__ == "__main__":
    main()
