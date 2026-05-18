"""
v7 leak-free ensemble training (WS1).

The v5 ensemble (``data/models/ensemble_v5.pkl``, CV AUROC 0.997) is
contaminated: its dominant features ``kg_score`` / ``transe_rank_log`` were
computed from a TransE model trained on the *same* DRKG ``treats`` edges used
as ensemble training positives. The model was graded on memorising its own
training graph.

This script removes the leak structurally rather than cosmetically:

  * KG features are scored with ``data/models/drkg_transE_clean`` — the
    edge-stripped model that has NEVER seen the held-out treatment edges.
  * The ensemble is trained AND tested only on pairs the clean model never
    saw: the 993-pair random held-out split (train + 5-fold CV) and the
    210-pair time-sliced split (test). Every ``kg_score`` feeding the model
    is therefore genuinely leak-free, for both train and test rows.

Outputs:
  data/models/ensemble_v7.pkl          calibrated classifier + metadata
  data/models/ensemble_v7_report.json  leak-free CV / time-sliced metrics

Whatever AUROC this produces is the honest number — it is expected to be far
below 0.997, because edge-stripped KG embeddings do not extrapolate to unseen
treatment links (see experiments/eval/v5_unified_heldout.json).

Runtime: a few minutes on CPU (no GPU needed).
"""

from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np


CLEAN_MODEL_DIR = Path("data/models/drkg_transE_clean")
RANDOM_HOLDOUT = Path("data/eval/holdout_test.jsonl")
TIME_SLICED = Path("data/eval/time_sliced_test.jsonl")
GROUND_TRUTH = Path("data/eval/ground_truth.jsonl")
ACTIVITIES_PATH = Path("data/drkg/drug_target_activities.json")
CHEMBL_PHASE_PATH = Path("data/drkg/chembl_phase.json")
OT_TRIPLETS_PATH = Path("data/open_targets/ot_triplets.tsv")

MODEL_OUT = Path("data/models/ensemble_v7.pkl")
REPORT_OUT = Path("data/models/ensemble_v7_report.json")

# Same 6 features as v5, but kg_score / transe_rank_log are now scored with
# the edge-stripped clean model. The other four never leaked.
FEATURE_KEYS = (
    "kg_score",
    "degree_penalty",
    "n_drug_targets",
    "is_fda_approved",
    "n_disease_genes",
    "transe_rank_log",
)

NEG_PER_POS = 5
SEED = 42

# DRKG treats-like relations (clean model retains the relation; only the
# specific held-out compound-disease edges were stripped).
TREATS_RELS = (
    "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease",
    "GNBR::T::Compound:Disease",
)


# ---- small feature loaders (mirrors scripts/phase_c_pipeline.py) ----------

def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.open():
        d = json.loads(line)
        pairs.append((d["compound"], d["disease"]))
    return pairs


def load_drug_target_count() -> dict[str, int]:
    if not ACTIVITIES_PATH.exists():
        return {}
    d = json.loads(ACTIVITIES_PATH.read_text())
    return {db: len(t) for db, t in d.items()}


def load_chembl_phase() -> dict[str, float]:
    if not CHEMBL_PHASE_PATH.exists():
        return {}
    return json.loads(CHEMBL_PHASE_PATH.read_text())


def count_disease_genes() -> dict[str, int]:
    if not OT_TRIPLETS_PATH.exists():
        return {}
    from collections import Counter
    c: Counter[str] = Counter()
    for line in OT_TRIPLETS_PATH.open():
        p = line.rstrip("\n").split("\t")
        if len(p) == 3 and p[1] == "OT::assoc::Gene:Disease" and p[2].startswith("Disease::"):
            c[p[2]] += 1
    return dict(c)


# ---- KG scoring with the edge-stripped clean model ------------------------

def score_kg_ranks(pairs_by_disease: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    """For each (drug, disease) pair, return the drug's 0-indexed rank among
    all compounds for the treats relation, using the CLEAN edge-stripped
    model. Pairs whose entities are absent from the model vocab are omitted.
    """
    import torch
    from pykeen.triples import TriplesFactory

    model = torch.load(CLEAN_MODEL_DIR / "trained_model.pkl",
                        map_location="cpu", weights_only=False)
    tf = TriplesFactory.from_path_binary(CLEAN_MODEL_DIR / "training_triples")
    ent2id, rel2id = tf.entity_to_id, tf.relation_to_id
    model.eval()

    rel_ids = [rel2id[r] for r in TREATS_RELS if r in rel2id]
    if not rel_ids:
        raise SystemExit("clean model vocab has no treats relation")

    compounds = [e for e in ent2id if e.startswith("Compound::")]
    cand_ids = np.array([ent2id[c] for c in compounds], dtype=np.int64)
    n_cand = len(compounds)

    ranks: dict[tuple[str, str], int] = {}
    with torch.no_grad():
        for dis, drugs in pairs_by_disease.items():
            if dis not in ent2id:
                continue
            s_id = ent2id[dis]
            # best treats-rel score for every candidate compound
            best = np.full(n_cand, -1e9)
            for rid in rel_ids:
                heads = torch.as_tensor(cand_ids)
                rels = torch.full_like(heads, rid)
                tails = torch.full_like(heads, s_id)
                batch = torch.stack([heads, rels, tails], dim=1)
                sc = model.score_hrt(batch).squeeze(-1).cpu().numpy()
                best = np.maximum(best, sc)
            for drug in drugs:
                if drug not in ent2id:
                    continue
                d_id = ent2id[drug]
                drug_score = -1e9
                for rid in rel_ids:
                    t = torch.as_tensor([[d_id, rid, s_id]])
                    drug_score = max(drug_score, model.score_hrt(t).item())
                ranks[(drug, dis)] = int(np.sum(best > drug_score))
    return ranks, n_cand, compounds


# ---- feature assembly -----------------------------------------------------

def build_features(pairs: list[tuple[str, str, int]],
                    ranks: dict[tuple[str, str], int],
                    n_cand: int,
                    drug_n_targets: dict,
                    chembl_phase: dict,
                    disease_genes: dict) -> tuple[np.ndarray, np.ndarray]:
    from opencure.scoring.hub_normalize import degree_penalty

    X, y = [], []
    for drug, dis, label in pairs:
        if (drug, dis) not in ranks:
            continue
        rk = ranks[(drug, dis)]
        kg_score = max(0.0, 1.0 - rk / max(n_cand - 1, 1))
        bare = drug.split("::", 1)[1] if "::" in drug else drug
        phase = chembl_phase.get(bare, 0) or 0
        try:
            is_fda = 1 if float(phase) >= 2 else 0
        except (TypeError, ValueError):
            is_fda = 0
        X.append([
            kg_score,
            degree_penalty(drug),
            drug_n_targets.get(bare, 0),
            is_fda,
            disease_genes.get(dis, 0),
            np.log1p(rk),
        ])
        y.append(label)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def make_split(positives: list[tuple[str, str]],
               drug_pool: list[str],
               forbidden: set[tuple[str, str]],
               rng: random.Random) -> list[tuple[str, str, int]]:
    """Positives + NEG_PER_POS HARD negatives.

    A hard negative is (real approved drug, disease) where the drug treats
    *some other* disease but not this one — drawn from ``drug_pool`` (drugs
    that appear as a treatment head in the ground truth). This is far harder
    than random-compound negatives (mostly obscure research chemicals with no
    graph presence) and yields an honest, non-inflated AUROC.
    """
    rows = [(c, d, 1) for c, d in positives]
    diseases = list({d for _, d in positives})
    n_target = NEG_PER_POS * len(positives)
    tries = 0
    while sum(1 for r in rows if r[2] == 0) < n_target and tries < n_target * 40:
        tries += 1
        c = rng.choice(drug_pool)
        d = rng.choice(diseases)
        if (c, d) in forbidden:
            continue
        rows.append((c, d, 0))
    return rows


def main() -> None:
    if not (CLEAN_MODEL_DIR / "trained_model.pkl").exists():
        raise SystemExit(f"Missing clean model: {CLEAN_MODEL_DIR}/trained_model.pkl")

    rng = random.Random(SEED)
    t0 = time.time()

    random_pos = load_pairs(RANDOM_HOLDOUT)
    time_pos = load_pairs(TIME_SLICED)
    ground_truth = load_pairs(GROUND_TRUTH)
    forbidden = set(ground_truth) | set(random_pos) | set(time_pos)
    # Hard-negative drug pool: every drug that treats *something* in the
    # ground truth — i.e. real approved drugs, not obscure research chemicals.
    drug_pool = sorted({c for c, _ in ground_truth + random_pos + time_pos})
    print(f"Held-out positives: {len(random_pos)} random, {len(time_pos)} time-sliced")
    print(f"Forbidden (known treatment) pairs: {len(forbidden)}")
    print(f"Hard-negative drug pool: {len(drug_pool)} real treatment drugs")

    drug_n_targets = load_drug_target_count()
    chembl_phase = load_chembl_phase()
    disease_genes = count_disease_genes()
    print(f"Feature helpers: {len(drug_n_targets)} drugs w/ targets, "
          f"{len(chembl_phase)} w/ phase, {len(disease_genes)} diseases w/ gene counts")

    # KG ranks first (also gives us the candidate-compound pool).
    by_disease: dict[str, list[str]] = {}
    for c, d in random_pos + time_pos:
        by_disease.setdefault(d, []).append(c)
    print(f"Scoring KG ranks for {len(by_disease)} diseases with clean model...")
    ranks, n_cand, _ = score_kg_ranks(by_disease)
    print(f"  scored {len(ranks)} (drug,disease) pairs against {n_cand} compounds "
          f"({time.time()-t0:.0f}s)")

    # Negatives also need KG ranks → score them in a second pass.
    train_rows = make_split(random_pos, drug_pool, forbidden, rng)
    test_rows = make_split(time_pos, drug_pool, forbidden, rng)
    neg_by_disease: dict[str, list[str]] = {}
    for c, d, lab in train_rows + test_rows:
        if lab == 0:
            neg_by_disease.setdefault(d, []).append(c)
    print(f"Scoring KG ranks for {sum(len(v) for v in neg_by_disease.values())} negatives...")
    neg_ranks, _, _ = score_kg_ranks(neg_by_disease)
    ranks.update(neg_ranks)

    X_tr, y_tr = build_features(train_rows, ranks, n_cand,
                                drug_n_targets, chembl_phase, disease_genes)
    X_te, y_te = build_features(test_rows, ranks, n_cand,
                                drug_n_targets, chembl_phase, disease_genes)
    print(f"Train: {len(X_tr)} rows ({int(y_tr.sum())} pos) | "
          f"Test (time-sliced): {len(X_te)} rows ({int(y_te.sum())} pos)")

    try:
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:
        raise SystemExit(f"Missing dep: {e} — pip install xgboost scikit-learn")

    def new_model() -> "xgb.XGBClassifier":
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=SEED,
        )

    # 5-fold CV on the random held-out split (leak-free).
    print("\n5-fold CV on random held-out split (leak-free features)...")
    cv_auc, cv_ap = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(X_tr, y_tr)):
        m = new_model()
        m.fit(X_tr[tr], y_tr[tr])
        p = m.predict_proba(X_tr[te])[:, 1]
        cv_auc.append(roc_auc_score(y_tr[te], p))
        cv_ap.append(average_precision_score(y_tr[te], p))
        print(f"  fold {fi+1}: AUC={cv_auc[-1]:.4f}  AP={cv_ap[-1]:.4f}")
    cv_auc_mean, cv_auc_std = float(np.mean(cv_auc)), float(np.std(cv_auc))
    cv_ap_mean = float(np.mean(cv_ap))

    # Train on the full random split, evaluate on the time-sliced split.
    base = new_model()
    base.fit(X_tr, y_tr)
    importances = dict(zip(FEATURE_KEYS, base.feature_importances_.tolist()))
    p_te = base.predict_proba(X_te)[:, 1]
    ts_auc = float(roc_auc_score(y_te, p_te))
    ts_ap = float(average_precision_score(y_te, p_te))

    calibrated = CalibratedClassifierCV(new_model(), method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump({
            "model": calibrated,
            "feature_keys": FEATURE_KEYS,
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "time_sliced_auc": ts_auc,
            "seed": SEED,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "leak_free": True,
            "kg_model": str(CLEAN_MODEL_DIR),
        }, f)

    report = {
        "description": "v7 leak-free ensemble. KG features scored with the "
                       "edge-stripped clean model; trained and tested only on "
                       "held-out pairs the clean model never saw.",
        "n_train": len(X_tr),
        "n_train_positives": int(y_tr.sum()),
        "n_test": len(X_te),
        "n_test_positives": int(y_te.sum()),
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "cv_ap_mean": cv_ap_mean,
        "time_sliced_auc": ts_auc,
        "time_sliced_ap": ts_ap,
        "feature_importances": importances,
        "feature_keys": list(FEATURE_KEYS),
        "v5_contaminated_cv_auc": 0.997,
        "honest_note": "v5's 0.997 CV AUROC was inflated by KG-feature leakage. "
                       "These numbers are the leak-free replacement.",
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print("v7 LEAK-FREE ENSEMBLE")
    print(f"{'='*60}")
    print(f"  CV AUROC (random held-out):  {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    print(f"  Time-sliced AUROC (held-out): {ts_auc:.4f}")
    print(f"  (v5 contaminated CV AUROC was 0.997)")
    print("\n  Feature importances:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22s} {v:.4f}  {'#' * int(v * 40)}")
    print(f"\n  Saved: {MODEL_OUT}")
    print(f"  Saved: {REPORT_OUT}")
    print(f"  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
