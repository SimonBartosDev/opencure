"""Train per-class ensemble heads for the v7 routing layer.

Used by ``scripts/phase_c_pipeline.py`` after the shared head finishes
training. For each of the six classes defined in
``opencure/eval/disease_classes.yaml`` we:

1. Filter the training matrix to rows whose disease entity resolves to
   that class (via DRKG name lookup).
2. Train a logistic-regression head on the same 6 features.
3. Save the head as ``data/models/ensemble_v7_<class>.pkl``.

Logistic regression is the right choice here: the underlying KG features
are mostly linear in log-odds (per the AUC 0.9968 of the calibrated
XGBoost — there's not much non-linear gain left to squeeze) and a
linear head per class is small (KB-scale), interpretable, and
inexpensive to retrain when the disease taxonomy shifts.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from opencure.scoring.per_class_ensemble import (
    PER_CLASS_MODEL_DIR,
    load_disease_class_map,
    _normalize,
)


# Module-private to avoid sklearn import at import time
def _train_one_class(
    X: np.ndarray, y: np.ndarray, *, seed: int = 42,
):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    base = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed,
    )
    # Isotonic calibration so the per-class probabilities stay
    # comparable to the shared head's probabilities. cv=3 keeps it
    # cheap on small per-class slices.
    cv_folds = 3 if len(np.unique(y)) == 2 and (y == 1).sum() >= 6 else 2
    if cv_folds < 2:
        # Too few positives; fit raw logistic without calibration.
        base.fit(X, y)
        return base
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
    calibrated.fit(X, y)
    return calibrated


def resolve_entity_class(
    disease_entity: str,
    *,
    entity_to_name: dict[str, str],
    class_map: dict[str, str],
) -> str | None:
    """Map ``Disease::MESH:Dxxx`` → class name, or ``None`` if unmapped."""
    if not disease_entity:
        return None
    name = entity_to_name.get(disease_entity, "")
    if not name:
        return None
    return class_map.get(_normalize(name))


def train_per_class_heads(
    X: np.ndarray,
    y: np.ndarray,
    disease_entities: list[str],
    *,
    entity_to_name: dict[str, str],
    feature_keys: tuple[str, ...],
    out_dir: Path = PER_CLASS_MODEL_DIR,
    seed: int = 42,
    min_class_positives: int = 50,
) -> dict[str, dict]:
    """Train + save one logistic head per disease class.

    Returns ``{class_name: {"path": ..., "n_pos": ..., "n_neg": ...}}``
    so the caller (phase_c_pipeline) can print a summary.
    Classes with fewer than ``min_class_positives`` positive examples
    are skipped — the routing layer falls back to the shared head for
    these (the safer default than overfitting on a thin slice).
    """
    class_map = load_disease_class_map()
    if not class_map:
        return {}

    # Map every row to its class.
    row_classes: list[str | None] = [
        resolve_entity_class(
            d, entity_to_name=entity_to_name, class_map=class_map,
        )
        for d in disease_entities
    ]

    # Group rows by class.
    classes = sorted({c for c in row_classes if c})
    summary: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        idx = np.array([i for i, c in enumerate(row_classes) if c == class_name])
        if idx.size == 0:
            continue
        Xc = X[idx]
        yc = y[idx]
        n_pos = int((yc == 1).sum())
        n_neg = int((yc == 0).sum())
        if n_pos < min_class_positives:
            print(f"  [skip] {class_name}: only {n_pos} positives "
                  f"(<{min_class_positives}); fallback to shared head")
            continue

        print(f"  Training {class_name}: {n_pos} pos / {n_neg} neg")
        model = _train_one_class(Xc, yc, seed=seed)
        out_path = out_dir / f"ensemble_v7_{class_name}.pkl"
        with out_path.open("wb") as fh:
            pickle.dump({
                "model": model,
                "feature_keys": feature_keys,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "class": class_name,
                "seed": seed,
            }, fh)
        summary[class_name] = {
            "path": str(out_path), "n_pos": n_pos, "n_neg": n_neg,
        }

    return summary


def collect_disease_names(entity_to_id: dict[str, int]) -> dict[str, str]:
    """Best-effort entity → human name map from DRKG side data.

    Used by ``train_per_class_heads`` to convert the disease entity in
    each training row to a class name. Falls back to the entity ID
    string when no human name is available — those rows then resolve to
    ``None`` and are excluded from per-class training.
    """
    # The mapping logic lives in opencure.data.drkg already; we re-use it.
    try:
        from opencure.data.drkg import load_disease_pool
        pool = load_disease_pool()
    except Exception:
        return {}
    out: dict[str, str] = {}
    for entry in pool:
        # disease_pool entries vary in shape; tolerate both list-of-dict
        # and dict-of-list forms.
        if isinstance(entry, dict):
            ent = entry.get("entity") or entry.get("disease_entity")
            name = entry.get("name") or entry.get("disease_name")
            if ent and name:
                out[ent] = name
    return out
