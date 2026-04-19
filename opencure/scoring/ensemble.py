"""
v5 ensemble inference.

The canonical ensemble is trained by ``scripts/phase_c_pipeline.py`` and
serialized to ``data/models/ensemble_v5.pkl``. This module loads that
artifact and exposes the feature-extraction + scoring primitives so both
the live search pipeline and the post-processor
(``scripts/score_ensemble_v5.py``) share one codepath.

Pre-v5 heuristic and XGBoost-native loaders were removed in the v5.1
cleanup; history is in git.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


MODEL_PATH = Path("data/models/ensemble_v5.pkl")
REPORT_PATH = Path("data/models/ensemble_v5_report.json")

DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    "kg_score",
    "degree_penalty",
    "n_drug_targets",
    "is_fda_approved",
    "n_disease_genes",
    "transe_rank_log",
)


def load_model(path: Path = MODEL_PATH) -> tuple[Any, tuple[str, ...]]:
    """Return (sklearn CalibratedClassifierCV, feature_keys).

    Raises FileNotFoundError if the trained model is absent — callers should
    treat that as "ensemble disabled" and fall back to the grouped combiner's
    ``combined_score``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found at {path}. Run "
            "scripts/phase_c_pipeline.py to train it."
        )
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], tuple(bundle.get("feature_keys", DEFAULT_FEATURE_KEYS))
    # Fallback: bare model — lean on the side-car report for feature keys.
    keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text())
            keys = tuple(report.get("feature_keys", DEFAULT_FEATURE_KEYS))
        except Exception:
            pass
    return bundle, keys


def build_features(
    *,
    compound_entity: str,
    disease_entity: str,
    rank_map: dict[str, int],
    n_compounds: int,
    drug_n_targets: dict[str, int],
    chembl_phase: dict[str, float],
    disease_gene_counts: dict[str, int],
    degree_penalty_fn,
) -> dict[str, float]:
    """Compute the 6-feature vector for (drug, disease).

    ``rank_map`` and ``n_compounds`` come from
    ``opencure.scoring.transe.score_drugs_for_disease_vectorized`` for the
    given disease. ``degree_penalty_fn`` is ``opencure.scoring.hub_normalize.degree_penalty``
    — passed in to avoid importing torch-heavy dependencies at module import time.
    """
    rk = rank_map.get(compound_entity, n_compounds)
    kg_score = max(0.0, 1.0 - rk / max(n_compounds - 1, 1))
    bare = compound_entity.split("::", 1)[1] if "::" in compound_entity else compound_entity
    phase = chembl_phase.get(bare, 0) or 0
    try:
        is_fda = 1 if float(phase) >= 2 else 0
    except (TypeError, ValueError):
        is_fda = 0
    return {
        "kg_score": kg_score,
        "degree_penalty": degree_penalty_fn(compound_entity),
        "n_drug_targets": drug_n_targets.get(bare, 0),
        "is_fda_approved": is_fda,
        "n_disease_genes": disease_gene_counts.get(disease_entity, 0),
        "transe_rank_log": float(np.log1p(rk)),
    }


def score(model, feature_keys: tuple[str, ...], feats: dict[str, float]) -> float:
    """Return the calibrated positive-class probability for a single candidate."""
    X = np.asarray([[feats[k] for k in feature_keys]], dtype=float)
    return float(model.predict_proba(X)[0, 1])
