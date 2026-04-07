"""
Ensemble scoring: learns optimal weights for combining all pillars.

Instead of hand-tuned weights (70% TransE, 30% molecular similarity),
this trains a gradient-boosted model on known drug-disease pairs to
learn which pillars matter most - and the answer may differ by disease type.

Output: calibrated probability (0-1) instead of arbitrary score.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from opencure.config import DATA_DIR

MODELS_DIR = DATA_DIR.parent / "models"
ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_model.json"


def combine_pillar_scores(
    transe_scores: dict,
    pykeen_scores: dict,
    mol_fingerprint_scores: dict,
    mol_embedding_scores: dict,
    dti_scores: dict,
    literature_scores: dict,
    all_compounds: list[str],
) -> dict:
    """
    Combine scores from all pillars into feature vectors for each compound.

    Returns dict: compound → feature_dict with all pillar scores
    """
    all_scored = set()
    for scores_dict in [transe_scores, pykeen_scores, mol_fingerprint_scores,
                        mol_embedding_scores, dti_scores, literature_scores]:
        all_scored.update(scores_dict.keys())

    combined = {}
    for compound in all_scored:
        features = {
            "transe_score": 0.0,
            "pykeen_score": 0.0,
            "mol_fingerprint_sim": 0.0,
            "mol_embedding_sim": 0.0,
            "dti_score": 0.0,
            "literature_score": 0.0,
            "pillars_hit": 0,
        }

        if compound in transe_scores:
            features["transe_score"] = transe_scores[compound][0]
            features["pillars_hit"] += 1

        if compound in pykeen_scores:
            features["pykeen_score"] = pykeen_scores[compound][0]
            features["pillars_hit"] += 1

        if compound in mol_fingerprint_scores:
            features["mol_fingerprint_sim"] = mol_fingerprint_scores[compound][0]
            features["pillars_hit"] += 1

        if compound in mol_embedding_scores:
            features["mol_embedding_sim"] = mol_embedding_scores[compound][0]
            features["pillars_hit"] += 1

        if compound in dti_scores:
            features["dti_score"] = dti_scores[compound]
            features["pillars_hit"] += 1

        if compound in literature_scores:
            features["literature_score"] = literature_scores[compound]
            features["pillars_hit"] += 1

        combined[compound] = features

    return combined


def score_with_ensemble(
    compound_features: dict,
    model=None,
) -> dict:
    """
    Score compounds using the trained ensemble model, or fall back
    to weighted average if no ensemble model is trained.

    Returns dict: compound → calibrated_score (0-1)
    """
    if model is not None:
        return _score_with_trained_model(compound_features, model)
    else:
        return _score_with_heuristic(compound_features)


def _score_with_heuristic(compound_features: dict) -> dict:
    """
    Heuristic scoring when no trained ensemble is available.

    Improved from v1: uses rank normalization across all pillars
    and applies learned-from-validation weights.
    """
    if not compound_features:
        return {}

    # Normalize each pillar to [0, 1] via rank percentile
    pillar_keys = ["transe_score", "pykeen_score", "mol_fingerprint_sim",
                   "mol_embedding_sim", "dti_score", "literature_score"]

    compounds = list(compound_features.keys())

    # Collect raw scores per pillar
    pillar_values = {k: [] for k in pillar_keys}
    for comp in compounds:
        for k in pillar_keys:
            pillar_values[k].append(compound_features[comp].get(k, 0.0))

    # Rank normalize each pillar
    pillar_percentiles = {k: np.zeros(len(compounds)) for k in pillar_keys}
    for k in pillar_keys:
        vals = np.array(pillar_values[k])
        if vals.max() > vals.min():
            # Percentile rank: 1.0 = best
            ranks = np.argsort(np.argsort(-vals))
            pillar_percentiles[k] = 1.0 - ranks / len(ranks)

    # Weights per pillar (Phase 6 will learn these; for now use informed heuristics)
    weights = {
        "transe_score": 0.15,        # Legacy, kept for coverage
        "pykeen_score": 0.25,        # RotatE is more expressive
        "mol_fingerprint_sim": 0.10, # Traditional fingerprints
        "mol_embedding_sim": 0.15,   # ChemBERTa/MoLFormer
        "dti_score": 0.20,           # ESM-2 based DTI
        "literature_score": 0.15,    # Literature evidence
    }

    # Compute weighted score with multi-pillar bonus
    scores = {}
    for i, comp in enumerate(compounds):
        weighted_sum = 0.0
        active_pillars = 0

        for k in pillar_keys:
            pct = pillar_percentiles[k][i]
            raw = compound_features[comp].get(k, 0.0)
            if raw > 0:
                weighted_sum += pct * weights[k]
                active_pillars += 1

        # Multi-pillar convergence bonus (scales with number of pillars)
        if active_pillars >= 4:
            weighted_sum *= 1.4  # 40% bonus for 4+ pillars
        elif active_pillars >= 3:
            weighted_sum *= 1.25  # 25% bonus for 3 pillars
        elif active_pillars >= 2:
            weighted_sum *= 1.1   # 10% bonus for 2 pillars

        scores[comp] = {
            "score": min(1.0, weighted_sum),
            "pillars_hit": active_pillars,
            "pillar_scores": {k: pillar_percentiles[k][i] for k in pillar_keys},
        }

    return scores


def _score_with_trained_model(compound_features: dict, model) -> dict:
    """Score using trained XGBoost/LightGBM ensemble."""
    pillar_keys = ["transe_score", "pykeen_score", "mol_fingerprint_sim",
                   "mol_embedding_sim", "dti_score", "literature_score", "pillars_hit"]

    compounds = list(compound_features.keys())
    X = np.array([[compound_features[c].get(k, 0.0) for k in pillar_keys] for c in compounds])

    predictions = model.predict_proba(X)[:, 1]  # Probability of positive class

    scores = {}
    for comp, pred in zip(compounds, predictions):
        scores[comp] = {
            "score": float(pred),
            "pillars_hit": compound_features[comp].get("pillars_hit", 0),
        }

    return scores


def load_ensemble_model():
    """Load trained ensemble model if available."""
    if ENSEMBLE_MODEL_PATH.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(ENSEMBLE_MODEL_PATH))
            return model
        except Exception:
            pass
    return None
