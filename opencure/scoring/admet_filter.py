"""
ADMET/Toxicity filtering and drug-likeness scoring.

Uses admet-ai (Chemprop-based) to predict 77+ ADMET endpoints for each
compound, including toxicity flags, drug-likeness properties, and
pharmacokinetic parameters. Acts as both a FILTER (remove toxic compounds)
and a SCORING PILLAR (drug-likeness score 0-1).

Install: pip install admet-ai
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Cache for predictions
_admet_cache: dict[str, dict] = {}
_model = None

CACHE_PATH = Path("data/drkg/admet_predictions.json")

# Toxicity thresholds — compounds must exceed MULTIPLE thresholds to be filtered.
# Individual threshold crossings add to a toxicity count; a compound is only
# filtered if it exceeds the FILTER_MIN_FLAGS count. This avoids filtering
# FDA-approved drugs with known but manageable toxicity profiles.
TOXICITY_ENDPOINTS = {
    "hERG": 0.7,            # hERG channel inhibition (cardiotoxicity)
    "AMES": 0.7,            # Ames mutagenicity (strong positive)
    "DILI": 0.85,           # Drug-induced liver injury (very high risk only)
    "Skin_Reaction": 0.8,   # Severe skin reaction
}
FILTER_MIN_FLAGS = 2  # Must exceed at least 2 thresholds to be filtered

# Endpoints used for drug-likeness scoring (higher = better)
POSITIVE_ENDPOINTS = [
    "Caco2_Wang",                    # Intestinal permeability (higher = better absorption)
    "Lipophilicity_AstraZeneca",     # LogD (moderate is best)
    "Solubility_AqSolDB",           # Aqueous solubility (higher = better)
    "HydrationFreeEnergy_FreeSolv",  # Hydration energy
]

# Endpoints used for drug-likeness scoring (lower = better)
NEGATIVE_ENDPOINTS = [
    "AMES",                          # Mutagenicity (lower = safer)
    "hERG",                          # Cardiotoxicity (lower = safer)
    "DILI",                          # Liver injury (lower = safer)
    "CYP1A2_Veith",                  # CYP inhibition (lower = fewer interactions)
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "Clearance_Hepatocyte_AZ",       # Clearance (lower = longer half-life)
]


def _get_model():
    """Lazy-load the ADMET model."""
    global _model
    if _model is None:
        try:
            from admet_ai import ADMETModel
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model = ADMETModel()
        except ImportError:
            raise ImportError(
                "admet-ai is required for ADMET filtering. "
                "Install with: pip install admet-ai"
            )
    return _model


def predict_admet(smiles: str) -> dict:
    """Predict ADMET properties for a single SMILES string."""
    if smiles in _admet_cache:
        return _admet_cache[smiles]

    model = _get_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(smiles=smiles)

    _admet_cache[smiles] = preds
    return preds


def predict_admet_batch(smiles_list: list[str]) -> dict[str, dict]:
    """Predict ADMET for a batch of SMILES. Returns {smiles: predictions}."""
    # Check cache first
    uncached = [s for s in smiles_list if s not in _admet_cache]

    if uncached:
        model = _get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for smiles in uncached:
                try:
                    preds = model.predict(smiles=smiles)
                    _admet_cache[smiles] = preds
                except Exception:
                    _admet_cache[smiles] = {}

    return {s: _admet_cache.get(s, {}) for s in smiles_list}


def is_toxic(predictions: dict) -> bool:
    """Check if a compound exceeds multiple toxicity thresholds.

    A compound is only filtered if it triggers at least FILTER_MIN_FLAGS
    toxicity endpoints. This avoids filtering FDA-approved drugs with
    known but manageable side effects.
    """
    if not predictions:
        return False

    flag_count = 0
    for endpoint, threshold in TOXICITY_ENDPOINTS.items():
        value = predictions.get(endpoint, 0)
        if isinstance(value, (int, float)) and value > threshold:
            flag_count += 1

    return flag_count >= FILTER_MIN_FLAGS


def get_toxicity_flags(predictions: dict) -> list[str]:
    """Return list of toxicity flags that are triggered."""
    flags = []
    if not predictions:
        return flags

    for endpoint, threshold in TOXICITY_ENDPOINTS.items():
        value = predictions.get(endpoint, 0)
        if isinstance(value, (int, float)) and value > threshold:
            flags.append(f"{endpoint}={value:.2f}")

    return flags


def compute_drug_likeness_score(predictions: dict) -> float:
    """
    Compute a 0-1 drug-likeness score from ADMET predictions.

    Higher score = more drug-like (good absorption, low toxicity, reasonable PK).
    """
    if not predictions:
        return 0.5  # No data = neutral

    scores = []

    # Negative endpoints: lower is better → score = 1 - value (clamped)
    for endpoint in NEGATIVE_ENDPOINTS:
        value = predictions.get(endpoint)
        if value is not None and isinstance(value, (int, float)):
            scores.append(max(0.0, 1.0 - value))

    # Use percentile endpoints where available (0-100 scale, higher = more drug-like)
    for endpoint in POSITIVE_ENDPOINTS:
        pctile_key = f"{endpoint}_drugbank_approved_percentile"
        pctile = predictions.get(pctile_key)
        if pctile is not None and isinstance(pctile, (int, float)):
            scores.append(pctile / 100.0)

    if not scores:
        return 0.5

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def load_cached_predictions() -> dict[str, dict]:
    """Load pre-computed ADMET predictions from cache file."""
    global _admet_cache
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text())
            _admet_cache.update(data)
            return data
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_cached_predictions():
    """Save current predictions to cache file."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_admet_cache, indent=2))


def score_drugs_for_disease_admet(
    compound_entities: list[str],
    smiles_map: dict[str, str],
) -> tuple[dict, set]:
    """
    Score all compounds by drug-likeness and identify toxic ones.

    Returns:
        scores: dict[compound_entity] -> (drug_likeness_score, toxicity_flags_str, "admet")
        toxic_compounds: set of compound entities that should be filtered
    """
    # Try loading cached predictions first
    load_cached_predictions()

    scores = {}
    toxic = set()

    for compound in compound_entities:
        smiles = smiles_map.get(compound)
        if not smiles:
            continue

        try:
            preds = predict_admet(smiles)
        except Exception:
            continue

        if is_toxic(preds):
            toxic.add(compound)
            continue

        dl_score = compute_drug_likeness_score(preds)
        flags = get_toxicity_flags(preds)
        flags_str = "; ".join(flags) if flags else "clean"
        scores[compound] = (dl_score, flags_str, "admet")

    return scores, toxic
