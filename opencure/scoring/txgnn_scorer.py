"""
TxGNN scoring using pre-computed predictions.

TxGNN (Harvard, Nature Medicine 2024) is the state-of-the-art GNN for
drug repurposing with 49% improvement over prior methods and zero-shot
capability for diseases with no known treatments.

Since TxGNN requires Python 3.8/3.9 + DGL, we pre-compute predictions
in a separate environment and load them as a static lookup table here.

Pre-compute with:
    source data/txgnn_env/bin/activate
    python scripts/precompute_txgnn.py
"""

import pandas as pd
from pathlib import Path

from opencure.config import DATA_DIR

TXGNN_PREDICTIONS = DATA_DIR.parent / "txgnn_predictions.tsv"

# Cache
_txgnn_cache = {}


def load_txgnn_predictions() -> dict:
    """
    Load pre-computed TxGNN predictions.

    Returns dict: disease_name → list of (drug_name, score) tuples sorted by score
    """
    if "data" in _txgnn_cache:
        return _txgnn_cache["data"]

    if not TXGNN_PREDICTIONS.exists():
        return {}

    df = pd.read_csv(TXGNN_PREDICTIONS, sep="\t")
    predictions = {}
    for disease, group in df.groupby("disease"):
        sorted_drugs = group.sort_values("score", ascending=False)
        predictions[disease] = list(zip(sorted_drugs["drug"], sorted_drugs["score"]))

    print(f"  TxGNN predictions loaded: {len(predictions)} diseases, {len(df)} total predictions")
    _txgnn_cache["data"] = predictions
    return predictions


def score_drugs_for_disease_txgnn(
    disease_name: str,
    compound_entities: list[str],
    drug_names: dict,
) -> dict:
    """
    Score drugs for a disease using pre-computed TxGNN predictions.

    Args:
        disease_name: Human-readable disease name
        compound_entities: List of compound entity IDs to score
        drug_names: Dict mapping entity → human name (for matching)

    Returns:
        Dict: compound_entity → (score, rank)
    """
    predictions = load_txgnn_predictions()
    if not predictions:
        return {}

    # Find matching disease in TxGNN predictions (fuzzy match)
    disease_lower = disease_name.lower()
    matched_disease = None
    for txgnn_disease in predictions:
        if disease_lower in txgnn_disease.lower() or txgnn_disease.lower() in disease_lower:
            matched_disease = txgnn_disease
            break

    if not matched_disease:
        return {}

    txgnn_drugs = predictions[matched_disease]

    # Build reverse map: drug name → entity
    name_to_entity = {}
    for entity, name in drug_names.items():
        name_to_entity[name.lower()] = entity

    # Match TxGNN drug names to our compound entities
    results = {}
    for rank, (drug_name, score) in enumerate(txgnn_drugs, 1):
        drug_lower = drug_name.lower()
        if drug_lower in name_to_entity:
            entity = name_to_entity[drug_lower]
            if entity in set(compound_entities):
                results[entity] = (score, rank)

    return results
