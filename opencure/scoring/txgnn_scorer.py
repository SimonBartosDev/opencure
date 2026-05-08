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

    # Find matching disease in TxGNN predictions (fuzzy match, apostrophe-normalized)
    def _norm(s: str) -> str:
        import re
        s = s.lower()
        # Remove possessive 's (with apostrophe or curly quote)
        s = re.sub(r"['\u2019]s\b", "", s)
        # Remove any remaining apostrophes
        s = s.replace("'", "").replace("\u2019", "")
        return s.strip()

    disease_norm = _norm(disease_name)
    matched_disease = None
    for txgnn_disease in predictions:
        tx_norm = _norm(txgnn_disease)
        if disease_norm == tx_norm or disease_norm in tx_norm or tx_norm in disease_norm:
            matched_disease = txgnn_disease
            break

    if not matched_disease:
        return {}

    txgnn_drugs = predictions[matched_disease]

    # Build name → entity map. TxGNN trained on PrimeKG uses parent-compound
    # names ("metformin"); DRKG/DrugBank stores salt forms ("metformin
    # hydrochloride"). Augment the lookup with salt-stripped aliases — same
    # approach as gene_signatures.py — to lift the match rate.
    SALT_SUFFIXES = (
        " hydrochloride", " sulfate", " sodium", " maleate", " citrate",
        " tartrate", " fumarate", " acetate", " phosphate", " chloride",
        " bromide", " succinate", " mesylate", " besylate", " tosylate",
        " nitrate", " hydrobromide", " calcium", " potassium", " disodium",
    )
    name_to_entity: dict[str, str] = {}
    for entity, name in drug_names.items():
        low = name.lower().strip()
        if low and low not in name_to_entity:
            name_to_entity[low] = entity
        for suffix in SALT_SUFFIXES:
            if low.endswith(suffix):
                stripped = low[: -len(suffix)].strip()
                # First hit wins — canonical full names beat salt-stripped aliases.
                name_to_entity.setdefault(stripped, entity)

    candidate_set = set(compound_entities)
    results: dict[str, tuple[float, int]] = {}
    for rank, (drug_name, score) in enumerate(txgnn_drugs, 1):
        drug_lower = drug_name.lower().strip()
        entity = name_to_entity.get(drug_lower)
        if entity and entity in candidate_set:
            # Keep the best (smallest) rank if the same entity matches
            # multiple alternate names.
            existing = results.get(entity)
            if existing is None or rank < existing[1]:
                results[entity] = (score, rank)

    return results
