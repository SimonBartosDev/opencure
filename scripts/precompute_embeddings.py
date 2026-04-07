"""
Pre-compute molecular embeddings for all DRKG compounds.

Usage:
    python3 scripts/precompute_embeddings.py chemberta   # ChemBERTa embeddings
    python3 scripts/precompute_embeddings.py molformer   # MoLFormer embeddings
    python3 scripts/precompute_embeddings.py both        # Both

Embeddings are cached to data/drkg/embeddings/ as .npz files.
First run downloads the model (~300MB for ChemBERTa, ~500MB for MoLFormer).
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from tqdm import tqdm

from opencure.config import DATA_DIR
from opencure.scoring.molecular import SMILES_CACHE
from opencure.scoring.molecular_embeddings import (
    get_chemberta_embeddings,
    get_molformer_embeddings,
    save_cached_embeddings,
    EMBEDDINGS_DIR,
)


def load_smiles() -> tuple[list[str], list[str]]:
    """Load SMILES data. Returns (entity_ids, smiles_strings)."""
    if not SMILES_CACHE.exists():
        print("ERROR: No SMILES cache found. Run scripts/fetch_smiles.py first.")
        sys.exit(1)

    df = pd.read_csv(SMILES_CACHE, sep="\t")
    # Filter out invalid SMILES
    valid = df.dropna(subset=["smiles"])
    valid = valid[valid["smiles"].str.len() > 0]

    print(f"Loaded {len(valid)} compounds with SMILES")
    return valid["entity"].tolist(), valid["smiles"].tolist()


def precompute(model_type: str):
    """Pre-compute embeddings for all compounds with SMILES."""
    entities, smiles_list = load_smiles()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nComputing {model_type} embeddings for {len(smiles_list)} compounds...")
    start = time.time()

    if model_type == "chemberta":
        embeddings = get_chemberta_embeddings(smiles_list, batch_size=64)
    elif model_type == "molformer":
        embeddings = get_molformer_embeddings(smiles_list, batch_size=32)
    else:
        print(f"Unknown model type: {model_type}")
        return

    elapsed = time.time() - start
    print(f"Computed {embeddings.shape} embeddings in {elapsed:.1f}s")

    save_cached_embeddings(embeddings, entities, model_type)
    print(f"Saved to {EMBEDDINGS_DIR}/")


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "chemberta"

    if model_type == "both":
        precompute("chemberta")
        precompute("molformer")
    else:
        precompute(model_type)
