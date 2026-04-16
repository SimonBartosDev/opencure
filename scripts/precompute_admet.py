#!/usr/bin/env python3
"""Pre-compute ADMET predictions for all compounds with SMILES.

Runs ADMET-AI on all compounds in compound_smiles.tsv and caches results
to data/drkg/admet_predictions.json. Takes ~5-10 minutes on CPU.

Usage:
    python3 scripts/precompute_admet.py
"""

import sys
import os
import json
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from opencure.scoring.admet_filter import predict_admet, save_cached_predictions, load_cached_predictions, is_toxic

SMILES_PATH = Path("data/drkg/compound_smiles.tsv")
CACHE_PATH = Path("data/drkg/admet_predictions.json")


def main():
    # Load existing cache
    cached = load_cached_predictions()
    print(f"Loaded {len(cached)} cached predictions")

    # Load SMILES
    smiles_map = {}
    with open(SMILES_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                smiles_map[parts[0]] = parts[1]

    print(f"Total compounds with SMILES: {len(smiles_map)}")

    # Filter out already-cached
    to_predict = {k: v for k, v in smiles_map.items() if v not in cached}
    print(f"Need to predict: {len(to_predict)}")

    if not to_predict:
        print("All predictions cached. Nothing to do.")
        return

    warnings.filterwarnings("ignore")

    done = 0
    failed = 0
    toxic_count = 0
    start = time.time()

    for drug_id, smiles in to_predict.items():
        try:
            preds = predict_admet(smiles)
            if is_toxic(preds):
                toxic_count += 1
            done += 1
        except Exception as e:
            failed += 1

        if (done + failed) % 100 == 0:
            elapsed = time.time() - start
            rate = (done + failed) / elapsed if elapsed > 0 else 0
            remaining = (len(to_predict) - done - failed) / rate if rate > 0 else 0
            print(f"  Progress: {done + failed}/{len(to_predict)} "
                  f"({done} ok, {failed} failed, {toxic_count} toxic) "
                  f"~{remaining/60:.0f} min remaining")

    # Save cache
    save_cached_predictions()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Predicted: {done}")
    print(f"  Failed: {failed}")
    print(f"  Toxic: {toxic_count}")
    print(f"  Cache saved: {CACHE_PATH}")


if __name__ == "__main__":
    main()
