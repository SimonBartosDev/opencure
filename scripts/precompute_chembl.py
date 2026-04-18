#!/usr/bin/env python3
"""Pre-compute ChEMBL max_phase for all DrugBank compounds.

Queries ChEMBL API for each compound and caches the result.
Takes ~30-60 min for 9,425 compounds (rate-limited to 0.1s/query).

Usage:
    python3 scripts/precompute_chembl.py
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.data.chembl_phase import load_cache, save_cache, fetch_chembl_by_name

SMILES_PATH = Path("data/drkg/compound_smiles.tsv")
NAMES_PATH = Path("data/drkg/drug_names_cache.tsv")


def main():
    # Load DrugBank ID -> Name map
    name_map = {}
    with open(NAMES_PATH) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name_map[parts[0]] = parts[1]

    # Load all DrugBank IDs from SMILES file (have SMILES = worth looking up)
    drugbank_ids = []
    with open(SMILES_PATH) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split("\t")
            if parts and parts[0] in name_map:
                drugbank_ids.append(parts[0])

    print(f"Total DrugBank IDs with names: {len(drugbank_ids)}")

    cache = load_cache()
    print(f"Already cached: {len(cache)}")

    to_fetch = [d for d in drugbank_ids if d not in cache]
    print(f"To fetch: {len(to_fetch)}")

    if not to_fetch:
        print("All cached. Nothing to do.")
        return

    start = time.time()
    done = 0
    found = 0
    not_found = 0

    for db_id in to_fetch:
        name = name_map.get(db_id, db_id)
        phase = fetch_chembl_by_name(name)

        cache[db_id] = phase
        done += 1
        if phase is not None:
            found += 1
        else:
            not_found += 1

        if done % 100 == 0:
            save_cache(cache)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(to_fetch) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(to_fetch)} done ({found} found, {not_found} not found) "
                  f"~{remaining/60:.0f} min remaining")

        time.sleep(0.1)

    save_cache(cache)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Found in ChEMBL: {found}")
    print(f"  Not found: {not_found}")

    # Distribution by phase
    from collections import Counter
    phases = Counter(cache.values())
    print("\nPhase distribution:")
    for phase in sorted(phases.keys(), key=lambda x: (x is None, x)):
        print(f"  Phase {phase}: {phases[phase]}")


if __name__ == "__main__":
    main()
