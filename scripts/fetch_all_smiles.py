"""
Batch fetch SMILES structures for ALL DrugBank compounds in DRKG.

Current coverage: 2,268 / 10,551 (21.5%)
Target: 90%+ coverage

Uses PubChem REST API with two endpoints:
  1. Primary: /compound/xref/RegistryID/{db_id}/property/CanonicalSMILES/JSON
  2. Fallback: /compound/name/{db_id}/property/IsomericSMILES/JSON

Rate limit: 5 req/sec (0.25s sleep between requests)
Saves incrementally every 200 drugs to avoid losing progress.

Usage:
    python3 scripts/fetch_all_smiles.py
    python3 scripts/fetch_all_smiles.py --max 500  # Test with 500
"""

from __future__ import annotations

import sys
import os
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DATA_DIR = Path("data/drkg")
SMILES_CACHE = DATA_DIR / "compound_smiles.tsv"
ENTITY_MAP = DATA_DIR / "embed" / "entities.tsv"

# Use a session for connection pooling (faster)
session = requests.Session()
session.headers.update({"Accept": "application/json"})


def load_existing_smiles() -> dict:
    """Load already-cached SMILES."""
    if SMILES_CACHE.exists() and SMILES_CACHE.stat().st_size > 10:
        df = pd.read_csv(SMILES_CACHE, sep="\t")
        return dict(zip(df["drugbank_id"], df["smiles"]))
    return {}


def get_all_drugbank_ids() -> list:
    """Extract all DrugBank IDs from DRKG entity map."""
    df = pd.read_csv(ENTITY_MAP, sep="\t", header=None, names=["entity", "idx"])
    compounds = df[df["entity"].str.startswith("Compound::DB", na=False)]["entity"]
    db_ids = [c.split("::")[1] for c in compounds]
    return sorted(set(db_ids))


def fetch_smiles_pubchem(db_id: str) -> str | None:
    """Fetch SMILES for a DrugBank ID from PubChem."""
    # PubChem returns SMILES under different field names depending on the compound:
    # CanonicalSMILES, IsomericSMILES, ConnectivitySMILES, SMILES
    smiles_fields = ["CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES", "SMILES"]

    # Primary: cross-reference lookup (RegistryID = DrugBank ID)
    try:
        url = f"{PUBCHEM_BASE}/compound/xref/RegistryID/{db_id}/property/CanonicalSMILES,IsomericSMILES/JSON"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                for field in smiles_fields:
                    smiles = props[0].get(field)
                    if smiles:
                        return smiles
    except Exception:
        pass

    # Fallback: name-based lookup
    try:
        url = f"{PUBCHEM_BASE}/compound/name/{db_id}/property/CanonicalSMILES,IsomericSMILES/JSON"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                for field in smiles_fields:
                    smiles = props[0].get(field)
                    if smiles:
                        return smiles
    except Exception:
        pass

    return None


def save_smiles(all_smiles: dict):
    """Save SMILES cache atomically (write to .tmp, then rename)."""
    tmp_path = SMILES_CACHE.with_suffix(".tmp")
    rows = []
    for db_id, smiles in sorted(all_smiles.items()):
        entity = f"Compound::{db_id}"
        rows.append({"drugbank_id": db_id, "entity": entity, "smiles": smiles})

    df = pd.DataFrame(rows)
    df.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.rename(SMILES_CACHE)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max drugs to fetch (0=all)")
    args = parser.parse_args()

    # Load existing
    existing = load_existing_smiles()
    print(f"Existing SMILES: {len(existing)}")

    # Get all DrugBank IDs
    all_ids = get_all_drugbank_ids()
    print(f"Total DrugBank IDs in DRKG: {len(all_ids)}")

    # Find missing
    missing = [db_id for db_id in all_ids if db_id not in existing]
    print(f"Missing SMILES: {len(missing)}")

    if args.max > 0:
        missing = missing[:args.max]
        print(f"Fetching first {args.max} only (test mode)")

    if not missing:
        print("All SMILES already cached!")
        return

    # Fetch with rate limiting
    fetched = 0
    failed = 0
    all_smiles = dict(existing)  # Start with existing
    start_time = time.time()

    for i, db_id in enumerate(missing):
        smiles = fetch_smiles_pubchem(db_id)

        if smiles:
            all_smiles[db_id] = smiles
            fetched += 1
        else:
            failed += 1

        # Rate limit: ~2 req/sec to avoid PubChem throttling
        time.sleep(0.5)

        # Progress every 100
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(missing)}] fetched={fetched} failed={failed} "
                f"rate={rate:.1f}/s ETA={remaining/60:.0f}min "
                f"total={len(all_smiles)}"
            )

        # Save incrementally every 200
        if (i + 1) % 200 == 0:
            save_smiles(all_smiles)

    # Final save
    save_smiles(all_smiles)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Fetched: {fetched}")
    print(f"  Failed: {failed}")
    print(f"  Total SMILES: {len(all_smiles)}")
    print(f"  Coverage: {100*len(all_smiles)/len(all_ids):.1f}%")


if __name__ == "__main__":
    main()
