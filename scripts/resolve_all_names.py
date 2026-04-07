"""
Batch resolve ALL drug names from PubChem.

Current: 7,770 / 10,551 (73.6%)
Target: 95%+

Uses same PubChem API as drugnames.py but with no cap and incremental saving.
"""
from __future__ import annotations

import sys
import os
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DATA_DIR = Path("data/drkg")
CACHE_FILE = DATA_DIR / "drug_names_cache.tsv"
ENTITY_MAP = DATA_DIR / "embed" / "entities.tsv"

session = requests.Session()


def fetch_name(db_id: str) -> str | None:
    """Fetch drug name from PubChem."""
    try:
        url = f"{PUBCHEM_REST}/compound/xref/RegistryID/{db_id}/property/Title/JSON"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                title = props[0].get("Title")
                if title:
                    return title
    except Exception:
        pass

    # Fallback: name-based lookup
    try:
        url = f"{PUBCHEM_REST}/compound/name/{db_id}/property/Title/JSON"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                title = props[0].get("Title")
                if title:
                    return title
    except Exception:
        pass

    return None


def save_cache(names: dict):
    """Save name cache atomically."""
    tmp = CACHE_FILE.with_suffix(".tmp")
    df = pd.DataFrame([{"drugbank_id": k, "name": v} for k, v in sorted(names.items())])
    df.to_csv(tmp, sep="\t", index=False)
    tmp.rename(CACHE_FILE)


def main():
    # Load existing
    existing = {}
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, sep="\t")
        existing = dict(zip(df["drugbank_id"], df["name"]))
    print(f"Existing names: {len(existing)}")

    # Get all DrugBank IDs
    entities = pd.read_csv(ENTITY_MAP, sep="\t", header=None, names=["entity", "idx"])
    all_ids = sorted(set(
        c.split("::")[1] for c in entities["entity"]
        if str(c).startswith("Compound::DB")
    ))
    print(f"Total DrugBank IDs: {len(all_ids)}")

    missing = [d for d in all_ids if d not in existing]
    print(f"Missing names: {len(missing)}")

    if not missing:
        print("All names resolved!")
        return

    all_names = dict(existing)
    fetched = 0
    failed = 0
    start = time.time()

    for i, db_id in enumerate(missing):
        name = fetch_name(db_id)
        if name:
            all_names[db_id] = name
            fetched += 1
        else:
            failed += 1

        time.sleep(0.5)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(missing)}] fetched={fetched} failed={failed} "
                  f"rate={rate:.1f}/s ETA={remaining/60:.0f}min total={len(all_names)}")

        if (i + 1) % 200 == 0:
            save_cache(all_names)

    save_cache(all_names)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f}min. Fetched: {fetched}, Failed: {failed}, Total: {len(all_names)}")


if __name__ == "__main__":
    main()
