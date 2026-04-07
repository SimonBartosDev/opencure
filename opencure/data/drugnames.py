"""
Resolve DrugBank IDs to human-readable drug names.

Uses PubChem's REST API (free, no key needed) with local caching.
On first run, resolves all DRKG compound names and saves to a local TSV.
Subsequent runs load from cache.
"""
from __future__ import annotations

import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from opencure.config import DATA_DIR, DRKG_ENTITY_MAP

CACHE_FILE = DATA_DIR / "drug_names_cache.tsv"

# PubChem can resolve DrugBank IDs via cross-reference
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def load_drug_names() -> dict[str, str]:
    """
    Load DrugBank ID → drug name mapping.
    Uses cached file if available, otherwise builds from curated list.

    Returns dict mapping DrugBank ID (e.g., 'DB00945') → name (e.g., 'Aspirin')
    """
    # Try loading from cache first
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, sep="\t")
        return dict(zip(df["drugbank_id"], df["name"]))

    # Fall back to the curated list in search.py + attempt to resolve more
    from opencure.search import DRUGBANK_NAMES
    return DRUGBANK_NAMES.copy()


def resolve_drugbank_via_pubchem(drugbank_id: str) -> str | None:
    """Resolve a single DrugBank ID to a drug name via PubChem."""
    try:
        url = f"{PUBCHEM_REST}/compound/xref/RegistryID/{drugbank_id}/property/Title/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("Title")
    except Exception:
        pass
    return None


def build_name_cache(batch_size: int = 100, max_resolve: int = 2000):
    """
    Build a comprehensive drug name cache by resolving DrugBank IDs via PubChem.

    This is slow (API rate-limited) but only needs to run once.
    Results are saved to CACHE_FILE.
    """
    # Start with curated names
    from opencure.search import DRUGBANK_NAMES
    names = DRUGBANK_NAMES.copy()

    # Get all DrugBank IDs from DRKG
    entity_df = pd.read_csv(DRKG_ENTITY_MAP, sep="\t", header=None, names=["entity", "id"])
    all_db_ids = [
        e.split("::")[1]
        for e in entity_df["entity"]
        if e.startswith("Compound::DB")
    ]

    # Find IDs we don't have names for yet
    unresolved = [db_id for db_id in all_db_ids if db_id not in names]
    print(f"Total DrugBank compounds: {len(all_db_ids)}")
    print(f"Already named: {len(names)}")
    print(f"Unresolved: {len(unresolved)}")
    print(f"Will attempt to resolve up to {min(len(unresolved), max_resolve)} via PubChem...")

    resolved_count = 0
    failed_count = 0

    for i, db_id in enumerate(tqdm(unresolved[:max_resolve], desc="Resolving names")):
        name = resolve_drugbank_via_pubchem(db_id)
        if name:
            names[db_id] = name
            resolved_count += 1
        else:
            failed_count += 1

        # Rate limiting: PubChem allows 5 req/sec
        if (i + 1) % 5 == 0:
            time.sleep(1.1)

    print(f"\nResolved: {resolved_count}, Failed: {failed_count}")
    print(f"Total named drugs: {len(names)}")

    # Save cache
    df = pd.DataFrame(
        [{"drugbank_id": k, "name": v} for k, v in sorted(names.items())]
    )
    df.to_csv(CACHE_FILE, sep="\t", index=False)
    print(f"Cache saved to {CACHE_FILE}")

    return names


if __name__ == "__main__":
    build_name_cache(max_resolve=2000)
