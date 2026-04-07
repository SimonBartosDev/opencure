"""
Fetch SMILES strings for DRKG compounds from PubChem.

Uses PubChem's batch retrieval endpoint for efficiency.
Saves results to data/drkg/compound_smiles.tsv.
"""

import sys
import os
import time
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.config import DATA_DIR, DRKG_ENTITY_MAP

SMILES_CACHE = DATA_DIR / "compound_smiles.tsv"
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_smiles_batch(drugbank_ids: list[str], batch_size: int = 10) -> dict[str, str]:
    """
    Fetch SMILES from PubChem for a list of DrugBank IDs.
    Uses individual lookups with rate limiting.
    """
    results = {}
    failed = 0

    for i in tqdm(range(0, len(drugbank_ids), 1), desc="Fetching SMILES"):
        db_id = drugbank_ids[i]
        try:
            # PubChem indexes DrugBank IDs as compound names/synonyms
            url = f"{PUBCHEM_REST}/compound/name/{db_id}/property/IsomericSMILES/JSON"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    # PubChem returns SMILES under various key names
                    p = props[0]
                    smiles = p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
                    if smiles:
                        results[db_id] = smiles
            else:
                failed += 1
        except Exception:
            failed += 1

        # Rate limit: PubChem allows 5 req/sec, be conservative
        if (i + 1) % 4 == 0:
            time.sleep(1.2)

    print(f"\nResolved: {len(results)}, Failed: {failed}")
    return results


def main():
    # Load existing cache
    existing = {}
    if SMILES_CACHE.exists():
        df = pd.read_csv(SMILES_CACHE, sep="\t")
        existing = dict(zip(df["drugbank_id"], df["smiles"]))
        print(f"Existing cache: {len(existing)} compounds")

    # Get all DrugBank IDs from DRKG
    entity_df = pd.read_csv(DRKG_ENTITY_MAP, sep="\t", header=None, names=["entity", "id"])
    all_db_ids = sorted(set(
        e.split("::")[1]
        for e in entity_df["entity"]
        if e.startswith("Compound::DB")
    ))
    print(f"Total DrugBank compounds in DRKG: {len(all_db_ids)}")

    # Filter to ones we haven't resolved yet
    unresolved = [db_id for db_id in all_db_ids if db_id not in existing]
    print(f"Unresolved: {len(unresolved)}")

    if not unresolved:
        print("All compounds already resolved!")
        return

    # Limit how many we fetch in one go
    max_fetch = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    to_fetch = unresolved[:max_fetch]
    print(f"Fetching SMILES for {len(to_fetch)} compounds from PubChem...")

    new_smiles = fetch_smiles_batch(to_fetch)

    # Merge with existing
    all_smiles = {**existing, **new_smiles}

    # Save
    df = pd.DataFrame([
        {"drugbank_id": k, "entity": f"Compound::{k}", "smiles": v}
        for k, v in sorted(all_smiles.items())
    ])
    df.to_csv(SMILES_CACHE, sep="\t", index=False)
    print(f"\nTotal SMILES cached: {len(all_smiles)}")
    print(f"Saved to {SMILES_CACHE}")


if __name__ == "__main__":
    main()
