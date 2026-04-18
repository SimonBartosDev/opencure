"""
ChEMBL clinical phase lookup via ChEMBL API.

Queries ChEMBL for max_phase of each DrugBank compound:
  0.5 = preclinical
  1   = Phase I trials
  2   = Phase II trials
  3   = Phase III trials
  4   = Approved

Used to filter out compounds that never entered clinical trials
(metabolites, research tools, experimental cofactors).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import requests

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CACHE_PATH = Path("data/drkg/chembl_phase.json")


def fetch_chembl_by_name(drug_name: str, timeout: int = 15) -> float | None:
    """
    Query ChEMBL by drug name (search endpoint). Returns max_phase or None.
    """
    url = f"{CHEMBL_API}/molecule/search.json"
    try:
        resp = requests.get(
            url,
            params={"q": drug_name, "limit": 1},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    molecules = data.get("molecules", [])
    if not molecules:
        return None

    max_phase = molecules[0].get("max_phase")
    if max_phase is None:
        return None

    try:
        return float(max_phase)
    except (ValueError, TypeError):
        return None


# Legacy aliases for backwards compat
def fetch_chembl_max_phase(drugbank_id: str, timeout: int = 15) -> float | None:
    return fetch_chembl_by_name(drugbank_id, timeout)


def fetch_chembl_by_xref(drugbank_id: str, timeout: int = 15) -> float | None:
    return fetch_chembl_by_name(drugbank_id, timeout)


def load_cache() -> dict:
    """Load cached phase data."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict):
    """Persist cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def lookup_phase(drugbank_id: str) -> float | None:
    """Look up phase with caching. Public interface."""
    cache = load_cache()
    if drugbank_id in cache:
        return cache[drugbank_id]

    # Try xref first (more reliable), then synonym
    phase = fetch_chembl_by_xref(drugbank_id)
    if phase is None:
        phase = fetch_chembl_max_phase(drugbank_id)

    cache[drugbank_id] = phase
    save_cache(cache)
    return phase


def batch_lookup(drugbank_ids: list[str], delay: float = 0.1) -> dict:
    """Lookup phase for many compounds with rate limiting."""
    cache = load_cache()
    updates = {}
    to_fetch = [d for d in drugbank_ids if d not in cache]

    for i, db_id in enumerate(to_fetch):
        phase = fetch_chembl_by_xref(db_id)
        if phase is None:
            phase = fetch_chembl_max_phase(db_id)
        updates[db_id] = phase
        time.sleep(delay)
        if (i + 1) % 50 == 0:
            cache.update(updates)
            save_cache(cache)
            updates = {}

    cache.update(updates)
    save_cache(cache)
    return cache
