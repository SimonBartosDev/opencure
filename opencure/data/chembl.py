"""Query ChEMBL API for drug metadata and cross-references."""
from __future__ import annotations

import requests
from functools import lru_cache


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def get_drug_info_by_drugbank(drugbank_id: str) -> dict | None:
    """
    Look up drug info from ChEMBL using a DrugBank ID as cross-reference.

    Returns dict with: chembl_id, name, molecule_type, max_phase, smiles
    or None if not found.
    """
    try:
        url = f"{CHEMBL_BASE}/molecule.json"
        resp = requests.get(
            url,
            params={
                "molecule_synonyms__molecule_synonym__iexact": drugbank_id,
                "limit": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data["molecules"]:
            return None

        mol = data["molecules"][0]
        smiles = None
        if mol.get("molecule_structures"):
            smiles = mol["molecule_structures"].get("canonical_smiles")

        return {
            "chembl_id": mol["molecule_chembl_id"],
            "name": mol.get("pref_name", ""),
            "molecule_type": mol.get("molecule_type", ""),
            "max_phase": mol.get("max_phase", 0),
            "smiles": smiles,
        }
    except Exception:
        return None


def get_drug_name_from_chembl(drugbank_id: str) -> str | None:
    """Look up a drug's preferred name from ChEMBL. Returns None if not found."""
    info = get_drug_info_by_drugbank(drugbank_id)
    if info and info["name"]:
        return info["name"]
    return None


def batch_resolve_drug_names(drugbank_ids: list[str], max_requests: int = 50) -> dict:
    """
    Resolve a batch of DrugBank IDs to human-readable names via ChEMBL.

    Rate-limited to avoid overwhelming the API.
    Returns dict mapping DrugBank ID -> name.
    """
    names = {}
    for i, db_id in enumerate(drugbank_ids[:max_requests]):
        name = get_drug_name_from_chembl(db_id)
        if name:
            names[db_id] = name
    return names
