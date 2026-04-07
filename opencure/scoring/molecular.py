"""
Pillar 1: Molecular similarity scoring.

Given a disease, find drugs already known to treat similar diseases,
then find other approved drugs with similar molecular structures.

Uses RDKit Morgan fingerprints + Tanimoto similarity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from opencure.config import DATA_DIR, TREATMENT_RELATIONS

SMILES_CACHE = DATA_DIR / "compound_smiles.tsv"


def load_smiles_from_drkg_relations(triplets: pd.DataFrame) -> dict[str, list[str]]:
    """
    Extract drug-disease treatment relationships from DRKG triplets.

    Returns dict mapping disease_entity → list of compound entities that treat it.
    """
    treatment_mask = triplets["relation"].isin(TREATMENT_RELATIONS)
    treatment_df = triplets[treatment_mask]

    # Group compounds by disease
    disease_drugs = {}
    for _, row in treatment_df.iterrows():
        head, tail = row["head"], row["tail"]
        # Relations go Compound → Disease
        if head.startswith("Compound::") and tail.startswith("Disease::"):
            disease_drugs.setdefault(tail, []).append(head)

    return disease_drugs


def get_known_treatments(
    disease_entity: str,
    triplets: pd.DataFrame,
) -> list[str]:
    """
    Get compounds with known treatment relationships to a disease in DRKG.

    Returns list of compound entity IDs.
    """
    treatment_mask = triplets["relation"].isin(TREATMENT_RELATIONS)
    disease_mask = triplets["tail"] == disease_entity
    compound_mask = triplets["head"].str.startswith("Compound::")

    matches = triplets[treatment_mask & disease_mask & compound_mask]
    return matches["head"].unique().tolist()


def compute_fingerprint_similarity(
    query_compounds: list[str],
    all_compounds: list[str],
    smiles_map: dict[str, str],
    top_k: int = 100,
) -> list[tuple[str, float, str]]:
    """
    Find compounds most similar to query compounds using Morgan fingerprints.

    Args:
        query_compounds: Compound entities to use as reference
        all_compounds: All compound entities to score
        smiles_map: Mapping from compound entity → SMILES string
        top_k: Number of top similar compounds to return

    Returns:
        List of (compound_entity, max_similarity, most_similar_to) tuples
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator, DataStructs
    except ImportError:
        print("  [WARN] RDKit not installed. Skipping molecular similarity.")
        print("  Install with: pip install rdkit")
        return []

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Compute fingerprints for query compounds
    query_fps = {}
    for comp in query_compounds:
        smiles = smiles_map.get(comp)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            query_fps[comp] = mfpgen.GetFingerprint(mol)

    if not query_fps:
        return []

    # Score all compounds against query compounds
    results = []
    query_fp_list = list(query_fps.values())
    query_comp_list = list(query_fps.keys())

    for comp in all_compounds:
        if comp in query_fps:
            continue  # Skip query compounds themselves
        smiles = smiles_map.get(comp)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        fp = mfpgen.GetFingerprint(mol)

        # Compute similarity to each query compound
        sims = DataStructs.BulkTanimotoSimilarity(fp, query_fp_list)
        max_idx = int(np.argmax(sims))
        max_sim = sims[max_idx]

        if max_sim > 0.1:  # Minimum threshold
            results.append((comp, max_sim, query_comp_list[max_idx]))

    # Sort by similarity descending
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def load_smiles_cache() -> dict[str, str]:
    """Load cached SMILES strings for compounds."""
    if SMILES_CACHE.exists() and SMILES_CACHE.stat().st_size > 10:
        try:
            df = pd.read_csv(SMILES_CACHE, sep="\t")
            if "entity" in df.columns and "smiles" in df.columns:
                return dict(zip(df["entity"], df["smiles"]))
        except Exception:
            pass
    return {}


def save_smiles_cache(smiles_map: dict[str, str]):
    """Save SMILES cache."""
    df = pd.DataFrame([
        {"entity": k, "smiles": v} for k, v in sorted(smiles_map.items())
    ])
    df.to_csv(SMILES_CACHE, sep="\t", index=False)


def fetch_smiles_from_pubchem(drugbank_ids: list[str], max_fetch: int = 500) -> dict[str, str]:
    """
    Fetch SMILES strings for DrugBank compounds from PubChem.

    Returns dict mapping compound entity → SMILES.
    """
    import requests
    import time

    smiles_map = {}
    to_fetch = drugbank_ids[:max_fetch]

    for i, db_id in enumerate(tqdm(to_fetch, desc="Fetching SMILES from PubChem")):
        entity = f"Compound::{db_id}"
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{db_id}/property/CanonicalSMILES/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    smiles = props[0].get("CanonicalSMILES")
                    if smiles:
                        smiles_map[entity] = smiles
        except Exception:
            pass

        # Rate limit: 5 req/sec for PubChem
        if (i + 1) % 4 == 0:
            time.sleep(1.1)

    return smiles_map


def score_by_molecular_similarity(
    disease_entity: str,
    triplets: pd.DataFrame,
    all_compounds: list[str],
    smiles_map: dict[str, str],
    top_k: int = 200,
) -> list[tuple[str, float, str]]:
    """
    Score drugs by molecular similarity to known treatments for a disease.

    1. Find drugs with known DRKG treatment relations to the disease
    2. Find other drugs that are molecularly similar to those known treatments
    3. Return ranked list

    Returns list of (compound_entity, similarity_score, similar_to_compound)
    """
    # Step 1: Get known treatments from DRKG
    known_treatments = get_known_treatments(disease_entity, triplets)

    if not known_treatments:
        return []

    # Filter to those with SMILES
    known_with_smiles = [c for c in known_treatments if c in smiles_map]

    if not known_with_smiles:
        return []

    # Step 2: Find similar compounds
    results = compute_fingerprint_similarity(
        query_compounds=known_with_smiles,
        all_compounds=all_compounds,
        smiles_map=smiles_map,
        top_k=top_k,
    )

    return results
