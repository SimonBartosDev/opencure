"""
PrimeKG Knowledge Graph Scorer.

Uses PrimeKG (Precision Medicine Knowledge Graph, Harvard) as an independent
knowledge graph source alongside DRKG. PrimeKG has 8.1M relationships across
30 relation types, 7,957 drugs, and 17,080 diseases.

Requires pre-trained TransE embeddings (run scripts/train_primekg.py first).
"""

from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
from typing import Optional

PRIMEKG_PATH = Path("data/primekg/kg.csv")
MODEL_DIR = Path("data/models/primekg")
ALIGNMENT_PATH = Path("data/primekg/entity_alignment.json")

_cache: dict = {}


def load_primekg_embeddings() -> tuple[Optional[np.ndarray], Optional[dict], Optional[dict]]:
    """Load pre-trained PrimeKG TransE embeddings.

    Returns (embeddings, entity_to_id, relation_to_id) or (None, None, None).
    """
    if "embeddings" in _cache:
        return _cache["embeddings"], _cache["entity_to_id"], _cache["relation_to_id"]

    emb_path = MODEL_DIR / "entity_embeddings.npy"
    ent_path = MODEL_DIR / "entity_to_id.json"
    rel_path = MODEL_DIR / "relation_to_id.json"

    if not all(p.exists() for p in [emb_path, ent_path, rel_path]):
        return None, None, None

    embeddings = np.load(emb_path)
    entity_to_id = json.loads(ent_path.read_text())
    relation_to_id = json.loads(rel_path.read_text())

    _cache["embeddings"] = embeddings
    _cache["entity_to_id"] = entity_to_id
    _cache["relation_to_id"] = relation_to_id

    return embeddings, entity_to_id, relation_to_id


def load_entity_alignment() -> dict:
    """Load mapping from DRKG entity IDs to PrimeKG entity IDs.

    Returns dict: {drkg_compound_entity: primekg_entity_key, ...}
    """
    if "alignment" in _cache:
        return _cache["alignment"]

    if not ALIGNMENT_PATH.exists():
        return {}

    alignment = json.loads(ALIGNMENT_PATH.read_text())
    _cache["alignment"] = alignment
    return alignment


def build_entity_alignment():
    """Build mapping between DRKG compound IDs and PrimeKG drug IDs.

    DRKG uses: Compound::DB00001
    PrimeKG uses: drug_DB00001 (DrugBank IDs in x_id/y_id fields)

    Also maps DRKG Disease::MESH:Dxxxxxx to PrimeKG disease IDs.
    """
    from opencure.data.drkg import load_entity_map

    drkg_entities = load_entity_map()
    alignment = {"compounds": {}, "diseases": {}}

    # Read PrimeKG to build drug ID set and disease ID set
    primekg_drugs = {}  # drugbank_id -> primekg_name
    primekg_diseases = {}  # primekg_id -> disease_name

    with open(PRIMEKG_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["x_type"] == "drug":
                primekg_drugs[row["x_id"]] = row["x_name"]
            if row["y_type"] == "drug":
                primekg_drugs[row["y_id"]] = row["y_name"]
            if row["x_type"] == "disease":
                primekg_diseases[row["x_id"]] = row["x_name"]
            if row["y_type"] == "disease":
                primekg_diseases[row["y_id"]] = row["y_name"]

    # Map DRKG compounds to PrimeKG drugs (via DrugBank ID)
    for entity_name, entity_id in drkg_entities.items():
        if entity_name.startswith("Compound::"):
            db_id = entity_name.split("::")[1]  # e.g., DB00001
            if db_id in primekg_drugs:
                primekg_key = f"drug_{db_id}"
                alignment["compounds"][entity_name] = primekg_key

    # Map diseases by name matching (fuzzy)
    drkg_disease_names = {}
    for entity_name in drkg_entities:
        if entity_name.startswith("Disease::"):
            drkg_disease_names[entity_name] = entity_name

    # Store disease name mappings for later use
    alignment["primekg_diseases"] = {k: v for k, v in primekg_diseases.items()}

    ALIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_PATH.write_text(json.dumps(alignment, indent=2))

    return alignment


def score_drugs_for_disease_primekg(
    disease_name: str,
    compound_entities: list[str],
    **kwargs,
) -> dict:
    """Score compounds for a disease using PrimeKG TransE embeddings.

    Returns dict[drkg_compound_entity] -> (score, relation, disease_entity)
    """
    embeddings, entity_to_id, relation_to_id = load_primekg_embeddings()
    if embeddings is None:
        return {}

    alignment = load_entity_alignment()
    if not alignment:
        return {}

    # Find disease in PrimeKG (by name match)
    disease_lower = disease_name.lower().strip()
    primekg_diseases = alignment.get("primekg_diseases", {})
    disease_id = None
    disease_key = None

    for pid, pname in primekg_diseases.items():
        if pname.lower() == disease_lower or disease_lower in pname.lower():
            disease_key = f"disease_{pid}"
            disease_id = pid
            break

    if disease_key is None or disease_key not in entity_to_id:
        return {}

    disease_idx = entity_to_id[disease_key]
    disease_emb = embeddings[disease_idx]

    # Find the "indication" relation (drug treats disease)
    treat_rel = None
    for rel_name, rel_id in relation_to_id.items():
        if "indication" in rel_name.lower():
            treat_rel = rel_id
            break

    if treat_rel is None:
        return {}

    # Load relation embeddings
    rel_emb_path = MODEL_DIR / "relation_embeddings.npy"
    if not rel_emb_path.exists():
        return {}
    rel_embeddings = np.load(rel_emb_path)
    relation_emb = rel_embeddings[treat_rel]

    # Score all aligned compounds: score = -||h + r - t||
    compound_alignment = alignment.get("compounds", {})
    results = {}

    for compound in compound_entities:
        primekg_key = compound_alignment.get(compound)
        if primekg_key is None or primekg_key not in entity_to_id:
            continue

        comp_idx = entity_to_id[primekg_key]
        comp_emb = embeddings[comp_idx]

        # TransE scoring: h + r ≈ t → score = -||h + r - t||
        diff = comp_emb + relation_emb - disease_emb
        distance = float(np.linalg.norm(diff))
        score = 1.0 / (1.0 + distance)  # Convert to 0-1 (closer = higher)

        results[compound] = (score, "indication", f"primekg_disease_{disease_id}")

    return results
