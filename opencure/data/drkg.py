"""Load and query the Drug Repurposing Knowledge Graph (DRKG)."""

import numpy as np
import pandas as pd
from functools import lru_cache

from opencure.config import (
    DRKG_TRIPLETS,
    DRKG_ENTITY_EMB,
    DRKG_RELATION_EMB,
    DRKG_ENTITY_MAP,
    DRKG_RELATION_MAP,
)


def load_triplets() -> pd.DataFrame:
    """Load DRKG triplets (head, relation, tail) from TSV."""
    if not DRKG_TRIPLETS.exists():
        raise FileNotFoundError(
            f"DRKG triplets not found at {DRKG_TRIPLETS}. "
            "Run: bash setup_data.sh"
        )
    return pd.read_csv(
        DRKG_TRIPLETS, sep="\t", header=None, names=["head", "relation", "tail"]
    )


def load_embeddings():
    """
    Load pretrained TransE embeddings and entity/relation mappings.

    Returns:
        entity_emb: np.ndarray of shape (num_entities, embedding_dim)
        relation_emb: np.ndarray of shape (num_relations, embedding_dim)
        entity_to_id: dict mapping entity string -> integer index
        id_to_entity: dict mapping integer index -> entity string
        relation_to_id: dict mapping relation string -> integer index
    """
    for path, name in [
        (DRKG_ENTITY_EMB, "entity embeddings"),
        (DRKG_RELATION_EMB, "relation embeddings"),
        (DRKG_ENTITY_MAP, "entity mapping"),
        (DRKG_RELATION_MAP, "relation mapping"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"DRKG {name} not found at {path}. Run: bash setup_data.sh"
            )

    entity_emb = np.load(str(DRKG_ENTITY_EMB))
    relation_emb = np.load(str(DRKG_RELATION_EMB))

    # Load entity mapping: entity_name \t id
    entity_df = pd.read_csv(DRKG_ENTITY_MAP, sep="\t", header=None, names=["entity", "id"])
    entity_to_id = dict(zip(entity_df["entity"], entity_df["id"]))
    id_to_entity = dict(zip(entity_df["id"], entity_df["entity"]))

    # Load relation mapping: relation_name \t id
    relation_df = pd.read_csv(DRKG_RELATION_MAP, sep="\t", header=None, names=["relation", "id"])
    relation_to_id = dict(zip(relation_df["relation"], relation_df["id"]))

    return entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id


def get_entity_type(entity: str) -> str:
    """Extract entity type from DRKG entity string. e.g., 'Compound::DB00945' -> 'Compound'."""
    return entity.split("::")[0]


def get_entities_by_type(entity_to_id: dict, entity_type: str) -> list[str]:
    """Get all entities of a given type (e.g., 'Compound', 'Disease', 'Gene')."""
    return [e for e in entity_to_id if get_entity_type(e) == entity_type]


def get_compound_entities(entity_to_id: dict, drugbank_only: bool = True) -> list[str]:
    """
    Get all Compound entities (drugs) in DRKG.

    Args:
        entity_to_id: Entity mapping dict
        drugbank_only: If True, only return DrugBank compounds (DB*),
                       filtering out MESH and CHEBI chemical entries.
                       DrugBank entries are actual drugs; MESH/CHEBI may be
                       generic chemical classes.
    """
    compounds = get_entities_by_type(entity_to_id, "Compound")
    if drugbank_only:
        compounds = [c for c in compounds if "::DB" in c]
    return compounds


def get_disease_entities(entity_to_id: dict) -> list[str]:
    """Get all Disease entities in DRKG."""
    return get_entities_by_type(entity_to_id, "Disease")


def find_disease_entities(entity_to_id: dict, query: str) -> list[tuple[str, float]]:
    """
    Find disease entities matching a query string (fuzzy search).

    DRKG disease entities look like:
      - Disease::MESH:D000544  (Alzheimer's disease)
      - Disease::DOID:10652

    Since DRKG entity IDs don't contain human-readable names, we search by:
    1. Exact MESH/DOID ID match if query looks like an ID
    2. Build a name mapping from the triplet data or known mappings

    For the PoC, we provide a curated mapping of common diseases to DRKG IDs,
    plus accept direct MESH/DOID IDs.

    Returns list of (entity_id, match_score) sorted by score descending.
    """
    query_lower = query.lower().strip()

    # Direct ID match (user provides MESH or DOID ID)
    if query_lower.startswith("mesh:") or query_lower.startswith("doid:"):
        entity_id = f"Disease::{query.upper().strip()}"
        if entity_id in entity_to_id:
            return [(entity_id, 1.0)]
        return []

    # Search the curated disease name mapping
    matches = []
    for name, ids in DISEASE_NAME_MAP.items():
        if query_lower in name.lower() or name.lower() in query_lower:
            score = len(name) / max(len(query), len(name))  # simple overlap score
            for disease_id in ids:
                # Some IDs are MESH/DOID (need Disease:: prefix), some are raw (e.g., SARS-CoV2)
                if disease_id.startswith("MESH:") or disease_id.startswith("DOID:"):
                    entity_id = f"Disease::{disease_id}"
                else:
                    entity_id = f"Disease::{disease_id}"
                if entity_id in entity_to_id:
                    matches.append((entity_id, score))

    # Deduplicate and sort
    seen = set()
    unique = []
    for entity_id, score in sorted(matches, key=lambda x: -x[1]):
        if entity_id not in seen:
            seen.add(entity_id)
            unique.append((entity_id, score))

    return unique


# Curated mapping of common disease names to DRKG MESH/DOID IDs.
# This is a bootstrap - will be replaced by a proper name resolution service.
DISEASE_NAME_MAP = {
    # Neurodegenerative
    "Alzheimer's disease": ["MESH:D000544"],
    "Alzheimer disease": ["MESH:D000544"],
    "Parkinson's disease": ["MESH:D010300"],
    "Parkinson disease": ["MESH:D010300"],
    "Huntington's disease": ["MESH:D006816"],
    "Amyotrophic lateral sclerosis": ["MESH:D000690"],
    "ALS": ["MESH:D000690"],
    "Multiple sclerosis": ["MESH:D009103"],

    # Cancer
    "Breast cancer": ["MESH:D001943"],
    "Lung cancer": ["MESH:D008175"],
    "Colorectal cancer": ["MESH:D015179"],
    "Prostate cancer": ["MESH:D011471"],
    "Pancreatic cancer": ["MESH:D010190"],
    "Leukemia": ["MESH:D007938"],
    "Lymphoma": ["MESH:D008223"],
    "Multiple myeloma": ["MESH:D009101"],
    "Melanoma": ["MESH:D008545"],
    "Glioblastoma": ["MESH:D005909"],
    "Ovarian cancer": ["MESH:D010051"],

    # Cardiovascular
    "Hypertension": ["MESH:D006973"],
    "Pulmonary arterial hypertension": ["MESH:D006976"],
    "Pulmonary hypertension": ["MESH:D006976"],
    "Heart failure": ["MESH:D006333"],
    "Atrial fibrillation": ["MESH:D001281"],
    "Atherosclerosis": ["MESH:D050197"],
    "Coronary artery disease": ["MESH:D003324"],

    # Metabolic
    "Diabetes mellitus": ["MESH:D003920"],
    "Type 2 diabetes": ["MESH:D003924"],
    "Obesity": ["MESH:D009765"],

    # Infectious
    # DRKG was built in 2020 - COVID entries are SARS-CoV-2 proteins, not MESH disease IDs
    "COVID-19": ["SARS-CoV2 Spike", "SARS-CoV2 nsp12", "SARS-CoV2 nsp5"],
    "SARS-CoV-2": ["SARS-CoV2 Spike", "SARS-CoV2 nsp12", "SARS-CoV2 nsp5"],
    "Malaria": ["MESH:D008288"],
    "Tuberculosis": ["MESH:D014376"],
    "HIV": ["MESH:D015658"],
    "Hepatitis C": ["MESH:D006526"],
    "Dengue": ["MESH:D003715"],
    "Chagas disease": ["MESH:D014355"],
    "Leishmaniasis": ["MESH:D007896"],
    "Schistosomiasis": ["MESH:D012552"],

    # Autoimmune / Inflammatory
    "Rheumatoid arthritis": ["MESH:D001172"],
    "Systemic lupus erythematosus": ["MESH:D008180"],
    "Lupus": ["MESH:D008180"],
    "Crohn's disease": ["MESH:D003424"],
    "Ulcerative colitis": ["MESH:D003093"],
    "Inflammatory bowel disease": ["MESH:D015212"],
    "Psoriasis": ["MESH:D011565"],

    # Respiratory
    "Asthma": ["MESH:D001249"],
    "COPD": ["MESH:D029424"],
    "Idiopathic pulmonary fibrosis": ["MESH:D054990"],
    "Cystic fibrosis": ["MESH:D003550"],

    # Rare diseases
    "Sickle cell disease": ["MESH:D000755"],
    "Fragile X syndrome": ["MESH:D005600"],
    "Duchenne muscular dystrophy": ["MESH:D020388"],
    "Neurofibromatosis": ["MESH:D009456"],
    "Marfan syndrome": ["MESH:D008382"],
    "Ehlers-Danlos syndrome": ["MESH:D004535"],
    "Gaucher disease": ["MESH:D005776"],
    "Fabry disease": ["MESH:D000795"],

    # Psychiatric / Neurological
    "Depression": ["MESH:D003866"],
    "Schizophrenia": ["MESH:D012559"],
    "Epilepsy": ["MESH:D004827"],
    "Bipolar disorder": ["MESH:D001714"],
    "Anxiety": ["MESH:D001008"],

    # Other
    "Chronic kidney disease": ["MESH:D051436"],
    "Liver cirrhosis": ["MESH:D008103"],
    "Osteoporosis": ["MESH:D010024"],
    "Endometriosis": ["MESH:D004715"],
    "Sepsis": ["MESH:D018805"],
}
