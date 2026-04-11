"""Configuration and paths for OpenCure."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories (configurable via env var for cloud deployment)
_data_root = Path(os.environ.get("OPENCURE_DATA_DIR", str(PROJECT_ROOT / "data")))
DATA_DIR = _data_root / "drkg"
EMBED_DIR = DATA_DIR / "embed"

# DRKG files
DRKG_TRIPLETS = DATA_DIR / "drkg.tsv"

# Pretrained TransE embeddings
DRKG_ENTITY_EMB = EMBED_DIR / "DRKG_TransE_l2_entity.npy"
DRKG_RELATION_EMB = EMBED_DIR / "DRKG_TransE_l2_relation.npy"
DRKG_ENTITY_MAP = EMBED_DIR / "entities.tsv"
DRKG_RELATION_MAP = EMBED_DIR / "relations.tsv"

# API endpoints
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"

# DRKG treatment-type relations (Compound → Disease)
# These relations indicate a drug treats, palliates, or is indicated for a disease
TREATMENT_RELATIONS = [
    "GNBR::T::Compound:Disease",       # Treats
    "Hetionet::CtD::Compound:Disease",  # Compound treats Disease
    "GNBR::C::Compound:Disease",        # Inhibits / treats
    "GNBR::Pa::Compound:Disease",       # Palliates
    "GNBR::J::Compound:Disease",        # Role in disease pathogenesis (relevant)
    "GNBR::Mp::Compound:Disease",       # Mechanism related
]
