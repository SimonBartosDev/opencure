"""Regression tests for the 4 v5.1 post-screen fixes.

Each test locks in the honest-verification pass that caught these issues:

  1. is_known_treatment via DRKG treats-edge lookup (was: heuristic with
     5+ trials AND 500+ PubMed; missed Oxamniquine/Schisto — the
     flagship positive control).
  2. ensemble_v5.pkl is loadable and its inference module exposes
     build_features + score with stable signatures.
  3. tissue_context scoring produces the canonical dict shape even
     when no genes overlap, so downstream consumers don't have to
     guess whether the field is present.
  4. Docking scaffold — the schema defines the field shape; the
     proxy can be absent (real Vina is v6) but every candidate from
     finalize_v5 carries a docking block.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from opencure.scoring.common import CANDIDATE_FIELDS


# ---- Fix #1: is_known_treatment via DRKG ---------------------------------

@pytest.mark.skipif(
    not Path("data/drkg/drkg.tsv").exists(),
    reason="DRKG source TSV not present (CI without data download)",
)
def test_known_treatment_finds_schisto_oxamniquine() -> None:
    from opencure.evidence.novelty import is_known_treatment
    # Oxamniquine is the reference Schisto treatment and appears with a
    # DRUGBANK::treats edge in DRKG.
    assert is_known_treatment({
        "drug_id": "DB01058",
        "disease_entity": "Disease::MESH:D012552",
        "pubmed_total": 0, "clinical_trials_total": 0,
    }) is True


@pytest.mark.skipif(
    not Path("data/drkg/drkg.tsv").exists(),
    reason="DRKG source TSV not present",
)
def test_known_treatment_rejects_cimetidine_schisto() -> None:
    """Cimetidine is a hub drug that ranks high but has no Schisto treats edge."""
    from opencure.evidence.novelty import is_known_treatment
    assert is_known_treatment({
        "drug_id": "DB00501",
        "disease_entity": "Disease::MESH:D012552",
        "pubmed_total": 0, "clinical_trials_total": 0,
    }) is False


def test_known_treatment_gracefully_handles_missing_disease_entity() -> None:
    """No disease_entity → fall through to the heuristic; empty evidence → False."""
    from opencure.evidence.novelty import is_known_treatment
    assert is_known_treatment({
        "drug_id": "DB01058", "disease_entity": "",
        "pubmed_total": 0, "clinical_trials_total": 0,
    }) is False


def test_known_treatment_labeling_relations_include_drugbank_treats() -> None:
    """Labeling must see DRUGBANK::treats (training KG does not)."""
    from opencure.config import KNOWN_TREATMENT_RELATIONS, TREATMENT_RELATIONS
    assert "DRUGBANK::treats::Compound:Disease" in KNOWN_TREATMENT_RELATIONS
    assert "DRUGBANK::treats::Compound:Disease" not in TREATMENT_RELATIONS


# ---- Fix #2: ensemble inference module ----------------------------------

@pytest.mark.skipif(
    not Path("data/models/ensemble_v5.pkl").exists(),
    reason="ensemble_v5.pkl not trained (run scripts/phase_c_pipeline.py first)",
)
def test_ensemble_loads_with_canonical_features() -> None:
    from opencure.scoring.ensemble import DEFAULT_FEATURE_KEYS, load_model
    model, keys = load_model()
    assert model is not None
    assert isinstance(keys, tuple) and len(keys) == 6
    # Sanity: the six features we train on.
    assert set(keys) >= {"kg_score", "transe_rank_log"}
    # predict_proba surface is what scripts/score_ensemble_v5.py relies on.
    assert hasattr(model, "predict_proba")


def test_ensemble_build_features_shape() -> None:
    from opencure.scoring.ensemble import DEFAULT_FEATURE_KEYS, build_features
    feats = build_features(
        compound_entity="Compound::DB00001",
        disease_entity="Disease::MESH:D012552",
        rank_map={"Compound::DB00001": 10},
        n_compounds=1000,
        drug_n_targets={"DB00001": 5},
        chembl_phase={"DB00001": 4.0},
        disease_gene_counts={"Disease::MESH:D012552": 42},
        degree_penalty_fn=lambda c: 1.0,
    )
    assert set(feats) == set(DEFAULT_FEATURE_KEYS)
    assert 0.0 <= feats["kg_score"] <= 1.0
    assert feats["is_fda_approved"] == 1
    assert feats["n_disease_genes"] == 42


# ---- Fix #3: tissue_context fallback -----------------------------------

def test_tissue_context_empty_gene_set_still_returns_canonical_dict() -> None:
    from opencure.scoring.tissue_context import score_tissue_context
    result = score_tissue_context("Tuberculosis", set())
    # Must always expose these keys so downstream consumers don't branch.
    assert "context_modifier" in result
    assert "tissues" in result
    assert "n_genes" in result
    # Empty input → neutral modifier.
    assert result["context_modifier"] == 1.0


def test_tissue_context_maps_schisto_to_liver() -> None:
    from opencure.scoring.tissue_context import DISEASE_TISSUE_MAP
    tissues = DISEASE_TISSUE_MAP.get("Schistosomiasis", [])
    assert "Liver" in tissues, "Schisto must map to Liver (schistosome lives in liver portal)"


# ---- Fix #4: docking scaffold schema ------------------------------------

def test_docking_field_is_in_canonical_schema() -> None:
    """Every candidate should be allowed to carry a 'docking' block."""
    assert "docking" in CANDIDATE_FIELDS
    assert "ensemble_prob" in CANDIDATE_FIELDS
    assert "ensemble_rank" in CANDIDATE_FIELDS


def test_legacy_triangulation_name_is_forbidden() -> None:
    from opencure.scoring.common import LEGACY_FIELDS
    # Flat ``triangulation_score`` at top level predates the v5 nested dict.
    assert "triangulation_score" in LEGACY_FIELDS


def test_pgx_flags_is_forbidden() -> None:
    from opencure.scoring.common import LEGACY_FIELDS
    assert "pgx_flags" in LEGACY_FIELDS
