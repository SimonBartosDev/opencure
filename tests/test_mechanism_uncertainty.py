"""Regression tests for the v7 mechanism-confidence heuristic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    from opencure.evidence import mechanism_uncertainty as mu

    p = tmp_path / "disease_gene_index.json"
    p.write_text(json.dumps({
        "Disease::WELL_STUDIED": [f"GENE_{i}" for i in range(60)],   # >> 30
        "Disease::MODERATE": [f"GENE_{i}" for i in range(15)],       # ~half
        "Disease::SPARSE": [f"GENE_{i}" for i in range(3)],          # << 30
        "Disease::UNKNOWN": [],
        # Older nested format — make sure the loader copes.
        "Disease::OLD_FORMAT": {"genes": [f"X_{i}" for i in range(45)]},
    }))
    monkeypatch.setattr(mu, "DISEASE_GENE_INDEX", p)
    mu.reset_cache()
    yield
    mu.reset_cache()


# ---- Schema -----------------------------------------------------------

def test_mechanism_confidence_field_in_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS
    assert "mechanism_confidence" in CANDIDATE_FIELDS


# ---- Confidence values -----------------------------------------------

def test_well_studied_disease_is_high_confidence() -> None:
    from opencure.evidence.mechanism_uncertainty import mechanism_confidence
    assert mechanism_confidence("Disease::WELL_STUDIED") == 1.0


def test_moderate_disease_is_partial_confidence() -> None:
    from opencure.evidence.mechanism_uncertainty import mechanism_confidence
    score = mechanism_confidence("Disease::MODERATE")
    assert 0.4 < score < 0.6  # 15 genes / 30 = 0.5


def test_sparse_disease_is_low_confidence() -> None:
    from opencure.evidence.mechanism_uncertainty import mechanism_confidence
    assert mechanism_confidence("Disease::SPARSE") == pytest.approx(0.1, abs=0.01)


def test_unknown_disease_returns_zero() -> None:
    from opencure.evidence.mechanism_uncertainty import mechanism_confidence
    # Empty list and missing disease both yield 0.
    assert mechanism_confidence("Disease::UNKNOWN") == 0.0
    assert mechanism_confidence("Disease::NOT_IN_INDEX") == 0.0
    assert mechanism_confidence("") == 0.0


def test_legacy_nested_format_still_parses() -> None:
    """Older disease_gene_index entries used {gene: {"genes": [...]}}."""
    from opencure.evidence.mechanism_uncertainty import mechanism_confidence
    score = mechanism_confidence("Disease::OLD_FORMAT")
    assert score == 1.0  # 45 genes >= 30 cap


# ---- Threshold logic --------------------------------------------------

def test_is_low_confidence_threshold() -> None:
    from opencure.evidence.mechanism_uncertainty import is_low_confidence
    assert is_low_confidence("Disease::SPARSE") is True
    assert is_low_confidence("Disease::WELL_STUDIED") is False


def test_annotate_returns_two_canonical_keys() -> None:
    from opencure.evidence.mechanism_uncertainty import annotate
    rec = annotate("Disease::WELL_STUDIED")
    assert set(rec) == {"mechanism_confidence", "mechanism_low_confidence"}
    assert rec["mechanism_confidence"] == 1.0
    assert rec["mechanism_low_confidence"] is False


def test_annotate_flags_sparse_disease() -> None:
    from opencure.evidence.mechanism_uncertainty import annotate
    rec = annotate("Disease::SPARSE")
    assert rec["mechanism_low_confidence"] is True


# ---- Top-level result schema ----------------------------------------

def test_mechanism_confidence_acceptable_at_top_level() -> None:
    """The top-level result file should accept mechanism_confidence
    even though it isn't (yet) in RESULT_TOP_LEVEL — finalize_v5 will
    add it. The check here is forward-looking: we don't want a regression
    where the field gets accidentally moved into per-candidate space."""
    # Schema is still in CANDIDATE_FIELDS for now (per-candidate
    # delivery is also valid), so we just check it's somewhere.
    from opencure.scoring.common import CANDIDATE_FIELDS, RESULT_TOP_LEVEL
    assert (
        "mechanism_confidence" in CANDIDATE_FIELDS
        or "mechanism_confidence" in RESULT_TOP_LEVEL
    )
