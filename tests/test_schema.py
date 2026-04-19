"""Validate every experiments/results/<Disease>.json against the canonical
schema in opencure.scoring.common. Catches the class of bug where a pillar
starts writing to a new field name without a corresponding read-path update
(the silent-zero regression that bit v3/v4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencure.scoring.common import (
    LEGACY_FIELDS,
    REQUIRED_CANDIDATE_FIELDS,
    validate_candidate,
    validate_result_file,
)


RESULTS_DIR = Path("experiments/results")
# Files produced by post-processors / aggregation, not single-disease
# screen outputs. Skipped.
NON_DISEASE_FILES = {"screening_summary", "novel_candidates",
                     "opencure_database"}


def _result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return [p for p in sorted(RESULTS_DIR.glob("*.json"))
            if p.stem not in NON_DISEASE_FILES]


@pytest.mark.parametrize("path", _result_files(), ids=lambda p: p.stem)
def test_result_json_matches_schema(path: Path) -> None:
    data = json.loads(path.read_text())
    warnings = validate_result_file(data)
    assert not warnings, (
        f"{path.name} violates canonical schema:\n  - "
        + "\n  - ".join(warnings[:20])
    )


def test_schema_constants_are_frozen() -> None:
    """Forbidden-legacy field names must not leak into the allowed sets."""
    from opencure.scoring.common import CANDIDATE_FIELDS
    assert not (LEGACY_FIELDS & CANDIDATE_FIELDS), (
        "LEGACY_FIELDS overlap CANDIDATE_FIELDS — validator will never reject them."
    )


def test_required_fields_are_canonical() -> None:
    """REQUIRED must be a subset of the overall CANDIDATE schema."""
    from opencure.scoring.common import CANDIDATE_FIELDS
    assert REQUIRED_CANDIDATE_FIELDS <= CANDIDATE_FIELDS


def test_validate_candidate_rejects_legacy_name() -> None:
    bad = {
        "drug_id": "DB00001", "drug_name": "X", "disease_name": "Y",
        "combined_score": 0.5, "pillars_hit": 1, "confidence": "LOW",
        "pgx_flags": {},  # legacy
    }
    warnings = validate_candidate(bad)
    assert any("legacy" in w for w in warnings)


def test_validate_candidate_rejects_missing_required() -> None:
    warnings = validate_candidate({"drug_id": "DB00001"})
    assert any("missing required" in w for w in warnings)


def test_validate_candidate_accepts_minimal_record() -> None:
    ok = {
        "drug_id": "DB00001", "drug_name": "X", "disease_name": "Y",
        "combined_score": 0.5, "pillars_hit": 1, "confidence": "LOW",
    }
    assert validate_candidate(ok) == []
