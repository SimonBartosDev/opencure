"""Regression tests for the v7 selectivity / off-target panel."""
from __future__ import annotations

import pytest

from opencure.scoring.selectivity_panel import (
    PENALTY_MAX,
    POTENCY_NM,
    SATURATION_OFF_TARGETS,
    compute_drug_selectivity,
    compute_selectivity_table,
    selectivity_penalty,
)


# ---- Schema ------------------------------------------------------------

def test_selectivity_fields_in_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS, V7_FIELDS
    for f in ("selectivity_score", "n_off_targets", "primary_target", "primary_nM"):
        assert f in V7_FIELDS, f"{f} missing from V7_FIELDS"
        assert f in CANDIDATE_FIELDS


# ---- Per-drug selectivity ---------------------------------------------

def test_no_targets_returns_optimistic_default() -> None:
    """A drug with no ChEMBL data → selectivity 1.0, n_off_targets 0.

    'No data' is the optimistic interpretation; this matches the rest
    of v5's fail-open posture.
    """
    rec = compute_drug_selectivity({})
    assert rec["selectivity_score"] == 1.0
    assert rec["n_off_targets"] == 0
    assert rec["primary_target"] == ""


def test_single_potent_hit_is_perfectly_selective() -> None:
    rec = compute_drug_selectivity(
        {"EGFR": {"median_nM": 5.0, "n": 3, "activity_types": ["IC50"]}}
    )
    assert rec["selectivity_score"] == 1.0
    assert rec["n_off_targets"] == 0
    assert rec["primary_target"] == "EGFR"
    assert rec["primary_nM"] == pytest.approx(5.0)


def test_inactive_targets_are_excluded() -> None:
    """Targets with median > POTENCY_NM don't count as off-targets."""
    rec = compute_drug_selectivity({
        "EGFR": {"median_nM": 5.0},  # primary
        "JAK2": {"median_nM": POTENCY_NM * 5},  # well above threshold — ignored
    })
    assert rec["primary_target"] == "EGFR"
    assert rec["n_off_targets"] == 0
    assert rec["selectivity_score"] == 1.0


def test_promiscuous_compound_loses_selectivity_proportionally() -> None:
    """Adding off-targets monotonically reduces selectivity_score until 0."""
    targets = {f"GENE_{i}": {"median_nM": 100.0} for i in range(10)}
    targets["EGFR"] = {"median_nM": 5.0}  # primary
    rec = compute_drug_selectivity(targets, saturation=20)

    # 10 off-targets, saturation=20 → selectivity = 1 - 10/20 = 0.5.
    assert rec["selectivity_score"] == pytest.approx(0.5, abs=0.01)
    assert rec["n_off_targets"] == 10


def test_pan_inhibitor_floors_at_zero() -> None:
    """50 off-targets at saturation=20 → selectivity clamped to 0."""
    targets = {f"GENE_{i}": {"median_nM": 100.0} for i in range(50)}
    rec = compute_drug_selectivity(targets, saturation=20)

    assert rec["selectivity_score"] == 0.0
    assert rec["n_off_targets"] == 49  # primary excluded


def test_primary_target_picks_best_potency() -> None:
    """Primary = lowest median_nM, regardless of dict insertion order."""
    rec = compute_drug_selectivity({
        "GENE_A": {"median_nM": 1000.0},
        "GENE_B": {"median_nM": 5.0},  # best
        "GENE_C": {"median_nM": 100.0},
    })
    assert rec["primary_target"] == "GENE_B"
    assert rec["primary_nM"] == pytest.approx(5.0)


def test_malformed_target_records_are_skipped() -> None:
    """Defensive: garbage in shouldn't crash the panel."""
    rec = compute_drug_selectivity({
        "GENE_A": {"median_nM": 5.0},
        "GENE_B": "not a dict",  # malformed
        "GENE_C": {"median_nM": "not a number"},  # bad value
        "GENE_D": {},  # missing median
    })
    assert rec["primary_target"] == "GENE_A"
    assert rec["n_off_targets"] == 0


# ---- Bulk table -------------------------------------------------------

def test_bulk_table_returns_one_record_per_drug() -> None:
    activities = {
        "DB00001": {"EGFR": {"median_nM": 5.0}},
        "DB00002": {f"G{i}": {"median_nM": 100.0} for i in range(50)},
    }
    table = compute_selectivity_table(activities)
    assert set(table) == {"DB00001", "DB00002"}
    assert table["DB00001"]["selectivity_score"] == 1.0
    assert table["DB00002"]["selectivity_score"] < 0.1


# ---- Penalty ----------------------------------------------------------

def test_penalty_unchanged_for_perfectly_selective() -> None:
    assert selectivity_penalty(0.8, selectivity=1.0) == pytest.approx(0.8)


def test_penalty_reduces_score_for_promiscuous() -> None:
    """At selectivity 0, score is multiplied by (1 - PENALTY_MAX)."""
    out = selectivity_penalty(1.0, selectivity=0.0)
    assert out == pytest.approx(1.0 - PENALTY_MAX)


def test_penalty_is_monotone_in_selectivity() -> None:
    s1 = selectivity_penalty(1.0, selectivity=0.2)
    s2 = selectivity_penalty(1.0, selectivity=0.6)
    s3 = selectivity_penalty(1.0, selectivity=0.9)
    assert s1 < s2 < s3


def test_penalty_never_amplifies() -> None:
    """No selectivity value should ever scale the score above 1.0."""
    for sel in (-1.0, 0.0, 0.3, 0.5, 0.99, 1.0, 1.5):
        assert selectivity_penalty(1.0, selectivity=sel) <= 1.0
