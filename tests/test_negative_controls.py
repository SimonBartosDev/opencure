"""Regression tests for the v7 negative-control suite.

Locks in the contract:
- YAML loads to the canonical NegativeControl shape.
- universal_hubs entries are merged into every per-disease list.
- Lead diseases (Schistosomiasis, Chagas, Sickle Cell, Niemann-Pick)
  each carry >=3 controls so the four-disease outreach push has
  defensible negative-control coverage from day one.
- The verifier flags a disease whose synthetic top-K puts a control
  above the median, and passes when the controls land below.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencure.eval.negative_control import (
    NEGATIVE_CONTROLS_PATH,
    NegativeControl,
    UNIVERSAL_KEY,
    controls_for_disease,
    load_negative_controls,
    verify_disease_result,
)


# ---- YAML loading -------------------------------------------------------

def test_negative_controls_yaml_exists() -> None:
    assert NEGATIVE_CONTROLS_PATH.exists(), \
        "tests/data/negative_controls.yaml is the canonical control list"


def test_load_returns_negative_control_dataclasses() -> None:
    catalog = load_negative_controls()
    assert UNIVERSAL_KEY in catalog
    for entries in catalog.values():
        for c in entries:
            assert isinstance(c, NegativeControl)
            assert c.drug_id.startswith("DB"), f"bad DrugBank id: {c.drug_id}"
            assert c.drug_name
            assert c.rationale, f"missing rationale for {c.drug_id}"


def test_universal_hubs_present() -> None:
    """Three known hub-leakers must always be in universal_hubs."""
    catalog = load_negative_controls()
    hub_ids = {c.drug_id for c in catalog[UNIVERSAL_KEY]}
    assert {"DB00143", "DB14288", "DB12515"}.issubset(hub_ids)


# ---- Lead-disease coverage ---------------------------------------------

@pytest.mark.parametrize(
    "disease",
    ["Schistosomiasis", "Chagas_disease", "Sickle_cell_disease",
     "Niemann-Pick_disease"],
)
def test_lead_diseases_have_three_curated_controls(disease: str) -> None:
    catalog = load_negative_controls()
    assert disease in catalog, f"missing curated controls for lead disease {disease}"
    assert len(catalog[disease]) >= 3, \
        f"{disease} needs >=3 curated negatives for outreach-grade coverage"


def test_controls_for_disease_merges_universal_hubs() -> None:
    """Per-disease + universal_hubs flow through controls_for_disease()."""
    merged = controls_for_disease("Schistosomiasis")
    drug_ids = {c.drug_id for c in merged}
    # Universal hubs included.
    assert {"DB00143", "DB14288", "DB12515"}.issubset(drug_ids)
    # Disease-specific controls included (DB00030 = insulin, etc.).
    assert "DB00030" in drug_ids


def test_controls_for_disease_dedupes_overlapping_ids() -> None:
    """A drug appearing in both universal_hubs and per-disease should land once."""
    merged = controls_for_disease("Schistosomiasis")
    seen: set[str] = set()
    for c in merged:
        assert c.drug_id not in seen, f"duplicate {c.drug_id}"
        seen.add(c.drug_id)


def test_controls_for_disease_unknown_returns_only_universal() -> None:
    """Unknown disease keys yield universal hubs only — never an exception."""
    merged = controls_for_disease("not_a_real_disease_xyz")
    assert {c.drug_id for c in merged} == {"DB00143", "DB14288", "DB12515"}


# ---- Verifier behavior on synthetic results ----------------------------

def _write_result(
    path: Path,
    *,
    candidates: list[tuple[str, int]],  # (drug_id_no_prefix, rank)
) -> None:
    data = {
        "disease": path.stem,
        "candidates": [
            {
                "rank": rk,
                "drug_id": did,
                "drug_name": did,
                "disease_name": path.stem,
                "combined_score": 0.5,
                "pillars_hit": 3,
                "confidence": "MEDIUM",
            }
            for did, rk in candidates
        ],
    }
    path.write_text(json.dumps(data))


def test_verifier_passes_when_controls_below_median(tmp_path: Path) -> None:
    """Schistosomiasis with insulin (DB00030) at rank 80/100 → passes."""
    result = tmp_path / "Schistosomiasis.json"
    cands = [(f"DB{1000+i:04d}", i + 1) for i in range(50)]  # decoys ranked 1-50
    cands += [("DB00030", 80), ("DB00451", 85), ("DB00682", 90)]  # neg controls deep
    cands += [(f"DB{2000+i:04d}", 51 + i) for i in range(47)]  # filler 51-97 (excluding 80/85/90)
    _write_result(result, candidates=cands)

    report = verify_disease_result("Schistosomiasis", result)
    assert report is not None
    assert report.passed
    assert report.pass_rate == 1.0


def test_verifier_fails_when_control_lands_above_median(tmp_path: Path) -> None:
    """Insulin at rank 5/100 for Schistosomiasis → FAIL."""
    result = tmp_path / "Schistosomiasis.json"
    cands = [("DB00030", 5)]  # insulin in top-5 — bad
    cands += [(f"DB{1000+i:04d}", i + 1) for i in range(100) if i != 4]
    _write_result(result, candidates=cands)

    report = verify_disease_result("Schistosomiasis", result)
    assert report is not None
    assert not report.passed
    # DB00030 should be the offending entry.
    failed_ids = {f[0] for f in report.failures}
    assert "DB00030" in failed_ids


def test_verifier_treats_missing_drug_as_pass(tmp_path: Path) -> None:
    """A negative control that doesn't appear in candidates can't be ranked
    high — counts as a pass (safest interpretation)."""
    result = tmp_path / "Schistosomiasis.json"
    # Only decoys; none of the curated negatives appear.
    cands = [(f"DB{1000+i:04d}", i + 1) for i in range(20)]
    _write_result(result, candidates=cands)

    report = verify_disease_result("Schistosomiasis", result)
    assert report is not None
    # Every control was "missing" → all counted as below median → passes.
    assert report.passed


def test_verifier_returns_none_for_empty_candidates(tmp_path: Path) -> None:
    result = tmp_path / "Schistosomiasis.json"
    result.write_text(json.dumps({"disease": "Schistosomiasis", "candidates": []}))

    report = verify_disease_result("Schistosomiasis", result)
    assert report is None


def test_verifier_returns_none_when_no_controls(tmp_path: Path) -> None:
    """If a disease has no curated controls and no universal hubs match,
    the verifier returns None (skip rather than synthesize-failure)."""
    result = tmp_path / "fictional_disease.json"
    _write_result(result, candidates=[(f"DB{1000+i:04d}", i + 1) for i in range(20)])
    # Patch the catalog to be empty so even universal hubs vanish.
    report = verify_disease_result("fictional_disease", result, catalog={})
    assert report is None
