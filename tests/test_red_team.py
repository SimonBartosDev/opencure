"""Regression tests for the v7 red-team agent.

The deterministic critic must:
- Always return a non-empty string (no candidate ever ships without a
  critique attached).
- Catch the canonical failure modes: single-pillar artifact, low
  selectivity, essentiality warning, hub damping, low mechanism
  confidence, evidence shortage, failed-trial history.
- Never raise on malformed input — defensive parsing.
"""
from __future__ import annotations

import pytest

from opencure.scoring.red_team import (
    HUB_DEGREE_DAMPING_THRESHOLD,
    LOW_MECHANISM_CONFIDENCE,
    LOW_SELECTIVITY_THRESHOLD,
    SINGLE_PILLAR_RATIO,
    assess_candidate,
    critique_candidate,
)


def _baseline_candidate() -> dict:
    """A 'reasonable' candidate that should produce no critique."""
    return {
        "drug_id": "DB00001", "drug_name": "Test", "disease_name": "X",
        "combined_score": 0.6, "pillars_hit": 4, "confidence": "MEDIUM",
        "transe_score": 0.55, "pykeen_score": 0.50, "primekg_score": 0.45,
        "mol_emb_similarity": 0.40, "txgnn_score": 0.35,
        "selectivity_score": 0.8, "n_off_targets": 4,
        "primary_target": "EGFR", "essentiality_warning": False,
        "degree_penalty": 0.95,
        "pubmed_total": 12, "clinical_trials_total": 1,
    }


# ---- Always-non-empty contract ----------------------------------------

def test_assess_candidate_always_returns_string() -> None:
    """Every candidate produces a string — never None, never empty."""
    cand = _baseline_candidate()
    out = assess_candidate(cand, mechanism_confidence=0.8)
    assert isinstance(out, str)
    assert len(out) > 0


def test_assess_handles_completely_empty_candidate() -> None:
    """No fields populated → still returns a string (not a crash)."""
    out = assess_candidate({}, mechanism_confidence=None)
    assert isinstance(out, str)
    assert len(out) > 0


def test_assess_with_use_llm_falls_back_when_mlx_missing() -> None:
    """LLM path returns the deterministic critique when MLX isn't installed."""
    cand = _baseline_candidate()
    out = assess_candidate(cand, use_llm=True)
    assert isinstance(out, str)
    assert len(out) > 0


# ---- Failure modes the deterministic critic should catch -------------

def test_single_pillar_artifact_flagged() -> None:
    """One pillar dominates all others → flagged."""
    cand = _baseline_candidate()
    cand.update({
        "transe_score": 0.95,
        "pykeen_score": 0.05, "primekg_score": 0.0,
        "mol_emb_similarity": 0.05, "txgnn_score": 0.05,
    })
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("Single-pillar" in r for r in rep.risks)


def test_low_selectivity_flagged() -> None:
    cand = _baseline_candidate()
    cand["selectivity_score"] = LOW_SELECTIVITY_THRESHOLD - 0.05
    cand["n_off_targets"] = 20
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("selectivity" in r.lower() for r in rep.risks)


def test_essentiality_warning_flagged() -> None:
    cand = _baseline_candidate()
    cand.update({
        "essentiality_warning": True,
        "primary_target": "RPL5",
        "target_essentiality": -1.5,
    })
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("essential" in r.lower() for r in rep.risks)
    assert any("RPL5" in r for r in rep.risks)


def test_hub_damping_flagged() -> None:
    cand = _baseline_candidate()
    cand["degree_penalty"] = 0.3  # heavily damped
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("Hub-damping" in r for r in rep.risks)


def test_low_mechanism_confidence_flagged() -> None:
    cand = _baseline_candidate()
    rep = critique_candidate(cand, mechanism_confidence=0.2)
    assert any("mechanism poorly mapped" in r.lower() for r in rep.risks)


def test_zero_evidence_flagged() -> None:
    cand = _baseline_candidate()
    cand["pubmed_total"] = 0
    cand["clinical_trials_total"] = 0
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("Zero PubMed" in r or "no literature" in r.lower() for r in rep.risks)


def test_failed_trial_history_flagged() -> None:
    cand = _baseline_candidate()
    cand["has_failed_trial"] = True
    cand["failed_trial_phase"] = 3
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert any("FAILED" in r and "Phase 3" in r for r in rep.risks)


# ---- Clean candidate produces no critique ---------------------------

def test_baseline_candidate_has_no_red_flags() -> None:
    """A balanced candidate should not trip any deterministic flags."""
    cand = _baseline_candidate()
    rep = critique_candidate(cand, mechanism_confidence=0.8)
    assert rep.risks == []
    assert rep.to_text() == "No structural red flags detected."


# ---- Schema integration ----------------------------------------------

def test_red_team_field_in_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS
    assert "red_team_assessment" in CANDIDATE_FIELDS


# ---- Stability: garbage in shouldn't crash ---------------------------

def test_garbage_pillar_values_dont_crash() -> None:
    """String / None pillar values are tolerated."""
    cand = {
        "drug_id": "DBX", "drug_name": "Test",
        "transe_score": "not a number",
        "pykeen_score": None,
        "selectivity_score": "weird",
        "primary_target": None,
        "pubmed_total": "0",
        "clinical_trials_total": None,
    }
    out = assess_candidate(cand, mechanism_confidence=None)
    assert isinstance(out, str)
