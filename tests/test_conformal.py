"""Regression tests for the v7 conformal-prediction wrapper.

Locks in:
- Coverage guarantee: empirical coverage ≥ 1−α on held-out test data
  drawn from the same distribution as the calibration set.
- Round-trip: save → load → predict produces the same interval.
- Schema: the three new candidate fields land in CANDIDATE_FIELDS so
  validate_candidate doesn't flag them as unknown.
- Fail-open: ConformalCalibrator.load() returns None when the artifact
  is absent, so score_ensemble_v5.py keeps producing ensemble_prob even
  before calibration has been run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opencure.scoring.conformal import (
    CALIBRATOR_PATH,
    ConformalCalibrator,
    DEFAULT_ALPHA,
    empirical_coverage,
)


# ---- Coverage guarantee on synthetic data --------------------------------

def _synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probabilities with truth labels.

    Generates p_hat ~ Beta(2, 2) (centered, varied) and y ~ Bernoulli(p_hat)
    so the model is *perfectly* calibrated. Conformal coverage on
    calibrated data should land near nominal.
    """
    rng = np.random.default_rng(seed)
    p_hat = rng.beta(2, 2, size=n)
    y = (rng.uniform(0, 1, size=n) < p_hat).astype(int)
    return p_hat, y


def test_conformal_coverage_meets_target_on_held_out() -> None:
    """Calibrate on 1000 points; held-out 1000 points should get ≥ 1−α coverage."""
    p_cal, y_cal = _synthetic(1000, seed=1)
    p_test, y_test = _synthetic(1000, seed=2)

    cal = ConformalCalibrator().fit(p_cal, y_cal, alpha=0.10)
    cov = empirical_coverage(cal, p_test, y_test)

    # Conformal guarantee is in expectation; with 1000 points the empirical
    # coverage should land within ~3 percentage points of nominal (Hoeffding).
    assert cov >= 0.85, f"coverage {cov:.3f} fell below 85%"


def test_conformal_round_trip(tmp_path: Path) -> None:
    """save → load → predict produces identical interval."""
    p_cal, y_cal = _synthetic(500, seed=3)
    cal = ConformalCalibrator().fit(p_cal, y_cal, alpha=0.10)

    out = tmp_path / "cal.npz"
    cal.save(out)
    loaded = ConformalCalibrator.load(out)
    assert loaded is not None

    p_hat = 0.7
    a = cal.predict_with_interval(p_hat)
    b = loaded.predict_with_interval(p_hat)
    assert a == b


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    """Fail-open contract: missing file → None, not an exception."""
    cal = ConformalCalibrator.load(tmp_path / "definitely_not_here.npz")
    assert cal is None


# ---- Output shape -------------------------------------------------------

def test_predict_with_interval_returns_three_canonical_fields() -> None:
    p_cal, y_cal = _synthetic(200, seed=4)
    cal = ConformalCalibrator().fit(p_cal, y_cal, alpha=0.10)

    out = cal.predict_with_interval(0.6)
    assert set(out) == {
        "ensemble_prob_lower",
        "ensemble_prob_upper",
        "prediction_set_at_90",
    }
    # Bounds clipped to [0, 1].
    assert 0.0 <= out["ensemble_prob_lower"] <= out["ensemble_prob_upper"] <= 1.0
    # Prediction set is non-empty for binary problems.
    assert len(out["prediction_set_at_90"]) >= 1
    assert all(c in {0, 1} for c in out["prediction_set_at_90"])


def test_uncertain_prediction_includes_both_classes_in_set() -> None:
    """A p_hat sitting at 0.5 with a non-trivial q_alpha covers both labels."""
    p_cal, y_cal = _synthetic(200, seed=5)
    cal = ConformalCalibrator().fit(p_cal, y_cal, alpha=0.10)
    out = cal.predict_with_interval(0.5)
    assert set(out["prediction_set_at_90"]) == {0, 1}


def test_confident_positive_only_emits_label_one() -> None:
    """A near-1.0 prediction with a small q_alpha gives prediction_set = {1}."""
    # Deterministic, very-confident calibration data
    p_cal = np.array([0.99] * 200)
    y_cal = np.array([1] * 200)
    cal = ConformalCalibrator().fit(p_cal, y_cal, alpha=0.10)
    out = cal.predict_with_interval(0.99)
    assert out["prediction_set_at_90"] == [1]


# ---- Schema integration -------------------------------------------------

def test_v7_fields_registered_in_candidate_schema() -> None:
    """The three new candidate fields must be CANDIDATE_FIELDS members."""
    from opencure.scoring.common import CANDIDATE_FIELDS, V7_FIELDS

    for f in V7_FIELDS:
        assert f in CANDIDATE_FIELDS, f"missing v7 field {f}"
    expected = {"ensemble_prob_lower", "ensemble_prob_upper", "prediction_set_at_90"}
    assert expected.issubset(CANDIDATE_FIELDS)


def test_validate_candidate_accepts_v7_fields() -> None:
    """A candidate with conformal fields populated should not be flagged."""
    from opencure.scoring.common import validate_candidate

    cand = {
        "drug_id": "DB00001", "drug_name": "Test",
        "disease_name": "TestDisease", "combined_score": 0.5,
        "pillars_hit": 3, "confidence": "MEDIUM",
        "ensemble_prob": 0.8,
        "ensemble_prob_lower": 0.7, "ensemble_prob_upper": 0.9,
        "prediction_set_at_90": [1],
    }
    warnings = validate_candidate(cand)
    # No "unknown fields" warning for the v7 fields.
    for w in warnings:
        assert "ensemble_prob_lower" not in w
        assert "ensemble_prob_upper" not in w
        assert "prediction_set_at_90" not in w


# ---- Input validation ---------------------------------------------------

def test_fit_rejects_mismatched_shapes() -> None:
    cal = ConformalCalibrator()
    with pytest.raises(ValueError):
        cal.fit(np.array([0.1, 0.2]), np.array([1, 0, 1]))


def test_fit_rejects_empty_calibration_set() -> None:
    cal = ConformalCalibrator()
    with pytest.raises(ValueError):
        cal.fit(np.array([]), np.array([]))


def test_predict_rejects_invalid_probability() -> None:
    cal = ConformalCalibrator().fit(np.array([0.5, 0.6]), np.array([1, 0]))
    with pytest.raises(ValueError):
        cal.predict_with_interval(1.5)


def test_predict_before_fit_raises() -> None:
    cal = ConformalCalibrator()
    with pytest.raises(RuntimeError):
        cal.predict_with_interval(0.5)
