"""Conformal-prediction wrapper around the v5 ensemble.

The v5 ensemble (XGBoost + isotonic calibration) emits a calibrated
probability ``ensemble_prob``. That number is *calibrated in expectation*
— the empirical positive rate among predictions of 0.7 is ~0.7. What it
isn't is *individually* calibrated: the platform has no honest answer
to "how confident is this 0.85 — could it be 0.6?".

This module adds the missing piece: split conformal prediction. Given a
held-out calibration set ``(p_hat_i, y_i)`` we compute the empirical
quantile of the absolute residuals; for any new prediction we return a
distribution-free interval ``[p_hat - q_α, p_hat + q_α]`` that contains
the true label with probability ≥ 1 − α (over the calibration sample +
test point exchangeability assumption).

We also emit the conformal prediction *set* for binary classification:
the labels whose probability survives the conformal threshold. A
candidate whose set is ``{1}`` is conformally-confidently positive;
``{0, 1}`` means the platform genuinely doesn't know.

Why a 50-line in-tree implementation instead of MAPIE / crepes:
- numpy is the only runtime dependency (already in tree).
- The contract is small and frozen — three outputs, no model surgery.
- Easy to reason about for the methods paper.

v7 calibration set: ``data/eval/holdout_test.jsonl`` (993 positives) plus
matched random negatives sampled by ``scripts/calibrate_conformal.py``.
The fitted calibrator lives at ``data/models/conformal_v7.npz``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_ALPHA = 0.10  # 1 - α = 90% coverage
CALIBRATOR_PATH = Path("data/models/conformal_v7.npz")


class ConformalCalibrator:
    """Split conformal calibrator for a single calibrated probability classifier.

    Fit on ``(p_hat, y)`` pairs from a held-out set the model never saw.
    Stateless once fit — just remembers the conformal quantile and the
    target coverage. Persists as a tiny npz so it's CI-cheap to ship.
    """

    def __init__(self) -> None:
        self.q_alpha: Optional[float] = None
        self.cal_size: int = 0
        self.alpha: float = DEFAULT_ALPHA

    def fit(
        self,
        p_hats: np.ndarray,
        y: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
    ) -> "ConformalCalibrator":
        """Compute the conformal quantile from calibration residuals.

        Uses the standard finite-sample correction
        ⌈(n+1)(1−α)⌉ / n (Vovk et al., Lei et al.).
        """
        if p_hats.shape != y.shape:
            raise ValueError("p_hats and y must have the same shape")
        if p_hats.size == 0:
            raise ValueError("calibration set must be non-empty")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        residuals = np.abs(y.astype(float) - p_hats.astype(float))
        n = residuals.size
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        sorted_residuals = np.sort(residuals)
        self.q_alpha = float(sorted_residuals[k - 1])
        self.cal_size = int(n)
        self.alpha = float(alpha)
        return self

    def predict_with_interval(self, p_hat: float) -> dict:
        """Return the conformal interval and prediction set for one ``p_hat``.

        Output shape (always populated, never raises on a fit calibrator):

        - ``ensemble_prob_lower``: ``max(0, p_hat - q_α)``
        - ``ensemble_prob_upper``: ``min(1, p_hat + q_α)``
        - ``prediction_set_at_90``: list of int labels in {0, 1} that are
          consistent with the conformal interval at the configured
          coverage level. Empty set is impossible for binary problems
          (one of {0, 1} is always within distance q_α of p_hat).
        """
        if self.q_alpha is None:
            raise RuntimeError("Calibrator has not been fit; call fit() first.")
        if not (0.0 <= p_hat <= 1.0):
            raise ValueError("p_hat must be a probability in [0, 1]")

        lower = max(0.0, p_hat - self.q_alpha)
        upper = min(1.0, p_hat + self.q_alpha)

        pred_set: list[int] = []
        if upper >= 0.5:
            pred_set.append(1)
        if lower <= 0.5:
            pred_set.append(0)

        return {
            "ensemble_prob_lower": lower,
            "ensemble_prob_upper": upper,
            "prediction_set_at_90": pred_set,
        }

    def save(self, path: Path = CALIBRATOR_PATH) -> None:
        if self.q_alpha is None:
            raise RuntimeError("Cannot save an unfit calibrator")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            q_alpha=np.array(self.q_alpha),
            cal_size=np.array(self.cal_size),
            alpha=np.array(self.alpha),
        )

    @classmethod
    def load(cls, path: Path = CALIBRATOR_PATH) -> Optional["ConformalCalibrator"]:
        """Return a fitted calibrator from disk, or ``None`` if absent.

        Callers should treat ``None`` as "no conformal coverage available"
        and emit the bare ``ensemble_prob`` without bounds; the platform
        stays operational instead of failing closed.
        """
        if not path.exists():
            return None
        try:
            data = np.load(str(path))
            cal = cls()
            cal.q_alpha = float(data["q_alpha"])
            cal.cal_size = int(data["cal_size"])
            cal.alpha = float(data["alpha"])
            return cal
        except Exception:
            return None


def empirical_coverage(
    calibrator: ConformalCalibrator,
    p_hats: np.ndarray,
    y: np.ndarray,
) -> float:
    """Fraction of test points whose true label sits inside the interval.

    Should be ≥ 1 − α on a fresh test set drawn from the same
    distribution as the calibration set. Used by the calibration script
    and the methods paper to report empirical coverage alongside the
    nominal target.
    """
    if calibrator.q_alpha is None:
        raise RuntimeError("Calibrator not fit")
    lower = np.clip(p_hats - calibrator.q_alpha, 0.0, 1.0)
    upper = np.clip(p_hats + calibrator.q_alpha, 0.0, 1.0)
    inside = (y >= lower) & (y <= upper)
    return float(inside.mean())
