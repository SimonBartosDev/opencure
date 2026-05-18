"""Regression tests for the v7 per-class ensemble routing layer.

Locks in:
- Every disease in ``experiments/systematic_screening.py`` maps to a
  class (or is consciously left unmapped).
- The four lead diseases for outreach (Schistosomiasis, Chagas, Sickle
  Cell, Niemann-Pick) route to the right classes.
- ``route_disease`` is tolerant of case + apostrophe drift.
- ``load_class_head`` and ``score_with_routing`` fail open to the
  shared head when per-class artifacts are missing.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from opencure.scoring.per_class_ensemble import (
    DISEASE_CLASSES_PATH,
    load_class_head,
    load_disease_class_map,
    reset_caches,
    route_disease,
    score_with_routing,
)


@pytest.fixture(autouse=True)
def _isolate_caches():
    reset_caches()
    yield
    reset_caches()


# ---- Class map sanity ---------------------------------------------------

def test_disease_classes_yaml_exists() -> None:
    assert DISEASE_CLASSES_PATH.exists()


def test_load_disease_class_map_returns_normalized_keys() -> None:
    m = load_disease_class_map()
    assert m, "class map cannot be empty"
    # Every value should be one of the canonical 6 classes.
    expected_classes = {
        "parasitic", "viral", "bacterial",
        "oncology", "rare_metabolic", "chronic_systemic",
    }
    assert set(m.values()).issubset(expected_classes)


def test_class_map_is_six_buckets() -> None:
    m = load_disease_class_map()
    classes = set(m.values())
    assert len(classes) == 6


# ---- Lead-disease routing ----------------------------------------------

@pytest.mark.parametrize(
    "disease,expected",
    [
        ("Schistosomiasis", "parasitic"),
        ("Chagas disease", "parasitic"),
        ("Sickle cell disease", "rare_metabolic"),
        ("Niemann-Pick disease", "rare_metabolic"),
    ],
)
def test_lead_diseases_route_to_correct_class(disease: str, expected: str) -> None:
    assert route_disease(disease) == expected


@pytest.mark.parametrize(
    "disease,expected",
    [
        # Cancers → oncology
        ("Breast cancer", "oncology"),
        ("Glioblastoma", "oncology"),
        # Viral
        ("HIV", "viral"),
        ("COVID-19", "viral"),
        # Bacterial
        ("Tuberculosis", "bacterial"),
        ("Leprosy", "bacterial"),
        # Chronic systemic
        ("Alzheimer's disease", "chronic_systemic"),
        ("Rheumatoid arthritis", "chronic_systemic"),
    ],
)
def test_class_routing_breadth(disease: str, expected: str) -> None:
    assert route_disease(disease) == expected


def test_route_tolerates_case_and_apostrophe_drift() -> None:
    assert route_disease("schistosomiasis") == "parasitic"
    assert route_disease("ALZHEIMER'S DISEASE") == "chronic_systemic"
    assert route_disease("Alzheimer’s disease") == "chronic_systemic"  # smart-quote


def test_route_unknown_disease_returns_none() -> None:
    assert route_disease("not_a_real_disease_in_the_world") is None


def test_route_empty_string_returns_none() -> None:
    assert route_disease("") is None


# ---- Coverage check -----------------------------------------------------

def test_every_screened_disease_has_a_class() -> None:
    """Every disease in TARGET_DISEASES should resolve to a class.

    If this fails, someone added a disease to systematic_screening.py
    without updating disease_classes.yaml.
    """
    from experiments.systematic_screening import TARGET_DISEASES

    unmapped = []
    for diseases in TARGET_DISEASES.values():
        for d in diseases:
            if route_disease(d) is None:
                unmapped.append(d)
    assert not unmapped, (
        f"{len(unmapped)} screened diseases have no class assignment: "
        f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}"
    )


# ---- Loader fail-open --------------------------------------------------

def test_load_class_head_returns_none_when_artifact_missing() -> None:
    """Without ``data/models/ensemble_v7_parasitic.pkl`` the loader
    returns None — caller falls back to the shared head."""
    head = load_class_head("parasitic_definitely_not_present")
    assert head is None


# ---- score_with_routing -----------------------------------------------

class _FakeModel:
    """Minimal sklearn-style stub: predict_proba on a 6-feature input."""
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Returns 0.7 for any input — distinguishable from anything real.
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


def _feats() -> dict[str, float]:
    return {
        "kg_score": 0.5,
        "degree_penalty": 1.0,
        "n_drug_targets": 5,
        "is_fda_approved": 1,
        "n_disease_genes": 100,
        "transe_rank_log": 2.0,
    }


def test_score_with_routing_uses_shared_head_when_class_unmapped() -> None:
    shared = _FakeModel()
    keys = ("kg_score", "degree_penalty", "n_drug_targets",
            "is_fda_approved", "n_disease_genes", "transe_rank_log")

    p, head = score_with_routing(
        "totally_unknown_disease",
        _feats(),
        shared_model=shared,
        shared_feature_keys=keys,
    )
    assert head == "shared"
    assert pytest.approx(p, abs=1e-6) == 0.7


def test_score_with_routing_uses_shared_head_when_class_artifact_missing() -> None:
    """Class IS resolved (Schistosomiasis → parasitic) but the per-class
    head pkl doesn't exist → falls back to shared."""
    shared = _FakeModel()
    keys = ("kg_score", "degree_penalty", "n_drug_targets",
            "is_fda_approved", "n_disease_genes", "transe_rank_log")

    p, head = score_with_routing(
        "Schistosomiasis", _feats(),
        shared_model=shared, shared_feature_keys=keys,
    )
    assert head == "shared"
    assert pytest.approx(p, abs=1e-6) == 0.7


def test_score_with_routing_uses_class_head_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    """When the per-class pkl exists, the routing layer uses it.

    We fit a tiny sklearn DummyClassifier so the pickle is real (not
    a locally-defined stub class that pickle can't handle).
    """
    from sklearn.dummy import DummyClassifier
    from opencure.scoring import per_class_ensemble as pce

    keys = ("kg_score", "degree_penalty", "n_drug_targets",
            "is_fda_approved", "n_disease_genes", "transe_rank_log")
    # DummyClassifier with constant strategy → returns the constant class probs.
    clf = DummyClassifier(strategy="constant", constant=1)
    # Fit on a 2-class dummy dataset; predict_proba then returns ~[0, 1].
    clf.fit(np.zeros((2, len(keys))), np.array([0, 1]))

    pkl = tmp_path / "ensemble_v7_parasitic.pkl"
    with pkl.open("wb") as fh:
        pickle.dump({"model": clf, "feature_keys": keys}, fh)

    monkeypatch.setattr(pce, "PER_CLASS_MODEL_DIR", tmp_path)
    pce.reset_caches()

    p, head = pce.score_with_routing(
        "Schistosomiasis", _feats(),
        shared_model=_FakeModel(), shared_feature_keys=keys,
    )
    assert head == "parasitic"
    # constant=1 strategy → P(y=1) = 1.0
    assert p > 0.9
