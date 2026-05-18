"""Regression tests for the retrospective-prospective classifier.

The classifier is a heuristic — it shouldn't be perfect, but it must
catch the most-common positive and negative phrasings in PubMed
abstracts. These tests lock in the behaviour we expect on canonical
abstract excerpts so a future tweak doesn't silently regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrospective_prospective import (  # type: ignore
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    classify_paper,
    render_report,
)


# ---- Classifier ---------------------------------------------------------

def test_positive_paper_classified_as_confirms() -> None:
    title = "Praziquantel shows strong efficacy in adult schistosomiasis"
    abstract = (
        "We treated 200 patients and observed clinical improvement and "
        "successful clearance of S. mansoni eggs."
    )
    assert classify_paper(title, abstract) == "confirms"


def test_negative_paper_classified_as_refutes() -> None:
    title = "Drug X failed to provide benefit in Phase II trial"
    abstract = (
        "The compound showed no significant difference vs placebo on "
        "the primary endpoint; the trial was terminated early due to "
        "lack of efficacy."
    )
    assert classify_paper(title, abstract) == "refutes"


def test_mixed_paper_classified_as_ambiguous() -> None:
    title = "Mixed results for drug Y in disease Z"
    abstract = (
        "Some patients showed improvement while others reported no "
        "benefit; further studies are needed."
    )
    assert classify_paper(title, abstract) == "ambiguous"


def test_irrelevant_paper_classified_as_ambiguous() -> None:
    """No positive or negative terms → ambiguous, not crash."""
    title = "A bibliometric review of repurposing literature"
    abstract = (
        "We surveyed 200 papers across 10 disease areas and counted "
        "publications per year."
    )
    assert classify_paper(title, abstract) == "ambiguous"


def test_classification_is_case_insensitive() -> None:
    out = classify_paper(
        "EFFICACY of Drug X",
        "The TREATMENT was IMPROVED in 80% of patients.",
    )
    assert out == "confirms"


def test_term_lists_are_disjoint() -> None:
    """Sanity: a positive term and a negative term shouldn't both appear in
    the same wordlist (otherwise the classifier double-counts)."""
    pos = set(POSITIVE_TERMS)
    neg = set(NEGATIVE_TERMS)
    assert pos.isdisjoint(neg)


# ---- Report rendering --------------------------------------------------

def test_render_report_handles_empty_rows() -> None:
    out = render_report([])
    assert isinstance(out, str)
    assert "no diseases scored" in out


def test_render_report_includes_per_disease_table() -> None:
    rows = [
        {"disease": "Schistosomiasis", "n_predictions": 5,
         "confirms": 2, "refutes": 0, "ambiguous": 1, "untested": 2},
        {"disease": "Chagas disease", "n_predictions": 5,
         "confirms": 1, "refutes": 1, "ambiguous": 0, "untested": 3},
    ]
    out = render_report(rows)
    assert "Schistosomiasis" in out
    assert "Chagas disease" in out
    # Summary totals
    assert "Predictions evaluated:** 10" in out
    assert "Independent confirmations:** 3" in out


def test_render_report_sorts_by_confirmations_descending() -> None:
    rows = [
        {"disease": "A", "n_predictions": 5, "confirms": 0, "refutes": 0,
         "ambiguous": 0, "untested": 5},
        {"disease": "B", "n_predictions": 5, "confirms": 4, "refutes": 0,
         "ambiguous": 0, "untested": 1},
        {"disease": "C", "n_predictions": 5, "confirms": 2, "refutes": 0,
         "ambiguous": 0, "untested": 3},
    ]
    out = render_report(rows)
    pos_a = out.find("| A |")
    pos_b = out.find("| B |")
    pos_c = out.find("| C |")
    # B (4 confirms) before C (2) before A (0).
    assert pos_b < pos_c < pos_a
