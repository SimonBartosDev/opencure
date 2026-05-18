"""Regression tests for the v7 binding novelty re-ranking (WS4).

The scoring pipeline orders candidates by combined_score, which is blind to
whether a drug is already standard-of-care. apply_novelty_ranking() demotes
known treatments below genuine repurposing leads in the surfaced top-K.
"""
from __future__ import annotations

from opencure.scoring.novelty_rank import (
    apply_novelty_ranking,
    is_repurposing_candidate,
)


# ---- Schema ------------------------------------------------------------

def test_field_in_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS, V7_FIELDS
    assert "is_repurposing_candidate" in V7_FIELDS
    assert "is_repurposing_candidate" in CANDIDATE_FIELDS


# ---- is_repurposing_candidate -----------------------------------------

def test_known_treatment_edge_is_not_a_lead() -> None:
    assert is_repurposing_candidate({"is_known_treatment": True}) is False


def test_known_novelty_level_is_not_a_lead() -> None:
    assert is_repurposing_candidate({"novelty_level": "KNOWN"}) is False
    assert is_repurposing_candidate({"novelty_level": "ESTABLISHED"}) is False
    assert is_repurposing_candidate({"novelty_level": "known"}) is False  # case-insensitive


def test_novel_levels_are_leads() -> None:
    for lvl in ("NOVEL", "EMERGING", "BREAKTHROUGH"):
        assert is_repurposing_candidate({"novelty_level": lvl}) is True


def test_missing_signal_defaults_to_lead() -> None:
    """We only demote on positive evidence of being known."""
    assert is_repurposing_candidate({}) is True


# ---- apply_novelty_ranking --------------------------------------------

def test_known_treatment_demoted_out_of_top_k() -> None:
    """The sickle-cell failure mode: a KNOWN drug with the highest
    combined_score must not occupy the surfaced top-K."""
    candidates = [
        {"drug_name": "Hydromorphone", "combined_score": 0.287, "novelty_level": "KNOWN"},
        {"drug_name": "Bromfenac", "combined_score": 0.283, "novelty_level": "NOVEL"},
        {"drug_name": "Meclofenamic acid", "combined_score": 0.261, "novelty_level": "BREAKTHROUGH"},
    ]
    ordered = apply_novelty_ranking(candidates)

    top_k = ordered[:2]
    assert all(c.get("novelty_level") != "KNOWN" for c in top_k)
    assert ordered[0]["drug_name"] == "Bromfenac"      # highest-scoring lead
    assert ordered[-1]["drug_name"] == "Hydromorphone"  # demoted to tail


def test_known_treatment_kept_not_dropped() -> None:
    """Demoted candidates remain in the output, flagged."""
    candidates = [
        {"drug_name": "A", "combined_score": 0.9, "is_known_treatment": True},
        {"drug_name": "B", "combined_score": 0.1, "novelty_level": "NOVEL"},
    ]
    ordered = apply_novelty_ranking(candidates)
    assert len(ordered) == 2
    demoted = [c for c in ordered if not c["is_repurposing_candidate"]]
    assert len(demoted) == 1 and demoted[0]["drug_name"] == "A"


def test_combined_score_order_preserved_within_partition() -> None:
    candidates = [
        {"drug_name": "lead_lo", "combined_score": 0.2, "novelty_level": "NOVEL"},
        {"drug_name": "lead_hi", "combined_score": 0.8, "novelty_level": "NOVEL"},
        {"drug_name": "known_hi", "combined_score": 0.9, "novelty_level": "KNOWN"},
        {"drug_name": "known_lo", "combined_score": 0.5, "novelty_level": "KNOWN"},
    ]
    ordered = apply_novelty_ranking(candidates)
    assert [c["drug_name"] for c in ordered] == [
        "lead_hi", "lead_lo", "known_hi", "known_lo",
    ]


def test_rank_is_rewritten_1_indexed() -> None:
    candidates = [
        {"drug_name": "A", "combined_score": 0.5, "rank": None, "novelty_level": "NOVEL"},
        {"drug_name": "B", "combined_score": 0.9, "rank": None, "novelty_level": "KNOWN"},
    ]
    ordered = apply_novelty_ranking(candidates)
    assert [c["rank"] for c in ordered] == [1, 2]


def test_empty_list_is_safe() -> None:
    assert apply_novelty_ranking([]) == []


def test_malformed_combined_score_does_not_crash() -> None:
    candidates = [
        {"drug_name": "A", "combined_score": None, "novelty_level": "NOVEL"},
        {"drug_name": "B", "combined_score": "oops", "novelty_level": "NOVEL"},
    ]
    ordered = apply_novelty_ranking(candidates)
    assert len(ordered) == 2
