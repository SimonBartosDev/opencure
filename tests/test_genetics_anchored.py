"""Tests for the genetics-anchored production scorer.

Covers: the result schema, the honest not-assessed paths (no EFO mapping; no
genetics-datasource associations), a genetics-covered disease returning ranked
mechanism-grounded candidates, and precision discipline (no banned datasources,
no bioactivity-only targets — only curated ChEMBL drug_mechanism targets).

All tests run OFFLINE (allow_network=False) so they rely only on the on-disk
Open Targets parquet + the genetics caches that ship with the repo.
"""
from __future__ import annotations

import pytest

from opencure.scoring.genetics_anchored import (
    GENETICS_USE,
    _BANNED,
    DrugCandidate,
    GeneticsResult,
    score_disease,
)

# A disease known to be genetics-covered in the shipped cache (SCN5A-anchored).
COVERED_DISEASE = "atrial fibrillation"
# A disease that maps to Open Targets but has NO genetics-datasource evidence.
MAPPED_NO_GENETICS = "gonorrhea"
# A string that matches no Open Targets disease at all.
UNMAPPED_DISEASE = "wibble syndrome xyz nonexistent"


@pytest.fixture(scope="module")
def covered_result() -> GeneticsResult:
    return score_disease(COVERED_DISEASE, allow_network=False)


# --------------------------------------------------------------------------
# result schema
# --------------------------------------------------------------------------
def test_result_schema(covered_result):
    r = covered_result
    assert isinstance(r, GeneticsResult)
    assert r.status in {"covered", "not_assessed"}
    assert r.covered == (r.status == "covered")
    assert r.not_assessed == (not r.covered)
    d = r.to_dict()
    for key in ("disease_query", "covered", "status", "reason",
                "ot_disease_ids", "candidates"):
        assert key in d


def test_candidate_schema(covered_result):
    assert covered_result.covered, "fixture disease should be covered"
    for c in covered_result.candidates:
        assert isinstance(c, DrugCandidate)
        assert c.drug_id and isinstance(c.drug_id, str)
        assert isinstance(c.score, float) and c.score > 0.0
        assert c.target_gene_id and c.target_gene_id.startswith("ENSG")
        assert isinstance(c.target_genetics_score, float)
        assert isinstance(c.evidence_datasources, list)
        # action_type comes from ChEMBL drug_mechanism — may be None for a
        # mechanism row with no recorded action, but the field must exist.
        cd = c.to_dict()
        assert "action_type" in cd and "target_gene_symbol" in cd


# --------------------------------------------------------------------------
# honest not-assessed path
# --------------------------------------------------------------------------
def test_unmapped_disease_is_not_assessed():
    r = score_disease(UNMAPPED_DISEASE, allow_network=False)
    assert r.not_assessed
    assert r.status == "not_assessed"
    assert r.candidates == []
    assert r.reason and "no EFO mapping" in r.reason


def test_mapped_but_no_genetics_is_not_assessed():
    r = score_disease(MAPPED_NO_GENETICS, allow_network=False)
    assert r.not_assessed
    assert r.status == "not_assessed"
    # it DID map to an Open Targets entity ...
    assert r.ot_disease_ids
    # ... but returns no ranking, with an explicit reason.
    assert r.candidates == []
    assert r.reason and "no genetics-datasource associations" in r.reason


def test_not_assessed_never_emits_a_ranking():
    """The core honest behavior: no signal -> empty candidate list, always."""
    for q in (UNMAPPED_DISEASE, MAPPED_NO_GENETICS):
        r = score_disease(q, allow_network=False)
        assert r.not_assessed and not r.candidates


# --------------------------------------------------------------------------
# covered disease -> ranked, mechanism-grounded candidates
# --------------------------------------------------------------------------
def test_covered_disease_returns_ranked_candidates(covered_result):
    r = covered_result
    assert r.covered
    assert r.status == "covered"
    assert r.reason is None
    assert len(r.candidates) >= 1
    # ranked: scores are non-increasing.
    scores = [c.score for c in r.candidates]
    assert scores == sorted(scores, reverse=True)


def test_each_candidate_carries_a_target_gene_and_evidence(covered_result):
    for c in covered_result.candidates:
        # every candidate is grounded in a specific genetic target gene ...
        assert c.target_gene_id.startswith("ENSG")
        # ... and the score equals that gene's genetics score (max-over-targets).
        assert c.score == c.target_genetics_score
        # evidence_datasources is a list (may be empty for legacy score-cache
        # diseases, populated for API-fetched ones) — but must be present.
        assert isinstance(c.evidence_datasources, list)


def test_top_k_truncates(covered_result):
    full = len(covered_result.candidates)
    if full >= 3:
        r = score_disease(COVERED_DISEASE, top_k=3, allow_network=False)
        assert len(r.candidates) == 3
        # truncation keeps the highest-scoring candidates.
        assert r.candidates[0].score == covered_result.candidates[0].score


# --------------------------------------------------------------------------
# precision discipline
# --------------------------------------------------------------------------
def test_genetics_whitelist_excludes_all_banned_datasources():
    # the leak-control whitelist must never overlap the banned set.
    assert GENETICS_USE.isdisjoint(_BANNED)
    # chembl (known-drug) and literature are banned by construction.
    assert "chembl" in _BANNED
    assert "europepmc" in _BANNED


def test_no_banned_datasource_appears_in_candidate_evidence(covered_result):
    for c in covered_result.candidates:
        for ds in c.evidence_datasources:
            assert ds in GENETICS_USE, f"non-genetics datasource leaked: {ds}"
            assert ds not in _BANNED


def test_mechanism_targets_only_no_bioactivity_supplement():
    """Precision discipline: candidates must come from curated ChEMBL
    drug_mechanism only — the module never reads the promiscuous
    measured-bioactivity target file (which the research showed dilutes
    precision)."""
    import opencure.scoring.genetics_anchored as mod
    src = mod.__file__
    text = open(src).read()
    # the bioactivity file / potency threshold must not be referenced.
    assert "drug_target_activities" not in text
    assert "POTENCY_NM" not in text
    # and the curated mechanism crosswalk must be the target source.
    assert "_chembl_mechanism" in text
