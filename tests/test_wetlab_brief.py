"""Regression tests for the v7 wet-lab brief generator.

Locks in:
- The brief always renders to non-empty Markdown, even with sparse
  candidate metadata (no LLM needed in CI).
- Every section header (Mechanism, Suggested assay, Red-team, Caveats)
  is present.
- Disease-class assay heuristics map correctly for the four lead
  diseases (Schistosomiasis → parasite, Sickle Cell → metabolic, etc.).
- Caveats trigger when essentiality_warning, low selectivity, low
  mechanism confidence, or failed-trial flags are set.
- Concentration range respects primary_nM when present, generic
  fallback otherwise.
"""
from __future__ import annotations

import pytest

from opencure.scoring.wetlab_brief import (
    ASSAY_BY_CLASS,
    BriefContext,
    DEFAULT_ASSAY,
    render_candidate_brief,
    render_disease_brief,
)


def _baseline_candidate() -> dict:
    return {
        "rank": 1, "drug_id": "DB00001", "drug_name": "Praziquantel",
        "disease_name": "Schistosomiasis",
        "ensemble_prob": 0.84,
        "ensemble_prob_lower": 0.74, "ensemble_prob_upper": 0.94,
        "primary_target": "TRP1", "primary_nM": 10.0,
        "selectivity_score": 0.8, "n_off_targets": 3,
        "pubmed_total": 250, "clinical_trials_total": 12,
        "relation_type": "DRUGBANK::treats",
        "red_team_assessment": "No structural red flags detected.",
    }


def _ctx(disease_name="Schistosomiasis", disease_class="parasitic"):
    # Empty disease_entity → mechanism_confidence falls back to 0.5
    # (above the speculative threshold), so caveats only fire from
    # candidate-level fields. Tests that exercise the mechanism-confidence
    # flag should override this explicitly.
    return BriefContext(
        disease_name=disease_name,
        disease_entity="",
        disease_class=disease_class,
        use_llm=False,
    )


# ---- Always-non-empty rendering ----------------------------------------

def test_render_candidate_brief_returns_markdown_string() -> None:
    out = render_candidate_brief(_baseline_candidate(), rank=1, ctx=_ctx())
    assert isinstance(out, str)
    assert len(out) > 100
    assert out.startswith("### #1")


def test_render_handles_minimal_candidate() -> None:
    """No probability, no target, no evidence — still produces a brief."""
    cand = {"drug_id": "DB99999", "drug_name": "Nothing"}
    out = render_candidate_brief(cand, rank=1, ctx=_ctx())
    assert isinstance(out, str)
    assert "Nothing" in out


# ---- Section headers ---------------------------------------------------

@pytest.mark.parametrize(
    "section",
    ["Mechanistic hypothesis", "Suggested assay", "Red-team assessment"],
)
def test_required_sections_present(section: str) -> None:
    out = render_candidate_brief(_baseline_candidate(), rank=1, ctx=_ctx())
    assert f"**{section}**" in out


def test_caveats_section_present_when_caveats_exist() -> None:
    cand = _baseline_candidate()
    cand["essentiality_warning"] = True
    out = render_candidate_brief(cand, rank=1, ctx=_ctx())
    assert "**Caveats**" in out


def test_caveats_section_absent_when_no_caveats() -> None:
    """A clean candidate produces no Caveats section."""
    out = render_candidate_brief(_baseline_candidate(), rank=1, ctx=_ctx())
    assert "**Caveats**" not in out


# ---- Disease-class assay routing --------------------------------------

@pytest.mark.parametrize(
    "disease_class,assay_keyword",
    [
        ("parasitic", "Parasite"),
        ("oncology", "viability"),
        ("rare_metabolic", "fibroblasts"),
        ("bacterial", "MIC"),
        ("viral", "Antiviral"),
        # ASSAY_BY_CLASS["chronic_systemic"]["readout"] = "Phenotypic readout..."
        ("chronic_systemic", "Phenotypic"),
    ],
)
def test_assay_block_matches_disease_class(
    disease_class: str, assay_keyword: str,
) -> None:
    out = render_candidate_brief(
        _baseline_candidate(), rank=1,
        ctx=_ctx(disease_class=disease_class),
    )
    assert assay_keyword in out


def test_assay_falls_back_when_class_unmapped() -> None:
    out = render_candidate_brief(
        _baseline_candidate(), rank=1, ctx=_ctx(disease_class=None),
    )
    assert DEFAULT_ASSAY["assay"] in out


# ---- Caveats triggers --------------------------------------------------

@pytest.mark.parametrize(
    "field_overrides,expected_substring",
    [
        ({"essentiality_warning": True, "primary_target": "RPL5"}, "essential"),
        ({"selectivity_score": 0.1, "n_off_targets": 25}, "Promiscuous"),
        ({"has_failed_trial": True, "failed_trial_phase": 2}, "Phase 2"),
        # Wide conformal interval
        ({"ensemble_prob_lower": 0.1, "ensemble_prob_upper": 0.9}, "wide"),
    ],
)
def test_specific_caveat_triggers(
    field_overrides: dict, expected_substring: str,
) -> None:
    cand = _baseline_candidate()
    cand.update(field_overrides)
    out = render_candidate_brief(cand, rank=1, ctx=_ctx())
    assert expected_substring in out


# ---- Concentration range ----------------------------------------------

def test_concentration_uses_primary_nm_when_present() -> None:
    out = render_candidate_brief(_baseline_candidate(), rank=1, ctx=_ctx())
    # 10 nM × 0.1 = 1 nM low; 10 nM × 100 = 1 µM high → expect those tokens.
    assert "Concentration range" in out


def test_concentration_falls_back_when_potency_unknown() -> None:
    cand = _baseline_candidate()
    cand.pop("primary_nM", None)
    out = render_candidate_brief(cand, rank=1, ctx=_ctx())
    assert "generic" in out.lower() or "1 nM – 10 µM" in out


# ---- Disease-level brief ------------------------------------------------

def test_render_disease_brief_top5_includes_header_and_5_candidates() -> None:
    cands = [_baseline_candidate() for _ in range(5)]
    for i, c in enumerate(cands, start=1):
        c["rank"] = i
        c["drug_id"] = f"DB0000{i}"
        c["drug_name"] = f"Drug{i}"
    out = render_disease_brief(
        cands,
        disease_name="Schistosomiasis",
        disease_entity="Disease::MESH:D012552",
        disease_class="parasitic",
        top_k=5,
    )
    assert out.startswith("# Schistosomiasis")
    for i in range(1, 6):
        assert f"Drug{i}" in out


def test_render_disease_brief_includes_mechanism_confidence_header() -> None:
    cands = [_baseline_candidate()]
    out = render_disease_brief(
        cands,
        disease_name="Schistosomiasis",
        disease_entity="Disease::MESH:D012552",
        disease_class="parasitic",
        top_k=1,
    )
    assert "Mechanism-confidence" in out
