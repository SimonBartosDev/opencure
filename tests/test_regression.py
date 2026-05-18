"""Regression tests — catch the classes of bugs we've already shipped.

The 3-pillar silent-zero bug (v3-v5.0, caught April 2026) is the
canonical reminder why this file exists.
"""

import pytest


class TestPillarFieldNameRegression:
    """Previously, search.py wrote proximity_raw_score / dti_score_raw / no
    gene_sig_score but result-dict assembly read proximity_score /
    dti_score / gene_sig_score → silent zeros. Test the output contract.
    """

    @pytest.mark.integration
    @pytest.mark.slow
    def test_live_search_surfaces_all_pillars(self):
        """Running a real search must produce non-zero scores for at least
        the pillars that have broad coverage (TransE, ADMET, PrimeKG,
        Proximity, DTI)."""
        from opencure.search import search
        cands = search("Malaria", top_k=10)
        assert len(cands) > 0

        pillars_always_present = ["transe_score", "admet_score", "primekg_score"]
        for field in pillars_always_present:
            n = sum(1 for c in cands if c.get(field) not in (None, 0))
            assert n > 0, f"pillar {field!r} must fire on at least 1 of 10 Malaria top"

        # Previously-broken pillars must now fire on ≥5 of 10 Malaria top
        for field in ["proximity_score", "dti_score"]:
            n = sum(1 for c in cands if c.get(field) not in (None, 0))
            assert n >= 5, f"pillar {field!r} fires {n}/10 — regression?"

    @pytest.mark.integration
    def test_no_raw_leak_in_final_json(self):
        """No '*_raw' or '*_raw_score' fields should appear in final output."""
        from opencure.search import search
        cands = search("Tuberculosis", top_k=5)
        for c in cands:
            raw_fields = [k for k in c if k.endswith("_raw") or k.endswith("_raw_score")]
            assert not raw_fields, f"stray _raw fields in output: {raw_fields}"


class TestFieldNameCanonicality:
    """Field names declared in opencure/scoring/common.py must be the ONLY
    names used throughout the pipeline. Typos are regressions."""

    def test_pillar_fields_importable(self):
        from opencure.scoring.common import PILLAR_FIELDS, GROUP_FIELDS, FINAL_FIELDS, ALL_SCORE_FIELDS
        assert len(PILLAR_FIELDS) > 10
        assert len(GROUP_FIELDS) > 5
        assert len(FINAL_FIELDS) > 0
        # No accidental duplicates
        assert len(set(ALL_SCORE_FIELDS)) == len(ALL_SCORE_FIELDS)


class TestFilterRegression:
    """Any previously-observed leaker must stay blacklisted."""

    @pytest.mark.parametrize("name", [
        # Documented leakers from v3 results audit (should now all reject).
        # Note: Folic Acid intentionally NOT in this list — phase-4 bypass
        # lets approved folate supplementation through; we only reject when
        # it appears as a prediction against an unrelated disease via the
        # blacklist+phase gate in drug_filter.is_therapeutic_candidate.
        "Uric Acid", "Glutathione", "L-Alanine", "Cordycepin Triphosphate",
        "16,17-Androstene-3-Ol", "Creatinine",
        "Oxidized Glutathione Disulfide", "Dihydrofolic Acid",
    ])
    def test_documented_leakers_stay_blacklisted(self, name):
        from opencure.filters.metabolite_blacklist import is_blacklisted_metabolite
        rejected, _ = is_blacklisted_metabolite(name)
        assert rejected, f"regression: {name!r} must stay blacklisted"


class TestMechanisticReversalLiveFix:
    """The Apr-2026 fix added mechanistic_reversal.py as the gene-sig
    fallback when L1000CDS2 coverage is thin (~5-10 matched drugs per
    disease). Verify the module is importable and exposes the entry
    point.  Full live regression is in TestPillarFieldNameRegression.
    """

    def test_module_import(self):
        from opencure.scoring.mechanistic_reversal import (
            score_mechanistic_reversal, _load_activity_index, _load_disease_gene_map,
        )
        assert callable(score_mechanistic_reversal)

    @pytest.mark.integration
    def test_produces_scores_for_malaria(self):
        from opencure.scoring.mechanistic_reversal import (
            score_mechanistic_reversal, _load_activity_index,
        )
        activities = _load_activity_index()
        if not activities:
            pytest.skip("ChEMBL activities cache not present")
        compounds = [f"Compound::{c}" for c in list(activities.keys())[:500]]
        out = score_mechanistic_reversal(
            ["Disease::MESH:D008288"],  # Malaria
            compounds,
            top_k=50,
        )
        # Non-empty output means the pillar has coverage for this disease
        assert len(out) > 0, "mechanistic reversal should find at least SOME scorers for Malaria"
