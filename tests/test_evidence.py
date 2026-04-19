"""Unit tests for evidence modules: DDI, pharmacogenomics, dose plausibility,
triangulation, tissue context."""

import pytest
from opencure.evidence.ddi_warnings import get_ddi_warnings
from opencure.evidence.pharmacogenomics import get_pharmacogenomic_flags
from opencure.evidence.dose_plausibility import get_dose_plausibility
from opencure.evidence.triangulation import compute_triangulation_score


class TestDDI:

    @pytest.mark.integration
    def test_warfarin_has_many_interactions(self):
        w = get_ddi_warnings("DB00682", top_k=20)
        assert w["n_interactions"] > 100, f"Warfarin should have many DDIs, got {w['n_interactions']}"

    @pytest.mark.integration
    def test_unknown_drug_no_warnings(self):
        w = get_ddi_warnings("DB_BOGUS_ID")
        assert w["n_interactions"] == 0
        assert not w["has_warnings"]

    @pytest.mark.integration
    def test_high_severity_drugs_flagged(self):
        """Aspirin should surface high-severity interactions with common
        co-prescribed drugs (statins, warfarin, clarithromycin)."""
        w = get_ddi_warnings("DB00945", top_k=10)
        high = [t for t in w["top_interactions"] if t["severity"] == "high"]
        assert len(high) > 0


class TestPharmacogenomics:

    @pytest.mark.integration
    @pytest.mark.parametrize("name,expected_level,expected_gene", [
        ("Warfarin", "high_risk", "VKORC1|CYP2C9|CYP4F2"),
        ("Abacavir", "high_risk", "HLA-B"),
        ("Clopidogrel", "high_risk", "CYP2C19"),
        ("Codeine", "high_risk", "CYP2D6"),
    ])
    def test_known_high_risk_drugs(self, name, expected_level, expected_gene):
        f = get_pharmacogenomic_flags(name)
        assert f["has_flags"], f"{name} should have pharmacogenomic flags"
        assert f["highest_risk"] == expected_level
        genes = expected_gene.split("|")
        assert any(g in f["summary"] for g in genes), \
            f"{name}: expected one of {genes} in summary {f['summary']!r}"

    def test_unknown_drug_no_flags(self):
        f = get_pharmacogenomic_flags("TotallyMadeUpDrug")
        assert not f["has_flags"]
        assert f["highest_risk"] == ""


class TestDosePlausibility:

    @pytest.mark.integration
    def test_approved_drug_plausibility(self):
        p = get_dose_plausibility("DB00843")  # Donepezil, phase 4
        assert p["plausibility"] == "yes"
        assert p["confidence"] == "high"

    def test_unknown_drug_gets_unknown(self):
        p = get_dose_plausibility("DB_BOGUS")
        assert p["plausibility"] == "unknown"
        assert p["confidence"] == "low"

    @pytest.mark.integration
    def test_stage_2_upgrade_when_target_given(self):
        """If ChEMBL activity cache is present, passing a target upgrades
        the profile to stage 2 with Cmax/IC50 reasoning."""
        p = get_dose_plausibility("DB00843", target_symbol="ACHE")
        if p.get("target_affinity"):
            # Stage-2 data present
            assert p["stage"] == 2
            ta = p["target_affinity"]
            assert "median_ic50_nM" in ta
            assert "cmax_over_ic50_ratio" in ta
            assert ta["mechanism_feasible"] in ("yes", "borderline", "no")


class TestTriangulation:

    def test_all_four_axes_hit(self):
        """Strong candidate with KG + docking + Pharos + literature should
        score 'silver-standard' at 4/4 axes."""
        r = compute_triangulation_score(
            kg_score=0.7,
            docking_score=-9.0,
            pharos_tdl="Tclin",
            pubmed_total=80,
        )
        assert r["n_axes_agree"] == 4
        assert r["label"] == "silver-standard"

    def test_kg_only_no_label(self):
        r = compute_triangulation_score(
            kg_score=0.5,
            docking_score=None,
            pharos_tdl="",
            pubmed_total=0,
        )
        assert r["n_axes_agree"] == 1
        assert r["label"] == "kg-only"

    def test_zero_axes(self):
        r = compute_triangulation_score(
            kg_score=0.1, docking_score=None, pharos_tdl="", pubmed_total=0,
        )
        assert r["n_axes_agree"] == 0
        assert r["label"] == ""

    def test_score_in_unit_range(self):
        r = compute_triangulation_score(
            kg_score=0.8, docking_score=-10.0, pharos_tdl="Tchem", pubmed_total=200,
        )
        assert 0 <= r["triangulation_score"] <= 1


class TestTissueContext:

    @pytest.mark.integration
    def test_parkinson_brain_genes_boosted(self):
        from opencure.scoring.tissue_context import score_tissue_context
        # SNCA entrez=6622 is a classic Parkinson gene; should be highly
        # expressed in substantia nigra
        r = score_tissue_context("Parkinsons disease", {"Gene::6622"})
        if r.get("n_genes", 0) > 0:
            assert r["context_modifier"] > 1.0, \
                "PD-relevant gene in PD tissue should give boost >1.0"

    @pytest.mark.integration
    def test_irrelevant_gene_neutral(self):
        from opencure.scoring.tissue_context import score_tissue_context
        # Gene that doesn't exist
        r = score_tissue_context("Parkinsons disease", {"Gene::99999999"})
        # Should fall back to neutral modifier 1.0
        assert r.get("context_modifier", 1.0) == 1.0
