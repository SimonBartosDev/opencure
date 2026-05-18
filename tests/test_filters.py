"""Unit tests for the v5 hard filters (metabolite, name heuristics, drug filter)."""

import pytest
from opencure.filters.metabolite_blacklist import is_blacklisted_metabolite, blacklist_size
from opencure.filters.name_heuristics import looks_like_research_chemical
from opencure.filters.drug_filter import is_therapeutic_candidate, check_smiles_rules


class TestMetaboliteBlacklist:

    @pytest.mark.parametrize("name", [
        "Uric Acid", "Glutathione", "L-Alanine", "ATP",
        "Cordycepin Triphosphate", "16,17-Androstene-3-Ol",
        "Creatinine", "Oxygen", "Fluoride Ion",
        "Dihydrofolic Acid", "Oxidized Glutathione Disulfide",
    ])
    def test_known_metabolites_rejected(self, name):
        rejected, cat = is_blacklisted_metabolite(name)
        assert rejected, f"{name!r} should be blacklisted"
        assert cat, f"{name!r} rejection should have a category label"

    @pytest.mark.parametrize("name", [
        "Aspirin", "Donepezil", "Hydroxyurea", "Artemisinin",
        "Clarithromycin", "Tenofovir", "Paclitaxel",
    ])
    def test_real_drugs_pass(self, name):
        rejected, _ = is_blacklisted_metabolite(name)
        assert not rejected, f"{name!r} should NOT be blacklisted"

    def test_phase_4_bypass(self):
        """Approved drugs (phase 4) bypass blacklist even if they share a name."""
        rejected, _ = is_blacklisted_metabolite("Glutathione", chembl_phase=4.0)
        assert not rejected, "phase-4 bypass should let Glutathione through"

    def test_blacklist_size_sanity(self):
        assert 50 < blacklist_size() < 500, "blacklist should be a curated set"


class TestNameHeuristics:

    @pytest.mark.parametrize("name", [
        "2-(3-GUANIDINOPHENYL)-3-MERCAPTOPROPANOIC ACID",
        "5-Amidino-Benzimidazole",
        "1-(2,6-Dichlorophenyl)-5-(2,4-Difluorophenyl)-7-Piperidin-4-Yl-3,4-Dihydroquinolin-2(1h)-One",
        "6-[N-(4-(Aminomethyl)Phenyl)Carbamyl]-2-Naphthalenecarboxamidine",
        "16,17-Androstene-3-Ol",
    ])
    def test_iupac_rejected(self, name):
        assert looks_like_research_chemical(name), f"{name!r} should read as IUPAC"

    @pytest.mark.parametrize("name", [
        "Aspirin", "Donepezil", "Hydroxyurea", "Paclitaxel",
        "Artemisinin", "Pembrolizumab", "Celecoxib",
        "N-Acetylcysteine",      # IUPAC-ish prefix but valid INN
        "1,3-Butadiene",          # short locant-only; heuristic lets through
        "L-Alanine", "Uric Acid", "Folic Acid", "Creatinine",
    ])
    def test_real_drug_names_pass(self, name):
        assert not looks_like_research_chemical(name), f"{name!r} should NOT trip the IUPAC heuristic"


class TestDrugFilter:

    def test_aspirin_passes(self):
        # Aspirin SMILES
        ok, reason = is_therapeutic_candidate(
            "DB00945",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            drug_name="Aspirin",
        )
        assert ok, f"Aspirin rejected with reason: {reason}"

    def test_donepezil_passes(self):
        ok, reason = is_therapeutic_candidate(
            "DB00843",
            "O=C1C2=CC(OC)=C(OC)C=C2CC1CC1CCN(CC1)Cc1ccccc1",
            drug_name="Donepezil",
        )
        assert ok, f"Donepezil rejected with reason: {reason}"

    def test_uric_acid_rejected_as_metabolite(self):
        ok, reason = is_therapeutic_candidate(
            "DB_UNKNOWN",
            "O=C1NC(=O)NC2=C1NC(=O)N2",  # simplified uric-acid-ish
            drug_name="Uric Acid",
        )
        assert not ok
        assert reason.startswith("metabolite"), f"expected metabolite rejection, got {reason!r}"

    def test_tiny_molecule_rejected(self):
        ok, reason = is_therapeutic_candidate("X", "C", drug_name="Methane")
        assert not ok
        assert "smiles" in reason

    def test_smiles_rules_min_heavy_atoms(self):
        # SMILES must be >=3 chars AND >=4 heavy atoms; methane "C" fails
        # the string-length gate first; n-propane "CCC" clears length but
        # fails the heavy-atom count (3 < 4).
        ok, reason = check_smiles_rules("CCC")
        assert not ok
        assert "too_small" in reason

    def test_smiles_rules_needs_carbon(self):
        ok, reason = check_smiles_rules("N")  # nitrogen only, single atom — too small first
        assert not ok
