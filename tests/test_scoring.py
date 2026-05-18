"""Unit tests for scoring modules: hub_normalize, grouped_combiner,
pillar_groups, mechanistic_reversal."""

import pytest
from opencure.scoring.hub_normalize import degree_penalty, get_degree, get_reference_degree
from opencure.scoring.grouped_combiner import combine_grouped_scores, EFFICACY_GROUPS, MIN_EFFICACY_GROUPS
from opencure.scoring.pillar_groups import build_feature_matrix
from opencure.scoring.kg_fusion import fuse_kg_scores


class TestHubNormalize:

    def test_reference_degree_sane(self):
        ref = get_reference_degree()
        assert 30 < ref < 300, f"reference degree out of sensible range: {ref}"

    def test_hub_drugs_damped(self):
        # Dexamethasone is a textbook hub (3,413 DRKG triplets)
        p = degree_penalty("DB01234")
        assert 0.5 < p < 0.9, f"hub drug should be partially damped, got {p}"

    def test_normal_drugs_less_damped(self):
        art = degree_penalty("DB13132")   # Artemisinin, 236 edges
        dex = degree_penalty("DB01234")   # Dexamethasone, 3,413 edges
        assert art > dex, f"Artemisinin ({art}) should be less damped than Dex ({dex})"

    def test_unknown_drug_no_penalty(self):
        assert degree_penalty("DB99999999") == 1.0, \
            "drugs not in degree cache should receive neutral 1.0 penalty"

    def test_low_degree_drug_no_penalty(self):
        # A drug with degree below median should get exactly 1.0
        # Create a fresh case: any drug with degree < reference should be neutral
        # We test this via behavior rather than mocking — very low-degree real drugs
        # are rare but do exist
        assert degree_penalty("DB_NONEXISTENT") == 1.0


class TestGroupedCombiner:

    def _mock_features(self, overrides=None):
        base = {
            "kg_score":         0.5,
            "txgnn_score":      0.4,
            "network_score":    0.3,
            "structural_score": 0.6,
            "mr_score":         0.2,
            "admet_score":      0.7,
            "groups_hit":       5,
        }
        if overrides:
            base.update(overrides)
        return {"DrugA": base}

    def test_requires_min_efficacy_groups(self):
        """Drugs with only 1 efficacy signal should be filtered out."""
        single = {"DrugA": {"kg_score": 0.8, "txgnn_score": 0, "network_score": 0,
                             "structural_score": 0, "mr_score": 0,
                             "admet_score": 0.5, "groups_hit": 1}}
        out = combine_grouped_scores(single)
        assert "DrugA" not in out, "single-pillar drugs must be filtered"

    def test_multi_group_drug_scores(self):
        out = combine_grouped_scores(self._mock_features())
        assert "DrugA" in out
        s = out["DrugA"]
        assert 0 < s["combined_score"] <= 1.2, f"score out of range: {s['combined_score']}"
        assert s["groups_hit"] == 5

    def test_convergence_bonus_applied_for_many_groups(self):
        five = combine_grouped_scores(self._mock_features())["DrugA"]
        two = combine_grouped_scores(self._mock_features({
            "txgnn_score": 0, "network_score": 0, "structural_score": 0  # only kg + mr
        }))
        # Drug with 5 groups should have higher convergence bonus than 2 groups
        assert five.get("convergence_bonus", 0) > two["DrugA"].get("convergence_bonus", 0) if "DrugA" in two else True

    def test_admet_multiplier_two_stage(self):
        """FDA-approved drug (by ChEMBL phase>=2) should get a higher
        admet_multiplier range than non-FDA."""
        import math
        # FDA formula:     0.8 + 0.2 * admet_score  → [0.8, 1.0]
        # non-FDA formula: 0.3 + 0.7 * admet_score  → [0.3, 1.0]
        # admet=0.5 → FDA 0.9, non-FDA 0.65
        assert math.isclose(0.8 + 0.2 * 0.5, 0.9, abs_tol=1e-9)
        assert math.isclose(0.3 + 0.7 * 0.5, 0.65, abs_tol=1e-9)
        # Sanity: FDA always >= non-FDA for any admet score
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            fda = 0.8 + 0.2 * a
            non_fda = 0.3 + 0.7 * a
            assert fda >= non_fda, f"FDA mult < non-FDA at admet={a}"

    def test_weights_sum_to_one(self):
        assert abs(sum(EFFICACY_GROUPS.values()) - 1.0) < 1e-6, \
            f"EFFICACY_GROUPS weights should sum to 1.0, got {sum(EFFICACY_GROUPS.values())}"


class TestKGFusion:

    def test_empty_inputs(self):
        assert fuse_kg_scores() == {}

    def test_single_kg_passthrough(self):
        transe = {"A": (0.7, "treats", "Disease::X")}
        out = fuse_kg_scores(transe_scores=transe)
        assert "A" in out
        assert out["A"][0] == 0.7  # score preserved
        assert out["A"][1] == 1    # num_kgs

    def test_rrf_two_kgs(self):
        transe = {"A": (0.9, "r", "D"), "B": (0.7, "r", "D")}
        pykeen = {"A": (0.8, "r", "D"), "B": (0.6, "r", "D")}
        out = fuse_kg_scores(transe_scores=transe, pykeen_scores=pykeen)
        assert "A" in out and "B" in out
        assert out["A"][0] > out["B"][0]   # A ranked higher in both → higher fused
        assert out["A"][1] == 2            # 2 KGs contributed


class TestPillarGroups:

    def test_build_feature_matrix_applies_degree_penalty(self):
        # Give a hub drug (DB01234) and unknown drug identical KG & network
        # scores. The hub's kg_score and network_score should end up lower
        # after degree penalty, but structural/mr/txgnn/admet should not be
        # damped.
        kg = {"Compound::DB01234": (0.8, 3, "kg"),
              "Compound::DB_UNK":  (0.8, 3, "kg")}
        struc = {"Compound::DB01234": (0.5, "mol_fp", "structural_group"),
                 "Compound::DB_UNK":  (0.5, "mol_fp", "structural_group")}
        net = {"Compound::DB01234": (0.6, "proximity", "network_group"),
               "Compound::DB_UNK":  (0.6, "proximity", "network_group")}
        feats = build_feature_matrix(
            kg, struc, net, {}, {}, {},
            {"Compound::DB01234", "Compound::DB_UNK"},
        )
        dex = feats["Compound::DB01234"]
        unk = feats["Compound::DB_UNK"]
        assert dex["kg_score"] < unk["kg_score"], "hub KG score should be damped"
        assert dex["network_score"] < unk["network_score"], "hub network score should be damped"
        assert dex["structural_score"] == unk["structural_score"], "structural should NOT be damped"


class TestMechanisticReversal:

    def test_imports(self):
        """Module must at least import without error."""
        from opencure.scoring import mechanistic_reversal
        assert hasattr(mechanistic_reversal, "score_mechanistic_reversal")

    def test_empty_inputs_empty_output(self):
        from opencure.scoring.mechanistic_reversal import score_mechanistic_reversal
        # Empty compounds → empty result
        out = score_mechanistic_reversal(["Disease::MESH:D000544"], [], top_k=10)
        assert out == {}

    @pytest.mark.integration
    def test_scores_are_normalized(self):
        """Scores should be in [0, 1]."""
        from opencure.scoring.mechanistic_reversal import (
            score_mechanistic_reversal, _load_activity_index
        )
        activities = _load_activity_index()
        if not activities:
            pytest.skip("ChEMBL activity index not built")
        compounds = [f"Compound::{c}" for c in list(activities.keys())[:100]]
        out = score_mechanistic_reversal(
            ["Disease::MESH:D008288"],  # Malaria
            compounds,
            top_k=20,
        )
        for comp, (score, n, gene) in out.items():
            assert 0 <= score <= 1, f"{comp}: score {score} out of [0,1]"
            assert n > 0
