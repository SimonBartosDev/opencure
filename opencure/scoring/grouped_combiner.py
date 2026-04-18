"""
Grouped-feature score combiner for OpenCure v3.

Replaces _combine_scores_v2 with a principled approach:
  1. Pillars already grouped (kg, structural, network) to avoid double-counting
  2. Each of 6 groups weighted once (no double-count bonuses)
  3. Convergence bonus scaled by GROUP diversity, not pillar count
  4. MR appears once, not twice
  5. ADMET contributes to drug-likeness, not filtering (filtering is separate)

Output: calibrated score in [0, 1.5] range where:
  - 1.0+ = strong multi-group evidence
  - 0.5-1.0 = moderate evidence (some groups)
  - <0.5 = weak evidence
"""

from __future__ import annotations

# Group weights — chosen for orthogonal signal strength.
# ADMET is NOT an efficacy signal — treated as a multiplier, not summand.
# Keys MUST match the feature dict keys produced by build_feature_matrix.
EFFICACY_GROUPS = {
    "kg_score":         0.25,  # Fused TransE + RotatE + PrimeKG
    "txgnn_score":      0.22,  # Best-in-class GNN
    "network_score":    0.20,  # Proximity + GeneSig
    "structural_score": 0.18,  # MolFP + ChemBERTa + DTI
    "mr_score":         0.15,  # Causal genetic evidence
}
# Weights sum to 1.0 exactly

# A compound MUST hit this many efficacy groups to be considered a candidate
MIN_EFFICACY_GROUPS = 2

# Convergence bonus: reward drugs supported by MANY orthogonal groups
CONVERGENCE_BONUS_PER_GROUP = 0.05
CONVERGENCE_START = 3  # Bonus kicks in at 3rd group


def combine_grouped_scores(features: dict) -> dict:
    """
    Combine per-compound group features into final score.

    Args:
        features: dict[compound] -> {"kg_score", "structural_score", "network_score",
                                     "txgnn_score", "mr_score", "admet_score",
                                     "groups_hit"}

    Returns:
        dict[compound] -> {"combined_score", "base_weighted_sum",
                           "convergence_bonus", "groups_hit", **group_scores}
    """
    combined = {}

    for compound, feats in features.items():
        # Collect EFFICACY groups (signals that the drug could work)
        active_groups = []
        for group_name, group_weight in EFFICACY_GROUPS.items():
            score = feats.get(group_name, 0)
            if score > 0:
                active_groups.append((group_name, group_weight, score))

        # Require minimum efficacy signals to be considered
        groups_hit = len(active_groups)
        if groups_hit < MIN_EFFICACY_GROUPS:
            continue

        # Use FIXED weights (not renormalized): missing pillar = 0 contribution.
        # This properly penalizes drugs with narrow evidence vs. drugs with
        # broad multi-group support.
        base_weighted_sum = sum(w * s for _, w, s in active_groups)

        # Convergence bonus for multi-group agreement
        convergence_bonus = max(0, groups_hit - CONVERGENCE_START + 1) * CONVERGENCE_BONUS_PER_GROUP

        # ADMET as multiplier (drug-likeness scales the score)
        # - admet=1.0 → full score
        # - admet=0.5 → half score
        # - admet=0 (missing) → pass through (0.7 default)
        admet_score = feats.get("admet_score", 0)
        admet_multiplier = (0.5 + admet_score * 0.5) if admet_score > 0 else 0.7

        efficacy_score = base_weighted_sum + convergence_bonus
        final_score = efficacy_score * admet_multiplier

        combined[compound] = {
            "combined_score": round(final_score, 4),
            "efficacy_score": round(efficacy_score, 4),
            "base_weighted_sum": round(base_weighted_sum, 4),
            "convergence_bonus": round(convergence_bonus, 4),
            "admet_multiplier": round(admet_multiplier, 4),
            "groups_hit": groups_hit,
            "pillars_hit": groups_hit,  # compatibility
            # Individual group scores for transparency
            "kg_score": feats.get("kg_score", 0),
            "txgnn_score": feats.get("txgnn_score", 0),
            "network_score": feats.get("network_score", 0),
            "structural_score": feats.get("structural_score", 0),
            "mr_score": feats.get("mr_score", 0),
            "admet_score": admet_score,
        }

    return combined
