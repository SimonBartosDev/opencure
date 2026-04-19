"""
Canonical types and field-name constants for the scoring pipeline.

Single source of truth — any pillar producing scores should use these
names, and any consumer (report.py, dashboard, mass_screen) should read
only these names. Eliminates the class of bug where v3 refactor left
two field names for the same concept (`proximity_raw_score` vs
`proximity_score`) and caused silent zeros downstream.
"""

from __future__ import annotations

from typing import Optional, TypedDict


# -------- Canonical score-dict field names --------
# Per-pillar scores (all normalized to [0, 1] unless noted)
PILLAR_FIELDS: tuple[str, ...] = (
    "transe_score",        # DRKG TransE  (raw log-score, not normalized)
    "transe_rank",         # int rank among compounds
    "pykeen_score",        # DRKG RotatE
    "pykeen_rank",
    "primekg_score",       # PrimeKG TransE
    "unified_score",       # v5 Phase 5 unified-KG TransE (0-1 rank-normalized)
    "unified_rank",
    "txgnn_score",         # rank-normalized to 0-1
    "txgnn_rank",
    "mol_similarity",      # Morgan fingerprint (0-1)
    "similar_to",          # str — most-similar known treatment
    "mol_emb_similarity",  # ChemBERTa (0-1)
    "mol_emb_similar_to",
    "gene_sig_score",      # L1000 + mechanistic reversal
    "gene_sig_rank",
    "proximity_score",     # STRING PPI (0-1)
    "proximity_distance",
    "mr_score",            # Mendelian randomization (0-1)
    "mr_genetic_targets",
    "admet_score",         # Chemprop drug-likeness (0-1)
    "admet_flags",
    "dti_score",           # DeepPurpose (0-1)
    "dti_best_target",
    "rgcn_score",          # v5 heterogeneous GNN (trained model required)
)


# Group-level scores (emitted by grouped_combiner)
GROUP_FIELDS: tuple[str, ...] = (
    "kg_group_score",          # RRF of transe+pykeen+primekg+unified
    "structural_group_score",  # max(mol_fp, mol_emb, dti)
    "network_group_score",     # max(proximity, gene_sig)
    # individual ungrouped scores re-echoed at group level for clarity
    "txgnn_group_score",       # = txgnn_score
    "mr_group_score",          # = mr_score
    # derived
    "admet_multiplier",        # [0.3, 1.0] multiplier
    "efficacy_score",          # weighted_sum + convergence_bonus (pre-ADMET)
    "degree_penalty",          # hub damping multiplier for kg+network
    "groups_hit",
    "pillars_hit",
)


# Combined final
FINAL_FIELDS: tuple[str, ...] = (
    "combined_score",          # efficacy_score × admet_multiplier
    "base_weighted_sum",
    "convergence_bonus",
)


class PillarScore(TypedDict, total=False):
    """Canonical per-compound scoring payload.

    All keys in PILLAR_FIELDS + GROUP_FIELDS + FINAL_FIELDS are permitted.
    Type checker will complain about typos when TypedDict is used.
    """
    # Pillars
    transe_score: float
    transe_rank: int
    pykeen_score: float
    pykeen_rank: int
    primekg_score: float
    unified_score: float
    unified_rank: int
    txgnn_score: float
    txgnn_rank: int
    mol_similarity: float
    similar_to: str
    mol_emb_similarity: float
    mol_emb_similar_to: str
    gene_sig_score: float
    gene_sig_rank: int
    proximity_score: float
    proximity_distance: int
    mr_score: float
    mr_genetic_targets: int
    admet_score: float
    admet_flags: str
    dti_score: float
    dti_best_target: str
    rgcn_score: float
    # Groups
    kg_group_score: float
    structural_group_score: float
    network_group_score: float
    txgnn_group_score: float
    mr_group_score: float
    admet_multiplier: float
    efficacy_score: float
    degree_penalty: float
    groups_hit: int
    pillars_hit: int
    # Final
    combined_score: float
    base_weighted_sum: float
    convergence_bonus: float
    # Metadata
    disease_entity: str
    transe_relation: str


ALL_SCORE_FIELDS: frozenset[str] = frozenset(PILLAR_FIELDS + GROUP_FIELDS + FINAL_FIELDS)
