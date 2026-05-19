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
    "rgcn_score",          # v6.1 heterogeneous GNN (trained model required)
    "rgcn_rank",           # rank under R-GCN scoring
    "rgcn_relation",       # which treats-relation gave the best DistMult score
    "jump_score",          # v7 Pillar 13: JUMP-CP morphological similarity (0-1)
    "jump_rank",           # rank by morphological similarity to known treatments
    "jump_similar_to",     # the known-treatment with closest morphological profile
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


# -------- Candidate record schema (full JSON output) --------
# A candidate is a drug-disease prediction row as emitted by the evidence
# pipeline into experiments/results/<Disease>.json. Fields below are the
# canonical names; anything missing is either optional or a bug.

IDENTITY_FIELDS: tuple[str, ...] = (
    "rank",
    "drug_id",
    "drug_entity",
    "drug_name",
    "disease_name",
    "disease_entity",
)

EVIDENCE_FIELDS: tuple[str, ...] = (
    "pubmed_total",
    "pubmed_treatment_total",
    "pubmed_repurposing_total",
    "key_papers",
    "most_cited_paper",
    "max_citations",
    "semantic_scholar_papers",
    "abstract_analyses",
    "repurposing_papers",
    "clinical_trials",
    "clinical_trials_total",
    "trial_phases",
    "failed_trial_count",
    "failed_trial_phase",
    "failed_trial_penalty",
    "has_failed_trial",
    "faers_total_reports",
    "faers_cooccurrences",
    "faers_signal",
    "faers_interpretation",
    "shared_targets",
    "shared_target_count",
    "direct_relations",
    "validation_experiments",
    "ai_hypothesis",
    "mechanistic_hypothesis",
    "kg_paths_text",
    "relation_type",
    "signature_reversal_found",
    "signature_reversal_rank",
    "signature_top_reversers",
    "signature_interpretation",
)

LABEL_FIELDS: tuple[str, ...] = (
    "is_known_treatment",
    "novelty_score",
    "novelty_level",
    "confidence",
    "confidence_reasons",
)

# v5 clinical guardrails (always present; may be empty dict if the
# underlying data isn't available for a candidate).
CLINICAL_FIELDS: tuple[str, ...] = (
    "dose_plausibility",
    "ddi_warnings",
    "pharmacogenomics",
    "triangulation",
    "tissue_context",
)

# v5.1/v5.2 post-processor outputs (attached by scripts/score_ensemble_v5.py,
# scripts/add_docking_proxy_axis.py, scripts/annotate_structural_alerts.py).
V51_FIELDS: tuple[str, ...] = (
    "ensemble_prob",
    "ensemble_rank",
    "ensemble_features",
    "docking",
    "structural_alerts",
)

# v7 post-processor outputs.
# - conformal interval: ensemble_prob_lower / _upper (90% coverage by default)
# - prediction_set_at_90: list[int] of conformally-consistent labels
#   ([1] = confident positive, [0,1] = uncertain, [0] = confident negative)
# - ensemble_head: which per-class head scored this candidate
#   ("parasitic", "viral", "bacterial", "oncology", "rare_metabolic",
#   "chronic_systemic", or "shared" when no per-class head matched).
# - selectivity_*: off-target panel — high selectivity = drug binds few targets;
#   promiscuous binders get combined_score dampened.
# - target_essentiality / essentiality_warning: DepMap CRISPR essentiality
#   flag for the candidate's primary target — high-essentiality genes
#   are druggable in oncology but risky elsewhere.
# - mechanism_confidence: how well-known the disease's molecular biology
#   is. Low confidence → every prediction tagged speculative in the
#   wet-lab brief.
# - red_team_assessment: short adversarial critique per candidate.
# - is_repurposing_candidate: False when the drug is already an established
#   treatment for this disease (demoted below the surfaced top-K by
#   scripts/apply_novelty_ranking.py); True for genuine repurposing leads.
V7_FIELDS: tuple[str, ...] = (
    "ensemble_prob_lower",
    "ensemble_prob_upper",
    "prediction_set_at_90",
    "ensemble_head",
    "selectivity_score",
    "n_off_targets",
    "primary_target",
    "primary_nM",
    "target_essentiality",
    "essentiality_warning",
    "mechanism_confidence",
    "red_team_assessment",
    "is_repurposing_candidate",
)

CANDIDATE_FIELDS: frozenset[str] = frozenset(
    IDENTITY_FIELDS + PILLAR_FIELDS + GROUP_FIELDS + FINAL_FIELDS
    + EVIDENCE_FIELDS + LABEL_FIELDS + CLINICAL_FIELDS + V51_FIELDS + V7_FIELDS
)

# Fields required on every candidate (missing = schema violation).
REQUIRED_CANDIDATE_FIELDS: frozenset[str] = frozenset({
    "drug_id", "drug_name", "disease_name", "combined_score",
    "pillars_hit", "confidence",
})

# Legacy / obsolete names that must NOT appear in any new output.
# Presence indicates an un-migrated codepath.
LEGACY_FIELDS: frozenset[str] = frozenset({
    "txgnn_raw_score",       # v3 leftover
    "proximity_raw_score",   # v3 leftover
    "dti_raw_score",         # v3 leftover
    "gene_sig_raw_score",    # v3 leftover
    "pgx_flags",             # predates v5 rename to "pharmacogenomics"
    "triangulation_score",   # predates v5 nested dict
    "mechanism_path",        # predates v5 rename to "mechanistic_hypothesis"
    "top_candidates",        # file-level; canonical is "candidates"
})


# Aggregate / non-per-disease files emitted into experiments/results/.
# Post-processors iterating over the directory must skip these — they have
# different shapes (lists, summary dicts) and the per-disease validators
# would crash. Centralized here so adding a new aggregate is a one-line
# change instead of six.
AGGREGATE_RESULT_FILES: frozenset[str] = frozenset({
    "screening_summary",
    "novel_candidates",
    "opencure_database",
    "mechanism_clusters",
})


# Top-level (file) keys — everything a result JSON should have.
RESULT_TOP_LEVEL: frozenset[str] = frozenset({
    "disease", "status", "timestamp", "elapsed_seconds", "total_searched",
    "evidence_generated", "confidence_counts", "pipeline_version",
    "data_manifest_hash", "candidates",
    # optional (added by post-processors / finalize)
    "ensemble_version", "ensemble_model_path", "docking_axis",
    "disease_entity",
    # failure mode: when systematic_screening.py can't resolve the disease
    # name to a DRKG entity, it writes status="failed" + error="...". The
    # validator should accept this without rejecting the whole record.
    "error",
})


def validate_candidate(cand: dict) -> list[str]:
    """Return a list of warning strings for a candidate dict.

    Empty list ⇒ candidate conforms to the canonical schema.
    """
    warnings: list[str] = []
    missing = REQUIRED_CANDIDATE_FIELDS - set(cand)
    if missing:
        warnings.append(f"missing required: {sorted(missing)}")
    legacy = LEGACY_FIELDS & set(cand)
    if legacy:
        warnings.append(f"legacy fields present: {sorted(legacy)}")
    unknown = set(cand) - CANDIDATE_FIELDS
    if unknown:
        warnings.append(f"unknown fields: {sorted(unknown)}")
    return warnings


def validate_result_file(data: dict) -> list[str]:
    """Validate a full result-file dict (top-level + every candidate).

    Returns warnings as strings; empty list ⇒ conformant. If ``data`` is
    not even shaped like a per-disease result file (e.g. a list emitted
    by an aggregate post-processor), a single warning is returned
    instead of raising.
    """
    warnings: list[str] = []
    if not isinstance(data, dict):
        return [f"not a result-file dict (got {type(data).__name__})"]
    unknown_top = set(data) - RESULT_TOP_LEVEL
    if unknown_top:
        warnings.append(f"unknown top-level keys: {sorted(unknown_top)}")
    if "candidates" not in data:
        warnings.append("missing 'candidates' key at top level")
        return warnings
    for i, c in enumerate(data["candidates"]):
        for w in validate_candidate(c):
            warnings.append(f"candidate[{i}] {c.get('drug_name', '?')}: {w}")
    return warnings
