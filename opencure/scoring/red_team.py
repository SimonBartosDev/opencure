"""Adversarial red-team agent — v7.

For every top-K prediction, run a structured "what would have to be true
for this to be a bad lead?" pass. Two implementations:

1. **Deterministic critic** (always runs in CI):
   Inspects pillar disagreement, hub-score artifacts, low selectivity,
   essentiality warnings, low mechanism confidence, and emits a
   plain-text critique. No model dependency, fast, reproducible.

2. **LLM critic** (opt-in, runs when ``mlx_lm`` is importable):
   Local Llama-3.1-8B via MLX on M4 Max. Prompts the model to argue
   *against* each prediction with at least 3 specific risks. Used by
   ``scripts/red_team_v7.py`` to enrich the deterministic critique
   into prose for the wet-lab briefs.

Both produce a single ``red_team_assessment`` string per candidate that
flows into the result JSON via the canonical V7_FIELDS schema.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- Deterministic critic ------------------------------------------------

# Score thresholds that trigger a critique line.
SINGLE_PILLAR_RATIO = 4.0   # candidate's best score is N× any other pillar
LOW_SELECTIVITY_THRESHOLD = 0.3
LOW_MECHANISM_CONFIDENCE = 0.4
HIGH_DEGREE_PENALTY_MIN = 0.95  # near 1 means damping caught nothing — rarely a critique
HUB_DEGREE_DAMPING_THRESHOLD = 0.5  # candidate is a damped hub


@dataclass
class CandidateCritique:
    risks: list[str]

    def to_text(self) -> str:
        if not self.risks:
            return "No structural red flags detected."
        return " ".join(f"({i+1}) {r}" for i, r in enumerate(self.risks))


def _candidate_pillar_scores(cand: dict) -> dict[str, float]:
    """Pull canonical pillar scores from a candidate dict, defaulting to 0."""
    keys = (
        "transe_score", "pykeen_score", "primekg_score", "unified_score",
        "rgcn_score", "txgnn_score", "mol_similarity", "mol_emb_similarity",
        "gene_sig_score", "proximity_score", "mr_score", "dti_score",
        "jump_score",
    )
    out: dict[str, float] = {}
    for k in keys:
        v = cand.get(k, 0)
        try:
            out[k] = float(v) if v else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def critique_candidate(
    cand: dict,
    *,
    mechanism_confidence: float | None = None,
) -> CandidateCritique:
    """Deterministic adversarial critique of a single candidate.

    Surfaces structural concerns the ranker can't:
    - Single-pillar artifact: one pillar dominates, others near-zero
    - Low selectivity: promiscuous binder
    - Pan-essential primary target: systemic toxicity risk
    - Hub-damping caught the candidate: KG topology is doing the lifting
    - Low mechanism confidence: disease biology not well-mapped
    - No disease-relevant evidence: zero pubmed / zero clinical trials

    Returns a CandidateCritique whose ``risks`` list is empty when no
    flags trigger.
    """
    risks: list[str] = []

    # 1. Single-pillar artifact — biggest pillar > N× any other
    pillars = _candidate_pillar_scores(cand)
    nonzero = sorted([s for s in pillars.values() if s > 0], reverse=True)
    if len(nonzero) >= 2:
        if nonzero[0] >= SINGLE_PILLAR_RATIO * nonzero[1]:
            top_pillar = max(pillars, key=pillars.get)
            risks.append(
                f"Single-pillar artifact: {top_pillar}={nonzero[0]:.2f} dominates "
                f"the candidate's signal; all other pillars are weak."
            )
    elif len(nonzero) == 1:
        only_pillar = next(k for k, v in pillars.items() if v > 0)
        risks.append(
            f"Single-pillar evidence: only {only_pillar} fires for this "
            f"candidate; risks an isolated KG-path or hub artifact."
        )

    # 2. Selectivity — promiscuous binder is unlikely a clean repurposing lead
    sel = cand.get("selectivity_score")
    if isinstance(sel, (int, float)) and sel < LOW_SELECTIVITY_THRESHOLD:
        n_off = cand.get("n_off_targets", "many")
        risks.append(
            f"Low selectivity ({sel:.2f}; ~{n_off} measurable off-targets); "
            f"clinical risk dominated by off-target effects rather than the "
            f"intended mechanism."
        )

    # 3. DepMap essentiality
    if cand.get("essentiality_warning"):
        target = cand.get("primary_target") or "primary target"
        ess = cand.get("target_essentiality")
        ess_str = f"{ess:.2f}" if isinstance(ess, (int, float)) else "broadly essential"
        risks.append(
            f"Primary target {target} is broadly essential "
            f"(median Chronos {ess_str}); systemic toxicity risk in non-oncology indications."
        )

    # 4. Hub-degree damping caught the candidate
    dp = cand.get("degree_penalty")
    if isinstance(dp, (int, float)) and dp < HUB_DEGREE_DAMPING_THRESHOLD:
        risks.append(
            f"Hub-damping factor {dp:.2f}: candidate is densely connected in DRKG. "
            f"High pillar scores may reflect graph topology rather than disease-"
            f"specific biology."
        )

    # 5. Mechanism confidence
    if (mechanism_confidence is not None
            and mechanism_confidence < LOW_MECHANISM_CONFIDENCE):
        risks.append(
            f"Disease mechanism poorly mapped (confidence {mechanism_confidence:.2f}); "
            f"every gene-overlap-driven pillar inherits that uncertainty."
        )

    # 6. Evidence shortage — no pubmed, no trials → speculative
    pubmed = int(cand.get("pubmed_total", 0) or 0)
    trials = int(cand.get("clinical_trials_total", 0) or 0)
    if pubmed == 0 and trials == 0:
        risks.append(
            "Zero PubMed papers and zero registered trials connect this "
            "drug-disease pair; the prediction is purely model-driven without "
            "any literature corroboration."
        )

    # 7. Failed-trial history — directly contradicts the prediction
    if cand.get("has_failed_trial"):
        phase = cand.get("failed_trial_phase", "unknown")
        risks.append(
            f"This drug previously FAILED a Phase {phase} trial for this "
            f"indication; the current prediction must explain what's different."
        )

    return CandidateCritique(risks=risks)


# ---- LLM critic (opt-in) ------------------------------------------------

def _build_llm_prompt(cand: dict, mechanism_confidence: float | None) -> str:
    """Structured prompt asking a local LLM to argue against the prediction."""
    drug = cand.get("drug_name") or cand.get("drug_id", "?")
    disease = cand.get("disease_name", "?")
    rationale_bits = []
    for k in ("ensemble_prob", "combined_score", "transe_score", "primary_target"):
        v = cand.get(k)
        if v not in (None, "", 0):
            rationale_bits.append(f"{k}={v}")
    summary = ", ".join(rationale_bits) or "no salient features"
    mech = (
        f"; disease mechanism confidence {mechanism_confidence:.2f}"
        if mechanism_confidence is not None else ""
    )
    return (
        "You are a critical drug-development scientist. The platform predicted "
        f"that {drug} could be repurposed for {disease}. Pillar summary: "
        f"{summary}{mech}.\n\n"
        "Argue AGAINST this prediction. List exactly three specific, "
        "concrete risks or reasons it might fail (mechanism mismatch, off-target "
        "burden, pharmacokinetics, toxicity, prior trial history, mechanism-"
        "incompatibility, etc.). Be skeptical and specific. Output as a numbered "
        "list, nothing else."
    )


def critique_candidate_with_llm(
    cand: dict,
    *,
    mechanism_confidence: float | None = None,
) -> str | None:
    """Optional LLM-narrated critique. Returns ``None`` when MLX isn't installed.

    Designed as an *additive* layer: the deterministic critic always
    runs; this enriches the output with prose for human reviewers when
    a local model is available.
    """
    try:
        from mlx_lm import load, generate  # type: ignore
    except ImportError:
        return None
    try:
        # Default to a small instruction-tuned model the user has already
        # installed. The actual model name is configurable via env var if
        # someone wants to swap it.
        import os
        model_name = os.environ.get(
            "OPENCURE_LLM_MODEL", "mlx-community/Llama-3.1-8B-Instruct-4bit",
        )
        model, tokenizer = load(model_name)
        prompt = _build_llm_prompt(cand, mechanism_confidence)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
        return response.strip()[:500]  # keep it punchy
    except Exception:
        return None


def assess_candidate(
    cand: dict,
    *,
    mechanism_confidence: float | None = None,
    use_llm: bool = False,
) -> str:
    """One-call entry point: deterministic critique, optionally LLM-narrated.

    Always returns a non-empty string suitable for the
    ``red_team_assessment`` field. Falls back to the deterministic
    critic when LLM isn't available, so CI never depends on a model
    being installed.
    """
    deterministic = critique_candidate(
        cand, mechanism_confidence=mechanism_confidence,
    ).to_text()
    if not use_llm:
        return deterministic
    llm_text = critique_candidate_with_llm(
        cand, mechanism_confidence=mechanism_confidence,
    )
    if llm_text:
        return f"{deterministic}\n\nLLM narrative:\n{llm_text}"
    return deterministic
