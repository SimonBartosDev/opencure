"""Wet-lab brief generator — v7.

For each disease's top-K candidates, produce a 1-page Markdown brief a
PI can read in under 10 minutes and forward to a graduate student.

Sections per candidate:
1. Header: rank, drug name, ensemble probability + conformal interval
2. Mechanism-of-action paragraph (cite-grounded)
3. Suggested assay matched to disease class
4. Concentration range (back-of-envelope from ChEMBL phase / activities)
5. Red-team summary (one-line distillation of red_team_assessment)
6. Caveats: essentiality warning, mechanism-confidence flag,
   selectivity warning
7. Existing-supply check (commercial availability, when known)

Designed deterministic-first. The MoA paragraph optionally uses a local
Llama-3.1-8B via MLX to enrich the citations; without it, the
deterministic version cites the candidate's evidence fields directly.

Used by ``scripts/generate_wetlab_briefs.py``; the consortium-output
location is ``experiments/results/briefs/<disease>_top5.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

from opencure.evidence.mechanism_uncertainty import (
    LOW_CONFIDENCE_THRESHOLD,
    mechanism_confidence,
)


# ---- Disease class → assay heuristic ------------------------------------

ASSAY_BY_CLASS: dict[str, dict[str, str]] = {
    "parasitic": {
        "assay": "Parasite phenotypic assay (e.g., WST-1 viability on adult parasites)",
        "model": "ex vivo schistosomules / live trypanosomes / Plasmodium-infected RBCs",
        "readout": "EC50 over 24-72 h",
    },
    "viral": {
        "assay": "Antiviral cell-based assay with reporter virus or RT-PCR",
        "model": "Vero / Huh-7.5 / Calu-3 cells infected with target strain",
        "readout": "Viral load (TCID50 or qRT-PCR) at 48 h",
    },
    "bacterial": {
        "assay": "MIC / MBC determination per CLSI broth microdilution",
        "model": "Type-strain isolates (e.g., M. tuberculosis H37Rv, M. ulcerans Agy99)",
        "readout": "MIC after 5-7 d incubation",
    },
    "oncology": {
        "assay": "Cell viability (CellTiter-Glo) on relevant cell-line panel",
        "model": "Disease-relevant cell lines from CCLE / DepMap",
        "readout": "IC50 over 72-96 h",
    },
    "rare_metabolic": {
        "assay": "Substrate accumulation / enzyme activity rescue in patient-derived fibroblasts",
        "model": "iPSC-derived organoids (when available) or patient fibroblasts",
        "readout": "Substrate clearance over 72 h",
    },
    "chronic_systemic": {
        "assay": "Disease-relevant cellular phenotype in primary cells or iPSC-derived organoids",
        "model": "Patient-derived iPSC organoids when feasible; otherwise primary cells",
        "readout": "Phenotypic readout matched to disease pathology",
    },
}

DEFAULT_ASSAY = {
    "assay": "Disease-relevant cellular phenotypic assay",
    "model": "Best-available disease-relevant cell or organism model",
    "readout": "Phenotypic readout at 48-72 h",
}


def _assay_block(disease_class: str | None) -> dict[str, str]:
    if disease_class and disease_class in ASSAY_BY_CLASS:
        return ASSAY_BY_CLASS[disease_class]
    return DEFAULT_ASSAY


# ---- Concentration range -------------------------------------------------

def _concentration_range(cand: dict) -> str:
    """Back-of-envelope range based on primary-target affinity.

    Uses the primary_target_nM from the selectivity panel when present;
    falls back to a generic "1 nM - 10 µM" range.
    """
    primary_nm = cand.get("primary_nM")
    if isinstance(primary_nm, (int, float)) and primary_nm > 0:
        # Span from 0.1× to 100× the primary potency
        low_nm = primary_nm * 0.1
        high_nm = primary_nm * 100.0
        return _format_nm_range(low_nm, high_nm)
    return "1 nM – 10 µM (generic, no primary-target potency on file)"


def _format_nm_range(low_nm: float, high_nm: float) -> str:
    def _fmt(nm: float) -> str:
        if nm < 1:
            return f"{nm * 1000:.0f} pM"
        if nm < 1000:
            return f"{nm:.0f} nM"
        if nm < 1_000_000:
            return f"{nm / 1000:.1f} µM"
        return f"{nm / 1_000_000:.1f} mM"
    return f"{_fmt(low_nm)} – {_fmt(high_nm)}"


# ---- Caveats -----------------------------------------------------------

def _caveats(cand: dict, disease_mechanism_confidence: float) -> list[str]:
    out: list[str] = []
    if cand.get("essentiality_warning"):
        target = cand.get("primary_target", "primary target")
        out.append(
            f"⚠ Primary target **{target}** is broadly essential per DepMap "
            f"(systemic toxicity risk in non-oncology indications)."
        )
    sel = cand.get("selectivity_score")
    if isinstance(sel, (int, float)) and sel < 0.3:
        n_off = cand.get("n_off_targets", "many")
        out.append(
            f"⚠ Promiscuous binder (selectivity {sel:.2f}, {n_off} off-targets); "
            f"clinical effects likely dominated by off-target activity."
        )
    if disease_mechanism_confidence < LOW_CONFIDENCE_THRESHOLD:
        out.append(
            f"⚠ Disease mechanism poorly mapped "
            f"(confidence {disease_mechanism_confidence:.2f}). "
            f"Treat this prediction as speculative; "
            f"validate the disease-relevant target before chasing the drug."
        )
    if cand.get("has_failed_trial"):
        phase = cand.get("failed_trial_phase", "?")
        out.append(
            f"⚠ This drug previously failed a Phase {phase} trial for this "
            f"indication. The current prediction needs to explain what's different."
        )
    lower = cand.get("ensemble_prob_lower")
    upper = cand.get("ensemble_prob_upper")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        if (upper - lower) > 0.4:
            out.append(
                f"Conformal interval is wide ({lower:.2f} – {upper:.2f}); "
                f"the platform is not confident in this candidate's score."
            )
    return out


# ---- Mechanism narrative ------------------------------------------------

def _mechanism_paragraph(cand: dict, disease_name: str) -> str:
    """Deterministic mechanism narrative grounded in the candidate's evidence.

    LLM-narrated version is produced by ``mechanism_paragraph_llm`` when
    MLX/Llama is installed; this function is the always-available
    fallback so briefs always have prose, even in CI.
    """
    drug = cand.get("drug_name", cand.get("drug_id", "this drug"))
    primary = cand.get("primary_target") or cand.get("dti_best_target") or ""
    similar_to = cand.get("mol_emb_similar_to") or cand.get("similar_to") or ""
    rxn = cand.get("relation_type") or ""

    parts = [f"**{drug}** is proposed for **{disease_name}**"]
    if rxn:
        parts.append(f"on the basis of a `{rxn}` relation in the unified KG")
    if primary:
        parts.append(
            f"with primary target **{primary}** "
            f"(median nM = {cand.get('primary_nM', '?')})"
        )
    if similar_to:
        bare = similar_to.split("::")[-1]
        parts.append(f"and morphological / chemistry similarity to {bare}")
    pubmed = cand.get("pubmed_total", 0)
    trials = cand.get("clinical_trials_total", 0)
    if pubmed or trials:
        parts.append(f"({pubmed} PubMed papers, {trials} registered trials)")
    return ". ".join(p for p in parts if p) + "."


def mechanism_paragraph_llm(cand: dict, disease_name: str) -> str | None:
    """Narrative mechanism paragraph from a local LLM. None if MLX absent."""
    try:
        from mlx_lm import load, generate  # type: ignore
    except ImportError:
        return None
    try:
        import os
        model_name = os.environ.get(
            "OPENCURE_LLM_MODEL", "mlx-community/Llama-3.1-8B-Instruct-4bit",
        )
        model, tokenizer = load(model_name)
        drug = cand.get("drug_name", cand.get("drug_id", "this drug"))
        primary = cand.get("primary_target") or cand.get("dti_best_target") or "an unknown target"
        evidence = []
        if cand.get("pubmed_total"):
            evidence.append(f"{cand['pubmed_total']} PubMed papers")
        if cand.get("relation_type"):
            evidence.append(f"a `{cand['relation_type']}` KG relation")
        if cand.get("similar_to"):
            evidence.append(f"chemistry similarity to {cand['similar_to']}")
        evidence_str = ", ".join(evidence) or "model-driven only"
        prompt = (
            f"Write a single paragraph (~80 words) explaining the proposed "
            f"mechanism by which {drug} could treat {disease_name}, given that "
            f"its primary target is {primary}. Evidence: {evidence_str}. Be "
            "factual, cite no journal names, no fabrications. End with a "
            "single-sentence caveat about what's still unknown."
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=180)
        return out.strip()[:800]
    except Exception:
        return None


# ---- Brief assembly -----------------------------------------------------

@dataclass
class BriefContext:
    disease_name: str
    disease_entity: str
    disease_class: str | None  # one of opencure/eval/disease_classes.yaml
    use_llm: bool = False


def render_candidate_brief(
    cand: dict,
    rank: int,
    ctx: BriefContext,
) -> str:
    """Markdown brief for a single candidate."""
    drug = cand.get("drug_name") or cand.get("drug_id", "?")
    db_id = cand.get("drug_id", "?")
    p = cand.get("ensemble_prob")
    p_lower = cand.get("ensemble_prob_lower")
    p_upper = cand.get("ensemble_prob_upper")
    interval = ""
    if isinstance(p_lower, (int, float)) and isinstance(p_upper, (int, float)):
        interval = f"  [CI: {p_lower:.2f} – {p_upper:.2f}]"
    score_line = f"Probability: {p:.2f}{interval}" if isinstance(p, (int, float)) else ""

    mc = mechanism_confidence(ctx.disease_entity) if ctx.disease_entity else 0.5

    moa = None
    if ctx.use_llm:
        moa = mechanism_paragraph_llm(cand, ctx.disease_name)
    moa = moa or _mechanism_paragraph(cand, ctx.disease_name)

    assay = _assay_block(ctx.disease_class)
    concentration = _concentration_range(cand)
    red_team = cand.get("red_team_assessment") \
        or "Red-team assessment not available."
    caveats = _caveats(cand, mc)

    parts = [
        f"### #{rank}. {drug} ({db_id})\n",
        f"{score_line}\n" if score_line else "",
        "**Mechanistic hypothesis**\n",
        f"{moa}\n",
        "**Suggested assay**\n",
        f"- Assay: {assay['assay']}",
        f"- Model: {assay['model']}",
        f"- Readout: {assay['readout']}",
        f"- Concentration range: {concentration}\n",
        "**Red-team assessment**\n",
        f"{red_team}\n",
    ]
    if caveats:
        parts.append("**Caveats**\n")
        for cav in caveats:
            parts.append(f"- {cav}")
        parts.append("")
    return "\n".join(parts)


def render_disease_brief(
    candidates: list[dict],
    *,
    disease_name: str,
    disease_entity: str,
    disease_class: str | None,
    top_k: int = 5,
    use_llm: bool = False,
) -> str:
    """Top-K Markdown brief for one disease."""
    ctx = BriefContext(
        disease_name=disease_name,
        disease_entity=disease_entity,
        disease_class=disease_class,
        use_llm=use_llm,
    )
    mc = mechanism_confidence(disease_entity) if disease_entity else 0.5

    header = (
        f"# {disease_name} — top-{top_k} repurposing candidates\n\n"
        f"Disease entity: `{disease_entity}`  \n"
        f"Disease class: `{disease_class or 'unmapped'}`  \n"
        f"Mechanism-confidence score: {mc:.2f} "
        f"({'speculative' if mc < LOW_CONFIDENCE_THRESHOLD else 'well-mapped'})  \n"
        f"Generated by OpenCure v7\n\n---\n\n"
    )
    body = "\n---\n\n".join(
        render_candidate_brief(c, rank=i + 1, ctx=ctx)
        for i, c in enumerate(candidates[:top_k])
    )
    return header + body
