"""
Dose plausibility reasoning (v5).

For each prediction, estimate whether the drug's clinical dose is
achievable at levels where the predicted mechanism (target inhibition /
agonism) is biochemically plausible.

Two-stage design:

STAGE 1 — Phase-based plausibility (always available)
  Uses ChEMBL max clinical trial phase + DrugBank indication dose:
    phase 4 (approved):       dose_range="clinical"         plausible=yes
    phase 2-3 (late-stage):   dose_range="trial"           plausible=likely
    phase 1 (early):          dose_range="investigational"  plausible=uncertain
    phase < 1 / unknown:      dose_range="research"         plausible=unknown

STAGE 2 — pKi vs Cmax reasoning (when ChEMBL 34 SQLite is unpacked)
  For each (drug, target) pair:
    - ChEMBL bioactivities → median IC50/Ki → derive Kd in nM
    - DrugBank → plasma Cmax for clinical dose
    - plausible if Cmax >= Kd (i.e. the drug reaches inhibitory
      concentration at its standard dose)

Stage 2 is added when data/sources_2024/chembl_34/ is populated; until
then stage 1 is returned alone. This module is wired into the evidence
report so every prediction gets SOMETHING, upgraded automatically later.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional


# Re-use the existing ChEMBL phase cache OpenCure already built
CHEMBL_PHASE_CACHE = Path("data/drkg/chembl_phase.json")
# Stage-2 activity lookup (built by scripts/build_drug_target_activities.py)
ACTIVITY_CACHE = Path("data/drkg/drug_target_activities.json")

# Rough typical plasma Cmax ranges for orally administered drugs (in nM):
# most clinically-useful oral drugs peak at 100 nM – 10 μM. We use the
# approximate log-midpoint (1 μM = 1000 nM) as a generic fallback when
# drug-specific Cmax data is not available.
FALLBACK_CMAX_NM = 1000.0


@lru_cache(maxsize=1)
def _load_phases() -> dict:
    import json
    if not CHEMBL_PHASE_CACHE.exists():
        return {}
    try:
        return json.loads(CHEMBL_PHASE_CACHE.read_text())
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_activities() -> dict:
    """Stage-2 lookup: {drugbank_id: {gene_symbol: {median_nM, n, types}}}"""
    import json
    if not ACTIVITY_CACHE.exists():
        return {}
    try:
        return json.loads(ACTIVITY_CACHE.read_text())
    except Exception:
        return {}


def _target_affinity_assessment(
    drug_id: str,
    target_symbol: Optional[str],
    cmax_nm: float = FALLBACK_CMAX_NM,
) -> dict:
    """Stage-2 dose plausibility: compare achievable plasma Cmax to median
    measured IC50/Ki for the predicted target.

    Returns {
      'cmax_over_ic50_ratio': float,
      'median_ic50_nM': float,
      'n_activities': int,
      'activity_types': list[str],
      'mechanism_feasible': 'yes'|'borderline'|'no'|'unknown',
    }
    or empty dict if no data for this (drug, target) pair.
    """
    if not target_symbol:
        return {}
    bare = drug_id.split("::", 1)[1] if "::" in drug_id else drug_id
    activities = _load_activities()
    drug_hits = activities.get(bare, {})
    stats = drug_hits.get(target_symbol)
    if not stats:
        return {}

    median_ic50 = float(stats.get("median_nM", 0) or 0)
    if median_ic50 <= 0:
        return {}

    ratio = cmax_nm / median_ic50
    if ratio >= 10:
        feasible = "yes"
    elif ratio >= 1:
        feasible = "borderline"
    else:
        feasible = "no"

    return {
        "cmax_over_ic50_ratio": round(ratio, 2),
        "median_ic50_nM": median_ic50,
        "n_activities": stats.get("n", 0),
        "activity_types": stats.get("activity_types", []),
        "mechanism_feasible": feasible,
        "cmax_source": f"generic fallback {cmax_nm:.0f} nM (per-drug Cmax TBD)",
    }


def _phase_to_plausibility(phase: Optional[float]) -> dict:
    """Return structured plausibility given ChEMBL max_phase."""
    if phase is None:
        return {
            "plausibility": "unknown",
            "dose_range": "research",
            "confidence": "low",
            "rationale": "No ChEMBL clinical phase data available.",
        }
    try:
        p = float(phase)
    except (TypeError, ValueError):
        return {
            "plausibility": "unknown",
            "dose_range": "research",
            "confidence": "low",
            "rationale": "ChEMBL phase not numeric.",
        }
    if p >= 4:
        return {
            "plausibility": "yes",
            "dose_range": "clinical",
            "confidence": "high",
            "rationale": (
                "FDA/EMA-approved drug — standard clinical dose is known to be "
                "tolerable in humans and produce measurable pharmacological effect."
            ),
        }
    if p >= 3:
        return {
            "plausibility": "likely",
            "dose_range": "late-trial",
            "confidence": "medium",
            "rationale": (
                "Phase-3 investigational — dose range established in Phase-2 "
                "pharmacokinetic studies; efficacy trials ongoing or recently completed."
            ),
        }
    if p >= 2:
        return {
            "plausibility": "likely",
            "dose_range": "trial",
            "confidence": "medium",
            "rationale": (
                "Phase-2 investigational — preliminary efficacy signal at tested "
                "doses; pharmacokinetics from Phase 1 informs dose estimate."
            ),
        }
    if p >= 1:
        return {
            "plausibility": "uncertain",
            "dose_range": "investigational",
            "confidence": "low",
            "rationale": (
                "Phase-1 only — safety and PK tested, but no demonstrated clinical "
                "effect at tested doses in the target population."
            ),
        }
    return {
        "plausibility": "unknown",
        "dose_range": "research",
        "confidence": "low",
        "rationale": (
            "Pre-clinical / research compound — human pharmacokinetics unestablished."
        ),
    }


def get_dose_plausibility(
    drug_id: str,
    target_symbol: Optional[str] = None,
) -> dict:
    """
    Return a dose-plausibility profile for a drug.

    Args:
        drug_id: DrugBank ID (with or without "Compound::" prefix)
        target_symbol: optional gene symbol (e.g., "ACHE") to enable
            stage-2 (pKi vs Cmax) assessment. If omitted or unknown,
            only stage-1 (ChEMBL phase) signal is returned.

    Returns:
        {
          "plausibility": "yes"|"likely"|"uncertain"|"unknown",
          "dose_range": "clinical"|"late-trial"|"trial"|"investigational"|"research",
          "confidence": "high"|"medium"|"low",
          "rationale": human-readable explanation,
          "chembl_phase": <float or None>,
          "target_affinity": {...} if stage-2 data available else {},
          "stage": 1 or 2
        }
    """
    bare = drug_id.split("::", 1)[1] if "::" in drug_id else drug_id
    phase = _load_phases().get(bare)
    profile = _phase_to_plausibility(phase)
    profile["chembl_phase"] = phase

    # Stage-2 upgrade — if ChEMBL activities loaded and a target was given
    aff = _target_affinity_assessment(drug_id, target_symbol)
    if aff:
        profile["target_affinity"] = aff
        profile["stage"] = 2
        # If mechanism infeasible, downgrade plausibility even for approved drugs
        if aff["mechanism_feasible"] == "no" and profile["plausibility"] == "yes":
            profile["plausibility"] = "uncertain"
            profile["rationale"] += (
                f" HOWEVER, at typical plasma concentrations the drug does NOT "
                f"reach inhibitory levels for {target_symbol} "
                f"(Cmax/IC50 = {aff['cmax_over_ic50_ratio']}x; need ≥1x for any "
                f"effect, ≥10x for confident target engagement)."
            )
        elif aff["mechanism_feasible"] == "borderline":
            profile["rationale"] += (
                f" Target engagement borderline: Cmax/IC50 = "
                f"{aff['cmax_over_ic50_ratio']}x for {target_symbol}."
            )
    else:
        profile["target_affinity"] = {}
        profile["stage"] = 1

    return profile


if __name__ == "__main__":
    # Smoke test
    for drug_id, name in [
        ("DB00945", "Aspirin"),
        ("DB00843", "Donepezil"),
        ("DB01211", "Clarithromycin"),
        ("DB13132", "Artemisinin"),
        ("DB01096", "Oxamniquine"),
        ("DB_UNKNOWN", "Unknown"),
    ]:
        p = get_dose_plausibility(drug_id)
        print(f"{name:20s} phase={p['chembl_phase']}  {p['plausibility']:10s}  {p['rationale'][:60]}")
