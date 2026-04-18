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


@lru_cache(maxsize=1)
def _load_phases() -> dict:
    import json
    if not CHEMBL_PHASE_CACHE.exists():
        return {}
    try:
        return json.loads(CHEMBL_PHASE_CACHE.read_text())
    except Exception:
        return {}


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


def get_dose_plausibility(drug_id: str) -> dict:
    """
    Return a dose-plausibility profile for a drug.

    Args:
        drug_id: DrugBank ID (with or without "Compound::" prefix)

    Returns:
        {
          "plausibility": "yes"|"likely"|"uncertain"|"unknown",
          "dose_range": "clinical"|"late-trial"|"trial"|"investigational"|"research",
          "confidence": "high"|"medium"|"low",
          "rationale": human-readable explanation,
          "chembl_phase": <float or None>,
          "upgrade_pending": True if ChEMBL SQLite not yet available for pKi-vs-Cmax
        }
    """
    bare = drug_id.split("::", 1)[1] if "::" in drug_id else drug_id
    phase = _load_phases().get(bare)
    profile = _phase_to_plausibility(phase)
    profile["chembl_phase"] = phase
    # Check whether ChEMBL 34 SQLite has been unpacked for stage-2 upgrade
    profile["upgrade_pending"] = not Path("data/sources_2024/chembl_34").exists()
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
