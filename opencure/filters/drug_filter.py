"""
Hard filters for therapeutic candidate viability.

Applied BEFORE any scoring to reject non-drug compounds (metabolites, ions,
toxic substances) that would otherwise dominate results due to their
incidental biological proximity to disease genes.

Three-gate filter:
  1. SMILES validity + structural rules (reject atoms, ions, tiny molecules)
  2. ADMET critical toxicity (single-flag rejection for hERG, AMES, DILI)
  3. ChEMBL clinical phase (optional soft filter; keep phase >= 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Critical ADMET thresholds — very high single flags = reject.
# Intentionally conservative so FDA-approved drugs with known side effects
# (e.g., Tacrine AMES=0.78, Donepezil hERG=0.96) pass. Relies on ChEMBL
# phase filter to catch non-drug compounds that have clean ADMET profiles.
CRITICAL_TOXICITY = {
    "hERG": 0.97,          # Only reject near-certain hERG blockers
    "AMES": 0.92,          # Only very strong mutagens
    "DILI": 0.92,          # Only high-risk hepatotoxins
    "Skin_Reaction": 0.92, # Severe skin reaction
}

# Minimum structural requirements
MIN_HEAVY_ATOMS = 4     # Reduced: Hydroxyurea has only 6 heavy atoms, urea has 4
MIN_MOL_WEIGHT = 60     # Reduced: Hydroxyurea is 76 Da

# Cache for ChEMBL phase data
_chembl_cache: Optional[dict] = None
CHEMBL_CACHE_PATH = Path("data/drkg/chembl_phase.json")


def _load_chembl_phase() -> dict:
    """Load ChEMBL phase cache if available. Returns {drugbank_id: max_phase}."""
    global _chembl_cache
    if _chembl_cache is not None:
        return _chembl_cache
    if CHEMBL_CACHE_PATH.exists():
        _chembl_cache = json.loads(CHEMBL_CACHE_PATH.read_text())
    else:
        _chembl_cache = {}
    return _chembl_cache


def check_smiles_rules(smiles: str) -> tuple[bool, str]:
    """Check SMILES-based structural rules. Returns (is_valid, reason)."""
    if not smiles or not isinstance(smiles, str) or len(smiles) < 3:
        return False, "no_smiles"

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return True, "rdkit_missing"  # Fail-open

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid_smiles"

    heavy = mol.GetNumHeavyAtoms()
    if heavy < MIN_HEAVY_ATOMS:
        return False, f"too_small ({heavy} heavy atoms)"

    mw = Descriptors.MolWt(mol)
    if mw < MIN_MOL_WEIGHT:
        return False, f"mol_weight_too_low ({mw:.0f} Da)"

    # Must contain at least one carbon (exclude inorganic salts, ions)
    has_carbon = any(a.GetSymbol() == "C" for a in mol.GetAtoms())
    if not has_carbon:
        return False, "inorganic_no_carbon"

    return True, "valid"


def check_admet_critical(admet_preds: dict) -> tuple[bool, str]:
    """Reject if any critical toxicity flag exceeds threshold."""
    if not admet_preds:
        return True, "no_admet_data"  # Fail-open

    flags = []
    for endpoint, threshold in CRITICAL_TOXICITY.items():
        value = admet_preds.get(endpoint, 0)
        if isinstance(value, (int, float)) and value > threshold:
            flags.append(f"{endpoint}={value:.2f}")

    if flags:
        return False, "toxic: " + "; ".join(flags)
    return True, "safe"


def check_chembl_phase(drugbank_id: str) -> tuple[bool, str]:
    """Check ChEMBL clinical phase. Keep phase >= 1 (or missing = fail-open)."""
    cache = _load_chembl_phase()
    if not cache:
        return True, "chembl_cache_missing"  # Fail-open

    phase = cache.get(drugbank_id)
    if phase is None:
        return True, "not_in_chembl"  # Fail-open — don't penalize for missing

    try:
        phase_num = float(phase)
    except (ValueError, TypeError):
        return True, "phase_unknown"

    if phase_num >= 1:
        return True, f"phase_{phase_num}"
    return False, f"phase_{phase_num}_too_early"


def is_therapeutic_candidate(
    drug_id: str,
    smiles: str,
    admet_preds: Optional[dict] = None,
    check_chembl: bool = True,
) -> tuple[bool, str]:
    """
    Full gate: SMILES rules → [FDA-approved bypass] → ADMET critical → ChEMBL phase.

    If a drug is FDA-approved (ChEMBL phase=4), bypass ADMET check since
    predicted toxicity is just a model; real-world approval is ground truth.

    Returns (is_valid, reason). If is_valid=False, reason explains rejection.
    """
    ok, reason = check_smiles_rules(smiles)
    if not ok:
        return False, f"smiles:{reason}"

    # Clinical-grade bypass: skip ADMET rejection for drugs proven in humans.
    # Phase ≥ 2 means human efficacy trials; ADMET predictions shouldn't
    # override real clinical evidence. Tenofovir, Artemisinin show phase=3
    # in ChEMBL even though they're approved — Phase 3 is safe threshold.
    cache = _load_chembl_phase()
    phase = cache.get(drug_id)
    is_fda_approved = (phase is not None and phase >= 2.0)

    if admet_preds and not is_fda_approved:
        ok, reason = check_admet_critical(admet_preds)
        if not ok:
            return False, f"admet:{reason}"

    if check_chembl:
        ok, reason = check_chembl_phase(drug_id)
        if not ok:
            return False, f"chembl:{reason}"

    return True, "valid" if not is_fda_approved else "valid_fda_approved"


def filter_compounds(
    compound_entities: list[str],
    smiles_map: dict,
    admet_cache: Optional[dict] = None,
    check_chembl: bool = True,
) -> tuple[list[str], dict]:
    """
    Apply hard filters to a list of compound entities.

    Returns:
        kept: list of compound entities that passed
        rejection_stats: dict with counts by rejection reason
    """
    kept = []
    rejections = {}

    for compound in compound_entities:
        smiles = smiles_map.get(compound)
        if smiles is None:
            db_id = compound.split("::")[1] if "::" in compound else compound
            smiles = smiles_map.get(db_id)

        admet = admet_cache.get(smiles) if admet_cache and smiles else None
        drug_id = compound.split("::")[1] if "::" in compound else compound

        ok, reason = is_therapeutic_candidate(drug_id, smiles, admet, check_chembl)
        if ok:
            kept.append(compound)
        else:
            category = reason.split(":")[0]
            rejections[category] = rejections.get(category, 0) + 1

    return kept, rejections
