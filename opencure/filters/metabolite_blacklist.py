"""
Endogenous-metabolite and cofactor blacklist.

Purpose: reject compounds that are normal human biochemistry (amino acids,
vitamins, cofactors, simple ions, nucleotide metabolites) UNLESS they are
also FDA-approved as drugs for real clinical use.

We match by normalized NAME (not DrugBank ID) because:
  - Names are more stable and auditable
  - Every entry here has been directly observed leaking into OpenCure top-10s
  - We don't want to accidentally blacklist real drugs by ID typos

Bypass rule: if the compound's ChEMBL max_phase >= 4, the compound IS an
approved drug and passes (user decides if the repurposing makes sense).
Folic Acid, Levothyroxine, Hydroxocobalamin, Thiamine etc. fall into this
bucket and are intentionally allowed through.

Category tags are for logging and dashboard "why was this rejected?" UI.
"""

from __future__ import annotations

from typing import Optional


# Curated name list. Keys are lowercased normalized names; values are categories.
# Every entry has been observed in v2/v3 top-10 results and verified as
# endogenous biochemistry rather than a therapeutic candidate.
_BLACKLIST_BY_NAME: dict[str, str] = {
    # Amino acids (standalone amino acids are not therapeutic candidates —
    # even if they have DrugBank entries for parenteral nutrition uses)
    "l-alanine": "amino_acid",
    "l-arginine": "amino_acid",
    "l-asparagine": "amino_acid",
    "l-aspartic acid": "amino_acid",
    "l-cysteine": "amino_acid",
    "l-cystine": "amino_acid",
    "l-glutamic acid": "amino_acid",
    "l-glutamine": "amino_acid",
    "l-histidine": "amino_acid",
    "l-isoleucine": "amino_acid",
    "l-leucine": "amino_acid",
    "l-lysine": "amino_acid",
    "l-methionine": "amino_acid",
    "l-ornithine": "amino_acid",
    "l-phenylalanine": "amino_acid",
    "l-proline": "amino_acid",
    "l-serine": "amino_acid",
    "l-threonine": "amino_acid",
    "l-tryptophan": "amino_acid",
    "l-tyrosine": "amino_acid",
    "l-valine": "amino_acid",
    "glycine": "amino_acid",
    "alanine": "amino_acid",
    "glutamic acid": "amino_acid",
    "aspartic acid": "amino_acid",
    "taurine": "amino_acid",

    # Nucleotide / nucleic-acid metabolites (building blocks, not drugs)
    "adenosine triphosphate": "nucleotide",
    "atp": "nucleotide",
    "adenosine monophosphate": "nucleotide",
    "amp": "nucleotide",
    "adenosine diphosphate": "nucleotide",
    "adp": "nucleotide",
    "cytidine triphosphate": "nucleotide",
    "cytidine monophosphate": "nucleotide",
    "guanosine triphosphate": "nucleotide",
    "guanosine monophosphate": "nucleotide",
    "uridine triphosphate": "nucleotide",
    "uridine monophosphate": "nucleotide",
    "inosine monophosphate": "nucleotide",
    "cordycepin triphosphate": "nucleotide",
    "cordycepin-triphosphate": "nucleotide",
    "2'-deoxyadenosine triphosphate": "nucleotide",
    "deoxyadenosine triphosphate": "nucleotide",

    # Core metabolites — normal biochemistry
    "glutathione": "metabolite",
    "l-glutathione": "metabolite",
    "uric acid": "metabolite",
    "urate": "metabolite",
    "creatinine": "metabolite",
    "creatine": "metabolite",
    "urea": "metabolite",
    "lactic acid": "metabolite",
    "pyruvic acid": "metabolite",
    "succinic acid": "metabolite",
    "citric acid": "metabolite",
    "fumaric acid": "metabolite",
    "malic acid": "metabolite",
    "oxalacetic acid": "metabolite",
    "acetyl-coa": "metabolite",
    "coenzyme a": "metabolite",
    "nadh": "metabolite",
    "nad": "metabolite",
    "nadph": "metabolite",
    "nadp": "metabolite",
    "fad": "metabolite",
    "fadh2": "metabolite",
    "s-adenosylmethionine": "metabolite",
    "s-adenosyl-l-methionine": "metabolite",

    # Endogenous steroids / sterols (not therapeutic on their own — most
    # therapeutic steroids are synthetic analogs)
    "16,17-androstene-3-ol": "steroid",
    "androsterone": "steroid",
    "pregnenolone": "steroid",
    "dehydroepiandrosterone": "steroid",
    "cholesterol": "steroid",

    # Simple ions / inorganic (beyond SMILES filter — some have carbon)
    "oxygen": "inorganic",
    "carbon dioxide": "inorganic",
    "bicarbonate": "inorganic",
    "hydrogen peroxide": "inorganic",
    "nitric oxide": "inorganic",  # borderline; phase-4 bypass handles therapeutic use
    "fluoride ion": "inorganic",
    "fluoride": "inorganic",
    "chloride": "inorganic",
    "ammonia": "inorganic",

    # Simple sugars (fuel, not drug)
    "d-glucose": "sugar",
    "alpha-d-glucose": "sugar",
    "beta-d-glucose": "sugar",
    "glucose-6-phosphate": "sugar",
    "fructose": "sugar",
    "sucrose": "sugar",
    "lactose": "sugar",
    "ribose": "sugar",

    # Polyamines
    "spermidine": "polyamine",
    "spermine": "polyamine",
    "putrescine": "polyamine",

    # Fatty acids / lipids as building blocks
    "stearic acid": "lipid",
    "palmitic acid": "lipid",
    "oleic acid": "lipid",
    "arachidonic acid": "lipid",
}


def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace for stable matching."""
    return " ".join(str(name).lower().split())


def is_blacklisted_metabolite(
    drug_name: str,
    chembl_phase: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Is this compound an endogenous metabolite / cofactor / ion?

    Args:
        drug_name: Display name (case-insensitive).
        chembl_phase: ChEMBL max_phase if known. Phase >= 4 bypasses the
            blacklist (compound has a real approved indication — e.g.
            Folic Acid as supplementation is not blacklisted here because
            it's phase 4; pure "Folic Acid" as repurposing *candidate* for
            an unrelated disease is dubious but we defer to ChEMBL).

    Returns:
        (is_blacklisted, category). category is "" if not blacklisted.
    """
    if not drug_name:
        return False, ""

    key = _normalize(drug_name)
    category = _BLACKLIST_BY_NAME.get(key, "")
    if not category:
        return False, ""

    # Phase-4 bypass: compound is a real approved drug
    try:
        if chembl_phase is not None and float(chembl_phase) >= 4.0:
            return False, ""
    except (TypeError, ValueError):
        pass

    return True, category


def blacklist_size() -> int:
    """Return number of curated blacklist entries (for logging)."""
    return len(_BLACKLIST_BY_NAME)
