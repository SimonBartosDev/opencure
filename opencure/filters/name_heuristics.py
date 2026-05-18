"""
Heuristic detection of "research chemical" naming patterns.

Drug-repurposing screens keep surfacing compounds with IUPAC-style names like
"2-(3-GUANIDINOPHENYL)-3-MERCAPTOPROPANOIC ACID" or
"1-(2,6-Dichlorophenyl)-5-(2,4-Difluorophenyl)-7-Piperidin-4-Yl-3,4-Dihydroquinolin-2(1h)-One".
These are experimental compounds or assay tools, not therapeutic candidates.
They typically lack ChEMBL clinical phase data and slip past fail-open filters.

Strategy: flag names whose structure looks like IUPAC rather than an INN/trade
name. Used as a soft gate: compound is rejected ONLY if the name looks like a
research chemical AND ChEMBL data is missing (i.e. we have no positive clinical
evidence to override the heuristic).

Passes (real drug names):
  "Aspirin", "Donepezil", "Hydroxyurea", "Paclitaxel", "Artemisinin"
Rejects:
  "2-(3-GUANIDINOPHENYL)-3-MERCAPTOPROPANOIC ACID"
  "5-Amidino-Benzimidazole"
  "1-(2,6-Dichlorophenyl)-5-(2,4-Difluorophenyl)-7-Piperidin-4-Yl-..."
"""

from __future__ import annotations

import re


# Tokens that almost exclusively appear in IUPAC/research-chem names
_IUPAC_TOKENS = re.compile(
    r"(?ix)"
    r"(benzimid|dihydroquin|guanidino|mercapto|amidino|"
    r"carbamyl|carboxamid|phenyl|piperidin|pyrrolidin|"
    r"tetrahydro|isoquinolin|sulfonamid|thiazol|triazol|imidazol)"
)

# Locants: comma-separated digit pairs like (2,4) or 2,6- or 3,4-
_LOCANT_PATTERN = re.compile(r"\(\d+,\d+\)|\d+,\d+-|\d+[a-z]?-\d")

# INN-style drug-name suffix hints (protective — these names are probably real drugs)
_INN_SUFFIXES = (
    "mab", "nib", "tinib", "zumab", "vir", "prazole", "sartan", "olol",
    "pril", "statin", "ine", "one", "ide", "ol", "an", "in", "um",
    "azole", "dine", "pine", "mycin", "cillin", "floxacin", "cycline",
    "azine", "afil", "caine", "parin", "zepam",
)


def looks_like_research_chemical(name: str) -> bool:
    """
    Heuristic: does this name look like an IUPAC / research-chem designation?

    Returns True for probable research chemicals (to be rejected when also
    missing ChEMBL clinical phase data). False for likely real drug names.
    """
    if not name or not isinstance(name, str):
        return False

    # Collapse whitespace, normalize case for analysis
    stripped = name.strip()
    if not stripped:
        return False

    # Very long names are almost always IUPAC
    if len(stripped) > 45:
        return True

    # Names starting with a numeric locant followed by an IUPAC token are
    # almost always research chemicals ("5-Amidino-Benzimidazole",
    # "6-Hydroxy-...", "2-Aminophenol"). Protective: skip if the name is short
    # AND a known bare molecule ("1,3-Butadiene" etc. will be rejected by
    # ChEMBL-missing gate instead).
    if re.match(r"^\d+-[A-Z]", stripped) and _IUPAC_TOKENS.search(stripped):
        return True

    # Strong signal: multiple locants
    locants = _LOCANT_PATTERN.findall(stripped)
    if len(locants) >= 2:
        return True

    # Strong signal: three or more hyphens with digits (IUPAC positional nomenclature)
    hyphen_count = stripped.count("-")
    has_digits = any(c.isdigit() for c in stripped)
    if hyphen_count >= 3 and has_digits and len(stripped) > 20:
        return True

    # IUPAC sub-token present
    if _IUPAC_TOKENS.search(stripped):
        # Protective: if also ends with a classic INN suffix, it's probably a real
        # drug (e.g. "celecoxib" has -ib, "nevirapine" has -ine)
        lower = stripped.lower()
        if any(lower.endswith(suf) for suf in _INN_SUFFIXES) and len(stripped) <= 25:
            return False
        # Long name with IUPAC token: likely research chem
        if len(stripped) > 22:
            return True

    # ALL-CAPS chemistry names are a strong signal ("2-(3-GUANIDINOPHENYL)-...")
    alpha = [c for c in stripped if c.isalpha()]
    if alpha and len(alpha) >= 15:
        uppers = sum(1 for c in alpha if c.isupper())
        if uppers / len(alpha) > 0.75 and hyphen_count >= 2:
            return True

    return False
