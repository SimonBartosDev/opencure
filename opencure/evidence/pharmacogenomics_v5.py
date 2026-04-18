"""
Pharmacogenomic flags (v5): CPIC + PharmGKB → clinical actionability warnings.

For every top-10 prediction, check whether the drug has any established
variant-drug interaction that affects dose, efficacy, or toxicity. This
moves OpenCure from "here's a prediction" to "here's a prediction + the
patient-subtype guardrails a prescriber must know".

Sources:
  - data/sources_2024/cpic_pairs.json (76 KB, CPIC tier-A/B/C/D pairs)
  - data/sources_2024/pharmgkb/clinical_annotations.tsv (5187 annotations)

Categories returned per drug:
  - high_risk: CPIC level A or B — actionable, must change dose/drug
  - moderate:  CPIC level C or PharmGKB level 1A/1B/2A
  - advisory:  PharmGKB level 2B/3
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


CPIC_PATH = Path("data/sources_2024/cpic_pairs.json")
PHARMGKB_PATH = Path("data/sources_2024/pharmgkb/clinical_annotations.tsv")


@lru_cache(maxsize=1)
def _load_cpic() -> dict[str, list[dict]]:
    """Load CPIC pairs keyed by drug name (lowercased).

    Returns: {drug_lower: [{'gene': X, 'level': A|B|C|D, 'guideline': N}]}
    """
    if not CPIC_PATH.exists():
        return {}
    try:
        data = json.loads(CPIC_PATH.read_text())
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    for row in data:
        drug = (row.get("drug") or {}).get("name", "").lower()
        gene = (row.get("gene") or {}).get("symbol", "")
        level = row.get("cpiclevel", "")
        guideline = (row.get("guideline") or {}).get("name", "")
        if drug and gene:
            out.setdefault(drug, []).append({
                "gene": gene, "level": level, "guideline": guideline
            })
    return out


@lru_cache(maxsize=1)
def _load_pharmgkb() -> dict[str, list[dict]]:
    """Load PharmGKB clinical annotations keyed by drug name (lowercased).

    Returns: {drug_lower: [{'gene': X, 'variant': V, 'level': 1A/1B/2A..,
                             'phenotype_category': Y, 'url': U}]}
    """
    if not PHARMGKB_PATH.exists():
        return {}
    out: dict[str, list[dict]] = {}
    with PHARMGKB_PATH.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_var = header.index("Variant/Haplotypes")
            i_gene = header.index("Gene")
            i_level = header.index("Level of Evidence")
            i_cat = header.index("Phenotype Category")
            i_drugs = header.index("Drug(s)")
            i_url = header.index("URL")
        except ValueError:
            return {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_var, i_gene, i_level, i_drugs):
                continue
            drugs_raw = parts[i_drugs]
            # Drug(s) column may contain multiple drugs separated by ';'
            for drug in drugs_raw.split(";"):
                drug = drug.strip().lower()
                if not drug:
                    continue
                out.setdefault(drug, []).append({
                    "gene": parts[i_gene],
                    "variant": parts[i_var],
                    "level": parts[i_level],
                    "phenotype_category": parts[i_cat],
                    "url": parts[i_url] if len(parts) > i_url else "",
                })
    return out


def _classify(cpic_level: str, pgkb_level: str) -> str:
    """Return 'high_risk' | 'moderate' | 'advisory' | '' based on levels."""
    c = (cpic_level or "").upper()
    p = (pgkb_level or "").upper()
    if c in ("A", "B"):
        return "high_risk"
    if c == "C" or p in ("1A", "1B", "2A"):
        return "moderate"
    if p in ("2B", "3"):
        return "advisory"
    return ""


def get_pharmacogenomic_flags(drug_name: str) -> dict:
    """
    Return a structured pharmacogenomic risk profile for a drug.

    {
      "has_flags": bool,
      "highest_risk": "high_risk"|"moderate"|"advisory"|"",
      "summary": "short human-readable string",
      "cpic": [...],
      "pharmgkb": [...]
    }
    """
    if not drug_name:
        return {"has_flags": False, "highest_risk": "", "summary": "", "cpic": [], "pharmgkb": []}

    name_key = drug_name.lower()
    cpic_hits = _load_cpic().get(name_key, [])
    pgkb_hits = _load_pharmgkb().get(name_key, [])

    levels = []
    for c in cpic_hits:
        r = _classify(c.get("level", ""), "")
        if r:
            levels.append(r)
    for p in pgkb_hits:
        r = _classify("", p.get("level", ""))
        if r:
            levels.append(r)

    rank = {"high_risk": 3, "moderate": 2, "advisory": 1, "": 0}
    highest = max(levels, key=lambda x: rank.get(x, 0)) if levels else ""

    summary_parts = []
    if cpic_hits:
        top = cpic_hits[0]
        summary_parts.append(f"CPIC-{top['level']} ({top['gene']})")
    if pgkb_hits:
        top = pgkb_hits[0]
        summary_parts.append(f"PharmGKB-{top['level']} ({top['gene']} / {top['variant']})")

    return {
        "has_flags": bool(cpic_hits or pgkb_hits),
        "highest_risk": highest,
        "summary": " • ".join(summary_parts),
        "cpic": cpic_hits[:5],
        "pharmgkb": pgkb_hits[:5],
    }


if __name__ == "__main__":
    # Smoke test
    for d in ["Warfarin", "Abacavir", "Clopidogrel", "Codeine", "Simvastatin", "Aspirin"]:
        f = get_pharmacogenomic_flags(d)
        print(f"{d:20s} highest={f['highest_risk']:10s}  {f['summary']}")
