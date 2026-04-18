"""
In-silico triangulation (v5): a prediction's confidence rises when
multiple independent evidence streams agree.

For each (drug, disease, target) candidate, we query up to four
independent signals:

  1. KG embedding   — already in combined_score (our internal)
  2. Docking        — AutoDock Vina score (via structure_docking.py)
  3. Pharos TDL     — NIH target-development-level classification
                      (Tclin > Tchem > Tbio > Tdark reflects
                       evidence that the gene is a validated drug target)
  4. Literature     — pubmed_total from evidence pipeline

A candidate scoring high across 3+ independent axes is "silver-standard
triangulated" — flagged prominently on the dashboard. This is how we
avoid the failure mode of one clever pillar over-ranking a false positive.

Pharos API: https://pharos-api.ncats.io (free, no auth)
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional


PHAROS_API = "https://pharos-api.ncats.io/targets"
PHAROS_CACHE = Path("data/drkg/pharos_tdl_cache.json")

# TDL rankings — Tclin = approved drug exists; Tchem = quality chem probe;
# Tbio = well-studied; Tdark = understudied
TDL_SCORES = {
    "Tclin": 1.00,
    "Tchem": 0.80,
    "Tbio":  0.55,
    "Tdark": 0.25,
}


@lru_cache(maxsize=1)
def _load_pharos_cache() -> dict[str, str]:
    if not PHAROS_CACHE.exists():
        return {}
    try:
        return json.loads(PHAROS_CACHE.read_text())
    except Exception:
        return {}


def _save_pharos_cache(cache: dict[str, str]) -> None:
    PHAROS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PHAROS_CACHE.write_text(json.dumps(cache, indent=1))


def get_pharos_tdl(gene_symbol: str) -> str:
    """Fetch Pharos target development level for a gene symbol.

    Returns 'Tclin'|'Tchem'|'Tbio'|'Tdark'|'' (empty = unknown / API failed).
    Cached to data/drkg/pharos_tdl_cache.json.
    """
    if not gene_symbol:
        return ""
    cache = _load_pharos_cache()
    if gene_symbol in cache:
        return cache[gene_symbol]

    url = f"{PHAROS_API}({urllib.parse.quote(gene_symbol)})?fields=tdl"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        tdl = data.get("tdl") or data.get("targetDevelopmentLevel") or ""
    except Exception:
        tdl = ""

    cache[gene_symbol] = tdl
    _save_pharos_cache(cache)
    return tdl


def compute_triangulation_score(
    kg_score: float,
    docking_score: Optional[float] = None,
    pharos_tdl: Optional[str] = None,
    pubmed_total: int = 0,
) -> dict:
    """
    Aggregate 4 independent axes into a triangulation profile.

    Args:
        kg_score:       OpenCure's combined_score (0-1)
        docking_score:  Vina best pose in kcal/mol (lower = better binding);
                        None if docking unavailable
        pharos_tdl:     Tclin/Tchem/Tbio/Tdark; None if unknown
        pubmed_total:   Total PubMed hits for (drug, disease)

    Returns:
        {
          "n_axes_agree": 0-4,
          "axes": {"kg": bool, "docking": bool, "pharos": bool, "literature": bool},
          "triangulation_score": 0-1,
          "label": "silver-standard" | "multi-axis" | "kg-only" | ""
        }
    """
    axes = {
        "kg":         kg_score >= 0.4,
        "docking":    docking_score is not None and docking_score <= -7.0,
        "pharos":     (pharos_tdl or "") in ("Tclin", "Tchem"),
        "literature": pubmed_total >= 10,
    }
    n = sum(1 for v in axes.values() if v)

    # Weighted aggregate (kg has highest weight; docking has more than pharos)
    w = {"kg": 0.35, "docking": 0.30, "pharos": 0.15, "literature": 0.20}
    # Scale each axis signal to [0,1] — kg directly, docking via clamp,
    # pharos via TDL_SCORES, literature via log
    import math
    kg_s = min(1.0, max(0.0, kg_score))
    dk_s = 0.0 if docking_score is None else min(1.0, max(0.0, (-docking_score) / 12.0))
    ph_s = TDL_SCORES.get(pharos_tdl or "", 0.0)
    lit_s = min(1.0, math.log1p(pubmed_total) / math.log1p(200))
    agg = w["kg"] * kg_s + w["docking"] * dk_s + w["pharos"] * ph_s + w["literature"] * lit_s

    if n >= 3:
        label = "silver-standard"
    elif n >= 2:
        label = "multi-axis"
    elif n >= 1:
        label = "kg-only"
    else:
        label = ""

    return {
        "n_axes_agree": n,
        "axes": axes,
        "triangulation_score": round(agg, 3),
        "label": label,
        "axis_values": {
            "kg": round(kg_s, 3),
            "docking": round(dk_s, 3),
            "pharos": round(ph_s, 3),
            "literature": round(lit_s, 3),
        },
    }


if __name__ == "__main__":
    # Smoke test
    for name, kg, dk, tdl, pm in [
        ("Strong candidate", 0.7, -9.0, "Tclin", 80),
        ("Literature-only",   0.2, None,  "Tdark",  300),
        ("Dark-target hit",   0.8, None,  "Tdark",  5),
        ("Lucky KG only",     0.5, None,  "",       2),
    ]:
        r = compute_triangulation_score(kg, dk, tdl, pm)
        print(f"{name:22s}  n_axes={r['n_axes_agree']}  score={r['triangulation_score']:.3f}  {r['label']}  axes={r['axes']}")
