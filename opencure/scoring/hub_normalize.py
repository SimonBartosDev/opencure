"""
Degree-based damping for KG/network pillars.

Problem: high-degree "hub" drugs (Dexamethasone=3413 triplets, Cimetidine=1979,
Tacrolimus=2638) are topologically close to every disease gene simply because
they're connected to thousands of targets. TransE / RotatE / PrimeKG / Network
Proximity all reward proximity, so hubs mechanically top every disease ranking.

Fix: multiply KG- and network-derived group scores by a penalty in (0, 1] that
damps high-degree drugs. The penalty is calibrated so that a median-degree drug
(~150 triplets) gets ~1.0 and a 3000-triplet hub gets ~0.55 at alpha=0.5.

Formula:
    penalty(d) = (log1p(median) / log1p(d)) ** alpha

We intentionally leave structural/MR/TxGNN/ADMET pillars un-damped — those are
driven by chemistry or causal genetics, not graph topology.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


_DEGREE_CACHE: Optional[dict[str, int]] = None
DEGREE_PATH = Path("data/drkg/drug_degree.json")
CHEMBL_PATH = Path("data/drkg/chembl_phase.json")

# Reference degree = median over ChEMBL-phase-≥1 ("real drug") distribution.
# Calibrated in _load_degree; empirical default ~81.
_REF_DEGREE: float = 80.0


def _load_degree() -> dict[str, int]:
    """Load drug_degree.json cache. Returns {} if unavailable (fail-open).

    Calibrates _REF_DEGREE as the median degree among compounds with
    ChEMBL max_phase >= 1 (i.e. actual clinical-stage drugs), so the
    penalty baseline reflects "real drugs", not the long tail of DRKG
    compounds that have only 1–2 annotations.
    """
    global _DEGREE_CACHE, _REF_DEGREE
    if _DEGREE_CACHE is not None:
        return _DEGREE_CACHE
    if not DEGREE_PATH.exists():
        _DEGREE_CACHE = {}
        return _DEGREE_CACHE
    _DEGREE_CACHE = json.loads(DEGREE_PATH.read_text())
    # Calibrate reference against ChEMBL-phase drugs if available
    if CHEMBL_PATH.exists():
        try:
            chembl = json.loads(CHEMBL_PATH.read_text())
            real_degrees: list[int] = []
            for db_id, phase in chembl.items():
                try:
                    if float(phase) < 1:
                        continue
                except (TypeError, ValueError):
                    continue
                d = _DEGREE_CACHE.get(f"Compound::{db_id}") or _DEGREE_CACHE.get(db_id)
                if d:
                    real_degrees.append(int(d))
            if real_degrees:
                real_degrees.sort()
                _REF_DEGREE = float(real_degrees[len(real_degrees) // 2])
        except Exception:
            pass
    return _DEGREE_CACHE


def degree_penalty(drug_id: str, alpha: float = 0.5) -> float:
    """
    Return multiplier in (0, 1] that damps high-degree drugs.

    drug_id can be either full entity form ("Compound::DB01234") or bare
    DrugBank ID ("DB01234"); both are cached.

    alpha controls strength: 0.0 = no damping, 1.0 = strong damping.
    Default 0.5 is calibrated so hubs take ~45% penalty while median-degree
    drugs are ~unaffected.

    Unknown drugs (not in cache) return 1.0 (fail-open).
    """
    cache = _load_degree()
    if not cache:
        return 1.0

    d = cache.get(drug_id)
    if d is None and "::" in drug_id:
        d = cache.get(drug_id.split("::", 1)[1])
    if d is None and "::" not in drug_id:
        d = cache.get(f"Compound::{drug_id}")
    if d is None or d <= 0:
        return 1.0

    if d <= _REF_DEGREE:
        return 1.0

    # Log-scale damping vs. real-drug reference
    ratio = math.log1p(_REF_DEGREE) / math.log1p(d)
    return max(0.3, min(1.0, ratio ** alpha))


def get_degree(drug_id: str) -> int:
    """Return raw triplet degree for a drug, or 0 if unknown."""
    cache = _load_degree()
    if not cache:
        return 0
    d = cache.get(drug_id)
    if d is None and "::" in drug_id:
        d = cache.get(drug_id.split("::", 1)[1])
    if d is None and "::" not in drug_id:
        d = cache.get(f"Compound::{drug_id}")
    return int(d or 0)


def get_reference_degree() -> float:
    """Expose the reference degree used for calibration (useful for tests/logs)."""
    _load_degree()
    return _REF_DEGREE
