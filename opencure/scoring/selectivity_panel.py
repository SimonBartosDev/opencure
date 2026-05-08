"""Selectivity / off-target panel — v7.

A drug that binds 50 different targets at sub-micromolar affinity is
either a tool compound or a future toxicity disaster, not a credible
repurposing lead. The selectivity panel surfaces the off-target count
and a normalized selectivity score so downstream consumers (the
grouped combiner, the wet-lab brief generator, the CI verifier) can
penalise promiscuous binders.

Source: ``data/drkg/drug_target_activities.json`` — ChEMBL-derived
median-nM affinities per drug × target. This file is already
maintained by the v5 ingest pipeline; the selectivity layer just
re-summarises it.

Contract:

    compute_selectivity(activities) -> dict[drug_id, dict]

Each returned record carries:
    selectivity_score: float in [0, 1] — 1.0 means highly selective
                       (≤ 1 measurable target); 0 means promiscuous
                       (saturating off-target list)
    n_off_targets:     int — count of targets with median ≤ POTENCY_NM
    primary_target:    str — gene symbol of the highest-affinity hit
    primary_nM:        float — median nM at that target

Used downstream:
- score_ensemble_v5.py post-processor attaches these per candidate.
- combined_score is dampened proportional to (1 - selectivity_score)
  by the grouped combiner via ``apply_selectivity_penalty`` (kept
  separate from this module so the pure-data layer stays cheap to load).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from opencure.config import DATA_DIR

ACTIVITIES_PATH = DATA_DIR / "drug_target_activities.json"

# A target is counted as "active" / off-target when median nM ≤ this threshold.
# 10 µM is the standard upper bound for biological relevance; below this and the
# target is plausibly hit at therapeutic doses.
POTENCY_NM = 10_000.0

# Selectivity falls to 0 when the drug saturates this many off-targets.
SATURATION_OFF_TARGETS = 30

# Soft-penalty cap: combined_score is multiplied by [1 - PENALTY_MAX, 1.0]
# based on selectivity_score. A perfectly promiscuous drug loses 30%; a
# highly selective drug loses nothing.
PENALTY_MAX = 0.30


def _load_activities(
    path: Path = ACTIVITIES_PATH,
) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def compute_drug_selectivity(
    targets: dict[str, dict],
    *,
    potency_nm: float = POTENCY_NM,
    saturation: int = SATURATION_OFF_TARGETS,
) -> dict[str, float | str | int]:
    """Per-drug selectivity record from a ``{target: {median_nM, ...}}`` map.

    Returns a flat dict with the four canonical fields. Returns
    selectivity_score=1.0 / n_off_targets=0 / primary_*='' when the
    drug has no measurable activities — same contract as a fully
    selective hit, since "no off-targets known" is the optimistic
    interpretation when ChEMBL has no data.
    """
    if not targets:
        return {
            "selectivity_score": 1.0,
            "n_off_targets": 0,
            "primary_target": "",
            "primary_nM": 0.0,
        }

    active: list[tuple[str, float]] = []  # (gene, median_nM)
    for gene, info in targets.items():
        if not isinstance(info, dict):
            continue
        median = info.get("median_nM")
        if median is None:
            continue
        try:
            nm = float(median)
        except (TypeError, ValueError):
            continue
        if nm <= potency_nm:
            active.append((gene, nm))

    if not active:
        return {
            "selectivity_score": 1.0,
            "n_off_targets": 0,
            "primary_target": "",
            "primary_nM": 0.0,
        }

    # Primary target = best (lowest nM) hit; off-targets = the rest.
    active.sort(key=lambda kv: kv[1])
    primary_gene, primary_nm = active[0]
    n_off = max(0, len(active) - 1)
    selectivity = 1.0 - min(n_off / saturation, 1.0)
    return {
        "selectivity_score": float(selectivity),
        "n_off_targets": int(n_off),
        "primary_target": primary_gene,
        "primary_nM": float(primary_nm),
    }


def compute_selectivity_table(
    activities: Optional[dict[str, dict[str, dict]]] = None,
) -> dict[str, dict]:
    """Bulk selectivity for every drug in the activities cache.

    Used by the score_ensemble_v5 post-processor to bulk-attach
    selectivity records without per-candidate disk hits.
    """
    if activities is None:
        activities = _load_activities()
    return {drug: compute_drug_selectivity(targets)
            for drug, targets in activities.items()}


def selectivity_penalty(score: float, selectivity: float) -> float:
    """Multiplicative penalty applied to combined_score.

    Linear interpolation: selectivity=1.0 → penalty=1.0 (unchanged);
    selectivity=0.0 → penalty=1 - PENALTY_MAX (30% off). Keeps
    combined_score monotone in selectivity, never negative.
    """
    if selectivity >= 1.0:
        return score
    factor = 1.0 - PENALTY_MAX * (1.0 - max(selectivity, 0.0))
    return score * factor
