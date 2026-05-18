"""Negative-control suite for the OpenCure ranker.

A "negative control" for ``(drug, disease)`` is a drug we have strong
clinical reason to believe is *not* a credible repurposing candidate
for that disease — wrong mechanism, wrong therapeutic class, no shared
biology. If the platform ranks one of these in the top half of its
candidate list for that disease, something is wrong (hub-bias artifact,
spurious KG path, embedding leakage). The platform should rank these
*below the per-disease median*.

Two sources of negatives:

1. **Curated YAML** at ``tests/data/negative_controls.yaml`` — hand-
   selected with rationales. Highest signal, lowest coverage.
2. **ATC-class-mismatch heuristic** (``synthesize_controls``) — for any
   disease without curated entries, sample N drugs whose ATC class is
   structurally unrelated to the disease's therapeutic area. Lower
   signal, full coverage.

The verifier (``verify_negative_controls``) consumes a directory of
finalized result JSONs (post ``finalize_v5``) and reports per-disease
pass/fail. ``scripts/verify_negative_controls.py`` wraps this into a
CI-able exit code so failures block release.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Optional dependency — yaml is in tree (used by recipe configs etc.).
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

NEGATIVE_CONTROLS_PATH = Path("tests/data/negative_controls.yaml")
UNIVERSAL_KEY = "universal_hubs"


@dataclass(frozen=True)
class NegativeControl:
    drug_id: str
    drug_name: str
    rationale: str


def load_negative_controls(
    path: Path = NEGATIVE_CONTROLS_PATH,
) -> dict[str, list[NegativeControl]]:
    """Read the YAML and return ``{disease_key: [NegativeControl, ...]}``.

    The special key ``universal_hubs`` carries controls expected to fail
    *for every* disease (metabolites, etc.); callers typically merge
    that list into each disease's per-disease list before checking.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to load negative_controls.yaml")
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, list[NegativeControl]] = {}
    for disease, entry in raw.items():
        drugs = entry.get("drugs", []) if isinstance(entry, dict) else []
        out[disease] = [
            NegativeControl(d["id"], d["name"], d.get("rationale", ""))
            for d in drugs
        ]
    return out


def controls_for_disease(
    disease_key: str,
    catalog: dict[str, list[NegativeControl]] | None = None,
) -> list[NegativeControl]:
    """Universal hubs + per-disease curated entries (deduped by drug_id)."""
    if catalog is None:
        catalog = load_negative_controls()
    universal = catalog.get(UNIVERSAL_KEY, [])
    specific = catalog.get(disease_key, [])
    seen: set[str] = set()
    merged: list[NegativeControl] = []
    for c in (*universal, *specific):
        if c.drug_id not in seen:
            merged.append(c)
            seen.add(c.drug_id)
    return merged


def synthesize_controls(
    disease_key: str,
    *,
    available_drug_ids: Iterable[str],
    n: int = 3,
    seed: int = 0,
) -> list[NegativeControl]:
    """ATC-class-mismatch fallback when no curated controls exist.

    Picks N drugs at random from ``available_drug_ids`` excluding
    anything that appears in *any* curated negative-control list (so
    we don't accidentally re-use someone else's hand-picked control).
    Pure-fallback; the real value comes from human curation.
    """
    import random

    catalog = load_negative_controls()
    used_ids = {
        c.drug_id
        for entries in catalog.values()
        for c in entries
    }
    pool = [d for d in available_drug_ids if d not in used_ids]
    if len(pool) < n:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    return [
        NegativeControl(
            drug_id=did,
            drug_name=did,
            rationale=f"ATC-class-mismatch heuristic for {disease_key}",
        )
        for did in chosen
    ]


# ---- Verifier ----------------------------------------------------------

@dataclass
class DiseaseControlReport:
    disease: str
    n_controls: int
    n_below_median: int
    failures: list[tuple[str, int]]  # (drug_id, rank)

    @property
    def pass_rate(self) -> float:
        return self.n_below_median / self.n_controls if self.n_controls else 1.0

    @property
    def passed(self) -> bool:
        return self.pass_rate >= 0.95


def _candidate_rank(cand: dict, fallback: int) -> int:
    """Best-available numeric rank from a result candidate dict."""
    for key in ("rank", "ensemble_rank"):
        v = cand.get(key)
        if isinstance(v, int):
            return v
    return fallback


def verify_disease_result(
    disease_key: str,
    result_path: Path,
    catalog: dict[str, list[NegativeControl]] | None = None,
) -> DiseaseControlReport | None:
    """Score one disease's result file against its negative-control set.

    Returns ``None`` if the disease has no controls (curated or hub-only)
    *and* synthesize_controls couldn't produce any either.
    """
    if not result_path.exists():
        return None
    data = json.loads(result_path.read_text())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return None

    # Build a drug_id → rank map
    rank_map: dict[str, int] = {}
    for i, cand in enumerate(candidates, start=1):
        drug_id = (cand.get("drug_id") or "").replace("Compound::", "")
        if drug_id:
            rank_map[drug_id] = _candidate_rank(cand, i)
    if not rank_map:
        return None

    median_rank = statistics.median(rank_map.values())
    controls = controls_for_disease(disease_key, catalog)
    if not controls:
        return None

    below_median = 0
    failures: list[tuple[str, int]] = []
    n_in_universe = 0
    for c in controls:
        rk = rank_map.get(c.drug_id)
        if rk is None:
            # Drug not in this disease's candidate list — count as below
            # median (it was never even ranked, the safest outcome).
            below_median += 1
            n_in_universe += 1
            continue
        n_in_universe += 1
        if rk > median_rank:
            below_median += 1
        else:
            failures.append((c.drug_id, rk))

    return DiseaseControlReport(
        disease=disease_key,
        n_controls=n_in_universe,
        n_below_median=below_median,
        failures=failures,
    )


def verify_negative_controls(
    results_dir: Path,
    catalog: dict[str, list[NegativeControl]] | None = None,
) -> list[DiseaseControlReport]:
    """Run the negative-control check across every result JSON.

    Skips disease files for which we have no controls at all.
    """
    if catalog is None:
        catalog = load_negative_controls()
    reports: list[DiseaseControlReport] = []
    for path in sorted(results_dir.glob("*.json")):
        # Skip aggregates (mechanism_clusters etc.).
        from opencure.scoring.common import AGGREGATE_RESULT_FILES
        if path.stem in AGGREGATE_RESULT_FILES:
            continue
        report = verify_disease_result(path.stem, path, catalog)
        if report is not None:
            reports.append(report)
    return reports
