"""
Post-processor: re-rank every v5/v7 result JSON so genuine repurposing leads
are surfaced ahead of already-known treatments.

The screening pipeline orders candidates by ``combined_score`` alone, which is
blind to whether a drug is already standard-of-care for the disease. This step
demotes known treatments (``is_known_treatment`` true, or ``novelty_level`` in
{KNOWN, ESTABLISHED}) to the tail of the candidate list — kept and flagged via
``is_repurposing_candidate``, never dropped — and rewrites ``rank``.

Run AFTER ``refresh_known_treatment_labels.py`` and ``score_ensemble_v5.py``
(neither re-sorts the candidate list) and BEFORE the dashboard / brief /
snapshot steps, which all consume candidates in list order.

Usage
-----
    python3 scripts/apply_novelty_ranking.py
    python3 scripts/apply_novelty_ranking.py Sickle_cell_disease
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.scoring.common import AGGREGATE_RESULT_FILES
from opencure.scoring.novelty_rank import apply_novelty_ranking


RESULTS_DIR = Path("experiments/results")


def rerank(path: Path) -> tuple[int, int]:
    """Re-rank one result JSON. Returns (n_surfaced, n_total)."""
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0, 0
    ordered = apply_novelty_ranking(candidates)
    if "candidates" in data:
        data["candidates"] = ordered
    else:
        data["top_candidates"] = ordered
    json.dump(data, path.open("w"), indent=2)
    n_surfaced = sum(1 for c in ordered if c.get("is_repurposing_candidate"))
    return n_surfaced, len(ordered)


def main() -> None:
    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)
    total_s = 0
    total_n = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        s, n = rerank(f)
        total_s += s
        total_n += n
        if n:
            print(f"  {f.name}: {s}/{n} surfaced as repurposing leads")
    print(f"\nDone. {total_s}/{total_n} candidates surfaced; "
          f"{total_n - total_s} known treatments demoted to tail.")


if __name__ == "__main__":
    main()
