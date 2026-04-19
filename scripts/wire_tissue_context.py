"""
Post-processor: populate ``tissue_context`` on every candidate in every v5
result JSON. Pre-fix results carried ``tissue_context: {}`` because the
pillar was only invoked when drug-disease shared targets were non-empty —
which happens on a minority of candidates. This backfill runs the same
GTEx/DISEASE_TISSUE_MAP scoring the live pipeline now runs, using
disease-associated genes when shared targets are absent.

Usage
-----
    python3 scripts/wire_tissue_context.py                # all result JSONs
    python3 scripts/wire_tissue_context.py Malaria
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.scoring.tissue_context import score_tissue_context


RESULTS_DIR = Path("experiments/results")


def _disease_genes(disease_name: str) -> set[str]:
    try:
        from opencure.data.opentargets import get_disease_targets
        tgts = get_disease_targets(disease_name) or []
        return {f"Gene::{g}" for g in tgts[:50] if g}
    except Exception:
        return set()


def backfill(path: Path) -> int:
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0
    disease_name = data.get("disease") or path.stem.replace("_", " ")

    # Disease-level gene set (cached once per file)
    disease_gene_set = _disease_genes(disease_name)

    n_populated = 0
    for cand in candidates:
        # Use shared_targets if the candidate has them; else fall back
        shared = cand.get("shared_targets") or []
        gene_set = {f"Gene::{g}" for g in shared if g} if shared else disease_gene_set
        ctx = score_tissue_context(disease_name, gene_set)
        cand["tissue_context"] = ctx
        if ctx.get("tissues"):
            n_populated += 1

    json.dump(data, path.open("w"), indent=2)
    return n_populated


def main() -> None:
    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in {"screening_summary", "novel_candidates",
                                          "opencure_database"})
    total = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n = backfill(f)
        print(f"  {f.name}: {n} tissue_context populated")
        total += n
    print(f"\nDone. {total} candidates now carry tissue_context.")


if __name__ == "__main__":
    main()
