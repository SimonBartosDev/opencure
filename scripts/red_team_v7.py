"""Adversarial red-team pass — attaches ``red_team_assessment`` to top-K
candidates of every result JSON under ``experiments/results/``.

Always runs the deterministic critic. Adds an LLM-narrated layer when
``--use-llm`` is set and ``mlx_lm`` + a local Llama model are present.

Usage:
    python3 scripts/red_team_v7.py                  # all result JSONs
    python3 scripts/red_team_v7.py Schistosomiasis  # one disease
    python3 scripts/red_team_v7.py --use-llm        # opt into LLM narrative
    python3 scripts/red_team_v7.py --top-k 30       # default 20

Designed to plug into ``scripts/finalize_v5.py`` as a new step right
before the wet-lab brief generator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.evidence.mechanism_uncertainty import mechanism_confidence
from opencure.scoring.common import AGGREGATE_RESULT_FILES
from opencure.scoring.red_team import assess_candidate

RESULTS_DIR = Path("experiments/results")


def red_team_file(path: Path, *, top_k: int, use_llm: bool) -> int:
    data = json.loads(path.read_text())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0

    # Disease-level mechanism confidence for the top-of-result file
    disease_entity = data.get("disease_entity") or ""
    if not disease_entity:
        for c in candidates:
            if c.get("disease_entity"):
                disease_entity = c["disease_entity"]; break
    mc = mechanism_confidence(disease_entity) if disease_entity else None

    annotated = 0
    for cand in candidates[:top_k]:
        critique = assess_candidate(cand, mechanism_confidence=mc, use_llm=use_llm)
        cand["red_team_assessment"] = critique
        annotated += 1

    data["mechanism_confidence"] = mc
    path.write_text(json.dumps(data, indent=2))
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diseases", nargs="*", help="Specific disease keys to process; empty = all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-llm", action="store_true",
                        help="Use local Llama-3.1-8B via MLX for narrative critique.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.diseases:
        files = [args.results_dir / f"{d}.json" for d in args.diseases]
    else:
        files = sorted(p for p in args.results_dir.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)

    total = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n = red_team_file(f, top_k=args.top_k, use_llm=args.use_llm)
        print(f"  {f.name}: {n} critiques")
        total += n
    print(f"\nDone. {total} candidates across {len(files)} files now carry "
          f"red_team_assessment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
