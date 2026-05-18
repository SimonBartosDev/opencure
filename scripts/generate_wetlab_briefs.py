"""Generate per-disease wet-lab briefs.

Reads every per-disease result JSON under ``experiments/results/`` and
emits a Markdown brief for the top-K candidates of each into
``experiments/results/briefs/<disease>_top<K>.md``.

Designed as the final step of ``scripts/finalize_v5.py``. Always
produces a brief — uses the deterministic mechanism narrative when
no local LLM is available, the LLM-narrated version when ``--use-llm``
is set and ``mlx_lm`` + a Llama model are installed.

Usage:
    python3 scripts/generate_wetlab_briefs.py
    python3 scripts/generate_wetlab_briefs.py Schistosomiasis Chagas_disease
    python3 scripts/generate_wetlab_briefs.py --top-k 10 --use-llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.scoring.common import AGGREGATE_RESULT_FILES
from opencure.scoring.per_class_ensemble import route_disease
from opencure.scoring.wetlab_brief import render_disease_brief

RESULTS_DIR = Path("experiments/results")


def brief_one(path: Path, *, top_k: int, use_llm: bool, out_dir: Path) -> Path | None:
    data = json.loads(path.read_text())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return None

    disease_name = data.get("disease") or path.stem.replace("_", " ")
    disease_entity = data.get("disease_entity") or ""
    if not disease_entity:
        for c in candidates:
            if c.get("disease_entity"):
                disease_entity = c["disease_entity"]; break

    disease_class = route_disease(disease_name)

    md = render_disease_brief(
        candidates,
        disease_name=disease_name,
        disease_entity=disease_entity,
        disease_class=disease_class,
        top_k=top_k,
        use_llm=use_llm,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_top{top_k}.md"
    out_path.write_text(md)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diseases", nargs="*",
                        help="Specific disease keys; empty = all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path,
                        default=RESULTS_DIR / "briefs")
    args = parser.parse_args()

    if args.diseases:
        files = [args.results_dir / f"{d}.json" for d in args.diseases]
    else:
        files = sorted(p for p in args.results_dir.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)

    n_written = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        out = brief_one(f, top_k=args.top_k, use_llm=args.use_llm,
                        out_dir=args.out_dir)
        if out:
            n_written += 1
            print(f"  {out}")
    print(f"\nWrote {n_written} briefs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
