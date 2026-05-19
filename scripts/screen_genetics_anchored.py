#!/usr/bin/env python3
"""
Screen a disease with OpenCure's genetics-anchored scorer.

Thin CLI over `opencure.scoring.genetics_anchored.score_disease`. It prints
either a ranked list of mechanism-grounded drug candidates (for a
genetics-covered disease) or the honest `not_assessed` message (for a disease
with no genetic signal). It NEVER fabricates a ranking.

Usage
-----
  python scripts/screen_genetics_anchored.py "type 2 diabetes mellitus"
  python scripts/screen_genetics_anchored.py Disease::MESH:D003924
  python scripts/screen_genetics_anchored.py EFO_0000095 --top 15
  python scripts/screen_genetics_anchored.py "gonorrhea" --offline
  python scripts/screen_genetics_anchored.py "Crohn disease" --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.scoring.genetics_anchored import score_disease


def _print_human(result) -> None:
    print(f"\nDisease query : {result.disease_query}")
    print(f"OT disease ids: {', '.join(result.ot_disease_ids) or '(none)'}")
    print(f"Verdict       : {result.status.upper()}")

    if result.not_assessed:
        print(f"\n  NOT ASSESSED — {result.reason}")
        print("\n  OpenCure has no genetic signal for this disease and will "
              "not\n  emit a ranking it cannot support.")
        return

    cands = result.candidates
    print(f"\n  COVERED — {len(cands)} mechanism-grounded drug candidate(s), "
          "ranked by\n  Open Targets genetics-datasource association score.\n")
    header = (f"  {'#':>3}  {'score':>7}  {'drug':<26}  {'gene':<10}  "
              f"{'action':<14}  evidence")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, c in enumerate(cands, 1):
        name = (c.drug_name or c.drug_id)[:26]
        gene = (c.target_gene_symbol or c.target_gene_id)[:10]
        action = (c.action_type or "-")[:14]
        ev = ", ".join(c.evidence_datasources) or "(score-cache; sources n/a)"
        print(f"  {i:>3}  {c.score:>7.4f}  {name:<26}  {gene:<10}  "
              f"{action:<14}  {ev}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("disease",
                    help="disease name, DRKG entity, MeSH id, or OT id")
    ap.add_argument("--top", type=int, default=20,
                    help="max ranked candidates to show (default 20)")
    ap.add_argument("--offline", action="store_true",
                    help="never call the OT API; caches only")
    ap.add_argument("--json", action="store_true",
                    help="emit the full result as JSON")
    args = ap.parse_args(argv)

    result = score_disease(
        args.disease, top_k=args.top, allow_network=not args.offline,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
