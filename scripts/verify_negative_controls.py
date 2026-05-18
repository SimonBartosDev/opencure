"""CI gate: assert that negative-control compounds rank below per-disease median.

Loads ``tests/data/negative_controls.yaml`` + every result JSON under
``experiments/results/`` and verifies that each curated negative
control sits below its disease's median rank.

Exits non-zero when the per-disease pass rate drops below the threshold
(default 95%), so CI catches regressions before they ship.

Usage:
    python3 scripts/verify_negative_controls.py
    python3 scripts/verify_negative_controls.py --threshold 0.90
    python3 scripts/verify_negative_controls.py --results-dir path/to/dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencure.eval.negative_control import verify_negative_controls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("experiments/results"))
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Minimum per-disease pass rate; below this fails CI.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory missing: {args.results_dir}")
        return 0  # not fatal — fresh checkouts have no results

    reports = verify_negative_controls(args.results_dir)
    if not reports:
        print(f"No diseases with negative-control coverage in {args.results_dir}.")
        return 0

    failed = []
    for r in reports:
        if not r.passed:
            failed.append(r)
            if not args.quiet:
                print(f"[FAIL] {r.disease}: {r.n_below_median}/{r.n_controls} "
                      f"below median ({r.pass_rate:.0%})")
                for drug_id, rk in r.failures[:5]:
                    print(f"        - {drug_id} ranked #{rk}")
        elif not args.quiet:
            print(f"[ok]   {r.disease}: {r.n_below_median}/{r.n_controls}")

    pass_rate = (len(reports) - len(failed)) / len(reports)
    print(f"\nTotal: {len(reports) - len(failed)}/{len(reports)} diseases pass "
          f"({pass_rate:.0%}); threshold {args.threshold:.0%}")
    return 0 if pass_rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
