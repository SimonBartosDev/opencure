#!/usr/bin/env python3
"""
Genetics-anchored screening across OpenCure's full disease set.

Runs `opencure.scoring.genetics_anchored.score_disease` over every disease in
`experiments.systematic_screening.TARGET_DISEASES` (~93 diseases) and writes:

  experiments/results/genetics_anchored/<disease>.json   one per disease
  experiments/results/genetics_anchored/genetics_anchored_summary.json

This is the platform's PRIMARY screening path. It reuses the validated scorer
verbatim — no scoring logic lives here. Diseases with no human-genetic signal
are recorded honestly as `not_assessed` with a machine-stable reason code; the
coverage number (covered vs not_assessed) is itself a real, reportable result.

Usage
-----
  python3 scripts/screen_genetics_anchored_all.py            # full run
  python3 scripts/screen_genetics_anchored_all.py --offline  # caches only
  python3 scripts/screen_genetics_anchored_all.py --top 40   # cap per disease
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.systematic_screening import TARGET_DISEASES
from opencure.scoring.genetics_anchored import score_disease

OUT_DIR = Path("experiments/results/genetics_anchored")


def _reason_code(reason: str | None) -> str:
    """Collapse a free-text not-assessed reason into a stable category code."""
    if not reason:
        return "unknown"
    r = reason.lower()
    if "no efo mapping" in r:
        return "no_efo_mapping"
    if "no genetics-datasource associations" in r:
        return "no_genetics_associations"
    if "no drug has" in r or "no curated chembl mechanism" in r:
        return "no_mechanism_drug"
    if "network fetch is disabled" in r:
        return "unresolved_offline"
    return "other"


def _safe_name(disease: str) -> str:
    return disease.replace(" ", "_").replace("'", "").replace("/", "_")


def screen_all(top_k: int | None, allow_network: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flat = [(cat, d) for cat, ds in TARGET_DISEASES.items() for d in ds]
    total = len(flat)

    print("=" * 70)
    print("  OpenCure — genetics-anchored screening (primary path)")
    print(f"  Diseases: {total}   network={'on' if allow_network else 'OFF'}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    summary: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "scorer": "opencure.scoring.genetics_anchored.score_disease",
        "total_diseases": total,
        "covered": 0,
        "not_assessed": 0,
        "not_assessed_reasons": {},
        "diseases": {},
    }

    for i, (category, disease) in enumerate(flat, 1):
        result = score_disease(
            disease, top_k=top_k, allow_network=allow_network,
        )
        rcode = None if result.covered else _reason_code(result.reason)

        record = {
            "disease": disease,
            "category": category,
            "status": result.status,
            "covered": result.covered,
            "reason": result.reason,
            "reason_code": rcode,
            "ot_disease_ids": result.ot_disease_ids,
            "n_candidates": len(result.candidates),
            "candidates": [c.to_dict() for c in result.candidates],
        }
        (OUT_DIR / f"{_safe_name(disease)}.json").write_text(
            json.dumps(record, indent=2)
        )

        if result.covered:
            summary["covered"] += 1
            top = result.candidates[0]
            print(f"[{i:>2}/{total}] COVERED      {disease:<32} "
                  f"{len(result.candidates):>4} leads  "
                  f"top: {(top.drug_name or top.drug_id)} -> "
                  f"{top.target_gene_symbol or top.target_gene_id}")
        else:
            summary["not_assessed"] += 1
            summary["not_assessed_reasons"][rcode] = (
                summary["not_assessed_reasons"].get(rcode, 0) + 1
            )
            print(f"[{i:>2}/{total}] not_assessed {disease:<32} ({rcode})")

        summary["diseases"][disease] = {
            "category": category,
            "status": result.status,
            "reason_code": rcode,
            "n_candidates": len(result.candidates),
        }

    (OUT_DIR / "genetics_anchored_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n" + "=" * 70)
    print("  COVERAGE SUMMARY")
    print("=" * 70)
    cov, na = summary["covered"], summary["not_assessed"]
    print(f"  covered      : {cov:>3} / {total}  "
          f"({100 * cov / total:.0f}%)")
    print(f"  not_assessed : {na:>3} / {total}  "
          f"({100 * na / total:.0f}%)")
    print("  not_assessed breakdown by reason:")
    for code, n in sorted(summary["not_assessed_reasons"].items(),
                          key=lambda kv: -kv[1]):
        print(f"    {code:<26} {n}")
    print(f"\n  Results: {OUT_DIR}/")
    print(f"  Summary: {OUT_DIR / 'genetics_anchored_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--top", type=int, default=None,
                    help="cap ranked candidates per disease (default: all)")
    ap.add_argument("--offline", action="store_true",
                    help="never call the OT API; on-disk caches only")
    args = ap.parse_args(argv)

    t0 = time.time()
    screen_all(top_k=args.top, allow_network=not args.offline)
    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
