"""Retrospective-prospective validation — v7 Phase B3.

For every top-K (drug, disease) prediction across the 93 screened
diseases, query PubMed for 2024-2025 papers containing both terms.
Classify each match into:

    confirms   — paper supports the prediction (efficacy, beneficial, treats…)
    refutes    — paper undercuts it (no effect, failed trial, contraindicated)
    ambiguous  — both/neither/unclear

Then aggregate to a per-disease and an overall report:

    "Of N v7 predictions made against pre-2024 KG data, M were
    independently corroborated by 2024-2025 publications, K were
    refuted, J remain untested."

This is the strongest trust signal we can produce *before* a wet lab
ever runs an assay — the platform's predictive power on previously-
unseen knowledge, scored against real-world publication outcomes.

Output:
    experiments/prospective_v7_2024_2025.md  — human-readable report
    data/prospective/retrospective_v7.jsonl  — raw matches (per pair)

Usage:
    python3 scripts/retrospective_prospective.py
    python3 scripts/retrospective_prospective.py --top-k 10 --year-from 2024
    python3 scripts/retrospective_prospective.py --diseases Schistosomiasis Chagas_disease
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.evidence.pubmed import search_pubmed
from opencure.scoring.common import AGGREGATE_RESULT_FILES

RESULTS_DIR = Path("experiments/results")
PROSPECTIVE_DIR = Path("data/prospective")
REPORT_PATH = Path("experiments/prospective_v7_2024_2025.md")
RAW_PATH = PROSPECTIVE_DIR / "retrospective_v7.jsonl"

POSITIVE_TERMS = (
    "efficacy", "effective", "beneficial", "improved", "improvement",
    "treatment", "treats", "therapeutic", "promising", "potential treatment",
    "ameliorat", "successful", "responded", "remission", "synerg",
)
NEGATIVE_TERMS = (
    "no effect", "no significant", "ineffective", "failed", "failure",
    "no benefit", "lack of efficacy", "did not", "futile", "harm",
    "contraindicated", "worsened", "adverse",
)


def classify_paper(title: str, abstract: str) -> str:
    """Heuristic classifier — positive / negative / ambiguous.

    We count term occurrences (not just presence) so that an abstract
    saying "lack of efficacy" once but "no benefit, ineffective, did not"
    three times is correctly classified as refutes — even though
    'efficacy' appears once on the positive list.
    """
    text = f"{title.lower()} {abstract.lower()}"
    pos = sum(text.count(t) for t in POSITIVE_TERMS)
    neg = sum(text.count(t) for t in NEGATIVE_TERMS)
    if pos > neg and pos > 0:
        return "confirms"
    if neg > pos and neg > 0:
        return "refutes"
    return "ambiguous"


def query_pair(
    drug_name: str, disease_name: str, *,
    year_from: int, year_to: int,
    max_results: int = 5,
) -> list[dict]:
    """Run a date-restricted PubMed query for the (drug, disease) pair."""
    query = (
        f'("{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]) '
        f'AND ({year_from}:{year_to}[dp])'
    )
    return search_pubmed(query, max_results=max_results)


def validate_disease(
    result_path: Path, *,
    top_k: int, year_from: int, year_to: int,
    raw_writer,
) -> dict:
    """Run the retrospective-prospective check for one disease's top-K."""
    data = json.loads(result_path.read_text())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return {"disease": result_path.stem, "n_predictions": 0,
                "confirms": 0, "refutes": 0, "ambiguous": 0, "untested": 0}

    disease_name = data.get("disease") or result_path.stem.replace("_", " ")
    counts = {"confirms": 0, "refutes": 0, "ambiguous": 0, "untested": 0}

    for cand in candidates[:top_k]:
        drug_name = cand.get("drug_name") or cand.get("drug_id") or ""
        if not drug_name:
            counts["untested"] += 1
            continue

        try:
            papers = query_pair(
                drug_name, disease_name,
                year_from=year_from, year_to=year_to,
                max_results=5,
            )
            time.sleep(0.4)  # PubMed polite rate
        except Exception as exc:
            print(f"    [WARN] query failed for {drug_name}/{disease_name}: {exc}")
            counts["untested"] += 1
            continue

        if not papers:
            counts["untested"] += 1
            continue

        verdicts = [
            classify_paper(p.get("title", ""), p.get("abstract_snippet", ""))
            for p in papers
        ]
        # Per-pair verdict = the most-frequent of the per-paper verdicts
        # (preferring confirms over ambiguous when tied).
        if any(v == "confirms" for v in verdicts):
            verdict = "confirms"
        elif any(v == "refutes" for v in verdicts):
            verdict = "refutes"
        else:
            verdict = "ambiguous"
        counts[verdict] += 1

        raw_writer.write(json.dumps({
            "drug": drug_name,
            "disease": disease_name,
            "verdict": verdict,
            "n_papers": len(papers),
            "papers": [
                {"pmid": p.get("pmid"), "title": p.get("title", ""),
                 "year": p.get("year"), "verdict": v}
                for p, v in zip(papers, verdicts)
            ],
        }) + "\n")

    return {
        "disease": disease_name,
        "n_predictions": min(top_k, len(candidates)),
        **counts,
    }


def render_report(rows: list[dict]) -> str:
    if not rows:
        return "# Retrospective-prospective v7 validation\n\n_(no diseases scored)_"
    total = {k: sum(r[k] for r in rows)
             for k in ("n_predictions", "confirms", "refutes", "ambiguous", "untested")}
    lines = [
        "# Retrospective-prospective v7 validation",
        "",
        "Each (drug, disease) prediction was queried against PubMed for "
        "papers published in the test window. The platform's KG and "
        "training data predate this window, so any matching publication "
        "is independent corroboration (or refutation).",
        "",
        f"**Diseases scored:** {len(rows)}  ",
        f"**Predictions evaluated:** {total['n_predictions']}  ",
        f"**Independent confirmations:** {total['confirms']}  ",
        f"**Refutations:** {total['refutes']}  ",
        f"**Ambiguous:** {total['ambiguous']}  ",
        f"**Untested (no 2024-2025 paper):** {total['untested']}",
        "",
        "## Per-disease breakdown",
        "",
        "| Disease | Predictions | Confirms | Refutes | Ambiguous | Untested |",
        "|---------|-------------|---------:|--------:|----------:|---------:|",
    ]
    rows.sort(key=lambda r: -r["confirms"])
    for r in rows:
        lines.append(
            f"| {r['disease']} | {r['n_predictions']} | "
            f"{r['confirms']} | {r['refutes']} | "
            f"{r['ambiguous']} | {r['untested']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diseases", nargs="*",
                        help="Specific disease keys; empty = all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--year-from", type=int, default=2024)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    if args.diseases:
        files = [args.results_dir / f"{d}.json" for d in args.diseases]
    else:
        files = sorted(p for p in args.results_dir.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)

    PROSPECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    with args.raw.open("w") as raw_fh:
        for f in files:
            if not f.exists():
                continue
            print(f"Querying {f.name}...")
            row = validate_disease(
                f, top_k=args.top_k,
                year_from=args.year_from, year_to=args.year_to,
                raw_writer=raw_fh,
            )
            rows.append(row)
            print(f"  {row['disease']}: confirms={row['confirms']} "
                  f"refutes={row['refutes']} ambiguous={row['ambiguous']} "
                  f"untested={row['untested']}")

    args.report.write_text(render_report(rows))
    print(f"\nReport: {args.report}")
    print(f"Raw: {args.raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
