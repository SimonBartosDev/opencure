#!/usr/bin/env python3
"""Literature Scout Agent — finds new papers mentioning OpenCure predictions.

Runs daily. For top predictions, queries PubMed for papers published in the
last 7 days. Flags papers that may validate or contradict predictions.
"""

from __future__ import annotations

import time
from datetime import datetime

from common import (
    load_config, load_predictions, write_outbox, log_run, days_ago
)
from opencure.evidence.pubmed import search_pubmed


def run():
    config = load_config()
    agent_cfg = config["agents"]["literature_scout"]
    top_n = agent_cfg.get("top_predictions", 50)
    lookback = agent_cfg.get("lookback_days", 7)

    predictions = load_predictions(top_n)
    date_from = days_ago(lookback)
    today = datetime.now().strftime("%Y-%m-%d")

    findings = []
    total_papers = 0
    checked = 0

    print(f"[Literature Scout] Checking {len(predictions)} predictions for papers since {date_from}")

    for pred in predictions:
        drug = pred["drug_name"]
        disease = pred["disease"]
        query = f'("{drug}"[Title/Abstract]) AND ("{disease}"[Title/Abstract]) AND ("{date_from}"[Date - Publication] : "3000"[Date - Publication])'

        try:
            results = search_pubmed(query, max_results=5)
        except Exception as e:
            print(f"  Error searching {drug} + {disease}: {e}")
            results = []

        if results:
            findings.append({
                "drug": drug,
                "disease": disease,
                "confidence": pred["confidence"],
                "novelty": pred["novelty_level"],
                "score": pred["combined_score"],
                "papers": results,
            })
            total_papers += len(results)

        checked += 1
        if checked % 10 == 0:
            print(f"  Checked {checked}/{len(predictions)}...")

        time.sleep(0.5)  # Rate limiting

    # Generate report
    report = f"# Literature Scout Report — {today}\n\n"
    report += f"Checked {checked} predictions for papers published in the last {lookback} days.\n\n"

    if not findings:
        report += "**No new papers found.** All predictions remain without recent literature updates.\n"
    else:
        report += f"**Found {total_papers} new paper(s) across {len(findings)} drug-disease pair(s).**\n\n"
        report += "---\n\n"

        for f in findings:
            report += f"## {f['drug']} + {f['disease']}\n"
            report += f"Prediction: {f['confidence']} confidence, {f['novelty']}, score {f['score']:.2f}\n\n"
            for p in f["papers"]:
                pmid = p.get("pmid", "")
                title = p.get("title", "Untitled")
                authors = p.get("authors", "")
                year = p.get("year", "")
                report += f"- **{title}**\n"
                report += f"  {authors} ({year})\n"
                if pmid:
                    report += f"  https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
                report += "\n"
            report += "---\n\n"

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("literature_report", report)
    log_run("literature_scout", f"Checked {checked} predictions, found {total_papers} new papers in {len(findings)} pairs")
    print(f"[Literature Scout] Done. {total_papers} papers found. Report: {path}")


if __name__ == "__main__":
    run()
