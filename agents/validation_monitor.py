#!/usr/bin/env python3
"""Validation Monitor Agent — tracks whether predictions are being tested.

Runs weekly. Queries ClinicalTrials.gov for new trials matching our
drug-disease pairs and PubMed for validation-related publications.
"""

from __future__ import annotations

import time
from datetime import datetime

from common import (
    load_config, load_predictions, write_outbox, log_run
)
from opencure.evidence.clinical_trials import search_clinical_trials
from opencure.evidence.pubmed import search_pubmed


def check_new_trials(drug: str, disease: str) -> list[dict]:
    """Check ClinicalTrials.gov for trials matching a drug-disease pair."""
    try:
        result = search_clinical_trials(drug, disease)
        if result and result.get("total_trials", 0) > 0:
            return [result]
    except Exception:
        pass
    return []


def check_validation_papers(drug: str, disease: str) -> list[dict]:
    """Search for recent papers about validating/testing this drug for this disease."""
    query = f'("{drug}"[Title/Abstract]) AND ("{disease}"[Title/Abstract]) AND (repurposing OR repositioning OR "in vitro" OR assay OR validation)'
    try:
        return search_pubmed(query, max_results=3)
    except Exception:
        return []


def run():
    config = load_config()
    predictions = load_predictions(30)  # Check top 30
    today = datetime.now().strftime("%Y-%m-%d")

    new_trials = []
    validation_papers = []
    checked = 0

    print(f"[Validation Monitor] Checking {len(predictions)} predictions for trials and validation papers...")

    for pred in predictions:
        drug = pred["drug_name"]
        disease = pred["disease"]

        # Check trials
        trials = check_new_trials(drug, disease)
        if trials:
            new_trials.append({"drug": drug, "disease": disease, "trials": trials, "prediction": pred})

        # Check validation papers
        papers = check_validation_papers(drug, disease)
        if papers:
            validation_papers.append({"drug": drug, "disease": disease, "papers": papers, "prediction": pred})

        checked += 1
        time.sleep(1)  # Rate limiting

    # Generate report
    report = f"# Validation Monitor Report — {today}\n\n"
    report += f"Checked {checked} top predictions for clinical trials and validation papers.\n\n"

    if new_trials:
        report += f"## New Clinical Trials Found ({len(new_trials)})\n\n"
        for t in new_trials:
            report += f"### {t['drug']} + {t['disease']}\n"
            report += f"Prediction: {t['prediction']['confidence']}, {t['prediction']['novelty_level']}\n"
            for trial in t["trials"]:
                report += f"- Total trials: {trial.get('total_trials', 0)}\n"
                phases = trial.get("trial_phases", {})
                if phases:
                    report += f"- Phases: {', '.join(f'{k}: {v}' for k, v in phases.items())}\n"
            report += f"- Search: https://clinicaltrials.gov/search?term={t['drug']}+{t['disease']}\n\n"
    else:
        report += "## Clinical Trials: None found\n\n"

    if validation_papers:
        report += f"## Validation-Related Papers Found ({len(validation_papers)})\n\n"
        for v in validation_papers:
            report += f"### {v['drug']} + {v['disease']}\n"
            for p in v["papers"]:
                report += f"- **{p.get('title', 'Untitled')}**\n"
                report += f"  {p.get('authors', '')} ({p.get('year', '')})\n"
                if p.get("pmid"):
                    report += f"  https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/\n"
                report += "\n"
    else:
        report += "## Validation Papers: None found\n\n"

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("validation_status", report)
    log_run("validation_monitor", f"Checked {checked}, found {len(new_trials)} trials, {len(validation_papers)} papers")
    print(f"[Validation Monitor] Done. Report: {path}")


if __name__ == "__main__":
    run()
