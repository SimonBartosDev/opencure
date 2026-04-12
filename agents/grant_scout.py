#!/usr/bin/env python3
"""Grant Scout Agent — finds funding opportunities for drug repurposing.

Runs weekly. Searches NIH Reporter and other sources for open grant
opportunities, then drafts summaries matching OpenCure predictions.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError

from common import (
    load_config, get_top_breakthrough_predictions, get_diseases,
    write_outbox, log_run
)


def search_nih_reporter(query: str, max_results: int = 10) -> list[dict]:
    """Search NIH Reporter for funded projects related to drug repurposing."""
    url = "https://api.reporter.nih.gov/v2/projects/search"
    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "terms",
                "search_text": query
            },
            "is_active": True
        },
        "offset": 0,
        "limit": max_results,
        "sort_field": "project_start_date",
        "sort_order": "desc"
    }

    try:
        req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("results", [])
    except (URLError, json.JSONDecodeError, TimeoutError):
        return []


def run():
    config = load_config()
    breakthroughs = get_top_breakthrough_predictions(5)
    diseases = get_diseases()
    today = datetime.now().strftime("%Y-%m-%d")

    report = f"# Grant Scout Report — {today}\n\n"
    report += "Funding opportunities and active NIH projects related to OpenCure predictions.\n\n"
    report += "---\n\n"

    # Search for drug repurposing grants
    report += "## Active NIH Drug Repurposing Projects\n\n"
    projects = search_nih_reporter("drug repurposing neglected diseases", max_results=10)
    time.sleep(1)

    if projects:
        for p in projects:
            title = p.get("project_title", "Untitled")
            pi = p.get("contact_pi_name", "Unknown PI")
            org = p.get("organization", {}).get("org_name", "")
            award = p.get("award_amount")
            mechanism = p.get("activity_code", "")
            start = p.get("project_start_date", "")[:10]
            end = p.get("project_end_date", "")[:10]
            abstract = (p.get("abstract_text") or "")[:300]

            report += f"### {title}\n"
            report += f"- **PI:** {pi} ({org})\n"
            report += f"- **Mechanism:** {mechanism}\n"
            if award:
                report += f"- **Award:** ${award:,.0f}\n"
            report += f"- **Period:** {start} to {end}\n"
            if abstract:
                report += f"- **Abstract:** {abstract}...\n"
            report += "\n"
    else:
        report += "No active projects found (API may be unavailable).\n\n"

    # Disease-specific searches
    report += "## Disease-Specific Funding Landscape\n\n"
    for bt in breakthroughs[:3]:
        disease = bt["disease"]
        drug = bt["drug_name"]
        projects = search_nih_reporter(f"{disease} drug repurposing", max_results=3)
        time.sleep(1)

        report += f"### {disease} (top prediction: {drug})\n"
        if projects:
            for p in projects:
                report += f"- {p.get('project_title', 'Untitled')} — {p.get('contact_pi_name', '')} ({p.get('activity_code', '')})\n"
        else:
            report += "- No active repurposing projects found — **gap = opportunity**\n"
        report += "\n"

    # Suggest grant targets
    report += "## Recommended Grant Programs\n\n"
    report += """- **NIH R21 (Exploratory)**: $275K over 2 years. Perfect for "AI-predicted drug repurposing validation." Apply via grants.nih.gov.
- **Gates Grand Challenges**: $100K exploration grants for global health. Our neglected disease predictions (Chagas, Malaria, Leishmaniasis) fit perfectly.
- **Wellcome Trust Discovery Awards**: Up to 8 years funding for bold science. Drug repurposing via AI qualifies.
- **Chan Zuckerberg Initiative**: Focus on rare diseases and open science. Our Apache 2.0 open-source approach aligns.
- **Fast Grants**: $10K-$500K with 48-hour decisions. For urgent science — frame as "AI-ready repurposing candidates awaiting validation."
- **Experiment.com**: Crowdfunding for science. Could fund a single cell assay ($5-10K) for top prediction.
- **CureDuchenne / CF Foundation / MJFF**: Disease-specific foundations with their own funding for repurposing research.

"""

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("grant_opportunities", report)
    log_run("grant_scout", f"Searched NIH Reporter, generated grant recommendations")
    print(f"[Grant Scout] Done. Report: {path}")


if __name__ == "__main__":
    run()
