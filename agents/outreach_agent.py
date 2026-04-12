#!/usr/bin/env python3
"""Outreach Agent — finds new researchers and drafts personalized emails.

Runs weekly. Searches PubMed for recent authors publishing on target diseases,
drafts personalized emails referencing their work + matching OpenCure predictions,
and drafts Reddit/forum posts.
"""

from __future__ import annotations

import time
import re
from datetime import datetime

from common import (
    load_config, load_all_predictions, get_top_breakthrough_predictions,
    get_diseases, write_outbox, log_run, days_ago
)
from opencure.evidence.pubmed import search_pubmed


def find_recent_authors(disease: str, lookback_days: int = 30, max_results: int = 10) -> list[dict]:
    """Find researchers who recently published on a disease."""
    date_from = days_ago(lookback_days)
    query = f'("{disease}"[Title]) AND ("{date_from}"[Date - Publication] : "3000"[Date - Publication])'

    try:
        results = search_pubmed(query, max_results=max_results)
    except Exception:
        return []

    authors = []
    for paper in results:
        author_str = paper.get("authors", "")
        if not author_str:
            continue
        # First author is likely a PhD student/postdoc
        first_author = author_str.split(",")[0].strip()
        if first_author and len(first_author) > 3:
            authors.append({
                "name": first_author,
                "paper_title": paper.get("title", ""),
                "journal": paper.get("journal", ""),
                "year": paper.get("year", ""),
                "pmid": paper.get("pmid", ""),
                "disease": disease,
            })

    return authors


def draft_email(researcher: dict, prediction: dict, config: dict) -> str:
    """Draft a personalized outreach email."""
    dashboard = config.get("dashboard_url", "https://simonbartosdev.github.io/opencure/")
    github = config.get("github_url", "https://github.com/SimonBartosDev/opencure")

    drug = prediction["drug_name"]
    disease = prediction["disease"]
    score = prediction["combined_score"]
    pillars = prediction["pillars_hit"]
    novelty = prediction["novelty_level"]
    confidence = prediction["confidence"]

    email = f"""**To:** {researcher['name']}
**Subject:** OpenCure AI prediction: {drug} for {disease} — seeking validation partner

Dear {researcher['name'].split()[-1] if ' ' in researcher['name'] else researcher['name']},

I came across your recent paper "{researcher['paper_title']}" and wanted to reach out about a computational drug repurposing prediction that aligns with your research on {disease}.

Our open-source platform OpenCure uses 8 independent AI scoring methods (knowledge graph embeddings, graph neural networks, molecular similarity, gene expression reversal, Mendelian Randomization, and more) to identify repurposing candidates. For {disease}, we've identified **{drug}** as a {novelty} prediction with {confidence} confidence, supported by {pillars} independent AI pillars (combined score: {score:.2f}).

This prediction has not been experimentally validated. We're looking for a researcher willing to run an initial cell-based assay — we'd be happy to co-author any resulting publication and can provide full computational evidence and methodology support.

You can explore all predictions interactively at: {dashboard}
Full code and data: {github}

Would you be interested in discussing this further?

Best regards,
OpenCure Team
"""
    return email


def draft_reddit_post(predictions: list[dict], config: dict) -> str:
    """Draft a Reddit post for r/labrats or r/bioinformatics."""
    dashboard = config.get("dashboard_url", "")
    github = config.get("github_url", "")

    top3 = predictions[:3]
    examples = "\n".join(
        f"- **{p['drug_name']}** for {p['disease']} ({p['novelty_level']}, {p['pillars_hit']} pillars, score {p['combined_score']:.2f})"
        for p in top3
    )

    post = f"""**Title:** We built an open-source AI platform that screens 10,500+ drugs across 60+ diseases for repurposing — looking for validation partners

**Body:**

Hey everyone,

We've been working on OpenCure, an open-source drug repurposing platform that combines 8 independent AI methods (TransE, RotatE, TxGNN, molecular fingerprints, ChemBERTa, gene signature reversal, network proximity, and Mendelian Randomization) to predict new uses for existing FDA-approved drugs.

We screened all ~10,500 approved drugs across 60+ diseases (focusing on neglected, rare, and underserved conditions) and identified **78 breakthrough predictions** — drug-disease pairs with zero existing published literature but strong computational support from multiple independent methods.

Some highlights:
{examples}

Everything is open-source (Apache 2.0):
- Interactive dashboard: {dashboard}
- GitHub: {github}

We're looking for researchers interested in validating any of these predictions with a cell-based assay. Happy to co-author and provide full computational evidence support.

Has anyone here worked with similar computational repurposing tools? Any suggestions for the most promising predictions to validate first?
"""
    return post


def run():
    config = load_config()
    agent_cfg = config["agents"]["outreach_agent"]
    max_researchers = agent_cfg.get("max_researchers_per_run", 20)

    diseases = get_diseases()
    all_preds = load_all_predictions()
    breakthroughs = get_top_breakthrough_predictions(10)
    today = datetime.now().strftime("%Y-%m-%d")

    # Find recent authors across diseases
    all_researchers = []
    seen_names = set()

    print(f"[Outreach Agent] Searching for recent authors across {len(diseases)} diseases...")

    for disease in diseases:
        authors = find_recent_authors(disease, lookback_days=30, max_results=5)
        for a in authors:
            if a["name"] not in seen_names:
                seen_names.add(a["name"])
                all_researchers.append(a)
        time.sleep(0.5)

    # Limit to max per run
    researchers = all_researchers[:max_researchers]

    print(f"[Outreach Agent] Found {len(all_researchers)} unique researchers, drafting for {len(researchers)}")

    # Generate report with email drafts
    report = f"# Outreach Agent Report — {today}\n\n"
    report += f"Found {len(all_researchers)} new researchers publishing on our target diseases in the last 30 days.\n"
    report += f"Drafted {len(researchers)} personalized emails below.\n\n"
    report += "**Action required:** Review each email, edit as needed, then approve for sending.\n\n"
    report += "---\n\n"

    # Draft emails
    for researcher in researchers:
        # Find best matching prediction for their disease
        matching = [p for p in all_preds if p["disease"] == researcher["disease"]]
        if not matching:
            continue
        matching.sort(key=lambda p: -p["combined_score"])
        best_pred = matching[0]

        email = draft_email(researcher, best_pred, config)
        report += f"### Email to: {researcher['name']}\n"
        report += f"Their paper: *{researcher['paper_title']}* ({researcher['journal']}, {researcher['year']})\n"
        if researcher["pmid"]:
            report += f"PubMed: https://pubmed.ncbi.nlm.nih.gov/{researcher['pmid']}/\n"
        report += f"\n{email}\n\n---\n\n"

    # Draft Reddit/forum posts
    report += "## Reddit / Forum Post Drafts\n\n"

    reddit = draft_reddit_post(breakthroughs, config)
    report += f"### r/labrats or r/bioinformatics\n\n{reddit}\n\n---\n\n"

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("outreach_drafts", report)
    log_run("outreach_agent", f"Found {len(all_researchers)} researchers, drafted {len(researchers)} emails + Reddit post")
    print(f"[Outreach Agent] Done. Report: {path}")


if __name__ == "__main__":
    run()
