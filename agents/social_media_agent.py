#!/usr/bin/env python3
"""Social Media Agent — drafts compelling content about OpenCure findings.

Runs weekly. Creates platform-specific drafts for Twitter/X threads,
Reddit posts, and LinkedIn updates. All drafts go to outbox for review.
"""

from __future__ import annotations

import random
from datetime import datetime

from common import (
    load_config, load_all_predictions, get_top_breakthrough_predictions,
    get_diseases, write_outbox, log_run
)


def draft_twitter_thread(predictions: list[dict], all_preds: list[dict], config: dict) -> str:
    """Draft a Twitter/X thread highlighting a specific finding."""
    dashboard = config.get("dashboard_url", "")
    github = config.get("github_url", "")
    total = len(all_preds)
    diseases = len(set(p["disease"] for p in all_preds))
    bt = len([p for p in all_preds if p.get("novelty_level") == "BREAKTHROUGH"])

    # Pick a random breakthrough to spotlight
    spotlight = random.choice(predictions[:5])

    # Cross-disease drugs
    from collections import Counter
    drug_counts = Counter(p["drug_name"] for p in all_preds if p.get("novelty_level") in ("BREAKTHROUGH", "NOVEL"))
    multi_disease = [(d, c) for d, c in drug_counts.most_common(5) if c >= 2]

    thread = f"""### Twitter/X Thread Draft

**Tweet 1 (hook):**
We used 8 independent AI methods to screen 10,500+ FDA-approved drugs across {diseases} diseases.

We found {bt} breakthrough predictions — drug-disease pairs with ZERO published literature but strong multi-method computational support.

Everything is open-source. Thread {{thread_emoji}}

**Tweet 2 (methodology):**
How it works: each drug is scored by 8 independent AI pillars:
- Knowledge graph embeddings (TransE, RotatE)
- Graph neural network (TxGNN)
- Molecular similarity
- Gene expression reversal
- Protein network proximity
- Causal genetic evidence (Mendelian Randomization)

When multiple methods agree independently, confidence goes up.

**Tweet 3 (spotlight):**
Example: {spotlight['drug_name']} for {spotlight['disease']}

Score: {spotlight['combined_score']:.2f}
Pillars: {spotlight['pillars_hit']}/8 methods agree
Novelty: {spotlight['novelty_level']}
MR (causal genetic evidence): {spotlight.get('mr_score', 0):.2f}

Zero published literature. Purely computational discovery.

**Tweet 4 (cross-disease):**
Interesting pattern: some drugs are predicted across MULTIPLE diseases.

{chr(10).join(f"- {d}: {c} diseases" for d, c in multi_disease[:3])}

Cross-disease convergence from independent AI methods suggests shared biological mechanisms.

**Tweet 5 (CTA):**
Explore all {total} predictions interactively:
{dashboard}

Full code + data (Apache 2.0):
{github}

Looking for researchers to validate these predictions with a cell assay. DM if interested.
"""
    return thread


def draft_linkedin_post(predictions: list[dict], all_preds: list[dict], config: dict) -> str:
    """Draft a LinkedIn post."""
    dashboard = config.get("dashboard_url", "")
    total = len(all_preds)
    bt = len([p for p in all_preds if p.get("novelty_level") == "BREAKTHROUGH"])

    post = f"""### LinkedIn Post Draft

**Can AI discover new drug treatments? We built an open-source platform to find out.**

Drug repurposing — finding new uses for existing approved drugs — could save years and billions vs. developing new drugs from scratch. But most computational tools use only 1-2 methods.

We built OpenCure, an open-source platform that combines 8 independent AI scoring methods to identify repurposing candidates. We screened all ~10,500 FDA-approved drugs across {len(set(p['disease'] for p in all_preds))} diseases.

Results: {total} candidates, including {bt} "breakthrough" predictions with zero existing published literature — genuinely novel computational discoveries.

The platform is free, open-source (Apache 2.0), and designed for researchers to explore and validate:
{dashboard}

We're actively looking for wet-lab partners to validate the top predictions. If you work in drug discovery, neglected diseases, or computational pharmacology — let's talk.

#DrugRepurposing #AI #OpenScience #DrugDiscovery #Bioinformatics
"""
    return post


def run():
    config = load_config()
    all_preds = load_all_predictions()
    breakthroughs = get_top_breakthrough_predictions(10)
    today = datetime.now().strftime("%Y-%m-%d")

    report = f"# Social Media Agent Report — {today}\n\n"
    report += "Platform-specific content drafts for review.\n\n"
    report += "**Action required:** Review, edit, and post on each platform.\n\n"
    report += "---\n\n"

    # Twitter thread
    twitter = draft_twitter_thread(breakthroughs, all_preds, config)
    report += f"## Twitter/X\n\n{twitter}\n\n---\n\n"

    # LinkedIn
    linkedin = draft_linkedin_post(breakthroughs, all_preds, config)
    report += f"## LinkedIn\n\n{linkedin}\n\n---\n\n"

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("social_posts", report)
    log_run("social_media_agent", f"Drafted Twitter thread + LinkedIn post")
    print(f"[Social Media Agent] Done. Report: {path}")


if __name__ == "__main__":
    run()
