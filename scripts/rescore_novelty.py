#!/usr/bin/env python3
"""Re-score novelty for all existing results using synonym-expanded PubMed search.

Reads each disease JSON, re-queries PubMed with synonyms for each candidate,
updates the novelty scores, and overwrites the JSON files. Then regenerates
the database, reports, and dashboard.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.evidence.pubmed import search_drug_disease_evidence
from opencure.evidence.novelty import compute_novelty_score

RESULTS_DIR = Path("experiments/results")


def rescore_disease(filepath: Path) -> dict:
    """Re-score novelty for all candidates in a disease file."""
    data = json.loads(filepath.read_text())
    disease = data.get("disease", filepath.stem)
    candidates = data.get("candidates", [])

    changes = {"upgraded": 0, "downgraded": 0, "unchanged": 0}

    for i, c in enumerate(candidates):
        drug = c["drug_name"]
        old_level = c.get("novelty_level", "UNKNOWN")
        old_pubmed = c.get("pubmed_total", c.get("pubmed_articles", 0))

        # Re-query PubMed with synonym expansion
        try:
            evidence = search_drug_disease_evidence(drug, disease, max_results=5)
            new_pubmed = evidence.get("total_articles", 0)
            new_treatment = evidence.get("total_treatment_articles", 0)
            new_repurposing = evidence.get("total_repurposing_articles", 0)
        except Exception as e:
            print(f"    Error querying {drug}: {e}")
            continue

        # Update pubmed counts
        c["pubmed_total"] = new_pubmed
        c["pubmed_treatment_total"] = new_treatment
        c["pubmed_repurposing_total"] = new_repurposing

        # Update key papers if we found new ones
        if new_pubmed > 0 and not c.get("key_papers"):
            c["key_papers"] = evidence.get("articles", [])
        if new_repurposing > 0 and not c.get("repurposing_papers"):
            c["repurposing_papers"] = evidence.get("repurposing_articles", [])

        # Re-compute novelty
        novelty_input = {
            "pubmed_total": new_pubmed,
            "clinical_trials_total": c.get("clinical_trials_total", 0),
            "pubmed_repurposing_total": new_repurposing,
            "max_citations": c.get("max_citations", 0),
            "faers_cooccurrences": c.get("faers_cooccurrences", 0),
            "combined_score": c.get("combined_score", 0),
            "pillars_hit": c.get("pillars_hit", 0),
            "shared_target_count": c.get("shared_target_count", 0),
            "direct_relations": c.get("direct_relations", 0),
            "mr_score": c.get("mr_score", 0),
        }

        result = compute_novelty_score(novelty_input)
        new_level = result["novelty_level"]

        c["novelty_score"] = result["novelty_score"]
        c["novelty_level"] = new_level
        c["novelty_interpretation"] = result["interpretation"]

        if new_level != old_level:
            direction = "downgraded" if _level_rank(new_level) > _level_rank(old_level) else "upgraded"
            changes[direction] += 1
            print(f"    {drug}: {old_level} -> {new_level} (pubmed: {old_pubmed} -> {new_pubmed})")
        else:
            changes["unchanged"] += 1

        time.sleep(0.5)  # Rate limiting

    # Write back
    filepath.write_text(json.dumps(data, indent=2))
    return changes


def _level_rank(level: str) -> int:
    return {"BREAKTHROUGH": 0, "NOVEL": 1, "EMERGING": 2, "KNOWN": 3, "ESTABLISHED": 4}.get(level, 5)


def main():
    disease_files = sorted([
        f for f in RESULTS_DIR.glob("*.json")
        if not f.name.startswith("opencure_")
        and f.name not in ("screening_summary.json", "novel_candidates.json")
    ])

    print(f"Re-scoring novelty for {len(disease_files)} diseases with synonym-expanded PubMed search...\n")

    total_changes = {"upgraded": 0, "downgraded": 0, "unchanged": 0}

    for i, f in enumerate(disease_files):
        disease_name = f.stem.replace("_", " ")
        print(f"[{i+1}/{len(disease_files)}] {disease_name}...")
        changes = rescore_disease(f)
        for k in total_changes:
            total_changes[k] += changes[k]

    print(f"\n{'='*50}")
    print(f"DONE. Results:")
    print(f"  Downgraded (was falsely novel): {total_changes['downgraded']}")
    print(f"  Upgraded: {total_changes['upgraded']}")
    print(f"  Unchanged: {total_changes['unchanged']}")

    # Regenerate database
    print("\nRegenerating database...")
    os.system("python3 experiments/generate_database.py")

    # Regenerate dashboard
    print("Regenerating Explorer dashboard...")
    os.system("python3 scripts/build_explorer.py")

    print("\nAll done. Review changes and push to GitHub.")


if __name__ == "__main__":
    main()
