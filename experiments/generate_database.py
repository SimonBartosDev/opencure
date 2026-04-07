"""
Compile screening results into a publishable database.

Reads per-disease JSON files from experiments/results/ and generates:
1. opencure_database.json - Full database with all evidence
2. opencure_database.csv - Flat CSV for researchers
3. novel_candidates.json - Only NOVEL confidence candidates (potential new discoveries)

Usage:
    python3 experiments/generate_database.py
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.evidence.novelty import compute_novelty_score

RESULTS_DIR = Path("experiments/results")


def load_all_results() -> list[dict]:
    """Load all per-disease result files."""
    all_results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name in ("screening_summary.json", "opencure_database.json",
                       "novel_candidates.json", "chagas_report.json"):
            continue
        with open(f) as fh:
            data = json.load(fh)
            if data.get("status") == "completed":
                all_results.append(data)
    return all_results


def generate_database():
    """Generate the publishable database files."""
    results = load_all_results()

    if not results:
        print("No results found. Run systematic_screening.py first.")
        return

    print(f"Loaded results for {len(results)} diseases")

    # Compile all candidates into flat list
    all_candidates = []
    novel_candidates = []

    for disease_result in results:
        disease = disease_result["disease"]
        for candidate in disease_result.get("candidates", []):
            if candidate.get("evidence_skipped") or candidate.get("confidence") in ("UNRESOLVED", "ERROR"):
                continue

            # Compute novelty score for this candidate
            novelty = compute_novelty_score({
                "pubmed_articles": candidate.get("pubmed_total", 0),
                "clinical_trials": candidate.get("clinical_trials_total", 0),
                "pubmed_repurposing": candidate.get("pubmed_repurposing_total", 0),
                "max_citations": candidate.get("max_citations", 0),
                "faers_cooccurrences": candidate.get("faers_cooccurrences", 0),
                "combined_score": candidate.get("combined_score", 0),
                "pillars_hit": candidate.get("pillars_hit", 0),
                "shared_targets": candidate.get("shared_target_count", 0),
                "direct_relations": len(candidate.get("direct_relations", [])),
            })

            entry = {
                "disease": disease,
                "drug_name": candidate.get("drug_name", ""),
                "drug_id": candidate.get("drug_id", ""),
                "confidence": candidate.get("confidence", ""),
                "novelty_level": novelty["novelty_level"],
                "novelty_score": novelty["novelty_score"],
                "is_known_treatment": novelty["novelty_level"] == "ESTABLISHED",
                "combined_score": candidate.get("combined_score", 0),
                "pillars_hit": candidate.get("pillars_hit", 0),
                "pubmed_articles": candidate.get("pubmed_total", 0),
                "pubmed_treatment": candidate.get("pubmed_treatment_total", 0),
                "pubmed_repurposing": candidate.get("pubmed_repurposing_total", 0),
                "clinical_trials": candidate.get("clinical_trials_total", 0),
                "faers_signal": candidate.get("faers_signal", ""),
                "faers_cooccurrences": candidate.get("faers_cooccurrences", 0),
                "faers_total_reports": candidate.get("faers_total_reports", 0),
                "signature_reversal": candidate.get("signature_reversal_found", False),
                "signature_rank": candidate.get("signature_reversal_rank", 0),
                "shared_targets": candidate.get("shared_target_count", 0),
                "direct_relations": len(candidate.get("direct_relations", [])),
                "max_citations": candidate.get("max_citations", 0),
                "mol_similarity": candidate.get("mol_similarity", 0),
                "similar_to": candidate.get("similar_to", ""),
                "transe_rank": candidate.get("transe_rank", 0),
                "ai_hypothesis": candidate.get("ai_hypothesis", ""),
                "novelty_interpretation": novelty["interpretation"],
                "confidence_reasons": "; ".join(candidate.get("confidence_reasons", [])),
            }

            all_candidates.append(entry)

            # Novel = BREAKTHROUGH or NOVEL novelty level
            if novelty["novelty_level"] in ("BREAKTHROUGH", "NOVEL"):
                novel_candidates.append(entry)

    print(f"Total drug-disease candidates: {len(all_candidates)}")

    # 1. Full JSON database
    db_json = RESULTS_DIR / "opencure_database.json"
    database = {
        "name": "OpenCure Drug Repurposing Database",
        "version": "0.1.0",
        "generated": datetime.now().isoformat(),
        "description": "AI-predicted drug repurposing candidates with multi-source evidence",
        "diseases_screened": len(results),
        "total_candidates": len(all_candidates),
        "novel_candidates": len(novel_candidates),
        "evidence_types": [
            "Knowledge graph (DRKG TransE embeddings)",
            "Molecular similarity (RDKit fingerprints + ChemBERTa)",
            "Published literature (PubMed + Semantic Scholar)",
            "Clinical trials (ClinicalTrials.gov)",
            "Real-world evidence (FDA FAERS adverse events)",
            "Gene signature reversal (L1000CDS2)",
        ],
        "candidates": all_candidates,
    }
    with open(db_json, "w") as f:
        json.dump(database, f, indent=2)
    print(f"Saved: {db_json}")

    # 2. CSV export
    db_csv = RESULTS_DIR / "opencure_database.csv"
    if all_candidates:
        fieldnames = list(all_candidates[0].keys())
        with open(db_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)
        print(f"Saved: {db_csv}")

    # 3. Novel candidates
    novel_json = RESULTS_DIR / "novel_candidates.json"
    novel_db = {
        "name": "OpenCure Novel Drug Repurposing Candidates",
        "description": "Candidates with strong computational support but NO existing published evidence. These are potential new discoveries.",
        "generated": datetime.now().isoformat(),
        "count": len(novel_candidates),
        "candidates": novel_candidates,
    }
    with open(novel_json, "w") as f:
        json.dump(novel_db, f, indent=2)
    print(f"Saved: {novel_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("  OpenCure Database Summary")
    print("=" * 60)

    # Confidence breakdown
    conf_counts = {}
    for c in all_candidates:
        conf = c["confidence"]
        conf_counts[conf] = conf_counts.get(conf, 0) + 1

    print(f"\n  Total candidates: {len(all_candidates)}")
    for conf in ["HIGH", "MEDIUM", "NOVEL", "LOW"]:
        count = conf_counts.get(conf, 0)
        if count:
            print(f"    {conf}: {count}")

    # Top diseases by HIGH confidence candidates
    disease_high = {}
    for c in all_candidates:
        if c["confidence"] == "HIGH":
            disease_high[c["disease"]] = disease_high.get(c["disease"], 0) + 1

    if disease_high:
        print(f"\n  Diseases with most HIGH confidence candidates:")
        for d, count in sorted(disease_high.items(), key=lambda x: -x[1])[:10]:
            print(f"    {d}: {count}")

    # Novelty breakdown
    novelty_counts = Counter(c.get("novelty_level", "UNKNOWN") for c in all_candidates)
    print(f"\n  Novelty breakdown:")
    for level in ["BREAKTHROUGH", "NOVEL", "EMERGING", "KNOWN", "ESTABLISHED"]:
        count = novelty_counts.get(level, 0)
        if count:
            print(f"    {level}: {count}")

    known = [c for c in all_candidates if c.get("is_known_treatment")]
    print(f"\n  Already-known treatments (filtered): {len(known)}")
    repurposing = [c for c in all_candidates if not c.get("is_known_treatment")]
    print(f"  Actual repurposing candidates: {len(repurposing)}")

    # Novel candidates detail
    if novel_candidates:
        print(f"\n  NOVEL/BREAKTHROUGH CANDIDATES (potential new discoveries):")
        for c in novel_candidates:
            print(f"    {c['drug_name']} → {c['disease']} (novelty: {c['novelty_score']:.3f}, score: {c['combined_score']:.3f})")
            print(f"      {c.get('novelty_interpretation', '')}")

    print(f"\n  Files:")
    print(f"    {db_json}")
    print(f"    {db_csv}")
    print(f"    {novel_json}")


if __name__ == "__main__":
    generate_database()
