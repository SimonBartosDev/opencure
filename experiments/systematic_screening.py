"""
OpenCure Systematic Screening: Neglected & Rare Diseases

Runs the full multi-pillar pipeline + evidence gathering across all target
diseases and produces a publishable database of drug repurposing predictions.

Usage:
    python3 experiments/systematic_screening.py              # Full run (25 diseases)
    python3 experiments/systematic_screening.py --test 3     # Test with 3 diseases
    python3 experiments/systematic_screening.py --resume     # Skip already-done diseases

Output: experiments/results/{disease_name}.json per disease
        experiments/results/screening_summary.json
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.search import search, search_simple
from opencure.evidence.report import generate_evidence_report
from opencure.evidence.novelty import compute_novelty_score

RESULTS_DIR = Path("experiments/results")

# Target diseases organized by category
TARGET_DISEASES = {
    "Neglected Tropical": [
        "Malaria",
        "Tuberculosis",
        "Dengue",
        "Chagas disease",
        "Leishmaniasis",
        "Schistosomiasis",
        "HIV",
        "Hepatitis C",
    ],
    "Rare Diseases": [
        "Sickle cell disease",
        "Fragile X syndrome",
        "Duchenne muscular dystrophy",
        "Neurofibromatosis",
        "Marfan syndrome",
        "Ehlers-Danlos syndrome",
        "Gaucher disease",
        "Fabry disease",
    ],
    "Neurodegenerative": [
        "Alzheimer's disease",
        "Parkinson's disease",
        "Huntington's disease",
        "Amyotrophic lateral sclerosis",
        "Multiple sclerosis",
    ],
    "Other Underserved": [
        "Idiopathic pulmonary fibrosis",
        "Cystic fibrosis",
        "Pulmonary hypertension",
        "Sepsis",
    ],
    "Cancer": [
        "Breast cancer",
        "Lung cancer",
        "Colorectal cancer",
        "Pancreatic cancer",
        "Prostate cancer",
        "Ovarian cancer",
        "Melanoma",
        "Glioblastoma",
        "Leukemia",
        "Lymphoma",
        "Multiple myeloma",
    ],
    "Cardiovascular & Metabolic": [
        "Heart failure",
        "Coronary artery disease",
        "Atrial fibrillation",
        "Hypertension",
        "Atherosclerosis",
        "Type 2 diabetes",
        "Obesity",
        "Chronic kidney disease",
        "Liver cirrhosis",
    ],
    "Autoimmune & Inflammatory": [
        "Rheumatoid arthritis",
        "Crohn's disease",
        "Ulcerative colitis",
        "Psoriasis",
        "Lupus",
        "Inflammatory bowel disease",
    ],
    "Respiratory & Other": [
        "Asthma",
        "COPD",
        "COVID-19",
        "Osteoporosis",
        "Endometriosis",
    ],
    "Neuropsychiatric": [
        "Depression",
        "Schizophrenia",
        "Bipolar disorder",
        "Epilepsy",
        "Anxiety",
    ],
}


def screen_disease(disease_name: str, top_k: int = 50, evidence_top_k: int = 10) -> dict:
    """
    Run full screening for a single disease.

    1. Search for top drug candidates (fast, TransE-only)
    2. Generate evidence reports for top candidates (slow, API calls)

    Returns dict with disease info, candidates, and summary.
    """
    start = time.time()

    # Step 1: Full multi-pillar search (all 7 pillars)
    results = search(disease_name, top_k=top_k, use_molecular_similarity=True, use_evidence=False)

    if not results:
        return {
            "disease": disease_name,
            "status": "no_results",
            "error": "No disease entity found in DRKG",
            "candidates": [],
            "timestamp": datetime.now().isoformat(),
        }

    # Step 2: Evidence reports for top candidates
    candidates = []
    for i, result in enumerate(results[:evidence_top_k]):
        drug_name = result["drug_name"]
        drug_id = result["drug_id"]

        # Skip drugs without resolved names (can't query APIs)
        if drug_name == drug_id:
            candidates.append({
                "rank": result["rank"],
                "drug_name": drug_name,
                "drug_id": drug_id,
                "combined_score": result.get("combined_score", 0),
                "confidence": "UNRESOLVED",
                "evidence_skipped": True,
            })
            continue

        print(f"    [{i+1}/{evidence_top_k}] Evidence for {drug_name} ({drug_id})...")

        try:
            report = generate_evidence_report(result, disease_name)
            candidates.append(report.to_dict())
        except Exception as e:
            print(f"    [WARN] Evidence report failed for {drug_name}: {e}")
            candidates.append({
                "rank": result["rank"],
                "drug_name": drug_name,
                "drug_id": drug_id,
                "combined_score": result.get("combined_score", 0),
                "confidence": "ERROR",
                "error": str(e),
            })

        # Rate limit between candidates
        time.sleep(1)

    elapsed = time.time() - start

    # Summary
    confidence_counts = {}
    for c in candidates:
        conf = c.get("confidence", "UNKNOWN")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    return {
        "disease": disease_name,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "total_searched": len(results),
        "evidence_generated": len([c for c in candidates if not c.get("evidence_skipped")]),
        "confidence_counts": confidence_counts,
        "candidates": candidates,
    }


def run_screening(test_count: int = 0, resume: bool = True):
    """Run systematic screening across all target diseases."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Flatten disease list
    all_diseases = []
    for category, diseases in TARGET_DISEASES.items():
        for d in diseases:
            all_diseases.append((category, d))

    if test_count > 0:
        all_diseases = all_diseases[:test_count]

    total = len(all_diseases)
    print("=" * 70)
    print("  OpenCure Systematic Screening")
    print(f"  Diseases: {total}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resume: {resume}")
    print("=" * 70)

    # Warm up data
    print("\nLoading data...")
    search("Alzheimer's disease", top_k=1, use_molecular_similarity=True, use_evidence=False)
    print()

    summary = {
        "start_time": datetime.now().isoformat(),
        "total_diseases": total,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "diseases": {},
    }

    for i, (category, disease) in enumerate(all_diseases):
        safe_name = disease.replace(" ", "_").replace("'", "")
        result_file = RESULTS_DIR / f"{safe_name}.json"

        # Resume: skip if already done
        if resume and result_file.exists():
            print(f"[{i+1}/{total}] {disease} — SKIPPED (already done)")
            summary["skipped"] += 1

            # Load existing result for summary
            with open(result_file) as f:
                existing = json.load(f)
            summary["diseases"][disease] = {
                "category": category,
                "status": existing.get("status", "unknown"),
                "confidence_counts": existing.get("confidence_counts", {}),
            }
            continue

        print(f"\n[{i+1}/{total}] Screening: {disease} ({category})")
        print("-" * 50)

        result = screen_disease(disease)

        # Save per-disease result (convert numpy types for JSON)
        def json_safe(obj):
            import numpy as np
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=json_safe)

        # Update summary
        status = result.get("status", "unknown")
        if status == "completed":
            summary["completed"] += 1
            conf = result.get("confidence_counts", {})
            print(f"  Done in {result['elapsed_seconds']}s — {conf}")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result.get('error', 'unknown')}")

        summary["diseases"][disease] = {
            "category": category,
            "status": status,
            "confidence_counts": result.get("confidence_counts", {}),
            "elapsed": result.get("elapsed_seconds", 0),
        }

        # Rate limit between diseases
        time.sleep(2)

    # Save summary
    summary["end_time"] = datetime.now().isoformat()
    summary_file = RESULTS_DIR / "screening_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("  SCREENING COMPLETE")
    print("=" * 70)
    print(f"  Diseases screened: {summary['completed']}")
    print(f"  Skipped (resume): {summary['skipped']}")
    print(f"  Failed: {summary['failed']}")
    print()

    # Aggregate confidence counts
    total_high = 0
    total_medium = 0
    total_novel = 0
    total_low = 0
    for d_info in summary["diseases"].values():
        cc = d_info.get("confidence_counts", {})
        total_high += cc.get("HIGH", 0)
        total_medium += cc.get("MEDIUM", 0)
        total_novel += cc.get("NOVEL", 0)
        total_low += cc.get("LOW", 0)

    print(f"  Across all diseases:")
    print(f"    HIGH confidence candidates: {total_high}")
    print(f"    MEDIUM confidence: {total_medium}")
    print(f"    NOVEL (potential new discoveries): {total_novel}")
    print(f"    LOW confidence: {total_low}")
    print()
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"  Summary: {summary_file}")
    print(f"\n  Next: python3 experiments/generate_database.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCure Systematic Disease Screening")
    parser.add_argument("--test", type=int, default=0, help="Test with N diseases only")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already-done diseases")
    args = parser.parse_args()

    run_screening(test_count=args.test, resume=not args.no_resume)
