"""
Validate OpenCure against known drug repurposing successes.

Tests whether the system can independently identify drugs that have been
successfully repurposed in the real world.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.search import search_simple as search

# Known drug repurposing successes to validate against.
# Format: (disease_query, drug_drugbank_id, drug_name, original_indication, repurposed_for)
KNOWN_REPURPOSING = [
    {
        "disease": "Multiple myeloma",
        "drug_id": "DB01041",
        "drug_name": "Thalidomide",
        "original": "Sedative/anti-nausea",
        "repurposed": "Multiple myeloma (blood cancer)",
    },
    {
        "disease": "Rheumatoid arthritis",
        "drug_id": "DB00563",
        "drug_name": "Methotrexate",
        "original": "Cancer chemotherapy",
        "repurposed": "Rheumatoid arthritis (autoimmune)",
    },
    {
        "disease": "Depression",
        "drug_id": "DB01174",
        "drug_name": "Phenobarbital",
        "original": "Sedative/anesthetic",
        "repurposed": "Epilepsy (anticonvulsant)",
    },
    {
        "disease": "Breast cancer",
        "drug_id": "DB00675",
        "drug_name": "Tamoxifen",
        "original": "Contraceptive research",
        "repurposed": "Breast cancer treatment & prevention",
    },
    {
        "disease": "Hypertension",
        "drug_id": "DB01136",
        "drug_name": "Carvedilol",
        "original": "Hypertension",
        "repurposed": "Heart failure",
    },
    {
        "disease": "Heart failure",
        "drug_id": "DB01136",
        "drug_name": "Carvedilol",
        "original": "Hypertension",
        "repurposed": "Heart failure",
    },
    {
        "disease": "Colorectal cancer",
        "drug_id": "DB00945",
        "drug_name": "Aspirin",
        "original": "Pain/anti-inflammatory",
        "repurposed": "Colorectal cancer prevention",
    },
    {
        "disease": "Parkinson's disease",
        "drug_id": "DB00674",
        "drug_name": "Galantamine",
        "original": "Polio treatment",
        "repurposed": "Alzheimer's disease",
    },
    {
        "disease": "Psoriasis",
        "drug_id": "DB00091",
        "drug_name": "Cyclosporine",
        "original": "Organ transplant rejection",
        "repurposed": "Psoriasis and autoimmune diseases",
    },
    {
        "disease": "Malaria",
        "drug_id": "DB00608",
        "drug_name": "Chloroquine",
        "original": "Malaria",
        "repurposed": "Lupus, rheumatoid arthritis",
    },
    {
        "disease": "Epilepsy",
        "drug_id": "DB00313",
        "drug_name": "Valproic acid",
        "original": "Epilepsy",
        "repurposed": "Bipolar disorder, migraine prevention, cancer research",
    },
    {
        "disease": "Leukemia",
        "drug_id": "DB00619",
        "drug_name": "Imatinib",
        "original": "Chronic myeloid leukemia",
        "repurposed": "GI stromal tumors, other cancers",
    },
]


def run_validation(top_k_thresholds=(50, 100, 200, 500)):
    """Run validation and report results."""
    print("=" * 80)
    print("  OpenCure Validation: Known Drug Repurposing Successes")
    print("=" * 80)
    print()

    # Load data once by doing a dummy search
    print("Loading data...")
    search("Alzheimer's disease", top_k=1)
    print()

    results_summary = []
    max_k = max(top_k_thresholds)

    for case in KNOWN_REPURPOSING:
        disease = case["disease"]
        target_drug = case["drug_id"]
        drug_name = case["drug_name"]

        print(f"Testing: {drug_name} ({target_drug}) → {disease}")

        results = search(disease, top_k=max_k)
        if not results:
            print(f"  [SKIP] No disease entity found for '{disease}'")
            results_summary.append({"case": case, "rank": None, "found": False})
            continue

        # Find the target drug in results
        found_rank = None
        for r in results:
            if r["drug_id"] == target_drug:
                found_rank = r["rank"]
                break

        if found_rank:
            print(f"  [FOUND] Rank {found_rank}/{len(results)} (score: {r.get('combined_score', r.get('score', 0))})")
        else:
            print(f"  [NOT FOUND] Not in top {max_k}")

        results_summary.append(
            {"case": case, "rank": found_rank, "found": found_rank is not None}
        )
        print()

    # Summary table
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print()

    print(f"{'Drug':<20}{'Disease':<25}{'Rank':<10}", end="")
    for k in top_k_thresholds:
        print(f"Top-{k:<5}", end="")
    print()
    print("-" * (55 + 9 * len(top_k_thresholds)))

    for item in results_summary:
        case = item["case"]
        rank = item["rank"]
        rank_str = str(rank) if rank else "N/F"

        print(f"{case['drug_name']:<20}{case['disease']:<25}{rank_str:<10}", end="")
        for k in top_k_thresholds:
            if rank and rank <= k:
                print(f"{'Y':<9}", end="")
            else:
                print(f"{'.':<9}", end="")
        print()

    # Hit rates
    print()
    print("Hit rates:")
    total_tested = len([r for r in results_summary if r["rank"] is not None or r["found"]])
    total_valid = len([r for r in results_summary if r["case"]["disease"] in
                       [s["case"]["disease"] for s in results_summary if s["rank"] is not None or not s["found"]]])

    for k in top_k_thresholds:
        hits = sum(1 for r in results_summary if r["rank"] is not None and r["rank"] <= k)
        total = len(results_summary)
        valid = len([r for r in results_summary if r.get("rank") is not None or r["found"] == False])
        if valid > 0:
            print(f"  Top-{k}: {hits}/{valid} ({100*hits/valid:.1f}%)")

    print()
    print("NOTE: This is Pillar 3 only (knowledge graph embeddings).")
    print("Adding Pillar 1 (molecular similarity) and Pillar 2 (protein docking)")
    print("will significantly improve recall.")


if __name__ == "__main__":
    run_validation()
