"""CLI entry point for OpenCure drug repurposing search."""

import argparse
import sys

from opencure.search import search


def main():
    parser = argparse.ArgumentParser(
        description="OpenCure - AI-powered drug repurposing search",
        epilog=(
            "Examples:\n"
            '  python3 -m opencure "Alzheimer\'s disease"\n'
            '  python3 -m opencure "COVID-19" --top 50\n'
            '  python3 -m opencure "MESH:D000544" --top 10\n'
            '  python3 -m opencure "Malaria" --evidence\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("disease", help="Disease name or MESH/DOID ID to search for")
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top candidates to show (default: 20)"
    )
    parser.add_argument(
        "--evidence", action="store_true", help="Show evidence details for each candidate"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: TransE only (skip molecular similarity)"
    )
    args = parser.parse_args()

    print(f"\n{'='*74}")
    print(f"  OpenCure - Drug Repurposing Search")
    print(f"  Query: {args.disease}")
    print(f"{'='*74}\n")

    results = search(
        args.disease,
        top_k=args.top,
        use_molecular_similarity=not args.fast,
        use_evidence=args.evidence,
    )

    if not results:
        print("No results found.")
        sys.exit(1)

    # Check if any results have molecular similarity data
    has_mol_sim = any("mol_similarity" in r for r in results)

    # Print results table
    print(f"\n{'='*74}")
    print(f"  Top {len(results)} Drug Repurposing Candidates")
    print(f"{'='*74}\n")

    if has_mol_sim:
        header = f"{'Rank':<5}{'Drug':<25}{'ID':<12}{'Score':<9}{'KG Rank':<9}{'MolSim':<8}{'Pillars':<8}{'Relation'}"
        print(header)
        print("-" * len(header))

        for r in results:
            name = r["drug_name"][:23]
            mol_sim = f"{r['mol_similarity']:.2f}" if "mol_similarity" in r else "-"
            kg_rank = str(r.get("transe_rank", "-"))
            print(
                f"{r['rank']:<5}"
                f"{name:<25}"
                f"{r['drug_id']:<12}"
                f"{r['combined_score']:<9.4f}"
                f"{kg_rank:<9}"
                f"{mol_sim:<8}"
                f"{r['pillars_hit']:<8}"
                f"{r['relation_type']}"
            )
    else:
        header = f"{'Rank':<5}{'Drug':<28}{'DrugBank ID':<14}{'Score':<10}{'Relation'}"
        print(header)
        print("-" * len(header))

        for r in results:
            name = r["drug_name"][:26]
            print(
                f"{r['rank']:<5}"
                f"{name:<28}"
                f"{r['drug_id']:<14}"
                f"{r['combined_score']:<10.4f}"
                f"{r['relation_type']}"
            )

    # Print evidence if requested
    if args.evidence:
        print(f"\n{'='*74}")
        print(f"  Evidence Details")
        print(f"{'='*74}")
        for r in results:
            evidence = r.get("evidence", [])
            if evidence:
                print(f"\n  {r['rank']}. {r['drug_name']} ({r['drug_id']})")
                for e in evidence:
                    print(f"     - {e}")
                if "mol_similarity" in r:
                    print(f"     - Molecular similarity: {r['mol_similarity']:.3f} to {r.get('similar_to', 'unknown')}")

    # Footer
    print(f"\n{'='*74}")
    print(f"  Disease: {results[0]['disease_entity']}")
    print(f"  Method: Multi-pillar scoring (TransE knowledge graph" +
          (f" + molecular similarity" if has_mol_sim else "") + ")")
    print(f"  Higher score = stronger predicted treatment relationship")
    print(f"{'='*74}\n")
    print("NOTE: These are computational predictions, not medical advice.")
    print("All candidates require experimental and clinical validation.\n")


if __name__ == "__main__":
    main()
