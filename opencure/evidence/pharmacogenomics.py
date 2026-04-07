"""
Pharmacogenomics evidence for drug repurposing candidates.

Queries PharmGKB for genetic variants that affect drug response,
helping identify which patient populations would benefit most.

This adds precision medicine context to repurposing predictions:
- CYP2D6 poor metabolizers may have reduced efficacy
- HLA variants may increase adverse event risk
- Specific genotypes may show enhanced response

Data source: PharmGKB REST API (free, no auth needed).
"""
from __future__ import annotations

import time
import requests
from typing import Optional

_pgkb_cache: dict[str, list] = {}

PHARMGKB_API = "https://api.pharmgkb.org/v1/data"


def get_pharmgkb_annotations(drug_name: str) -> list[dict]:
    """
    Get clinical pharmacogenomics annotations for a drug from PharmGKB.

    Returns list of variant-drug annotations with clinical significance.
    """
    cache_key = drug_name.lower()
    if cache_key in _pgkb_cache:
        return _pgkb_cache[cache_key]

    annotations = []

    try:
        # Search for drug in PharmGKB
        resp = requests.get(
            f"{PHARMGKB_API}/clinicalAnnotation",
            params={"relatedChemicals.name": drug_name, "view": "min"},
            timeout=15,
        )
        time.sleep(0.3)

        if resp.status_code == 200:
            data = resp.json()
            results = data.get("data", [])

            for ann in results[:10]:  # Limit to top 10
                annotation = {
                    "id": ann.get("id", ""),
                    "level": ann.get("level", ""),
                    "gene": "",
                    "variant": "",
                    "phenotype": "",
                    "url": f"https://www.pharmgkb.org/clinicalAnnotation/{ann.get('id', '')}",
                }

                # Extract gene/variant info
                genes = ann.get("relatedGenes", [])
                if genes:
                    annotation["gene"] = genes[0].get("symbol", "")

                variants = ann.get("relatedVariants", [])
                if variants:
                    annotation["variant"] = variants[0].get("name", "")

                phenotypes = ann.get("phenotypes", [])
                if phenotypes:
                    annotation["phenotype"] = phenotypes[0].get("name", "")

                annotations.append(annotation)

    except Exception:
        pass

    _pgkb_cache[cache_key] = annotations
    return annotations


def get_drug_interactions(drug_name: str) -> list[dict]:
    """
    Get known drug-drug interactions from PharmGKB.

    Important for combination therapy safety assessment.
    """
    interactions = []

    try:
        resp = requests.get(
            f"{PHARMGKB_API}/drugLabel",
            params={"relatedChemicals.name": drug_name, "view": "min"},
            timeout=15,
        )
        time.sleep(0.3)

        if resp.status_code == 200:
            data = resp.json()
            for label in data.get("data", [])[:5]:
                interactions.append({
                    "source": label.get("source", ""),
                    "name": label.get("name", ""),
                    "url": f"https://www.pharmgkb.org/drugLabel/{label.get('id', '')}",
                })

    except Exception:
        pass

    return interactions


def summarize_pharmacogenomics(drug_name: str) -> dict:
    """
    Generate a pharmacogenomics summary for a drug.

    Returns dict with annotations, key genes, and clinical implications.
    """
    annotations = get_pharmgkb_annotations(drug_name)

    if not annotations:
        return {
            "has_pgx_data": False,
            "annotations": [],
            "key_genes": [],
            "summary": f"No pharmacogenomics data found for {drug_name} in PharmGKB.",
        }

    # Extract key genes
    key_genes = list(set(a["gene"] for a in annotations if a["gene"]))

    # Count by evidence level
    levels = {}
    for a in annotations:
        lvl = a.get("level", "unknown")
        levels[lvl] = levels.get(lvl, 0) + 1

    summary_parts = [f"{drug_name} has {len(annotations)} pharmacogenomics annotation(s)"]
    if key_genes:
        summary_parts.append(f"affecting genes: {', '.join(key_genes[:5])}")
    if levels:
        summary_parts.append(f"evidence levels: {levels}")

    return {
        "has_pgx_data": True,
        "annotations": annotations[:5],
        "key_genes": key_genes,
        "evidence_levels": levels,
        "summary": ". ".join(summary_parts),
    }
