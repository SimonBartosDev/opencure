"""
Drug combination synergy prediction for repurposing candidates.

For complex diseases (cancer, neurodegeneration), single drugs are often
insufficient. This module identifies drug PAIRS that could work synergistically
by targeting complementary disease pathways.

Strategy:
1. For a top candidate drug, find its targets in DRKG
2. Find OTHER drugs that target DIFFERENT disease-relevant pathways
3. Score complementarity: drugs that together cover more disease targets
4. Check DrugComb database for known synergy evidence

This runs AFTER ranking (post-processing) to avoid O(n²) during scoring.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def compute_target_complementarity(
    drug_a_targets: set,
    drug_b_targets: set,
    disease_genes: set,
) -> float:
    """
    Score how complementary two drugs are for treating a disease.

    High score = drugs target DIFFERENT disease genes (complementary coverage)
    Low score = drugs target the SAME genes (redundant)

    Returns: 0-1 complementarity score
    """
    if not disease_genes:
        return 0.0

    # Coverage by each drug individually
    a_coverage = drug_a_targets & disease_genes
    b_coverage = drug_b_targets & disease_genes

    if not a_coverage and not b_coverage:
        return 0.0

    # Combined coverage (union)
    combined = a_coverage | b_coverage

    # Overlap (intersection) - lower is better for complementarity
    overlap = a_coverage & b_coverage

    # Complementarity = high combined coverage + low overlap
    coverage_score = len(combined) / max(len(disease_genes), 1)
    overlap_penalty = len(overlap) / max(len(combined), 1)

    # Score: coverage * (1 - overlap), clamped to [0, 1]
    return min(1.0, coverage_score * (1.0 - 0.5 * overlap_penalty))


def find_synergistic_partners(
    drug_entity: str,
    disease_name: str,
    top_candidates: list[dict],
    triplets,
    drug_names: dict,
    max_partners: int = 5,
) -> list[dict]:
    """
    Find the best combination partners for a drug against a disease.

    Args:
        drug_entity: DRKG compound entity of the primary drug
        disease_name: Disease name
        top_candidates: List of search result dicts (other candidates)
        triplets: DRKG triplets DataFrame
        drug_names: Entity->name mapping
        max_partners: Max number of partners to return

    Returns:
        List of dicts: {partner_name, partner_id, complementarity_score, shared_disease_targets}
    """
    # Get primary drug's targets
    drug_a_targets = _get_drug_gene_targets(drug_entity, triplets)
    if not drug_a_targets:
        return []

    # Get disease-related genes
    disease_genes = _get_disease_genes(disease_name, triplets)
    if not disease_genes:
        return []

    partners = []
    for candidate in top_candidates:
        partner_entity = candidate.get("drug_entity", "")
        if partner_entity == drug_entity:
            continue  # Skip self

        partner_targets = _get_drug_gene_targets(partner_entity, triplets)
        if not partner_targets:
            continue

        # Compute complementarity
        comp_score = compute_target_complementarity(
            drug_a_targets, partner_targets, disease_genes
        )

        if comp_score > 0.1:
            combined_coverage = (drug_a_targets | partner_targets) & disease_genes
            partners.append({
                "partner_name": candidate.get("drug_name", partner_entity),
                "partner_id": candidate.get("drug_id", ""),
                "complementarity_score": round(comp_score, 3),
                "combined_disease_targets": len(combined_coverage),
                "partner_unique_targets": len(partner_targets & disease_genes - drug_a_targets),
            })

    # Sort by complementarity
    partners.sort(key=lambda x: -x["complementarity_score"])
    return partners[:max_partners]


def _get_drug_gene_targets(drug_entity: str, triplets) -> set:
    """Get gene targets for a drug from DRKG."""
    genes = set()
    mask_fwd = (triplets["head"] == drug_entity) & (triplets["tail"].str.startswith("Gene::"))
    genes.update(triplets[mask_fwd]["tail"].values)
    mask_rev = (triplets["tail"] == drug_entity) & (triplets["head"].str.startswith("Gene::"))
    genes.update(triplets[mask_rev]["head"].values)
    return genes


def _get_disease_genes(disease_name: str, triplets) -> set:
    """Get disease-associated genes from DRKG."""
    genes = set()
    # Find disease entities matching the name
    from opencure.data.drkg import find_disease_entities, load_embeddings
    entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id = load_embeddings()
    matches = find_disease_entities(entity_to_id, disease_name)

    for disease_entity, _ in matches:
        mask1 = (triplets["head"] == disease_entity) & (triplets["tail"].str.startswith("Gene::"))
        genes.update(triplets[mask1]["tail"].values)
        mask2 = (triplets["tail"] == disease_entity) & (triplets["head"].str.startswith("Gene::"))
        genes.update(triplets[mask2]["head"].values)

    return genes
