"""TransE scoring for drug-disease relationship prediction using DRKG embeddings."""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from opencure.config import TREATMENT_RELATIONS


def score_triplet(
    head_emb: np.ndarray,
    rel_emb: np.ndarray,
    tail_emb: np.ndarray,
) -> float:
    """
    Compute TransE score for a (head, relation, tail) triplet.

    TransE scoring: score = -||h + r - t||_2
    Higher score (less negative) = stronger predicted relationship.
    """
    return -float(np.linalg.norm(head_emb + rel_emb - tail_emb))


def score_drugs_for_disease(
    disease_entity: str,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    entity_to_id: dict,
    relation_to_id: dict,
    compound_entities: list[str],
    treatment_relations: list[str] | None = None,
    show_progress: bool = False,
) -> list[tuple[str, float, str]]:
    """
    Score all compounds against a disease using TransE embeddings.

    For each compound, computes the TransE score using all treatment-type
    relations, and returns the best (highest) score per compound.

    Args:
        disease_entity: DRKG disease entity ID (e.g., "Disease::MESH:D000544")
        entity_emb: Entity embedding matrix
        relation_emb: Relation embedding matrix
        entity_to_id: Entity name to index mapping
        relation_to_id: Relation name to index mapping
        compound_entities: List of compound entity IDs to score
        treatment_relations: List of relation types to use (defaults to TREATMENT_RELATIONS)
        show_progress: Show progress bar

    Returns:
        Sorted list of (compound_entity, best_score, best_relation) tuples,
        highest score first.
    """
    if treatment_relations is None:
        treatment_relations = TREATMENT_RELATIONS

    # Get disease embedding index
    if disease_entity not in entity_to_id:
        raise ValueError(f"Disease entity '{disease_entity}' not found in DRKG embeddings")
    disease_idx = entity_to_id[disease_entity]
    disease_emb = entity_emb[disease_idx]

    # Get relation embedding indices (filter to those that exist in DRKG)
    rel_indices = {}
    for rel in treatment_relations:
        if rel in relation_to_id:
            rel_indices[rel] = relation_to_id[rel]

    if not rel_indices:
        raise ValueError("No treatment relations found in DRKG relation embeddings")

    # Pre-fetch relation embeddings
    rel_embs = {rel: relation_emb[idx] for rel, idx in rel_indices.items()}

    # Score all compounds
    results = []
    compounds_iter = tqdm(compound_entities, desc="Scoring drugs", disable=not show_progress)

    for compound in compounds_iter:
        if compound not in entity_to_id:
            continue

        compound_idx = entity_to_id[compound]
        compound_emb = entity_emb[compound_idx]

        # Score with each treatment relation, keep the best
        best_score = float("-inf")
        best_rel = ""

        for rel, r_emb in rel_embs.items():
            # TransE: compound + relation ≈ disease
            # score = -||compound_emb + rel_emb - disease_emb||
            score = score_triplet(compound_emb, r_emb, disease_emb)
            if score > best_score:
                best_score = score
                best_rel = rel

        results.append((compound, best_score, best_rel))

    # Sort by score descending (highest = most likely to treat)
    results.sort(key=lambda x: -x[1])
    return results


def score_drugs_for_disease_vectorized(
    disease_entity: str,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    entity_to_id: dict,
    relation_to_id: dict,
    compound_entities: list[str],
    treatment_relations: list[str] | None = None,
) -> list[tuple[str, float, str]]:
    """
    Vectorized version of score_drugs_for_disease. Much faster for large compound sets.
    """
    if treatment_relations is None:
        treatment_relations = TREATMENT_RELATIONS

    if disease_entity not in entity_to_id:
        raise ValueError(f"Disease entity '{disease_entity}' not found in DRKG embeddings")

    disease_idx = entity_to_id[disease_entity]
    disease_emb_vec = entity_emb[disease_idx]

    # Get valid relation indices
    rel_indices = {}
    for rel in treatment_relations:
        if rel in relation_to_id:
            rel_indices[rel] = relation_to_id[rel]
    if not rel_indices:
        raise ValueError("No treatment relations found in DRKG relation embeddings")

    # Get valid compound indices
    valid_compounds = []
    valid_indices = []
    for c in compound_entities:
        if c in entity_to_id:
            valid_compounds.append(c)
            valid_indices.append(entity_to_id[c])

    if not valid_indices:
        return []

    # Batch compute: compound_embs shape (N, dim)
    compound_embs = entity_emb[valid_indices]

    # For each relation, compute scores for all compounds at once
    best_scores = np.full(len(valid_compounds), float("-inf"))
    best_rels = [""] * len(valid_compounds)

    for rel, rel_idx in rel_indices.items():
        r_emb = relation_emb[rel_idx]
        # score = -||compound + relation - disease|| for all compounds
        diff = compound_embs + r_emb - disease_emb_vec
        scores = -np.linalg.norm(diff, axis=1)

        improved = scores > best_scores
        best_scores[improved] = scores[improved]
        for i in np.where(improved)[0]:
            best_rels[i] = rel

    # Build results and sort
    results = list(zip(valid_compounds, best_scores.tolist(), best_rels))
    results.sort(key=lambda x: -x[1])
    return results
