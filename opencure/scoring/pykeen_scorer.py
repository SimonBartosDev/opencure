"""
Knowledge graph scoring using PyKEEN models (RotatE, ComplEx, DistMult).

Replaces the basic TransE embeddings with more expressive models that can
capture complex relation patterns (symmetry, antisymmetry, inversion, composition).

RotatE models relations as rotations in complex space - significantly more
expressive than TransE's simple translations.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from opencure.config import DATA_DIR

MODELS_DIR = DATA_DIR.parent / "models"
PYKEEN_MODEL_DIR = MODELS_DIR / "pykeen"


def load_pykeen_model(model_name: str = "rotate"):
    """
    Load a trained PyKEEN model from disk.

    Args:
        model_name: Name of the model (rotate, complex, distmult)

    Returns:
        (model, triples_factory) or None if model not found
    """
    model_path = PYKEEN_MODEL_DIR / model_name
    if not model_path.exists():
        print(f"  [WARN] PyKEEN {model_name} model not found at {model_path}")
        print(f"  Run: python3 scripts/train_pykeen.py {model_name}")
        return None, None

    try:
        from pykeen.models import Model
        from pykeen.triples import TriplesFactory

        model = torch.load(model_path / "trained_model.pkl", map_location="cpu", weights_only=False)
        tf = TriplesFactory.from_path_binary(model_path / "training_triples")

        print(f"  Loaded PyKEEN {model_name}: {model.num_entities} entities, {model.num_relations} relations")
        return model, tf
    except Exception as e:
        print(f"  [WARN] Failed to load PyKEEN model: {e}")
        return None, None


def score_drugs_for_disease_pykeen(
    disease_entity: str,
    model,
    triples_factory,
    compound_entities: list[str],
    treatment_relations: list[str] | None = None,
    top_k: int = 500,
) -> list[tuple[str, float, str]]:
    """
    Score all compounds against a disease using a trained PyKEEN model.

    Args:
        disease_entity: DRKG disease entity ID
        model: Trained PyKEEN model
        triples_factory: TriplesFactory from training
        compound_entities: List of compound entity IDs to score
        treatment_relations: Relations to use (defaults to treatment relations)
        top_k: Return top-k results

    Returns:
        Sorted list of (compound_entity, score, relation) tuples
    """
    from opencure.config import TREATMENT_RELATIONS

    if treatment_relations is None:
        treatment_relations = TREATMENT_RELATIONS

    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id

    # Check disease exists
    if disease_entity not in entity_to_id:
        return []

    disease_id = entity_to_id[disease_entity]

    # Get valid treatment relations
    valid_rels = {}
    for rel in treatment_relations:
        if rel in relation_to_id:
            valid_rels[rel] = relation_to_id[rel]

    if not valid_rels:
        return []

    # Get valid compounds
    valid_compounds = []
    valid_compound_ids = []
    for c in compound_entities:
        if c in entity_to_id:
            valid_compounds.append(c)
            valid_compound_ids.append(entity_to_id[c])

    if not valid_compound_ids:
        return []

    # Score all compounds for each treatment relation
    model.eval()
    best_scores = {}

    with torch.no_grad():
        for rel_name, rel_id in valid_rels.items():
            # Build batch: all compounds as heads, fixed relation and disease as tail
            heads = torch.tensor(valid_compound_ids, dtype=torch.long)
            relations = torch.full_like(heads, rel_id)
            tails = torch.full_like(heads, disease_id)

            # Stack into (N, 3) tensor
            batch = torch.stack([heads, relations, tails], dim=1)

            # Score
            scores = model.score_hrt(batch).squeeze(-1).cpu().numpy()

            for compound, score in zip(valid_compounds, scores):
                if compound not in best_scores or score > best_scores[compound][0]:
                    best_scores[compound] = (float(score), rel_name)

    # Sort and return top-k
    results = [(comp, score, rel) for comp, (score, rel) in best_scores.items()]
    results.sort(key=lambda x: -x[1])
    return results[:top_k]
