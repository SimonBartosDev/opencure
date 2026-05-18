"""
Unified KG scorer — the fourth KG pillar (v4 Phase 5).

Loads a PyKEEN TransE model trained on the unified DRKG+PrimeKG+OpenTargets
graph (14M triples) and scores compounds against diseases via multiple
treatment-like relations across the three sources:

    DRUGBANK::treats::Compound:Disease        (DRKG)
    Hetionet::CtD::Compound:Disease          (DRKG)
    GNBR::T::Compound:Disease                (DRKG)
    PRIMEKG::indication                       (PrimeKG)
    PRIMEKG::off-label use                    (PrimeKG)
    OT::treats::Compound:Disease              (Open Targets, phase >= 3)
    OT::trialed::Compound:Disease             (Open Targets, phase 1-2)

Returns best-across-relations score per compound, for use as the fourth
input to kg_fusion.fuse_kg_scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


UNIFIED_MODEL_DIR = Path("data/models/unified_transE")

TREATMENT_RELATIONS_UNIFIED = (
    # DRKG
    "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease",
    "Hetionet::CpD::Compound:Disease",
    "GNBR::T::Compound:Disease",
    # PrimeKG
    "PRIMEKG::indication",
    "PRIMEKG::off-label use",
    # Open Targets
    "OT::treats::Compound:Disease",
    "OT::trialed::Compound:Disease",
)


def load_unified_model():
    """Load the trained unified-KG PyKEEN model.

    Returns (model, triples_factory) or (None, None) if model missing.
    """
    if not UNIFIED_MODEL_DIR.exists():
        return None, None
    try:
        import torch
        from pykeen.triples import TriplesFactory
        model_file = UNIFIED_MODEL_DIR / "trained_model.pkl"
        tf_dir = UNIFIED_MODEL_DIR / "training_triples"
        if not model_file.exists() or not tf_dir.exists():
            return None, None
        model = torch.load(model_file, map_location="cpu", weights_only=False)
        tf = TriplesFactory.from_path_binary(tf_dir)
        print(f"  Loaded unified-KG TransE: {model.num_entities:,} entities, "
              f"{model.num_relations:,} relations")
        return model, tf
    except Exception as e:
        print(f"  [WARN] unified KG model load failed: {e}")
        return None, None


def score_drugs_for_disease_unified(
    disease_entity: str,
    model,
    triples_factory,
    compound_entities: list[str],
    top_k: int = 500,
) -> dict[str, tuple[float, int, str]]:
    """
    Score compounds against a disease using the unified-KG model.

    Returns dict[compound] -> (score_normalized_0_to_1, rank, best_relation).
    Score is normalized so the top-1 compound = 1.0, top-K = 0.0, which
    matches the convention expected by kg_fusion.fuse_kg_scores.
    """
    import torch

    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id

    if disease_entity not in entity_to_id:
        return {}
    disease_id = entity_to_id[disease_entity]

    valid_rels = {r: relation_to_id[r] for r in TREATMENT_RELATIONS_UNIFIED if r in relation_to_id}
    if not valid_rels:
        return {}

    valid_compounds: list[str] = []
    valid_compound_ids: list[int] = []
    for c in compound_entities:
        cid = entity_to_id.get(c)
        if cid is not None:
            valid_compounds.append(c)
            valid_compound_ids.append(cid)

    if not valid_compound_ids:
        return {}

    model.eval()
    # Best raw TransE score + which relation produced it
    best: dict[str, tuple[float, str]] = {}

    with torch.no_grad():
        for rel_name, rel_id in valid_rels.items():
            heads = torch.tensor(valid_compound_ids, dtype=torch.long)
            relations = torch.full_like(heads, rel_id)
            tails = torch.full_like(heads, disease_id)
            batch = torch.stack([heads, relations, tails], dim=1)
            scores = model.score_hrt(batch).squeeze(-1).cpu().numpy()
            for c, s in zip(valid_compounds, scores):
                fs = float(s)
                if c not in best or fs > best[c][0]:
                    best[c] = (fs, rel_name)

    if not best:
        return {}

    # Normalize to 0-1 using rank position (top-K window)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    n = min(top_k, len(ranked))
    out: dict[str, tuple[float, int, str]] = {}
    for i, (comp, (_, rel)) in enumerate(ranked[:n]):
        norm_score = 1.0 - (i / max(n - 1, 1))  # rank-1 = 1.0, rank-n = 0.0
        out[comp] = (norm_score, i + 1, rel)
    return out
