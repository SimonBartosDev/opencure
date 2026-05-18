"""
Heterogeneous R-GCN pillar (v5 C1).

Replaces shallow embedding methods with a proper heterogeneous graph
neural network. The 14M-triple unified KG has 17 node types (Compound,
Disease, Gene, BiologicalProcess, Pathway, MolecularFunction, etc.) and
~250 relation types. A relational graph convolutional network (R-GCN)
message-passes along each edge type separately, producing node
embeddings that respect the heterogeneity of biomedical knowledge.

This is the standard architecture used by TxGNN and other published
SOTA systems. OpenCure's v5 adds it as the 12th pillar while keeping
TransE/RotatE/PrimeKG for diversity in RRF fusion.

Training (scripts/train_rgcn.py) requires a CUDA GPU for practicality
— ~4-8 hours on A100 vs days on Apple MPS. The scorer below loads
pre-trained weights from data/models/rgcn_v5/ when they exist, and
fail-opens (empty scores) when not.

This module is architected to be dropped in as score_drugs_for_disease_rgcn
alongside the other pillars in opencure/search.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


RGCN_MODEL_DIR = Path("data/models/rgcn_v5")
UNIFIED_KG_PATH = Path("data/unified_kg/unified.tsv")


TREATS_RELATIONS = (
    "DRUGBANK::treats::Compound:Disease",
    "OT::treats::Compound:Disease",
    "PRIMEKG::indication",
    "Hetionet::CtD::Compound:Disease",
)


def load_rgcn_model():
    """Load a trained R-GCN + node/relation embeddings.

    Returns dict:
        {
          "node_emb":         FloatTensor [num_entities, dim],
          "rel_emb":          FloatTensor [num_relations, dim],
          "entity_to_id":     dict[str, int],
          "relation_to_id":   dict[str, int],
        }
    or None if model file is absent / unreadable.
    """
    try:
        import torch
    except ImportError:
        return None

    path = RGCN_MODEL_DIR / "trained_model.pt"
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        # Validate shape — must contain everything DistMult needs
        for k in ("node_emb", "rel_emb", "entity_to_id", "relation_to_id"):
            if k not in state:
                return None
        return {
            "node_emb": state["node_emb"],
            "rel_emb": state["rel_emb"],
            "entity_to_id": state["entity_to_id"],
            "relation_to_id": state["relation_to_id"],
        }
    except Exception as exc:
        print(f"  [WARN] R-GCN load failed: {exc}")
        return None


def score_drugs_for_disease_rgcn(
    disease_entity: str,
    compound_entities: list[str],
    rgcn_state: dict | None = None,
    top_k: int = 500,
) -> dict[str, tuple[float, int, str]]:
    """Score compounds against a disease via DistMult on R-GCN embeddings.

    DistMult scoring head:  score(h, r, t) = <h_emb, r_emb, t_emb> over the
    relevant relation. We score each compound against the disease for every
    treats-like relation that exists in the model's vocabulary, then take
    the maximum score per compound (best-relation evidence).

    Returns dict[compound] -> (rank_normalized_score, rank, best_relation).
    Empty dict on any failure (fail-open).
    """
    if rgcn_state is None or disease_entity not in rgcn_state["entity_to_id"]:
        return {}
    try:
        import torch
    except ImportError:
        return {}

    ent2id = rgcn_state["entity_to_id"]
    rel2id = rgcn_state["relation_to_id"]
    node_emb = rgcn_state["node_emb"]      # [E, dim]
    rel_emb = rgcn_state["rel_emb"]        # [R, dim]

    # Pick the relations that exist in the trained vocab
    treats_rel_ids = [rel2id[r] for r in TREATS_RELATIONS if r in rel2id]
    if not treats_rel_ids:
        return {}

    valid = [(c, ent2id[c]) for c in compound_entities if c in ent2id]
    if not valid:
        return {}

    dis_id = ent2id[disease_entity]
    t_emb = node_emb[dis_id]                                # [dim]
    comp_ids = torch.tensor([v[1] for v in valid], dtype=torch.long)
    h_emb = node_emb[comp_ids]                              # [N, dim]

    # DistMult: score = sum( h * r * t ).  Compute over all treats-rels and
    # take the max per compound.
    best_scores = torch.full((len(valid),), -float("inf"))
    best_rel_per_compound = ["rgcn"] * len(valid)
    id_to_rel = {v: k for k, v in rel2id.items()}
    for rid in treats_rel_ids:
        r_emb = rel_emb[rid]                                # [dim]
        scores = (h_emb * r_emb.unsqueeze(0) * t_emb.unsqueeze(0)).sum(dim=-1)
        improved = scores > best_scores
        best_scores = torch.where(improved, scores, best_scores)
        for i, did_improve in enumerate(improved.tolist()):
            if did_improve:
                best_rel_per_compound[i] = id_to_rel[rid]

    # Rank descending; emit rank-normalized score in [0, 1] (1 = top)
    order = best_scores.argsort(descending=True).tolist()
    n_emit = min(top_k, len(order))
    out: dict[str, tuple[float, int, str]] = {}
    for rank_i, idx in enumerate(order[:n_emit]):
        comp = valid[idx][0]
        norm_score = 1.0 - rank_i / max(n_emit - 1, 1)
        out[comp] = (round(norm_score, 4), rank_i + 1, best_rel_per_compound[idx])
    return out
