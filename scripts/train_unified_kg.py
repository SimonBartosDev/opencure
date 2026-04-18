"""
Train RotatE on the unified (DRKG + PrimeKG + OT) knowledge graph.

Requires scripts/build_unified_kg.py to have produced
data/unified_kg/unified.tsv.

Uses PyKEEN. Runs ~4–6 h on Apple Silicon MPS, ~1.5 h on a good GPU.

Outputs data/models/unified_transE/trained_model.pkl and the entity/relation
embeddings as numpy arrays for fast loading at search time.

The search pipeline already supports multiple KG scores via kg_fusion;
we just need to register the new embedding as a fourth input in
opencure/scoring/pillar_groups.group_kg_scores.
"""

from __future__ import annotations

import sys
from pathlib import Path


UNIFIED_TSV = Path("data/unified_kg/unified.tsv")
OUT_DIR = Path("data/models/unified_transE")


def main() -> None:
    try:
        import pykeen
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
    except ImportError:
        raise SystemExit("PyKEEN not installed. pip install pykeen torch")

    if not UNIFIED_TSV.exists():
        raise SystemExit(
            f"Missing {UNIFIED_TSV}. Run:\n"
            "  python3 scripts/build_unified_kg.py"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading triplets from {UNIFIED_TSV}…")
    tf = TriplesFactory.from_path(str(UNIFIED_TSV))
    print(f"  {tf.num_entities:,} entities  {tf.num_relations:,} relations  "
          f"{tf.num_triples:,} triples")

    train, valid, test = tf.split([0.95, 0.025, 0.025], random_state=42)

    # MPS does NOT support complex-number norms required by RotatE. We use
    # TransE instead — same family, real-valued, works on MPS. Upgrade to
    # RotatE when running on CUDA. TransE is the reference model DRKG itself
    # was trained with, so consistency with the existing DRKG TransE scorer
    # is actually a plus.
    result = pipeline(
        training=train, validation=valid, testing=test,
        model="TransE",
        model_kwargs=dict(embedding_dim=128, scoring_fct_norm=2),
        training_kwargs=dict(num_epochs=20, batch_size=2048, use_tqdm=True),
        optimizer_kwargs=dict(lr=1e-3),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=20),
        random_seed=42,
        device="mps" if _has_mps() else "cpu",
        evaluation_kwargs=dict(use_tqdm=True),
    )

    result.save_to_directory(str(OUT_DIR))
    print(f"Saved to {OUT_DIR}")
    print(f"Final MRR: {result.metric_results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"Hits@10:   {result.metric_results.get_metric('hits_at_10'):.4f}")


def _has_mps() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
