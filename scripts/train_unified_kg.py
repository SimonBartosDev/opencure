"""
Train RotatE on the unified (DRKG + PrimeKG + OT) knowledge graph.

Requires scripts/build_unified_kg.py to have produced
data/unified_kg/unified.tsv.

Uses PyKEEN. Runs ~4–6 h on Apple Silicon MPS, ~1.5 h on a good GPU.

Outputs data/models/unified_rotatE/trained_model.pkl and the entity/relation
embeddings as numpy arrays for fast loading at search time.

The search pipeline already supports multiple KG scores via kg_fusion;
we just need to register the new embedding as a fourth input in
opencure/scoring/pillar_groups.group_kg_scores.
"""

from __future__ import annotations

import sys
from pathlib import Path


UNIFIED_TSV = Path("data/unified_kg/unified.tsv")
OUT_DIR = Path("data/models/unified_rotatE")


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

    train, valid, test = tf.split([0.90, 0.05, 0.05], random_state=42)

    result = pipeline(
        training=train, validation=valid, testing=test,
        model="RotatE",
        model_kwargs=dict(embedding_dim=256),
        training_kwargs=dict(num_epochs=40, batch_size=1024),
        optimizer_kwargs=dict(lr=5e-4),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=50),
        random_seed=42,
        device="mps" if _has_mps() else "cpu",
    )

    result.save_to_directory(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


def _has_mps() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
