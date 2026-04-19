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

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(UNIFIED_TSV))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=20)
    args, _ = ap.parse_known_args()
    data_path = Path(args.data)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading triplets from {data_path}…")
    tf = TriplesFactory.from_path(str(data_path))
    print(f"  {tf.num_entities:,} entities  {tf.num_relations:,} relations  "
          f"{tf.num_triples:,} triples")

    # SKIP PyKEEN's default pipeline() because its final evaluation on MPS
    # is unusably slow (~50 h for 350k triples). Train directly via
    # SLCWATrainingLoop and save the model; held-out metrics come from
    # scripts/run_heldout_eval.py (seconds, not hours).
    from pykeen.models import TransE
    from pykeen.training import SLCWATrainingLoop
    from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
    from torch.optim import Adam
    import torch
    import shutil

    device = torch.device("mps" if _has_mps() else "cpu")
    print(f"Device: {device}")

    model = TransE(
        triples_factory=tf,
        embedding_dim=128,
        scoring_fct_norm=2,
    ).to(device)

    optimizer = Adam(params=model.parameters(), lr=1e-3)
    trainer = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
        optimizer=optimizer,
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=20),
    )

    print(f"Training TransE 128-dim for {args.epochs} epochs, batch 2048...")
    losses = trainer.train(
        triples_factory=tf,
        num_epochs=args.epochs,
        batch_size=2048,
        use_tqdm=True,
    )
    print(f"Final loss: {losses[-1]:.4f}")

    torch.save(model, out_dir / "trained_model.pkl")
    tf_binary_dir = out_dir / "training_triples"
    if tf_binary_dir.exists():
        shutil.rmtree(tf_binary_dir)
    tf.to_path_binary(tf_binary_dir)
    print(f"Saved to {out_dir}")


def _has_mps() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
