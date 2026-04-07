"""
Train a PyKEEN knowledge graph embedding model on DRKG.

Usage:
    python3 scripts/train_pykeen.py rotate      # Train RotatE (recommended)
    python3 scripts/train_pykeen.py complex      # Train ComplEx
    python3 scripts/train_pykeen.py distmult     # Train DistMult

Models are saved to data/models/pykeen/<model_name>/
Training takes ~2-4 hours on GPU, ~8-12 hours on CPU.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from opencure.config import DRKG_TRIPLETS

MODELS_DIR = Path("data/models/pykeen")


def train_model(model_name: str = "rotate"):
    """Train a PyKEEN model on DRKG."""
    print(f"{'='*60}")
    print(f"  Training {model_name.upper()} on DRKG")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")
    print()

    # Load DRKG triplets
    print("Loading DRKG triplets...")
    import pandas as pd
    df = pd.read_csv(DRKG_TRIPLETS, sep="\t", header=None, names=["head", "relation", "tail"])
    print(f"  {len(df):,} triplets loaded")

    # Create TriplesFactory from pandas
    print("Creating TriplesFactory...")
    tf = TriplesFactory.from_labeled_triples(
        df[["head", "relation", "tail"]].values,
    )
    print(f"  {tf.num_entities:,} entities, {tf.num_relations} relations")

    # Split into train/valid/test
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)
    print(f"  Training: {training.num_triples:,} | Validation: {validation.num_triples:,} | Test: {testing.num_triples:,}")

    # Model configs
    model_configs = {
        "rotate": {
            "model": "RotatE",
            "model_kwargs": {"embedding_dim": 200},
            "training_kwargs": {
                "num_epochs": 100,
                "batch_size": 8192,
            },
            "optimizer_kwargs": {"lr": 1e-3},
            "negative_sampler_kwargs": {"num_negs_per_pos": 32},
        },
        "complex": {
            "model": "ComplEx",
            "model_kwargs": {"embedding_dim": 200},
            "training_kwargs": {
                "num_epochs": 100,
                "batch_size": 4096,
            },
            "optimizer_kwargs": {"lr": 1e-3},
        },
        "distmult": {
            "model": "DistMult",
            "model_kwargs": {"embedding_dim": 200},
            "training_kwargs": {
                "num_epochs": 100,
                "batch_size": 4096,
            },
            "optimizer_kwargs": {"lr": 1e-3},
        },
    }

    if model_name not in model_configs:
        print(f"Unknown model: {model_name}. Choose from: {list(model_configs.keys())}")
        return

    config = model_configs[model_name]

    # Train using low-level API (skip evaluation - too slow on CPU)
    print(f"\nTraining {config['model']} (no evaluation - saves hours on CPU)...")
    start = time.time()

    # RotatE uses complex numbers - MPS doesn't support complex norm ops
    # Must use CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    from pykeen.models import model_resolver
    from pykeen.training import SLCWATrainingLoop
    from pykeen.sampling import BasicNegativeSampler

    model_cls = model_resolver.lookup(config["model"])
    model = model_cls(
        triples_factory=training,
        **config.get("model_kwargs", {}),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), **config.get("optimizer_kwargs", {}))

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=config.get("negative_sampler_kwargs", {}),
    )

    losses = training_loop.train(
        triples_factory=training,
        **config.get("training_kwargs", {}),
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Final loss: {losses[-1]:.6f}")

    # Save model
    output_dir = MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    torch.save(model, output_dir / "trained_model.pkl")

    # Save the triples factory (needed for entity/relation mapping)
    training.to_path_binary(output_dir / "training_triples")

    print(f"\nModel saved to {output_dir}/")
    print(f"  trained_model.pkl ({os.path.getsize(output_dir / 'trained_model.pkl') / 1e6:.1f} MB)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "rotate"
    train_model(model_name)
