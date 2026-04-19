"""
Train a KG-embedding model on the unified (DRKG + PrimeKG + OT) graph.

Device detection
----------------
- ``cuda`` if an NVIDIA GPU is available (best; use this on an RTX laptop)
- ``mps``  on Apple Silicon (works, but PyKEEN on MPS tops out at 128-dim
  / 50 epochs in tractable time — not publication-grade)
- ``cpu``  as last-resort (days, not hours)

The script auto-picks sensible ``embedding_dim`` / ``num_epochs`` /
``batch_size`` / ``model`` for the detected device — override any of
them with CLI flags.

Recommended runs
----------------
On RTX 4070 laptop (8 GB VRAM):
    # TransE — sanity check, fast
    python3 scripts/train_unified_kg.py --model TransE --epochs 400
    # RotatE — the publishable retrain (10-16 h overnight)
    python3 scripts/train_unified_kg.py --model RotatE --epochs 400

On Apple MPS (current machine):
    python3 scripts/train_unified_kg.py           # 128-dim, 20 epochs,
                                                  # honest-baseline only

Requires ``scripts/build_unified_kg.py`` to have produced
``data/unified_kg/unified.tsv``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


UNIFIED_TSV = Path("data/unified_kg/unified.tsv")
OUT_DIR = Path("data/models/unified_transE_clean")


def detect_device() -> str:
    """Return the best available torch device as a string."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_defaults(device: str, model_name: str) -> dict:
    """Return sensible default hyperparameters for the given device/model.

    CUDA gets the full 400-dim / 400-epoch publication-grade config.
    MPS caps at 128-dim / 50 epochs (beyond that PyKEEN on MPS is untenable).
    CPU gets a tiny config that finishes in a day for smoke testing.
    """
    if device == "cuda":
        return {
            "embedding_dim": 400,
            "num_epochs": 400,
            "batch_size": 4096,
            "num_negs_per_pos": 64,
            "lr": 1e-3,
        }
    if device == "mps":
        # RotatE on MPS fails (complex norms unsupported); fall back to TransE.
        return {
            "embedding_dim": 128,
            "num_epochs": 50,
            "batch_size": 2048,
            "num_negs_per_pos": 20,
            "lr": 1e-3,
        }
    # cpu
    return {
        "embedding_dim": 64,
        "num_epochs": 5,
        "batch_size": 1024,
        "num_negs_per_pos": 10,
        "lr": 1e-3,
    }


def parse_args() -> argparse.Namespace:
    device = detect_device()
    defaults = device_defaults(device, "TransE")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default=str(UNIFIED_TSV))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--model", choices=("TransE", "RotatE"), default="TransE",
                    help="RotatE needs CUDA (complex norms); MPS/CPU force TransE.")
    ap.add_argument("--device", default=device, choices=("cuda", "mps", "cpu"))
    ap.add_argument("--embedding-dim", type=int, default=defaults["embedding_dim"])
    ap.add_argument("--epochs", type=int, default=defaults["num_epochs"])
    ap.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    ap.add_argument("--num-negs-per-pos", type=int, default=defaults["num_negs_per_pos"])
    ap.add_argument("--lr", type=float, default=defaults["lr"])
    return ap.parse_args()


def main() -> None:
    try:
        import pykeen  # noqa: F401
        from pykeen.triples import TriplesFactory
        from pykeen.models import TransE, RotatE
        from pykeen.training import SLCWATrainingLoop
        from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
        import torch
        from torch.optim import Adam
    except ImportError as exc:
        raise SystemExit(f"PyKEEN/torch not installed: {exc}\n"
                         "  pip install pykeen torch")

    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out)

    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Run:\n"
            "  python3 scripts/build_unified_kg.py"
        )

    # RotatE requires complex arithmetic — MPS/CPU either can't or shouldn't.
    if args.model == "RotatE" and args.device != "cuda":
        print(f"⚠  RotatE requested on {args.device}; falling back to TransE "
              f"(RotatE's complex-norm ops need CUDA).", file=sys.stderr)
        args.model = "TransE"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading triplets from {data_path}…")
    tf = TriplesFactory.from_path(str(data_path))
    print(f"  {tf.num_entities:,} entities  {tf.num_relations:,} relations  "
          f"{tf.num_triples:,} triples")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}  embedding_dim={args.embedding_dim}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}  "
          f"negs_per_pos={args.num_negs_per_pos}  lr={args.lr}")

    ModelCls = RotatE if args.model == "RotatE" else TransE
    model_kwargs: dict = {"embedding_dim": args.embedding_dim}
    if args.model == "TransE":
        model_kwargs["scoring_fct_norm"] = 2
    model = ModelCls(triples_factory=tf, **model_kwargs).to(device)

    optimizer = Adam(params=model.parameters(), lr=args.lr)
    trainer = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
        optimizer=optimizer,
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs_per_pos),
    )

    # Deliberately bypass pykeen.pipeline() — its on-device evaluator
    # was unusably slow on MPS (~50 h for 350k triples). Held-out
    # metrics are computed separately via scripts/run_heldout_eval.py.
    print(f"Training {args.model} ({args.embedding_dim}-dim) for {args.epochs} epochs...")
    losses = trainer.train(
        triples_factory=tf,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_tqdm=True,
    )
    print(f"Final loss: {losses[-1]:.4f}")

    torch.save(model, out_dir / "trained_model.pkl")
    tf_binary_dir = out_dir / "training_triples"
    if tf_binary_dir.exists():
        shutil.rmtree(tf_binary_dir)
    tf.to_path_binary(tf_binary_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
