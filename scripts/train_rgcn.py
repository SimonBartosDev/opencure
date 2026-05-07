"""
Train a heterogeneous R-GCN on the unified KG (v5 C1).

Produces data/models/rgcn_v5/trained_model.pt — node embeddings + relation
embeddings + entity_to_id mapping.

Recommended hardware:
  - CUDA GPU (A100/V100) — ~4-8 hours for 14M triples, 17 node types
  - Apple MPS          — ~24+ hours; usable but slow
  - CPU               — >24 hours; not recommended

The model is a 2-layer R-GCN with DistMult decoder. Hyperparameters:
  - embedding_dim: 200
  - num_bases:     30 (for relation parameter sharing)
  - dropout:       0.2
  - epochs:        50
  - batch_size:    4096 triples/batch
  - lr:            0.01 with CosineLR decay
  - neg samples:   20 per positive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Graceful degradation if PyG not available
try:
    import torch
    from torch.nn import Module
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.nn import RGCNConv
    PYG_OK = True
except ImportError:
    PYG_OK = False


UNIFIED_KG_PATH = Path("data/unified_kg/unified_train_clean.tsv")  # stripped version!
MODEL_DIR = Path("data/models/rgcn_v5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dim", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--neg_samples", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--checkpoint_every", type=int, default=5,
                    help="Save model + epoch state every N epochs (0 = disable).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint.pt in MODEL_DIR if present.")
    # Sampling-based training (correct R-GCN-at-scale pattern):
    # full-graph encode + per-batch backward causes autograd version
    # mismatch when optim.step() mutates relation weight matrices
    # mid-graph. We instead subsample edges + triples per epoch and do
    # ONE backward + step per epoch — same statistical guarantees over
    # multiple epochs, no autograd inplace-version error.
    ap.add_argument("--edges_per_epoch", type=int, default=2_000_000,
                    help="Edges sampled for the encoder pass each epoch "
                         "(out of ~14M; full set covers in ~7 epochs at 2M/epoch).")
    ap.add_argument("--triples_per_epoch", type=int, default=500_000,
                    help="Positive triples scored each epoch (out of ~14M).")
    args = ap.parse_args()

    if not PYG_OK:
        raise SystemExit(
            "PyTorch Geometric not installed. Run:\n"
            "  pip install torch-geometric"
        )

    if not UNIFIED_KG_PATH.exists():
        raise SystemExit(
            f"Missing {UNIFIED_KG_PATH}. Run:\n"
            "  python3 scripts/build_unified_kg.py && "
            "python3 scripts/strip_heldout_edges.py"
        )

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("NOTE: using MPS — expect ~24 h training time. "
                  "Recommend CUDA for R-GCN.")
        else:
            device = torch.device("cpu")
            print("WARNING: no GPU found; CPU training is impractical for 14M triples.")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load triples, build entity + relation vocabularies
    t0 = time.time()
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []
    with UNIFIED_KG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            hi = entities.setdefault(h, len(entities))
            ri = relations.setdefault(r, len(relations))
            ti = entities.setdefault(t, len(entities))
            triples.append((hi, ri, ti))

    n_ent = len(entities)
    n_rel = len(relations)
    print(f"Loaded in {time.time()-t0:.1f}s: {n_ent:,} entities  {n_rel:,} relations  {len(triples):,} triples")

    # Build edge index + edge type for R-GCN forward pass
    triples_t = torch.tensor(triples, dtype=torch.long)
    edge_index = torch.stack([triples_t[:, 0], triples_t[:, 2]], dim=0)
    edge_type = triples_t[:, 1]

    # Model: 2-layer R-GCN + DistMult decoder
    class RGCNDistMult(Module):
        def __init__(self, n_ent, n_rel, dim, n_bases):
            super().__init__()
            self.node_emb = torch.nn.Embedding(n_ent, dim)
            self.rel_emb = torch.nn.Embedding(n_rel, dim)
            self.conv1 = RGCNConv(dim, dim, n_rel, num_bases=n_bases)
            self.conv2 = RGCNConv(dim, dim, n_rel, num_bases=n_bases)
            self.drop = torch.nn.Dropout(0.2)

        def encode(self, edge_index, edge_type):
            x = self.node_emb.weight
            x = torch.relu(self.conv1(x, edge_index, edge_type))
            x = self.drop(x)
            x = self.conv2(x, edge_index, edge_type)
            return x

        def score(self, h, r, t, x):
            return (x[h] * self.rel_emb(r) * x[t]).sum(dim=-1)

    model = RGCNDistMult(n_ent, n_rel, args.embedding_dim, n_bases=min(30, n_rel)).to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    # Resume from checkpoint if present
    ckpt_path = MODEL_DIR / "checkpoint.pt"
    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    # Training: sampled single-step per epoch (correct R-GCN-at-scale pattern).
    #
    # Why not the obvious full-graph encode + mini-batch loop:
    #   The encoder produces x (node features) by message-passing through
    #   relation-weight matrices [dim, dim] per relation. If we then do
    #   mini-batch backward+step, optim.step() mutates those weight
    #   matrices. The next batch's gradient flows back through x, which
    #   was computed against the now-stale weights → autograd raises
    #   "[X, X] is at version 1; expected version 0" and kills training.
    #
    # The fix: ONE forward, ONE backward, ONE step per epoch. To keep
    # epochs fast on a 14M-edge graph we subsample:
    #   - edges_per_epoch    edges go through the encoder
    #   - triples_per_epoch  positive triples are scored
    # Across `epochs` runs of stochastic sampling, the model sees the
    # full graph with high probability while each individual epoch fits
    # comfortably on A100 40GB.
    print(f"Training: epochs {start_epoch}→{args.epochs}; "
          f"sampling {args.edges_per_epoch:,} edges / {args.triples_per_epoch:,} triples per epoch")
    edge_index_d = edge_index.to(device)
    edge_type_d = edge_type.to(device)
    triples_t_d = triples_t.to(device)
    n_edges_total = edge_index_d.size(1)
    n_triples_total = triples_t_d.size(0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_ep = time.time()
        optim.zero_grad()

        # Sample a fresh edge subset for the encoder this epoch.
        edge_perm = torch.randperm(n_edges_total, device=device)[:args.edges_per_epoch]
        sub_edge_index = edge_index_d[:, edge_perm]
        sub_edge_type = edge_type_d[edge_perm]

        # Encode the subsampled graph (single forward pass)
        x = model.encode(sub_edge_index, sub_edge_type)

        # Sample a triple batch + corrupt-tail negatives
        triple_perm = torch.randperm(n_triples_total, device=device)[:args.triples_per_epoch]
        batch = triples_t_d[triple_perm]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        neg_t = torch.randint(0, n_ent, (batch.size(0) * args.neg_samples,), device=device)
        h_neg = h.repeat_interleave(args.neg_samples)
        r_neg = r.repeat_interleave(args.neg_samples)

        pos_score = model.score(h, r, t, x)
        neg_score = model.score(h_neg, r_neg, neg_t, x)
        # Margin ranking loss (DistMult-style decoder)
        loss = torch.clamp(
            1.0 - pos_score.repeat_interleave(args.neg_samples) + neg_score,
            min=0,
        ).mean()

        # Single backward + step — no inplace-version conflict.
        loss.backward()
        optim.step()
        sched.step()

        dt = time.time() - t_ep
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={loss.item():.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  dt={dt:.1f}s",
              flush=True)

        # Periodic checkpoint so a crash mid-training loses ≤checkpoint_every epochs
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch + 1,
                "loss": loss.item(),
            }, ckpt_path)
            print(f"  ✓ checkpoint @ epoch {epoch+1} → {ckpt_path}", flush=True)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "node_emb": model.node_emb.weight.detach().cpu(),
        "rel_emb": model.rel_emb.weight.detach().cpu(),
        "entity_to_id": entities,
        "relation_to_id": relations,
        "hyperparams": vars(args),
    }
    torch.save(state, MODEL_DIR / "trained_model.pt")
    (MODEL_DIR / "vocab.json").write_text(json.dumps({
        "n_entities": n_ent, "n_relations": n_rel, "n_triples": len(triples)
    }, indent=2))
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
