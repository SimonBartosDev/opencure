"""
Held-out evaluation using the v5 unified-KG TransE models.

Evaluates BOTH:
  - data/models/unified_transE/       (contaminated — trained on all edges)
  - data/models/unified_transE_clean/ (clean — edge-stripped training)

against the 993-pair random holdout AND the 210-pair time-sliced test set.

Outputs: experiments/eval/v5_unified_heldout.json — a 2x2 matrix of
(model × test-set) Hit@10 / MRR / median rank.

The clean-model numbers on the time-sliced test set are the headline
metric for the methods paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


OUT = Path("experiments/eval/v5_unified_heldout.json")


def load_eval_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["compound"], d["disease"]))
    return pairs


def eval_model(model_dir: Path, pairs: list[tuple[str, str]], label: str) -> dict:
    import torch
    from pykeen.triples import TriplesFactory

    model_path = model_dir / "trained_model.pkl"
    tf_path = model_dir / "training_triples"
    if not model_path.exists() or not tf_path.exists():
        return {"error": f"model not trained: {model_dir}"}

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    tf = TriplesFactory.from_path_binary(tf_path)
    ent2id = tf.entity_to_id
    rel2id = tf.relation_to_id

    # Collect compound candidates from the training entities
    candidates = [e for e in ent2id if e.startswith("Compound::")]
    cand_ids = np.array([ent2id[c] for c in candidates])
    n_cands = len(candidates)

    treats_rels = [r for r in (
        "DRUGBANK::treats::Compound:Disease",
        "Hetionet::CtD::Compound:Disease",
        "PRIMEKG::indication",
        "OT::treats::Compound:Disease",
    ) if r in rel2id]

    if not treats_rels:
        return {"error": "no treats relations in model vocab"}

    # Evaluate
    hits = {10: 0, 50: 0, 100: 0, 500: 0}
    ranks: list[int] = []
    usable = [(d, s) for d, s in pairs if d in ent2id and s in ent2id]

    print(f"    {label}: {len(usable)}/{len(pairs)} pairs usable "
          f"({n_cands} candidates pool)")

    model.eval()
    with torch.no_grad():
        for drug_ent, dis_ent in usable:
            d_id = ent2id[drug_ent]
            s_id = ent2id[dis_ent]
            # Score this drug vs ALL candidates across treats-like relations;
            # take the best-rel score per candidate.
            best_drug_score = -1e9
            best_cand_scores = np.full(n_cands, -1e9)
            for r in treats_rels:
                rid = rel2id[r]
                # drug's score
                batch_drug = torch.tensor([[d_id, rid, s_id]], dtype=torch.long)
                sd = model.score_hrt(batch_drug).squeeze().item()
                best_drug_score = max(best_drug_score, sd)
                # all candidates
                heads = torch.tensor(cand_ids, dtype=torch.long)
                rels = torch.full_like(heads, rid)
                tails = torch.full_like(heads, s_id)
                batch = torch.stack([heads, rels, tails], dim=1)
                sc = model.score_hrt(batch).squeeze().cpu().numpy()
                best_cand_scores = np.maximum(best_cand_scores, sc)

            rank = int(np.sum(best_cand_scores > best_drug_score) + 1)
            ranks.append(rank)
            for k in hits:
                if rank <= k:
                    hits[k] += 1

    n = len(ranks) or 1
    mrr = sum(1.0 / r for r in ranks) / n
    return {
        "pairs_evaluated": len(ranks),
        "pairs_total": len(pairs),
        "candidate_pool": n_cands,
        "hit_at": {str(k): round(100 * v / n, 2) for k, v in hits.items()},
        "mrr": round(mrr, 4),
        "median_rank": int(np.median(ranks)) if ranks else -1,
        "mean_rank": round(float(np.mean(ranks)), 1) if ranks else -1,
    }


def main() -> None:
    print("Loading held-out sets...")
    random_holdout = load_eval_pairs(Path("data/eval/holdout_test.jsonl"))
    print(f"  random holdout:      {len(random_holdout)} pairs")
    time_sliced_path = Path("data/eval/time_sliced_test.jsonl")
    if time_sliced_path.exists():
        time_sliced = load_eval_pairs(time_sliced_path)
        print(f"  time-sliced (2020+): {len(time_sliced)} pairs")
    else:
        time_sliced = []

    results = {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    model_candidates = [
        ("drkg_transE_clean",          Path("data/models/drkg_transE_clean")),
        ("unified_transE_contaminated", Path("data/models/unified_transE")),
        ("unified_transE_clean",       Path("data/models/unified_transE_clean")),
    ]
    # Only evaluate models that exist
    model_candidates = [(lbl, d) for lbl, d in model_candidates
                         if (d / "trained_model.pkl").exists()]
    if not model_candidates:
        raise SystemExit("No trained models found in data/models/. Train one first.")
    print(f"Evaluating {len(model_candidates)} trained model(s)")
    for model_label, model_dir in model_candidates:
        print(f"\n=== {model_label} ===")
        t0 = time.time()
        results.setdefault(model_label, {})

        print("  running random held-out (993 pairs)...")
        results[model_label]["random_holdout"] = eval_model(
            model_dir, random_holdout, model_label + " / random"
        )

        if time_sliced:
            print("  running time-sliced (210 post-2020 approved)...")
            results[model_label]["time_sliced"] = eval_model(
                model_dir, time_sliced, model_label + " / time-sliced"
            )

        print(f"  done in {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_label in ("unified_transE_contaminated", "unified_transE_clean"):
        r = results.get(model_label, {})
        print(f"\n{model_label}:")
        for tset in ("random_holdout", "time_sliced"):
            m = r.get(tset, {})
            if "hit_at" not in m:
                continue
            print(f"  {tset:>18}  Hit@10 = {m['hit_at']['10']:>5}%  "
                  f"MRR = {m['mrr']}  median={m.get('median_rank')}")
    print(f"\nFull results → {OUT}")


if __name__ == "__main__":
    main()
