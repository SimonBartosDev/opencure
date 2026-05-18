"""
Held-out drug-disease benchmark using TransE embeddings.

For each held-out (drug_entity, disease_entity) pair, rank all compounds in
DRKG against the disease via the pretrained TransE score (||h + r - t||),
then measure where the held-out drug lands.

This is the cleanest single-pillar benchmark we can run without retraining.
Contamination caveat: the TransE model was trained on the full DRKG including
the held-out treats-edges, so these numbers are an upper bound. Interpret as:
"when the KG-embedding pillar is shown 993 disease queries whose answer it
has memorized, how well can it retrieve that answer?"

Reported metrics:
  - Hit@10, Hit@50, Hit@100, Hit@500
  - MRR over all held-out pairs
  - Per-disease breakdown

Output: experiments/eval/transe_heldout.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from opencure.data.drkg import load_embeddings, get_compound_entities
from opencure.eval.ground_truth import load_holdout


OUT_PATH = Path("experiments/eval/transe_heldout.json")


def main() -> None:
    print("Loading DRKG embeddings…")
    t0 = time.time()
    entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id = load_embeddings()
    print(f"  {len(entity_to_id):,} entities, {len(relation_to_id):,} relations  ({time.time()-t0:.1f}s)")

    # Relation used: DRUGBANK::treats is what the held-out edges are labeled with
    rel_key = "DRUGBANK::treats::Compound:Disease"
    rel_id = relation_to_id.get(rel_key)
    if rel_id is None:
        raise SystemExit(f"Relation {rel_key!r} not in relation_to_id")
    rel_vec = relation_emb[rel_id]

    # Collect the set of DrugBank-only compound candidate entities
    print("Selecting compound candidates…")
    compounds = get_compound_entities(entity_to_id, drugbank_only=True)
    cand_ids = np.array([entity_to_id[c] for c in compounds])
    print(f"  {len(compounds):,} DrugBank compound candidates")
    cand_emb = entity_emb[cand_ids]

    # Build reverse lookup from entity string
    pairs = load_holdout()
    print(f"Held-out pairs: {len(pairs)}")

    # Filter pairs where both ends exist in entity_to_id
    usable = [(d, s) for d, s in pairs if d in entity_to_id and s in entity_to_id]
    print(f"  Usable after entity check: {len(usable)}")

    ranks: list[int] = []
    per_disease: dict[str, list[int]] = {}
    hits = {10: 0, 50: 0, 100: 0, 500: 0}

    for drug_ent, dis_ent in usable:
        d_id = entity_to_id[drug_ent]
        s_id = entity_to_id[dis_ent]
        # TransE score: -||h + r - t||  (higher = better)
        tail = entity_emb[s_id]
        # Predicted tail = head + rel, compare to actual tails (candidates)
        # BUT for treats: head=compound, tail=disease. We rank compounds as heads.
        # score(c, dis) = -||emb[c] + rel - emb[dis]||
        diff = cand_emb + rel_vec - tail  # (N_cand, dim)
        scores = -np.linalg.norm(diff, axis=1)  # higher = better
        # rank = #candidates with score strictly greater than drug's score + 1
        drug_score = -np.linalg.norm(entity_emb[d_id] + rel_vec - tail)
        rank = int(np.sum(scores > drug_score) + 1)
        ranks.append(rank)
        per_disease.setdefault(dis_ent, []).append(rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1

    n = len(ranks)
    if n == 0:
        raise SystemExit("No evaluable pairs")

    mrr = float(sum(1.0 / r for r in ranks) / n)
    summary = {
        "evaluated_pairs": n,
        "total_heldout": len(pairs),
        "candidate_pool_size": len(compounds),
        "hit_at": {str(k): round(100 * v / n, 2) for k, v in hits.items()},
        "mrr": round(mrr, 4),
        "median_rank": int(np.median(ranks)),
        "mean_rank": round(float(np.mean(ranks)), 1),
        "note": (
            "Training-contaminated upper bound — TransE was trained on full DRKG "
            "including these treats edges. Retrain on edge-stripped graph for clean numbers."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("TransE held-out retrieval metrics")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
