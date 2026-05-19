"""
WS1 decisive evaluation: how well does the FULL 11-pillar pipeline rank
genuine repurposing events?

For each (old drug X, new disease B) in data/eval/repurposing_benchmark.jsonl,
run the real search() pipeline for B and record X's rank among all candidate
drugs — by combined_score and by each individual pillar. This is naturally
leak-free: DRKG is 2020-vintage and the post-2020 X->B edge is not in it.

This is the honest go/no-go number. Hit@K here is X landing in the top-K of
~8000 filtered drug candidates — the same task a wet lab would judge.

Output: experiments/eval/repurposing_fullstack.json
Runtime: ~1-1.5 h on M4 Max (one search() per disease, ~32 diseases).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = Path("data/eval/repurposing_benchmark.jsonl")
OUT = Path("experiments/eval/repurposing_fullstack.json")

# Pillar sub-scores recorded on every candidate (higher = better for all).
PILLARS = [
    "combined_score", "efficacy_score", "transe_score", "txgnn_score",
    "mol_emb_similarity", "dti_score", "proximity_score", "primekg_score",
]


def rank_by(cands: list[dict], drug_id: str, field: str) -> float | None:
    """Mid-rank of drug_id by `field` (higher=better). None if the drug is
    absent / has no value, OR if the pillar gave the SAME value to (nearly)
    every candidate — that is no-coverage, not a rank-1 finish.

    Mid-rank: ``higher + equal/2 + 1`` (standard tie handling). A pillar with
    zero coverage for a disease scores all candidates equally → mid-rank lands
    at the pool midpoint, not a spurious rank 1.
    """
    target = next((c for c in cands if c.get("drug_id") == drug_id), None)
    if target is None:
        return None
    tv = target.get(field)
    if tv is None:
        return None
    tv = float(tv)
    vals = [float(c[field]) for c in cands if c.get(field) is not None]
    if not vals:
        return None
    higher = sum(1 for v in vals if v > tv)
    equal = sum(1 for v in vals if v == tv)
    # No-coverage guard: if >90% of candidates share the positive's value,
    # the pillar has no opinion here — not evaluable.
    if equal > 0.9 * len(vals):
        return None
    return higher + equal / 2.0 + 1


def main() -> None:
    from opencure.search import search

    rows = [json.loads(ln) for ln in BENCH.open()]
    by_disease: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_disease[r["disease"]].append(r)
    print(f"{len(rows)} repurposing pairs across {len(by_disease)} diseases")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_pair: list[dict] = []
    t0 = time.time()

    for i, (disease_entity, pairs) in enumerate(sorted(by_disease.items()), 1):
        mesh = disease_entity.split("::", 1)[1] if "::" in disease_entity else disease_entity
        print(f"\n[{i}/{len(by_disease)}] {disease_entity} "
              f"({len(pairs)} positive(s))  {time.time()-t0:.0f}s")
        try:
            cands = search(mesh, top_k=100000,
                           use_molecular_similarity=True, use_evidence=False)
        except Exception as exc:
            print(f"  search failed: {type(exc).__name__}: {exc}")
            cands = []
        if not isinstance(cands, list):
            cands = []
        pool = len(cands)
        print(f"  pool: {pool} ranked candidates")

        for r in pairs:
            drug_id = r.get("drug_id") or r["compound"].split("::", 1)[-1]
            entry = {
                "drug_id": drug_id,
                "disease": disease_entity,
                "pool_size": pool,
                "prior_indications": r.get("prior_indications"),
                "filtered_out": rank_by(cands, drug_id, "combined_score") is None,
                "ranks": {p: rank_by(cands, drug_id, p) for p in PILLARS},
            }
            per_pair.append(entry)
            cr = entry["ranks"]["combined_score"]
            print(f"    {drug_id}: combined rank = {cr if cr else 'not ranked'}"
                  f"  (pool {pool})")

        # checkpoint after every disease
        OUT.write_text(json.dumps({"per_pair": per_pair}, indent=2))

    # ---- aggregate ----
    def summary(field: str) -> dict:
        ranks = [p["ranks"][field] for p in per_pair if p["ranks"][field]]
        n = len(per_pair)
        if not ranks:
            return {"evaluable": 0, "n": n}
        ranks_arr = np.array(ranks)
        return {
            "evaluable": len(ranks),
            "n": n,
            "hit_at_10": round(100 * np.mean(ranks_arr <= 10), 1),
            "hit_at_30": round(100 * np.mean(ranks_arr <= 30), 1),
            "hit_at_100": round(100 * np.mean(ranks_arr <= 100), 1),
            "mrr": round(float(np.mean(1.0 / ranks_arr)), 4),
            "median_rank": int(np.median(ranks_arr)),
        }

    result = {
        "benchmark": str(BENCH),
        "n_pairs": len(per_pair),
        "n_diseases": len(by_disease),
        "note": "Leak-free: DRKG is 2020-vintage; post-2020 repurposing edges "
                "are absent. Hit@K = rank among all filtered drug candidates. "
                "PrimeKG/TxGNN pillars may carry residual contamination if "
                "their source KG postdates 2020.",
        "by_pillar": {p: summary(p) for p in PILLARS},
        "per_pair": per_pair,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*64}\nREPURPOSING BENCHMARK — FULL STACK ({len(per_pair)} pairs)\n{'='*64}")
    for p in PILLARS:
        s = result["by_pillar"][p]
        if s.get("evaluable"):
            print(f"  {p:<20s} Hit@10={s['hit_at_10']:>5}%  "
                  f"Hit@100={s['hit_at_100']:>5}%  MRR={s['mrr']:.4f}  "
                  f"median={s['median_rank']}  (n={s['evaluable']}/{s['n']})")
        else:
            print(f"  {p:<20s} not evaluable")
    print(f"\nSaved: {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
