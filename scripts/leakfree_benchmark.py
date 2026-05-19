"""
Leak-free per-pillar held-out benchmark — the OpenCure verification instrument.

WHY THIS EXISTS
---------------
Every previous OpenCure "performance" number was contaminated: pillars were
scored from a knowledge graph that still contained the test edges. This script
is the honest replacement. It measures each *timeless* pillar — pillars whose
data is a fixed physical/measured property, not literature-derived — against a
held-out set, with the one possible leak path explicitly closed.

LEAK CONTROL
------------
A timeless similarity pillar scores a drug by similarity to a disease's *known
treatments*. The only way it can leak is if the held-out drug is itself in that
anchor set (it would then be "similar to itself"). `get_known_treatments` reads
anchors from TREATMENT_RELATIONS edges. So we strip, from the triplets the
pillar sees, every held-out (drug, disease) row in those relations. After
stripping, the held-out drug is a rankable candidate scored only against the
disease's *other* known treatments — genuinely leak-free.

The embedding model itself (ChemBERTa / MoLFormer) is pretrained on molecular
structure alone and has no temporal/literature component, so it cannot leak.

OUTPUT
------
experiments/eval/leakfree_pillar_scorecard.json — per-pillar leak-free
Hit@10/30/100, MRR, median rank, vs a popularity (graph-degree) baseline.
A pillar earns its place in the architecture only by beating that baseline.

Runtime: ~1-2 min on CPU.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOLDOUT = Path("data/eval/holdout_test.jsonl")
OUT = Path("experiments/eval/leakfree_pillar_scorecard.json")


def load_holdout() -> list[tuple[str, str]]:
    return [(d["compound"], d["disease"])
            for d in (json.loads(l) for l in HOLDOUT.open())]


def midrank(scores: np.ndarray, target_score: float) -> float:
    """Standard tie-aware rank (1-indexed): higher score = better rank."""
    higher = int(np.sum(scores > target_score))
    equal = int(np.sum(scores == target_score))
    return higher + equal / 2.0 + 1.0


def summarise(ranks: list[float], n_pairs: int) -> dict:
    if not ranks:
        return {"evaluable": 0, "n_pairs": n_pairs}
    a = np.array(ranks, dtype=float)
    return {
        "evaluable": len(a),
        "n_pairs": n_pairs,
        "hit_at_10": round(100 * float(np.mean(a <= 10)), 1),
        "hit_at_30": round(100 * float(np.mean(a <= 30)), 1),
        "hit_at_100": round(100 * float(np.mean(a <= 100)), 1),
        "mrr": round(float(np.mean(1.0 / a)), 4),
        "median_rank": int(np.median(a)),
    }


def main() -> None:
    from opencure.config import TREATMENT_RELATIONS
    from opencure.data.drkg import load_triplets
    from opencure.scoring.molecular import get_known_treatments
    from opencure.scoring.molecular_embeddings import (
        compute_cosine_similarity,
        load_best_molecular_embeddings,
    )

    heldout = load_holdout()
    heldout_set = set(heldout)
    by_disease: dict[str, list[str]] = defaultdict(list)
    for drug, dis in heldout:
        by_disease[dis].append(drug)
    print(f"Held-out pairs: {len(heldout)} across {len(by_disease)} diseases")

    # ---- leak control: strip held-out treatment edges -------------------
    triplets = load_triplets()
    treat_set = set(TREATMENT_RELATIONS)
    pair_idx = list(zip(triplets["head"], triplets["tail"]))
    is_heldout_pair = np.array([p in heldout_set for p in pair_idx])
    is_treat = triplets["relation"].isin(treat_set).to_numpy()
    strip_mask = is_heldout_pair & is_treat
    stripped = triplets[~strip_mask].reset_index(drop=True)
    print(f"Stripped {int(strip_mask.sum())} held-out treatment edges "
          f"({len(triplets)} -> {len(stripped)} triplets)")

    # popularity baseline: DRKG node degree
    degree: Counter[str] = Counter()
    for h, t in zip(triplets["head"], triplets["tail"]):
        degree[h] += 1
        degree[t] += 1

    # ---- molecular-embedding pillar (timeless) --------------------------
    embeddings, entities, model_name = load_best_molecular_embeddings()
    if embeddings is None:
        raise SystemExit("no molecular embeddings available")
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    all_compounds = [e for e in entities if e.startswith("Compound::")]
    print(f"Molecular embeddings: {model_name}, {len(all_compounds)} compounds")

    mol_ranks: list[float] = []
    pop_ranks: list[float] = []
    unevaluable = {"no_anchor": 0, "drug_no_embedding": 0}

    for dis, drugs in by_disease.items():
        known = set(get_known_treatments(dis, stripped))   # leak-free anchors
        known_with_emb = [c for c in known if c in entity_to_idx]
        if not known_with_emb:
            unevaluable["no_anchor"] += len(drugs)
            continue
        known_embs = embeddings[[entity_to_idx[c] for c in known_with_emb]]

        candidates = [c for c in all_compounds if c not in known]
        cand_idx = [entity_to_idx[c] for c in candidates]
        cand_embs = embeddings[cand_idx]
        # max cosine similarity of each candidate to the disease's anchors
        sim = compute_cosine_similarity(cand_embs, known_embs).max(axis=1)
        cand_pos = {c: i for i, c in enumerate(candidates)}
        cand_deg = np.array([degree.get(c, 0) for c in candidates], dtype=float)

        for drug in drugs:
            if drug not in cand_pos:
                unevaluable["drug_no_embedding"] += 1
                continue
            i = cand_pos[drug]
            mol_ranks.append(midrank(sim, sim[i]))
            pop_ranks.append(midrank(cand_deg, cand_deg[i]))

    scorecard = {
        "description": "Leak-free per-pillar held-out benchmark. Held-out "
                       "treatment edges stripped from anchor sets; ranks are "
                       "tie-aware mid-ranks against the full candidate pool.",
        "n_heldout_pairs": len(heldout),
        "leak_control": f"stripped {int(strip_mask.sum())} held-out "
                        f"TREATMENT_RELATIONS edges",
        "unevaluable": unevaluable,
        "pillars": {
            f"molecular_embedding ({model_name})": summarise(mol_ranks, len(heldout)),
            "popularity_baseline (DRKG degree)": summarise(pop_ranks, len(heldout)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2))

    print(f"\n{'='*64}\nLEAK-FREE PILLAR SCORECARD\n{'='*64}")
    for name, s in scorecard["pillars"].items():
        if s.get("evaluable"):
            print(f"  {name}")
            print(f"    Hit@10={s['hit_at_10']}%  Hit@30={s['hit_at_30']}%  "
                  f"Hit@100={s['hit_at_100']}%  MRR={s['mrr']}  "
                  f"median={s['median_rank']}  (n={s['evaluable']}/{s['n_pairs']})")
        else:
            print(f"  {name}: not evaluable")
    print(f"\n  unevaluable pairs: {unevaluable}")
    print(f"  Saved: {OUT}")
    print("\n  A pillar earns its place only by beating the popularity baseline.")


if __name__ == "__main__":
    main()
