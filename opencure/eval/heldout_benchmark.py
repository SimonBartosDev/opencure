"""
Held-out benchmark harness.

Two modes:

1. "fast" — evaluates the TransE embedding pillar alone on held-out pairs
   by ranking held-out drugs against random negatives. Seconds per 1000
   pairs. Useful for quick iteration on embedding changes.

2. "pipeline" — evaluates the full OpenCure pipeline by looking up each
   held-out (drug, disease) pair in a previously-computed screening result
   directory. Pipeline is not re-run; we measure where the held-out drug
   landed in the ranked output. Catches filter/combiner regressions.

Metrics: Hit@10, Hit@50, Hit@100, MRR, AUROC.

IMPORTANT — contamination disclosure:
The current TransE/RotatE/PrimeKG embeddings were trained on DRKG INCLUDING
the `treats` edges we now hold out. This inflates Hit@K. Until Phase 5
retrains on an edge-stripped graph, report these as "training-contaminated"
upper bounds. The benchmark is still useful for:
  - Comparing v3 vs v4 (both contaminated equally)
  - Detecting regressions (a drop in Hit@K under the same contamination
    still signals a real problem)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from opencure.eval.ground_truth import load_holdout


_ENTITY_TO_ID: Optional[dict] = None


def _get_entity_to_id() -> dict:
    global _ENTITY_TO_ID
    if _ENTITY_TO_ID is not None:
        return _ENTITY_TO_ID
    try:
        from opencure.data.drkg import load_embeddings
        _, _, entity_to_id, _, _ = load_embeddings()
        _ENTITY_TO_ID = entity_to_id
    except Exception as e:
        print(f"  could not load embeddings: {e}")
        _ENTITY_TO_ID = {}
    return _ENTITY_TO_ID


def _resolve_disease_entities(disease_name: str) -> set[str]:
    """Resolve a disease display name to its set of DRKG MeSH entities."""
    try:
        from opencure.data.drkg import find_disease_entities
        ent2id = _get_entity_to_id()
        if not ent2id:
            return set()
        matches = find_disease_entities(ent2id, disease_name)
        # find_disease_entities returns [(entity, score), ...]
        entities = {m[0] for m in matches} if matches else set()
        return entities
    except Exception:
        return set()


def evaluate_pipeline_results(
    results_dir: Path,
    holdout_pairs: Optional[list[tuple[str, str]]] = None,
    top_k: tuple[int, ...] = (10, 50, 100),
) -> dict:
    """
    Score the full pipeline on held-out pairs, using existing result JSONs.

    For each held-out (drug_entity, disease_entity) pair:
      - find the matching disease result file
      - look up the drug's rank in the ranked candidate list
      - record hits at each K threshold

    Returns summary metrics dict.
    """
    if holdout_pairs is None:
        holdout_pairs = load_holdout()

    # Build disease-entity -> candidate list from result JSONs.
    # Each file may map to MULTIPLE MeSH entities (aliases); we record all.
    results: dict[str, list[str]] = {}  # disease_entity -> [drug_id_in_rank_order]
    for jf in sorted(results_dir.glob("*.json")):
        if any(x in jf.name for x in ("opencure", "screening", "novel")):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        cands = data.get("candidates", []) if isinstance(data, dict) else data
        if not cands:
            continue
        disease_name = jf.stem.replace("_", " ")
        entities = _resolve_disease_entities(disease_name)
        if not entities:
            continue
        drug_order = [c.get("drug_id", "") for c in cands if c.get("drug_id")]
        for ent in entities:
            results[ent] = drug_order

    print(f"  Loaded {len(results)} disease-entity keys with ranked drug lists")

    # Score each held-out pair
    ranks: list[int] = []  # 1-indexed rank; len(list)+1 if not found
    hits = {k: 0 for k in top_k}
    matched = 0
    total = 0
    per_disease: dict[str, dict] = {}

    for drug_entity, disease_entity in holdout_pairs:
        total += 1
        if disease_entity not in results:
            continue
        drug_id = drug_entity.split("::", 1)[1] if "::" in drug_entity else drug_entity
        rank_list = results[disease_entity]
        try:
            r = rank_list.index(drug_id) + 1  # 1-indexed
        except ValueError:
            r = len(rank_list) + 1
        ranks.append(r)
        matched += 1
        for k in top_k:
            if r <= k:
                hits[k] += 1
        d_entry = per_disease.setdefault(disease_entity, {"tested": 0, "hit10": 0})
        d_entry["tested"] += 1
        if r <= 10:
            d_entry["hit10"] += 1

    if matched == 0:
        return {
            "error": "No held-out pairs matched any disease result — check disease name map",
            "total_pairs": total,
            "matched": 0,
        }

    mrr = sum(1.0 / r for r in ranks) / len(ranks)

    return {
        "total_holdout_pairs": total,
        "matched_to_results": matched,
        "coverage_pct": round(100 * matched / total, 1),
        "hit_at": {k: round(100 * hits[k] / matched, 2) for k in top_k},
        "mrr": round(mrr, 4),
        "diseases_scored": len(per_disease),
        "note": "Training-contaminated upper bound (embeddings saw these edges).",
    }


def main() -> None:
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results")
    print(f"Evaluating results in: {d}")
    metrics = evaluate_pipeline_results(d)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
