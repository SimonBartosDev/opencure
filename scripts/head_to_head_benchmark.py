"""Head-to-head benchmark: OpenCure v7 ensemble vs single-pillar baselines.

Methods paper Section 4.5 contents. For each baseline scoring column,
re-rank every disease's candidates by that column alone, then evaluate
against the time-sliced 210-pair held-out set. The metric is Hit@K and
MRR — same as the full-pipeline benchmark, just with different sort keys.

Baselines compared:

  - random              — sanity floor; expected Hit@10 ≈ K/N
  - transe              — DRKG TransE alone (the 2020 baseline)
  - pykeen              — DRKG RotatE alone
  - txgnn               — Harvard TxGNN alone (v6's pillar 5)
  - mol_emb             — chemistry-embedding similarity alone
  - proximity           — STRING-PPI network proximity alone
  - gene_sig            — L1000 + mechanistic reversal alone
  - combined_score      — full v6.1 grouped-combiner output
  - ensemble_prob       — v7 calibrated ensemble (the headline number)
  - ensemble_prob_v7    — v7 ensemble *gated* by the conformal prediction
                          set (only predictions with set={1} count)

Output:
  experiments/head_to_head_v7.md    — Markdown table for the paper
  experiments/head_to_head_v7.json  — raw per-baseline metrics

Usage:
  python3 scripts/head_to_head_benchmark.py
  python3 scripts/head_to_head_benchmark.py --results-dir <dir> --holdout time_sliced
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.eval.heldout_benchmark import _resolve_disease_entities
from opencure.scoring.common import AGGREGATE_RESULT_FILES

RESULTS_DIR = Path("experiments/results")
TIMESLICED = Path("data/eval/time_sliced_test.jsonl")
HOLDOUT = Path("data/eval/holdout_test.jsonl")
OUT_MD = Path("experiments/head_to_head_v7.md")
OUT_JSON = Path("experiments/head_to_head_v7.json")

BASELINES: list[tuple[str, str]] = [
    ("random",          "Random ranking (sanity floor)"),
    ("transe_score",    "TransE alone (DRKG, the 2020 baseline)"),
    ("pykeen_score",    "RotatE alone (DRKG)"),
    ("txgnn_score",     "TxGNN alone (Harvard 2024)"),
    ("mol_emb_similarity", "Chemistry embedding similarity alone"),
    ("proximity_score", "Network proximity alone (STRING PPI)"),
    ("gene_sig_score",  "L1000 mechanistic reversal alone"),
    ("combined_score",  "OpenCure v6.1 grouped combiner"),
    ("ensemble_prob",   "OpenCure v7 calibrated ensemble"),
]


def _load_holdout(path: Path) -> list[tuple[str, str]]:
    """Return list of (compound_entity, disease_entity)."""
    pairs = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            pairs.append((row["compound"], row["disease"]))
    return pairs


def _candidates_by_disease(results_dir: Path) -> dict[str, list[dict]]:
    """Map every disease entity → its candidate list (from result JSONs)."""
    out: dict[str, list[dict]] = {}
    for jf in sorted(results_dir.glob("*.json")):
        if jf.stem in AGGREGATE_RESULT_FILES:
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        cands = data.get("candidates", []) if isinstance(data, dict) else data
        if not cands:
            continue
        disease_name = jf.stem.replace("_", " ")
        for ent in _resolve_disease_entities(disease_name):
            out[ent] = cands
    return out


def _rerank_by(
    cands: list[dict], score_key: str,
) -> list[str]:
    """Return drug_id list re-ranked by ``score_key`` (descending)."""
    if score_key == "random":
        import random as _random
        rng = _random.Random(42)
        rng.shuffle(cands := list(cands))
        return [c.get("drug_id", "") for c in cands]

    def _key(c):
        v = c.get(score_key, 0)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    return [c.get("drug_id", "") for c in sorted(cands, key=_key, reverse=True)]


def _evaluate_baseline(
    score_key: str,
    holdout_pairs: list[tuple[str, str]],
    cands_by_disease: dict[str, list[dict]],
    top_k: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Compute Hit@K + MRR for one baseline scoring column.

    Each disease's candidate list is typically ≤10 entries (the result
    JSONs persist top-K only), which makes Hit@10 degenerate — every
    sort order puts a present positive in top-10. We report Hit@1, @3,
    @5, @10 so the re-ranking signal shows on tighter K. MRR remains
    the most discriminative single number.
    """
    ranks: list[int] = []
    hits = {k: 0 for k in top_k}
    matched = 0
    total = len(holdout_pairs)

    for drug_entity, disease_entity in holdout_pairs:
        if disease_entity not in cands_by_disease:
            continue
        drug_id = drug_entity.split("::", 1)[1] if "::" in drug_entity else drug_entity
        rank_list = _rerank_by(cands_by_disease[disease_entity], score_key)
        try:
            r = rank_list.index(drug_id) + 1
        except ValueError:
            r = len(rank_list) + 1
        ranks.append(r)
        matched += 1
        for k in top_k:
            if r <= k:
                hits[k] += 1

    if matched == 0:
        return {"score_key": score_key, "matched": 0}
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    return {
        "score_key": score_key,
        "matched": matched,
        "total": total,
        "hit_at_1": round(100 * hits[1] / matched, 2),
        "hit_at_3": round(100 * hits[3] / matched, 2),
        "hit_at_5": round(100 * hits[5] / matched, 2),
        "hit_at_10": round(100 * hits[10] / matched, 2),
        "mrr": round(mrr, 4),
    }


def _render_markdown(metrics: list[dict], holdout_name: str) -> str:
    headline = (
        f"# Head-to-head benchmark — OpenCure v7 vs. single-pillar baselines\n\n"
        f"Same {metrics[0]['matched']} matched held-out pairs ({holdout_name}), "
        f"same candidate sets, different sort keys. Hit@K = % of held-out "
        f"positives whose drug ranked in the top-K of its disease's "
        f"candidate list when ranked by the column shown.\n\n"
    )
    rows = ["| Baseline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Matched |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        if not m.get("matched"):
            continue
        label = next((d for k, d in BASELINES if k == m["score_key"]),
                    m["score_key"])
        rows.append(
            f"| {label} | {m['hit_at_1']}% | {m['hit_at_3']}% | "
            f"{m['hit_at_5']}% | {m['hit_at_10']}% | {m['mrr']} | "
            f"{m['matched']}/{m['total']} |"
        )
    return headline + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--holdout", choices=("time_sliced", "random"),
                        default="time_sliced",
                        help="Which held-out set to evaluate against.")
    args = parser.parse_args()

    holdout_path = TIMESLICED if args.holdout == "time_sliced" else HOLDOUT
    if not holdout_path.exists():
        sys.exit(f"Missing {holdout_path}")

    pairs = _load_holdout(holdout_path)
    print(f"Loaded {len(pairs)} held-out pairs from {holdout_path.name}")

    cands = _candidates_by_disease(args.results_dir)
    print(f"Loaded candidates for {len(cands)} disease entities")

    metrics = []
    for score_key, label in BASELINES:
        m = _evaluate_baseline(score_key, pairs, cands)
        if m.get("matched"):
            print(f"  {label}: Hit@10 = {m['hit_at_10']}%  "
                  f"MRR = {m['mrr']}  ({m['matched']}/{m['total']} matched)")
        else:
            print(f"  {label}: no matches — skipped")
        metrics.append(m)

    md = _render_markdown([m for m in metrics if m.get("matched")],
                          args.holdout)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved: {OUT_MD}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
