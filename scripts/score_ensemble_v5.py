"""
Post-processor: attach ``ensemble_prob`` and ``ensemble_rank`` to every
candidate in every v5 result JSON.

Wires the trained ``data/models/ensemble_v5.pkl`` model (XGBoost + isotonic
calibration, AUC-ROC 0.9968 ± 0.0004 in 5-fold CV on 23,814 pairs) into the
inference path. Until this script runs, results carry no calibrated
probability — only the grouped-combiner ``combined_score``.

Feature extraction mirrors ``scripts/phase_c_pipeline.py`` exactly so
train/serve skew stays at zero.

Usage
-----
    python3 scripts/score_ensemble_v5.py                      # all result JSONs
    python3 scripts/score_ensemble_v5.py Malaria Chagas_disease
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.data.drkg import load_embeddings, get_compound_entities
from opencure.scoring.common import AGGREGATE_RESULT_FILES
from opencure.scoring.ensemble import (
    MODEL_PATH,
    build_features,
    load_model,
    score as score_candidate,
)
from opencure.scoring.hub_normalize import degree_penalty
from opencure.scoring.transe import score_drugs_for_disease_vectorized


ACTIVITIES_PATH = Path("data/drkg/drug_target_activities.json")
CHEMBL_PHASE_PATH = Path("data/drkg/chembl_phase.json")
OT_TRIPLETS_PATH = Path("data/open_targets/ot_triplets.tsv")
RESULTS_DIR = Path("experiments/results")


_DISEASE_NAME_MAP: dict[str, str] | None = None


def _resolve_disease_entity(disease_name: str, ent2id: dict) -> str | None:
    """Best-effort name→entity lookup via DRKG's find_disease_entities."""
    try:
        from opencure.data.drkg import find_disease_entities
        matches = find_disease_entities(ent2id, disease_name)
        if matches:
            return matches[0][0] if isinstance(matches[0], tuple) else matches[0]
    except Exception as exc:
        print(f"  [warn] disease resolve failed for {disease_name!r}: {exc}")
    return None


def load_drug_target_count() -> dict[str, int]:
    if not ACTIVITIES_PATH.exists():
        return {}
    acts = json.load(ACTIVITIES_PATH.open())
    return {k: len(v) for k, v in acts.items()}


def load_chembl_phase() -> dict[str, float]:
    if not CHEMBL_PHASE_PATH.exists():
        return {}
    return json.load(CHEMBL_PHASE_PATH.open())


def count_disease_associated_genes() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not OT_TRIPLETS_PATH.exists():
        return counts
    with OT_TRIPLETS_PATH.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if "Disease" in h:
                counts[h] = counts.get(h, 0) + 1
    return counts


def score_file(
    path: Path, model, feature_keys, ent_emb, rel_emb, ent2id, rel2id,
    compounds, drug_n_targets, chembl_phase, disease_gene_counts,
) -> int:
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0
    # Resolve disease_entity: try top-level, else first candidate's disease_entity,
    # else fall back to name→entity lookup via disease pool, else skip.
    disease_entity = data.get("disease_entity")
    if not disease_entity:
        for c in candidates:
            if c.get("disease_entity"):
                disease_entity = c["disease_entity"]; break
    if not disease_entity:
        disease_entity = _resolve_disease_entity(
            data.get("disease", path.stem.replace("_", " ")), ent2id,
        )
    if not disease_entity or disease_entity not in ent2id:
        print(f"  {path.name}: no disease_entity resolvable, skipping")
        return 0

    # Rank every compound once for this disease
    try:
        ranked = score_drugs_for_disease_vectorized(
            disease_entity, ent_emb, rel_emb, ent2id, rel2id, compounds,
            ["DRUGBANK::treats::Compound:Disease"],
        )
    except Exception as exc:
        print(f"  {path.name}: transE score failed: {exc}")
        return 0
    rank_map = {c: rk for rk, (c, _, _) in enumerate(ranked)}
    n = len(ranked)
    n_disease_genes = disease_gene_counts.get(disease_entity, 0)

    probs = []
    for cand in candidates:
        drug_id = cand.get("drug_id", "")
        compound = drug_id if drug_id.startswith("Compound::") else f"Compound::{drug_id}"
        feats = build_features(
            compound_entity=compound,
            disease_entity=disease_entity,
            rank_map=rank_map,
            n_compounds=n,
            drug_n_targets=drug_n_targets,
            chembl_phase=chembl_phase,
            disease_gene_counts={disease_entity: n_disease_genes},
            degree_penalty_fn=degree_penalty,
        )
        p = score_candidate(model, feature_keys, feats)
        cand["ensemble_prob"] = round(p, 4)
        cand["ensemble_features"] = {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in feats.items()}
        probs.append((cand, p))

    # Stable secondary rank by ensemble_prob (does not re-sort existing output)
    probs.sort(key=lambda x: -x[1])
    for rnk, (cand, _) in enumerate(probs, 1):
        cand["ensemble_rank"] = rnk

    data["ensemble_version"] = "v5"
    data["ensemble_model_path"] = str(MODEL_PATH)

    json.dump(data, path.open("w"), indent=2)
    return len(candidates)


def main() -> None:
    try:
        model, feature_keys = load_model()
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {feature_keys}")

    print("Loading DRKG embeddings...")
    ent_emb, rel_emb, ent2id, id2ent, rel2id = load_embeddings()
    compounds = get_compound_entities(ent2id, drugbank_only=True)
    print(f"  {len(ent2id):,} entities, {len(compounds):,} compounds")

    print("Loading feature helpers...")
    drug_n_targets = load_drug_target_count()
    chembl_phase = load_chembl_phase()
    disease_gene_counts = count_disease_associated_genes()

    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in AGGREGATE_RESULT_FILES)

    total = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name} not found")
            continue
        n = score_file(f, model, feature_keys, ent_emb, rel_emb, ent2id, rel2id,
                       compounds, drug_n_targets, chembl_phase, disease_gene_counts)
        print(f"  {f.name}: {n} candidates scored")
        total += n
    print(f"\nDone. {total} candidates across {len(files)} files carry ensemble_prob.")


if __name__ == "__main__":
    main()
