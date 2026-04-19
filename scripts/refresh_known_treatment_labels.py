"""
Backfill: re-run ``is_known_treatment`` on every candidate in every v5
result JSON using the DRKG treats-edge lookup (``KNOWN_TREATMENT_RELATIONS``).
Pre-fix runs used a heuristic that required 5+ trials AND 500+ PubMed
articles, yielding 1/90 positives (Oxamniquine for Schistosomiasis —
the flagship positive control — was missed).

Usage
-----
    python3 scripts/refresh_known_treatment_labels.py
    python3 scripts/refresh_known_treatment_labels.py Malaria
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.evidence.novelty import is_known_treatment


RESULTS_DIR = Path("experiments/results")


def _resolve_disease_entity(disease_name: str) -> str | None:
    try:
        from opencure.data.drkg import load_embeddings, find_disease_entities
        _, _, ent2id, _, _ = load_embeddings()
        matches = find_disease_entities(ent2id, disease_name)
        if matches:
            m = matches[0]
            return m[0] if isinstance(m, tuple) else m
    except Exception:
        pass
    return None


def backfill(path: Path) -> tuple[int, int]:
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0, 0
    disease_name = data.get("disease") or path.stem.replace("_", " ")
    # Resolve disease_entity (use existing one if present, else DRKG search)
    disease_entity = data.get("disease_entity")
    if not disease_entity:
        for c in candidates:
            if c.get("disease_entity"):
                disease_entity = c["disease_entity"]
                break
    if not disease_entity:
        disease_entity = _resolve_disease_entity(disease_name)
    if not disease_entity:
        return 0, 0
    data["disease_entity"] = disease_entity

    n_known = 0
    for cand in candidates:
        was = cand.get("is_known_treatment", False)
        cand["disease_entity"] = disease_entity
        label = is_known_treatment({
            "drug_id": cand.get("drug_id", ""),
            "disease_entity": disease_entity,
            "pubmed_total": cand.get("pubmed_total", 0),
            "clinical_trials_total": cand.get("clinical_trials_total", 0),
        })
        cand["is_known_treatment"] = label
        if label:
            n_known += 1
    json.dump(data, path.open("w"), indent=2)
    return n_known, len(candidates)


def main() -> None:
    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in {"screening_summary", "novel_candidates",
                                          "opencure_database"})
    total_k = 0
    total_n = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        k, n = backfill(f)
        total_k += k
        total_n += n
        print(f"  {f.name}: {k}/{n} known-treatment positives")
    print(f"\nDone. {total_k}/{total_n} candidates flagged is_known_treatment.")


if __name__ == "__main__":
    main()
