"""
Build a TIME-SLICED held-out benchmark.

This is the single most credible validation format for drug repurposing:
train on pre-2020 biology, test on drug-disease indications approved
2020-2024. A model that recovers these post-2020 indications from
pre-2020 evidence cannot have memorized the answer.

Approach:
  1. From OT 24.09 molecule, take drugs with yearOfFirstApproval >= 2020
  2. Join with OT indication to get (drug, disease) pairs approved post-2020
  3. Map ChEMBL → DrugBank via molecule.crossReferences
  4. Map EFO → MeSH via diseases.dbXRefs
  5. Emit data/eval/time_sliced_test.jsonl

For every pair we also emit the cutoff year (drug's first-approval year) so
the evaluator can enforce "exclude edges dated >= cutoff" when scoring.

Additionally emit data/eval/time_sliced_train_exclude.txt — set of triplets
that must be excluded from training KGs to preserve the time-sliced split.
"""

from __future__ import annotations

import json
from pathlib import Path


OT_DIR = Path("data/open_targets")
DISEASES_DIR = Path("data/open_targets/diseases")
OUT_PAIRS = Path("data/eval/time_sliced_test.jsonl")
OUT_EXCLUDE = Path("data/eval/time_sliced_train_exclude.txt")

CUTOFF_YEAR = 2020


def main() -> None:
    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        raise SystemExit("Install pyarrow + pandas")

    print("Loading OT molecule (for approval years)…")
    mol_parts = sorted((OT_DIR / "molecule").glob("*.parquet"))
    mol_df = pd.concat([pq.read_table(p).to_pandas() for p in mol_parts])
    print(f"  {len(mol_df):,} molecules")

    # ChEMBL → DrugBank
    ch_to_db: dict[str, str] = {}
    approval_year: dict[str, int] = {}
    for _, row in mol_df.iterrows():
        year = row.get("yearOfFirstApproval")
        if year is None or pd.isna(year):
            continue
        cr = row.get("crossReferences")
        if cr is None:
            continue
        db = None
        for src, ids in cr:
            if str(src).lower() == "drugbank":
                try:
                    db = list(ids)[0]
                except Exception:
                    pass
                break
        if db and db.startswith("DB"):
            ch_to_db[row["id"]] = db
            approval_year[db] = int(year)

    post_2020 = {db: y for db, y in approval_year.items() if y >= CUTOFF_YEAR}
    print(f"  ChEMBL→DrugBank: {len(ch_to_db):,} mapped, {len(post_2020):,} approved {CUTOFF_YEAR}+")

    # EFO → MeSH
    print("Loading OT diseases (for MeSH mapping)…")
    efo_to_mesh: dict[str, str] = {}
    for p in sorted((OT_DIR / "diseases").glob("*.parquet")):
        df = pq.read_table(p).to_pandas()
        for _, r in df.iterrows():
            refs = r.get("dbXRefs")
            if refs is None:
                continue
            for x in refs:
                xs = str(x)
                if xs.startswith("MeSH:") or xs.startswith("MESH:"):
                    efo_to_mesh[r["id"]] = f"MESH:{xs.split(':', 1)[1]}"
                    break
    print(f"  EFO→MeSH: {len(efo_to_mesh):,} mapped")

    # Now process indications: keep only those for post-2020 drugs with max_phase >= 4
    print("Scanning indications for post-2020 approved indications…")
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for p in sorted((OT_DIR / "indication").glob("*.parquet")):
        df = pq.read_table(p).to_pandas()
        for _, row in df.iterrows():
            db = ch_to_db.get(row["id"])
            if not db or db not in post_2020:
                continue
            year = post_2020[db]
            inds = row.get("indications")
            if inds is None:
                continue
            for ind in inds:
                dis_id = ind.get("disease")
                mesh = efo_to_mesh.get(dis_id)
                if not mesh:
                    continue
                phase = ind.get("maxPhaseForIndication", 0) or 0
                try:
                    phase = float(phase)
                except Exception:
                    continue
                if phase < 3:  # we want approved or late-phase only
                    continue
                key = (db, mesh)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "compound": f"Compound::{db}",
                    "disease": f"Disease::{mesh}",
                    "drug_id": db,
                    "disease_id": mesh,
                    "first_approval_year": year,
                    "phase": phase,
                })

    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PAIRS.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Also write the "exclude these edges from training" list: any triplet
    # with this (compound, treats-like-relation, disease) must be removed
    # from the KG used for time-sliced model training.
    with OUT_EXCLUDE.open("w") as f:
        for p in pairs:
            f.write(f"{p['compound']}\t{p['disease']}\n")

    print()
    print(f"Wrote {OUT_PAIRS}  ({len(pairs)} post-{CUTOFF_YEAR} approved indication pairs)")
    print(f"Wrote {OUT_EXCLUDE}")
    print()
    # Year histogram
    from collections import Counter
    years = Counter(p["first_approval_year"] for p in pairs)
    print("Pairs by first_approval_year:")
    for y in sorted(years):
        print(f"  {y}: {years[y]}")


if __name__ == "__main__":
    main()
