"""
Build a DrugBank-indexed bioactivity lookup from ChEMBL 34.

Extracts median IC50 / Ki / Kd / EC50 (in nM) per (drug, target) pair for
every DrugBank-mapped compound. Used by stage-2 dose plausibility to
answer: at the drug's standard clinical dose (achievable Cmax), does the
drug reach inhibitory concentration at its predicted target?

Output: data/drkg/drug_target_activities.json
  {
    "DB00843": {                                 # drugbank_id
      "ACHE": {                                  # target symbol
        "n": 34,
        "median_nM": 6.3,
        "min_nM": 1.2,
        "max_nM": 50.0,
        "activity_types": ["IC50", "Ki"]
      },
      ...
    },
    ...
  }

Runtime: ~2 min on SSD (full-table scan of 3.8M rows).
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


DB = Path("data/sources_2024/chembl_34/chembl_34_sqlite/chembl_34.db")
OUT = Path("data/drkg/drug_target_activities.json")

# Build ChEMBL → DrugBank map from Open Targets molecule parquet
def _chembl_to_drugbank() -> dict[str, str]:
    import pyarrow.parquet as pq
    path = Path("data/open_targets/molecule")
    if not path.exists():
        raise SystemExit("Missing data/open_targets/molecule parquet")
    mp: dict[str, str] = {}
    for p in sorted(path.glob("*.parquet")):
        df = pq.read_table(p).to_pandas()
        for _, row in df.iterrows():
            cr = row.get("crossReferences")
            if cr is None:
                continue
            for src, ids in cr:
                if str(src).lower() == "drugbank":
                    try:
                        db = list(ids)[0]
                        if db.startswith("DB"):
                            mp[row["id"]] = db
                    except Exception:
                        pass
                    break
    return mp


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"Missing {DB}. Unpack ChEMBL 34 first.")

    print("Loading ChEMBL→DrugBank map from OT…")
    ch_to_db = _chembl_to_drugbank()
    print(f"  {len(ch_to_db):,} drugs mapped")

    if not ch_to_db:
        raise SystemExit("No ChEMBL→DrugBank mappings — check Open Targets data")

    print("Opening ChEMBL 34 SQLite (read-only)…")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Efficient one-shot SQL: join activities with molecule_dictionary
    # (chembl_id) and target_dictionary + component_sequences (symbol),
    # keep only numeric nM values for the 4 relevant activity types.
    query = """
    SELECT
        md.chembl_id   AS chembl_id,
        cs.component_synonym AS gene_symbol,
        a.standard_type      AS atype,
        a.standard_value     AS val_nm
    FROM activities a
    JOIN molecule_dictionary md ON md.molregno = a.molregno
    JOIN assays asy ON asy.assay_id = a.assay_id
    JOIN target_dictionary td ON td.tid = asy.tid
    JOIN target_components tc ON tc.tid = td.tid
    JOIN component_synonyms cs
        ON cs.component_id = tc.component_id
       AND cs.syn_type IN ('GENE_SYMBOL', 'HGNC_SYMBOL')
    WHERE a.standard_type IN ('IC50','Ki','Kd','EC50')
      AND a.standard_units = 'nM'
      AND a.standard_value IS NOT NULL
      AND a.standard_value > 0
      AND td.target_type = 'SINGLE PROTEIN'
    """

    print("Scanning activities (3.8M rows)…")
    t0 = time.time()

    # bucket per (drugbank_id, gene_symbol)
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"vals": [], "types": set()}
    )
    rows_seen = rows_kept = 0
    for row in cur.execute(query):
        rows_seen += 1
        chembl_id = row["chembl_id"]
        db_id = ch_to_db.get(chembl_id)
        if not db_id:
            continue
        gene = row["gene_symbol"]
        if not gene:
            continue
        val = row["val_nm"]
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v <= 0 or v > 1_000_000:  # drop implausible extremes (> 1 mM)
            continue
        b = buckets[(db_id, gene)]
        b["vals"].append(v)
        b["types"].add(row["atype"])
        rows_kept += 1

        if rows_seen % 500_000 == 0:
            print(f"  processed {rows_seen:,} rows, kept {rows_kept:,}  "
                  f"({time.time()-t0:.0f}s)")

    con.close()
    print(f"Done: {rows_seen:,} rows scanned, {rows_kept:,} kept, "
          f"{len(buckets):,} unique (drug, target) pairs  ({time.time()-t0:.0f}s)")

    # Summarize buckets
    out: dict[str, dict] = defaultdict(dict)
    for (db_id, gene), b in buckets.items():
        vals = b["vals"]
        if len(vals) < 1:
            continue
        vals_sorted = sorted(vals)
        out[db_id][gene] = {
            "n": len(vals),
            "median_nM": round(statistics.median(vals), 3),
            "min_nM":    round(vals_sorted[0], 3),
            "max_nM":    round(vals_sorted[-1], 3),
            "activity_types": sorted(b["types"]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f)

    n_drugs = len(out)
    n_pairs = sum(len(v) for v in out.values())
    print(f"Wrote {OUT}: {n_drugs:,} drugs × {n_pairs:,} drug-target pairs")

    # Sample output for verification
    for db_id in list(out.keys())[:3]:
        sample_targets = list(out[db_id].items())[:3]
        print(f"\n{db_id}:")
        for target, stats in sample_targets:
            print(f"  {target}: n={stats['n']}  median={stats['median_nM']} nM  "
                  f"types={stats['activity_types']}")


if __name__ == "__main__":
    main()
