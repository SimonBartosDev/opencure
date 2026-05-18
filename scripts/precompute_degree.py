"""
Count DRKG triplet degree per drug entity.

Produces data/drkg/drug_degree.json: {drug_entity_or_db_id: int_count}.

Keys are stored in two forms for convenient lookup:
  - Full DRKG entity string (e.g. "Compound::DB00501")
  - Bare DrugBank ID (e.g. "DB00501")

Both map to the same integer — the number of triplets in which the drug appears
as either head or tail. Used by opencure/scoring/hub_normalize.py to damp the
KG/network pillars for high-degree "hub" drugs (Dexamethasone, Cimetidine, etc.)
that would otherwise dominate every disease's top-10.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DRKG_PATH = Path("data/drkg/drkg.tsv")
OUT_PATH = Path("data/drkg/drug_degree.json")


def main() -> None:
    if not DRKG_PATH.exists():
        raise SystemExit(f"Missing {DRKG_PATH}")

    counts: Counter[str] = Counter()
    with DRKG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, _, tail = parts[0], parts[1], parts[2]
            if head.startswith("Compound::"):
                counts[head] += 1
            if tail.startswith("Compound::"):
                counts[tail] += 1

    # Duplicate keys under bare DrugBank ID for convenience
    expanded: dict[str, int] = {}
    for ent, n in counts.items():
        expanded[ent] = n
        if "::" in ent:
            db_id = ent.split("::", 1)[1]
            # If multiple entity forms resolve to same id, keep max
            expanded[db_id] = max(expanded.get(db_id, 0), n)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(expanded, indent=2))

    top = counts.most_common(15)
    print(f"Wrote {OUT_PATH} with {len(counts)} compound entities ({len(expanded)} keys total)")
    print("Top 15 hub drugs by triplet degree:")
    for ent, n in top:
        print(f"  {n:>6}  {ent}")


if __name__ == "__main__":
    main()
