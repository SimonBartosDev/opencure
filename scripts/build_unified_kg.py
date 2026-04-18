"""
Union DRKG + PrimeKG + Open Targets 24.09 into one triplet TSV.

After running:
  - opencure.data.open_targets download + parse (→ data/open_targets/ot_triplets.tsv)
  - PrimeKG is already loaded at data/primekg/kg.csv

This script produces data/unified_kg/unified.tsv, deduplicated by
(head, relation, tail). Relations are namespaced (DRKG::*, PRIMEKG::*,
OT::*) so the origin is preserved.

Next step after this: scripts/train_unified_rotate.py kicks off a PyKEEN
RotatE training run (~6 h on Apple Silicon MPS) → embeddings saved to
data/models/unified_rotatE/.

The search pipeline will then fuse the new embedding via kg_fusion.
"""

from __future__ import annotations

from pathlib import Path


DRKG_PATH = Path("data/drkg/drkg.tsv")
PRIMEKG_PATH = Path("data/primekg/kg.csv")
OT_PATH = Path("data/open_targets/ot_triplets.tsv")
OUT_PATH = Path("data/unified_kg/unified.tsv")


def main() -> None:
    if not DRKG_PATH.exists():
        raise SystemExit(f"Missing {DRKG_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    n_drkg = n_pkg = n_ot = 0

    with OUT_PATH.open("w") as out:
        # DRKG
        print(f"Loading {DRKG_PATH}…")
        with DRKG_PATH.open() as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    t = tuple(parts)
                    if t not in seen:
                        seen.add(t)
                        out.write("\t".join(parts) + "\n")
                        n_drkg += 1

        # PrimeKG — CSV with header, columns: relation, display_relation, x_id, x_type, ...
        if PRIMEKG_PATH.exists():
            print(f"Loading {PRIMEKG_PATH}…")
            import csv
            with PRIMEKG_PATH.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = f"{row.get('x_type','?')}::{row.get('x_id','')}"
                    r = f"PRIMEKG::{row.get('relation','?')}"
                    t_ = f"{row.get('y_type','?')}::{row.get('y_id','')}"
                    if "?" in (h + t_) or not row.get("x_id") or not row.get("y_id"):
                        continue
                    trip = (h, r, t_)
                    if trip not in seen:
                        seen.add(trip)
                        out.write("\t".join(trip) + "\n")
                        n_pkg += 1

        # Open Targets
        if OT_PATH.exists():
            print(f"Loading {OT_PATH}…")
            with OT_PATH.open() as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        t = tuple(parts)
                        if t not in seen:
                            seen.add(t)
                            out.write("\t".join(parts) + "\n")
                            n_ot += 1

    print()
    print(f"Wrote {OUT_PATH}")
    print(f"  DRKG:    {n_drkg:>10,}")
    print(f"  PrimeKG: {n_pkg:>10,}")
    print(f"  OT:      {n_ot:>10,}")
    print(f"  Total:   {len(seen):>10,}  (deduped)")


if __name__ == "__main__":
    main()
