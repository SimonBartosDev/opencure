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

        # PrimeKG — CSV with cols relation,display_relation,x_index,x_id,x_type,x_name,x_source,...
        # Map entity types to DRKG-compatible form so drug/gene nodes merge:
        #   gene/protein + NCBI source  →  Gene::<entrez>
        #   drug + DrugBank source      →  Compound::DB<id>
        #   disease                     →  Disease::MONDO_<index>  (won't merge, but trainable)
        #   (other types kept with their native prefix)
        if PRIMEKG_PATH.exists():
            print(f"Loading {PRIMEKG_PATH}…")
            import csv

            def primekg_entity(typ: str, ident: str, source: str) -> str:
                typ = (typ or "").strip()
                ident = (ident or "").strip()
                source = (source or "").strip()
                if not ident:
                    return ""
                if typ == "gene/protein":
                    # PrimeKG x_id for NCBI is the entrez ID directly
                    return f"Gene::{ident}"
                if typ == "drug":
                    # x_id is DrugBank accession when source==DrugBank
                    if source == "DrugBank" and ident.startswith("DB"):
                        return f"Compound::{ident}"
                    return f"Compound::{source}:{ident}"
                if typ == "disease":
                    return f"Disease::MONDO_{ident.split('_')[0].strip().replace(' ','')}"
                # Default: keep native PrimeKG type
                safe_typ = typ.replace(" ", "_").replace("/", "_")
                return f"{safe_typ}::{ident}"

            with PRIMEKG_PATH.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = primekg_entity(row.get("x_type", ""), row.get("x_id", ""), row.get("x_source", ""))
                    t_ = primekg_entity(row.get("y_type", ""), row.get("y_id", ""), row.get("y_source", ""))
                    if not h or not t_:
                        continue
                    r = f"PRIMEKG::{row.get('relation','unknown')}"
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
