"""
Build a unified disease → gene-symbols index by merging:

  1. Open Targets  (``data/open_targets/ot_triplets.tsv``, relation
     ``OT::assoc::Gene:Disease``) — curated, sparse for NTDs.
  2. DRKG GNBR     (``data/drkg/drkg.tsv``, relations
     ``GNBR::{D,G,J,L,Md,Te,U,Ud,X,Y}::Gene:Disease``) — text-mined,
     broader coverage but noisier.

Output: ``data/disease_gene_index.json`` mapping
``Disease::MESH:Dxxxxxx`` → sorted list of HGNC gene symbols.

Coverage typically jumps from ~0 to 20-100 genes for neglected-tropical
diseases that have thin OT curation (Schistosomiasis, Leishmaniasis,
Chagas, etc).

Run once after any data refresh; post-processors (tissue_context,
docking_proxy) read the cache and stay fast.

Usage
-----
    python3 scripts/build_disease_gene_index.py            # build + save
    python3 scripts/build_disease_gene_index.py --verify   # sanity-print
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


DRKG_PATH = Path("data/drkg/drkg.tsv")
OT_PATH = Path("data/open_targets/ot_triplets.tsv")
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")
OUTPUT_PATH = Path("data/disease_gene_index.json")


GNBR_GENE_DISEASE_RELATIONS = frozenset({
    "GNBR::D::Gene:Disease",    # drug-target / marker for disease
    "GNBR::G::Gene:Disease",    # genetic variation
    "GNBR::J::Gene:Disease",    # role in pathogenesis
    "GNBR::L::Gene:Disease",    # biomarker
    "GNBR::Md::Gene:Disease",   # model of disease
    "GNBR::Te::Gene:Disease",   # therapeutic effect
    "GNBR::U::Gene:Disease",    # gene causes/influences
    "GNBR::Ud::Gene:Disease",   # under-expression in disease
    "GNBR::X::Gene:Disease",    # over-expression in disease
    "GNBR::Y::Gene:Disease",    # polymorphism alters expression
})
OT_RELATION = "OT::assoc::Gene:Disease"


def load_entrez_to_symbol() -> dict[str, str]:
    if not HGNC_PATH.exists():
        raise SystemExit(f"Missing {HGNC_PATH}")
    mapping: dict[str, str] = {}
    with HGNC_PATH.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            ent = (row.get("entrez_id") or "").strip()
            sym = (row.get("symbol") or "").strip()
            if ent and sym:
                mapping[ent] = sym
    return mapping


def stream_disease_genes(
    path: Path, relations: frozenset[str], ent_to_sym: dict[str, str],
) -> dict[str, set[str]]:
    """Scan a TSV and return disease_entity → gene-symbol set."""
    import pandas as pd  # deferred — heavy import
    out: dict[str, set[str]] = {}
    for chunk in pd.read_csv(
        path, sep="\t", header=None,
        names=["head", "relation", "tail"], chunksize=500_000,
    ):
        m = chunk["relation"].isin(relations)
        sub = chunk[m]
        for _, row in sub.iterrows():
            gene_ent, dis_ent = row["head"], row["tail"]
            # Some OT edges may be oriented Disease→Gene — handle both.
            if gene_ent.startswith("Disease::") and dis_ent.startswith("Gene::"):
                gene_ent, dis_ent = dis_ent, gene_ent
            if not gene_ent.startswith("Gene::") or not dis_ent.startswith("Disease::"):
                continue
            entrez = gene_ent.split("::", 1)[1]
            sym = ent_to_sym.get(entrez)
            if not sym:
                continue
            out.setdefault(dis_ent, set()).add(sym)
    return out


def merge_dicts(*dicts: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for d in dicts:
        for k, v in d.items():
            merged.setdefault(k, set()).update(v)
    return merged


def verify_coverage(index: dict[str, list[str]], sample_diseases: list[tuple[str, str]]) -> None:
    print("\nCoverage sanity check:")
    for name, entity in sample_diseases:
        n = len(index.get(entity, []))
        status = "✓" if n >= 5 else ("~" if n >= 1 else "✗")
        sample_genes = (index.get(entity) or [])[:5]
        print(f"  {status} {name:30s} {entity:30s}  n={n:4d}  {sample_genes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        if not OUTPUT_PATH.exists():
            raise SystemExit(f"No index at {OUTPUT_PATH}. Build it first "
                             "(run without --verify).")
        index = json.loads(OUTPUT_PATH.read_text())
    else:
        print("Loading HGNC entrez→symbol mapping…")
        ent_to_sym = load_entrez_to_symbol()
        print(f"  {len(ent_to_sym):,} Entrez→symbol entries")

        t0 = time.time()
        print(f"Scanning OT triplets ({OT_PATH})…")
        ot = stream_disease_genes(OT_PATH, frozenset({OT_RELATION}), ent_to_sym) \
             if OT_PATH.exists() else {}
        print(f"  {sum(len(v) for v in ot.values()):,} gene-disease pairs across "
              f"{len(ot):,} diseases from OT ({time.time()-t0:.0f}s)")

        t0 = time.time()
        print(f"Scanning DRKG GNBR ({DRKG_PATH})…")
        drkg = stream_disease_genes(DRKG_PATH, GNBR_GENE_DISEASE_RELATIONS, ent_to_sym) \
               if DRKG_PATH.exists() else {}
        print(f"  {sum(len(v) for v in drkg.values()):,} gene-disease pairs across "
              f"{len(drkg):,} diseases from DRKG/GNBR ({time.time()-t0:.0f}s)")

        merged = merge_dicts(ot, drkg)
        # Persist as sorted lists (stable JSON diffs, smaller size).
        index = {k: sorted(v) for k, v in merged.items()}
        OUTPUT_PATH.write_text(json.dumps(index, separators=(",", ":")))
        print(f"\nSaved {len(index):,} diseases × "
              f"{sum(len(v) for v in index.values()):,} gene associations → {OUTPUT_PATH}")

    sample = [
        ("Schistosomiasis",  "Disease::MESH:D012552"),
        ("Leishmaniasis",    "Disease::MESH:D007896"),
        ("Chagas disease",   "Disease::MESH:D014355"),
        ("Malaria",          "Disease::MESH:D008288"),
        ("Tuberculosis",     "Disease::MESH:D014376"),
        ("Sickle cell",      "Disease::MESH:D000755"),
        ("Alzheimer's",      "Disease::MESH:D000544"),
    ]
    verify_coverage(index, sample)


if __name__ == "__main__":
    main()
