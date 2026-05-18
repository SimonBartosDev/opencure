"""
Open Targets 24.09 data loader → DRKG-compatible triplets.

Parses the downloaded parquet files in data/open_targets/ and emits a TSV
of (head, relation, tail) triplets that unions cleanly with DRKG. Uses
HGNC mapping for Ensembl→Entrez so gene nodes merge with DRKG's Gene::
schema, and OT diseases.dbXRefs for EFO→MeSH so diseases merge too.

Pipeline:
  1. Load molecule → map ChEMBL→DrugBank via crossReferences ('drugbank')
  2. Load diseases → map EFO/MONDO/DOID → MeSH via dbXRefs
  3. Load targets → map Ensembl → Entrez via HGNC TSV
  4. Emit four relation families:
       OT::treats::Compound:Disease   (from indication, phase>=3)
       OT::mechanism::Compound:Gene   (from mechanismOfAction)
       OT::assoc::Gene:Disease        (from associationByOverallDirect, score>=0.3)
       OT::phase1-2::Compound:Disease (from indication, phase 1-2 — lower weight)
  5. Write TSV to data/open_targets/ot_triplets.tsv

Runtime: ~3 min on M-series for the parse stage.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path


OT_DIR = Path("data/open_targets")
HGNC_TSV = Path("data/mappings/hgnc_complete_set.txt")
OT_TRIPLETS_OUT = OT_DIR / "ot_triplets.tsv"


def load_chembl_to_drugbank() -> dict[str, str]:
    """ChEMBL molecule ID → DrugBank ID via molecule/*.parquet crossReferences."""
    import pyarrow.parquet as pq
    mp: dict[str, str] = {}
    for p in sorted((OT_DIR / "molecule").glob("*.parquet")):
        df = pq.read_table(p).to_pandas()
        for _, row in df.iterrows():
            cr = row.get("crossReferences")
            if cr is None:
                continue
            for src, ids in cr:
                if str(src).lower() == "drugbank":
                    try:
                        db_id = list(ids)[0]
                    except Exception:
                        continue
                    if db_id and db_id.startswith("DB"):
                        mp[row["id"]] = db_id
                        break
    return mp


def load_efo_to_mesh() -> dict[str, str]:
    """EFO/MONDO/DOID disease ID → MeSH ID via diseases/*.parquet dbXRefs."""
    import pyarrow.parquet as pq
    mp: dict[str, str] = {}
    for p in sorted((OT_DIR / "diseases").glob("*.parquet")):
        df = pq.read_table(p).to_pandas()
        for _, row in df.iterrows():
            refs = row.get("dbXRefs")
            if refs is None:
                continue
            for x in refs:
                xs = str(x)
                if xs.startswith("MeSH:") or xs.startswith("MESH:"):
                    mesh_id = xs.split(":", 1)[1]
                    mp[row["id"]] = f"MESH:{mesh_id}"
                    break
    return mp


def load_ensembl_to_entrez() -> dict[str, str]:
    """Ensembl gene ID → Entrez gene ID via HGNC complete set TSV."""
    mp: dict[str, str] = {}
    if not HGNC_TSV.exists():
        raise SystemExit(f"Missing {HGNC_TSV}. Download from HGNC first.")
    with HGNC_TSV.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        i_entrez = header.index("entrez_id")
        i_ensembl = header.index("ensembl_gene_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_entrez, i_ensembl):
                continue
            ens = parts[i_ensembl].strip()
            ent = parts[i_entrez].strip()
            if ens and ent and ens.startswith("ENSG"):
                mp[ens] = ent
    return mp


def write_triplets(
    chembl_to_db: dict[str, str],
    efo_to_mesh: dict[str, str],
    ens_to_entrez: dict[str, str],
) -> dict[str, int]:
    """Emit the four relation families to ot_triplets.tsv. Returns counts."""
    import pyarrow.parquet as pq

    OT_TRIPLETS_OUT.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    written: set[tuple[str, str, str]] = set()

    with OT_TRIPLETS_OUT.open("w") as out:
        # --- indication: drug → disease ---
        for p in sorted((OT_DIR / "indication").glob("*.parquet")):
            df = pq.read_table(p).to_pandas()
            for _, row in df.iterrows():
                db_id = chembl_to_db.get(row["id"])
                if not db_id:
                    continue
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
                        phase = 0
                    if phase >= 3:
                        rel = "OT::treats::Compound:Disease"
                    elif phase >= 1:
                        rel = "OT::trialed::Compound:Disease"
                    else:
                        continue
                    trip = (f"Compound::{db_id}", rel, f"Disease::{mesh}")
                    if trip not in written:
                        written.add(trip)
                        out.write("\t".join(trip) + "\n")
                        counts[rel] += 1

        # --- mechanismOfAction: drug → gene ---
        for p in sorted((OT_DIR / "mechanismOfAction").glob("*.parquet")):
            df = pq.read_table(p).to_pandas()
            for _, row in df.iterrows():
                ch_raw = row.get("chemblIds")
                tg_raw = row.get("targets")
                chembls = list(ch_raw) if ch_raw is not None else []
                targets = list(tg_raw) if tg_raw is not None else []
                action = str(row.get("actionType") or "").upper() or "MOA"
                rel = f"OT::{action}::Compound:Gene"
                for ch in chembls:
                    db_id = chembl_to_db.get(ch)
                    if not db_id:
                        continue
                    for ens in targets:
                        entrez = ens_to_entrez.get(ens)
                        if not entrez:
                            continue
                        trip = (f"Compound::{db_id}", rel, f"Gene::{entrez}")
                        if trip not in written:
                            written.add(trip)
                            out.write("\t".join(trip) + "\n")
                            counts[rel] += 1

        # --- associationByOverallDirect: gene ↔ disease, score>=0.3 ---
        for p in sorted((OT_DIR / "associationByOverallDirect").glob("*.parquet")):
            df = pq.read_table(p).to_pandas()
            df = df[df["score"] >= 0.3]
            for _, row in df.iterrows():
                entrez = ens_to_entrez.get(row["targetId"])
                mesh = efo_to_mesh.get(row["diseaseId"])
                if not entrez or not mesh:
                    continue
                rel = "OT::assoc::Gene:Disease"
                trip = (f"Gene::{entrez}", rel, f"Disease::{mesh}")
                if trip not in written:
                    written.add(trip)
                    out.write("\t".join(trip) + "\n")
                    counts[rel] += 1

    return dict(counts)


def main() -> None:
    t0 = time.time()
    print("Building mappings…")
    chembl_to_db = load_chembl_to_drugbank()
    print(f"  ChEMBL→DrugBank: {len(chembl_to_db):,} mappings")

    efo_to_mesh = load_efo_to_mesh()
    print(f"  EFO/MONDO/DOID→MeSH: {len(efo_to_mesh):,} mappings")

    ens_to_entrez = load_ensembl_to_entrez()
    print(f"  Ensembl→Entrez: {len(ens_to_entrez):,} mappings")

    print(f"\nWriting triplets to {OT_TRIPLETS_OUT}…")
    counts = write_triplets(chembl_to_db, efo_to_mesh, ens_to_entrez)

    print(f"\nDone in {time.time()-t0:.1f}s")
    total = 0
    for rel, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {n:>10,}  {rel}")
        total += n
    print(f"  {total:>10,}  TOTAL")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--print-download-plan":
        print("Download OT 24.09 parquet via:")
        print("  wget -r -np -nH --cut-dirs=7 ...")
    else:
        main()
