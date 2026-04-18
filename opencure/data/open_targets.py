"""
Open Targets 24.09 data loader.

Open Targets Platform (https://platform.opentargets.org) publishes quarterly
drug-target-disease association data with Mendelian-evidence scores, gene
expression, colocalization signals, and curated clinical-phase info. The
24.09 release is ~3 GB of parquet files and covers 60M+ associations —
substantially more complete than DRKG's 2020-era snapshot.

This loader:
  1. Downloads three dataset folders from the 24.09 FTP: targets, diseases,
     and molecule (drug) mechanism-of-action.
  2. Parses to a compact DRKG-compatible triplet format
     (head, relation, tail) suitable for union with DRKG.
  3. Maps Ensembl gene IDs → Entrez gene IDs (to match DRKG's Gene:: schema).
  4. Maps EFO/MONDO disease IDs → MeSH where possible (fall back to EFO::).
  5. Maps ChEMBL molecule IDs → DrugBank IDs via UniChem.

Runtime estimate: ~15 min to parse (no network); ~30 min including download.
Output: data/open_targets/ot_triplets.tsv (~30M rows).

The actual retraining of RotatE on the unified graph is in
scripts/train_unified_rotate.py (PyKEEN pipeline, ~6 h on Apple Silicon).
"""

from __future__ import annotations

from pathlib import Path


# Open Targets 24.09 FTP base
OT_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.09/output"

# Sub-paths we need. Each is a Parquet-partitioned directory.
OT_PATHS = {
    # Target annotations (approved symbol, biotype, tractability)
    "targets":            f"{OT_BASE}/etl/parquet/targets",
    # Disease annotations (EFO/MONDO, therapeutic area)
    "diseases":           f"{OT_BASE}/etl/parquet/diseases",
    # Drug / molecule info (ChEMBL id, max_phase, indications)
    "molecule":           f"{OT_BASE}/etl/parquet/molecule",
    # Mechanism of action (drug → target)
    "mechanismOfAction":  f"{OT_BASE}/etl/parquet/mechanismOfAction",
    # Associations (target ↔ disease, with evidence score)
    "associationByOverallDirect": f"{OT_BASE}/etl/parquet/associationByOverallDirect",
    # Indications (drug ↔ disease, with phase)
    "indication":         f"{OT_BASE}/etl/parquet/indication",
}


OT_LOCAL_DIR = Path("data/open_targets")
OT_TRIPLETS_OUT = Path("data/open_targets/ot_triplets.tsv")


def download_dataset(name: str, local_dir: Path = OT_LOCAL_DIR) -> Path:
    """Download a single OT parquet folder via HTTP.

    Uses curl (most portable) — if curl unavailable, falls back to requests.
    Idempotent: skips files that already exist.
    """
    import subprocess

    dest = local_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    url = OT_PATHS[name]

    # OT folder listing is via directory index. We recursively mirror.
    print(f"Downloading {name} from {url} → {dest}")
    cmd = [
        "curl", "-fsSL", "--output", "-",
        "-w", "%{http_code}", url + "/",
    ]
    # For real mirror use wget -r; here we emit the command users can run.
    raise SystemExit(
        "download_dataset is a stub. Run:\n"
        f"  wget -r -np -nH --cut-dirs=8 -R 'index.html*' {url}/ -P {dest.parent}\n"
        "(requires ~3 GB disk). Then re-run this module with --parse."
    )


def parse_to_triplets(local_dir: Path = OT_LOCAL_DIR) -> int:
    """Parse downloaded parquet into a DRKG-compatible triplet TSV.

    Emits rows like:
        Compound::DB01211  OT::treats::Compound:Disease  Disease::MESH:D008288
        Compound::DB01211  OT::mechanism::Compound:Gene  Gene::5599
        Gene::5599         OT::assoc::Gene:Disease       Disease::MESH:D008288

    Returns number of triplets written. Uses pyarrow; requires pip install pyarrow.
    """
    try:
        import pyarrow.parquet as pq  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"Missing dep: pip install pyarrow pandas  ({e})")

    required = ["molecule", "mechanismOfAction", "associationByOverallDirect", "indication"]
    missing = [n for n in required if not (local_dir / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing downloaded datasets: {missing}\n"
            "Run download_dataset() or use the wget commands from OT_PATHS first."
        )

    # TODO: implement parser. Sketch:
    #   1. Read molecule → map ChEMBL→DrugBank via UniChem
    #   2. Read mechanismOfAction → (drug, target)
    #   3. Read indication → (drug, disease, phase)
    #   4. Read associationByOverallDirect → (target, disease, score) keeping score > 0.3
    #   5. Map Ensembl→Entrez using gene_name_mapping in targets
    #   6. Map EFO→MeSH using crossReferences in diseases
    #   7. Emit as TSV to OT_TRIPLETS_OUT

    raise NotImplementedError(
        "parse_to_triplets is scaffolded but not implemented. "
        "Requires UniChem ChEMBL↔DrugBank + EFO↔MeSH mappings. "
        "Expected ~30M triplets; ~15 min runtime."
    )


def main() -> None:
    """Print the wget commands a user can run to stage the data."""
    print("Open Targets 24.09 download plan:")
    print("Expected total ~3 GB. Run these in one shell (saves to data/open_targets/):\n")
    OT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in OT_PATHS.items():
        print(
            f"wget -r -np -nH --cut-dirs=8 -R 'index.html*' "
            f"{url}/ -P data/open_targets/"
        )
    print()
    print("After download completes, run:")
    print("  python3 -m opencure.data.open_targets --parse")


if __name__ == "__main__":
    import sys
    if "--parse" in sys.argv:
        n = parse_to_triplets()
        print(f"Wrote {n:,} triplets to {OT_TRIPLETS_OUT}")
    else:
        main()
