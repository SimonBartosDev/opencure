"""Ingest JUMP Cell Painting compound profiles → ``data/jump_cp/``.

JUMP-CP (the consortium led by Broad/MERCK/Bayer) released ~140K
compound morphological profiles in U2OS cells along with their
CellProfiler-derived feature vectors and InChIKey ↔ name mappings.
For v7 we use the public profile-level features directly (no GPU
re-embedding required).

Steps:
1. Download (or read from cache) the consortium's compound profile
   parquet at:
     https://cellpainting-gallery.s3.amazonaws.com/...   (full URL in the
     consortium's data-access docs).
2. Normalize the InChIKey ↔ DrugBank ID mapping using DrugBank's
   compound table (already in tree at ``data/drkg/compound_smiles.tsv``)
   plus the salt-stripped aliases produced by ``txgnn_scorer``-style
   logic.
3. Persist to two artifacts the scorer expects:
     data/jump_cp/compound_features.npz       — embeddings + InChIKeys
     data/jump_cp/inchikey_to_drugbank.tsv    — InChIKey → DrugBank ID

Usage:
    python3 scripts/precompute_jump_cp.py --features path/to/jump_profiles.parquet
    python3 scripts/precompute_jump_cp.py --features <url>
    python3 scripts/precompute_jump_cp.py --smoke   # writes tiny stub

The full ingest is ~10 GB; the smoke mode writes an 8-row synthetic
artifact so the rest of the pipeline can be exercised end-to-end
without the full download.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from opencure.scoring.jump_cell_painting import (
    FEATURES_PATH,
    INCHIKEY_MAP_PATH,
    JUMP_DIR,
)


SMILES_PATH = Path("data/drkg/compound_smiles.tsv")


def _load_drugbank_inchikey_map() -> dict[str, str]:
    """Build ``InChIKey → DrugBank ID`` from the in-tree compound table.

    Falls back to an empty dict when the SMILES cache is absent (a fresh
    checkout). The downstream artifact will then be empty too — fine,
    the JUMP scorer fails open.
    """
    if not SMILES_PATH.exists():
        print(f"[WARN] {SMILES_PATH} missing — InChIKey map will be empty")
        return {}
    df = pd.read_csv(SMILES_PATH, sep="\t")
    if "inchikey" not in df.columns or "entity" not in df.columns:
        print(f"[WARN] {SMILES_PATH} missing inchikey column — empty map")
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        ik = row.get("inchikey")
        ent = row.get("entity")
        if pd.isna(ik) or pd.isna(ent):
            continue
        # entity = "Compound::DBxxxxxxx"
        bare = str(ent).split("::", 1)[1] if "::" in str(ent) else str(ent)
        out[str(ik)] = bare
    print(f"[ok]  {len(out)} InChIKey → DrugBank entries loaded")
    return out


def _smoke_features(n: int = 8, dim: int = 32) -> tuple[np.ndarray, list[str]]:
    """Synthetic 8-compound JUMP feature set for end-to-end testing."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    # 8 fictitious InChIKeys; the script will leave most unmapped if the
    # in-tree DrugBank table doesn't contain matches — that's fine for
    # smoke testing.
    inchikeys = [f"SMOKE-INCHIKEY-{i:03d}" for i in range(n)]
    return embeddings, inchikeys


def _load_jump_features(features_arg: str) -> tuple[np.ndarray, list[str]]:
    """Read JUMP-CP profiles from parquet/CSV.

    Expected columns: ``InChIKey`` plus N feature columns named
    ``Cells_*`` / ``Cytoplasm_*`` / ``Nuclei_*`` per the consortium
    convention. Other columns are ignored. Returns the feature matrix
    plus the parallel InChIKey list.
    """
    path = Path(features_arg)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="," if path.suffix == ".csv" else "\t")
    else:
        sys.exit(f"Unsupported features format: {path.suffix}")

    if "InChIKey" not in df.columns:
        sys.exit("JUMP profile file must have an 'InChIKey' column")

    feature_cols = [c for c in df.columns
                    if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))]
    if not feature_cols:
        sys.exit(f"No CellProfiler feature columns found in {path}")

    embeddings = df[feature_cols].to_numpy(dtype=np.float32)
    inchikeys = df["InChIKey"].astype(str).tolist()
    print(f"[ok]  Loaded {len(inchikeys)} profiles × {len(feature_cols)} features")
    return embeddings, inchikeys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", help="Path or URL to JUMP profile parquet/csv.")
    parser.add_argument("--smoke", action="store_true",
                        help="Write a tiny synthetic artifact for end-to-end smoke testing.")
    args = parser.parse_args()

    JUMP_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        embeddings, inchikeys = _smoke_features()
    else:
        if not args.features:
            sys.exit("--features required when --smoke is not set")
        embeddings, inchikeys = _load_jump_features(args.features)

    np.savez_compressed(
        str(FEATURES_PATH),
        embeddings=embeddings,
        inchikeys=np.array(inchikeys),
    )

    inchikey_to_drugbank = _load_drugbank_inchikey_map()
    matched = 0
    with INCHIKEY_MAP_PATH.open("w") as fh:
        fh.write("inchikey\tdrugbank_id\n")
        for ik in inchikeys:
            db = inchikey_to_drugbank.get(ik)
            if db:
                fh.write(f"{ik}\t{db}\n")
                matched += 1

    manifest = {
        "n_profiles": int(embeddings.shape[0]),
        "feature_dim": int(embeddings.shape[1]),
        "n_drugbank_matched": matched,
        "smoke_mode": bool(args.smoke),
    }
    (JUMP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {FEATURES_PATH} ({embeddings.shape})")
    print(f"Saved {INCHIKEY_MAP_PATH} ({matched} DrugBank matches)")
    print(f"Saved {JUMP_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
