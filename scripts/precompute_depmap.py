"""Ingest DepMap gene essentiality → ``data/depmap/gene_essentiality.tsv``.

DepMap publishes a per-gene × per-cell-line Chronos essentiality matrix.
For v7 we don't need the full matrix — just per-gene summaries:

    gene_symbol, median_chronos, fraction_essential_lines

Where ``fraction_essential_lines`` = fraction of cell lines whose
Chronos score is < ESSENTIAL_THRESHOLD (-0.5).

Steps:
1. Read the DepMap CRISPR effect CSV (per-line × per-gene Chronos).
   Default expected location is ``--src`` argument; consortium
   distributes ``CRISPR_(DepMap_Public_*)_v2.csv`` files at
   https://depmap.org/portal/download/
2. Compute the two summary stats per gene.
3. Write the TSV the loader expects.

Usage:
    python3 scripts/precompute_depmap.py --src CRISPRGeneEffect.csv
    python3 scripts/precompute_depmap.py --smoke    # tiny synthetic stub
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from opencure.scoring.depmap_essentiality import (
    ESSENTIAL_THRESHOLD,
    ESSENTIALITY_PATH,
)


def _smoke_table() -> pd.DataFrame:
    """Hand-built stub: a few well-known essential and non-essential genes."""
    return pd.DataFrame([
        # Pan-essential (housekeeping) — should warn in non-oncology contexts
        ("RPL5", -1.5, 0.98),
        ("POLR2A", -1.4, 0.97),
        ("RPS6", -1.3, 0.96),
        # Cancer-context essential (selectively)
        ("MYC", -0.9, 0.65),
        ("KRAS", -0.7, 0.55),
        # Non-essential — safe targets
        ("EGFR", 0.05, 0.05),
        ("CFTR", 0.10, 0.02),
        ("HBB", 0.08, 0.01),  # Sickle cell target
        ("GBA", 0.04, 0.03),  # Gaucher
        ("NPC1", 0.06, 0.04),  # Niemann-Pick C
    ], columns=["gene_symbol", "median_chronos", "fraction_essential_lines"])


def _summarise_chronos(matrix_path: Path) -> pd.DataFrame:
    """Read the consortium CRISPR effect matrix and produce per-gene summaries."""
    df = pd.read_csv(matrix_path, index_col=0)  # rows = cell lines, cols = genes
    # Some columns include "(EntrezID)" suffix — strip it.
    df.columns = [c.split(" ")[0] for c in df.columns]

    median = df.median(axis=0)
    n_lines = df.shape[0]
    fraction = (df < ESSENTIAL_THRESHOLD).sum(axis=0) / n_lines

    return pd.DataFrame({
        "gene_symbol": median.index,
        "median_chronos": median.values,
        "fraction_essential_lines": fraction.values,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", help="DepMap CRISPRGeneEffect.csv (rows=lines, cols=genes)")
    parser.add_argument("--smoke", action="store_true",
                        help="Write a tiny synthetic table for end-to-end smoke testing.")
    args = parser.parse_args()

    if args.smoke:
        df = _smoke_table()
    else:
        if not args.src:
            sys.exit("--src required when --smoke is not set")
        df = _summarise_chronos(Path(args.src))

    ESSENTIALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ESSENTIALITY_PATH, sep="\t", index=False)
    print(f"Saved {len(df)} gene essentiality records to {ESSENTIALITY_PATH}")
    n_warning = (df["fraction_essential_lines"] >= 0.80).sum()
    print(f"  {n_warning} genes flagged as broadly essential (≥80% of lines)")


if __name__ == "__main__":
    main()
