"""
Pre-compute ESM-2 protein embeddings for every DRKG ``Gene::`` entity.

Mirrors ``scripts/precompute_embeddings.py`` for chemistry. Output is a
versioned ``.npz`` cache that ``dti_predictor.load_best_protein_embeddings()``
auto-discovers.

Usage:
    # Default: 150M variant (640-dim), MPS auto-detect, ~4-6 h on M4 Max.
    python3 scripts/precompute_esm2_embeddings.py

    # Smaller / smoke-test:
    python3 scripts/precompute_esm2_embeddings.py --variant 8M --limit 50

    # CUDA users with the budget:
    python3 scripts/precompute_esm2_embeddings.py --variant 650M

The fetch step calls UniProt's REST API (~20K genes × 0.3 s rate-limit ≈ 100 min).
A sequence-fetch cache lives at ``data/drkg/embeddings/_uniprot_sequence_cache.tsv``
so subsequent runs skip the network entirely.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make sibling package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from opencure.config import DATA_DIR
from opencure.scoring.dti_predictor import (
    ESM2_VARIANTS,
    _default_device,
    get_esm2_embeddings,
    save_protein_embeddings,
)

DRKG_PATH = DATA_DIR / "drkg.tsv"
SEQ_CACHE = DATA_DIR / "embeddings" / "_uniprot_sequence_cache.tsv"


def _load_drkg_genes(limit: int | None = None) -> list[str]:
    """Return unique ``Gene::<entrez_id>`` entities present in DRKG."""
    if not DRKG_PATH.exists():
        sys.exit(f"DRKG not found at {DRKG_PATH}; run scripts/fetch_drkg.py first.")

    # Stream-read just the columns we need; DRKG is large.
    df = pd.read_csv(DRKG_PATH, sep="\t", names=["head", "rel", "tail"], usecols=["head", "tail"])
    heads = df["head"][df["head"].str.startswith("Gene::", na=False)]
    tails = df["tail"][df["tail"].str.startswith("Gene::", na=False)]
    genes = pd.concat([heads, tails]).drop_duplicates().tolist()
    if limit:
        genes = genes[:limit]
    return genes


def _load_seq_cache() -> dict[str, str]:
    if not SEQ_CACHE.exists():
        return {}
    df = pd.read_csv(SEQ_CACHE, sep="\t")
    return dict(zip(df["gene"], df["sequence"]))


def _append_seq_cache(gene: str, sequence: str) -> None:
    SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SEQ_CACHE.exists()
    with SEQ_CACHE.open("a") as f:
        if write_header:
            f.write("gene\tsequence\n")
        f.write(f"{gene}\t{sequence}\n")


def _fetch_uniprot_sequence(gene_entity: str, entrez_to_symbol: dict[str, str]) -> str | None:
    """Resolve ``Gene::<entrez>`` → UniProt sequence via UniProt REST.

    Returns the sequence string or ``None`` if no canonical reviewed
    human entry exists. Honours UniProt's polite-rate convention.
    """
    entrez = gene_entity.split("::", 1)[1].split(";")[0]
    symbol = entrez_to_symbol.get(entrez, entrez)

    try:
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
                "format": "fasta",
                "size": 1,
            },
            timeout=10,
        )
    except Exception:
        return None
    if resp.status_code != 200 or ">" not in resp.text:
        return None

    lines = resp.text.strip().split("\n")
    sequence = "".join(lines[1:])
    if len(sequence) < 50:
        return None
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=tuple(ESM2_VARIANTS),
        default="150M",
        help="ESM-2 variant. 150M is the M4-Max sweet spot.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N genes (smoke test).")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model_name, dim = ESM2_VARIANTS[args.variant]
    device = _default_device()
    print(f"ESM-2 {args.variant} ({model_name}, {dim}-dim) on {device}")

    # --- Step 1: enumerate DRKG genes ---------------------------------
    genes = _load_drkg_genes(limit=args.limit)
    print(f"DRKG has {len(genes)} unique Gene:: entities")

    # --- Step 2: resolve sequences (with persistent cache) ------------
    seq_cache = _load_seq_cache()
    print(f"Sequence cache hit: {len(seq_cache)} / {len(genes)}")

    try:
        from opencure.scoring.mendelian_randomization import _load_entrez_to_symbol
        entrez_to_symbol = _load_entrez_to_symbol()
    except Exception:
        entrez_to_symbol = {}

    resolved: list[tuple[str, str]] = []
    fetched = 0
    for gene in tqdm(genes, desc="resolve sequences"):
        if gene in seq_cache:
            resolved.append((gene, seq_cache[gene]))
            continue
        seq = _fetch_uniprot_sequence(gene, entrez_to_symbol)
        if seq is not None:
            _append_seq_cache(gene, seq)
            resolved.append((gene, seq))
            fetched += 1
            time.sleep(0.3)  # polite rate
    print(f"Sequences resolved: {len(resolved)} ({fetched} freshly fetched)")

    if not resolved:
        sys.exit("No sequences resolved; aborting before embedding step.")

    # --- Step 3: ESM-2 inference --------------------------------------
    gene_ids, sequences = zip(*resolved)
    print(f"\nComputing ESM-2 {args.variant} embeddings for {len(sequences)} proteins...")
    start = time.time()
    embeddings = get_esm2_embeddings(
        list(sequences),
        model_name=model_name,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"  {embeddings.shape} in {time.time() - start:.1f}s")

    # --- Step 4: save -------------------------------------------------
    save_protein_embeddings(embeddings, list(gene_ids), variant=args.variant)
    print(f"Saved to {DATA_DIR}/embeddings/protein_embeddings_esm2_{args.variant}.npz")


if __name__ == "__main__":
    main()
