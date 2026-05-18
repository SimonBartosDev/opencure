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

    Slow path — used only when batch ID-mapping fails. Each call is
    one HTTP round-trip plus a 0.3s polite sleep regardless of outcome.
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


# ---- Fast path: UniProt batch ID-mapping API ----------------------------

def _batch_resolve_entrez_to_uniprot(
    entrez_ids: list[str],
    *,
    chunk_size: int = 5000,
) -> dict[str, str]:
    """Convert entrez Gene IDs → UniProt accessions via UniProt's bulk API.

    UniProt's ``/idmapping`` endpoint accepts up to 100K IDs per job; we
    chunk at 5K to keep individual jobs fast and visible in the polling
    output. Returns ``{entrez: uniprot_acc}``; entrez IDs without a
    reviewed human Swiss-Prot entry are absent from the result.

    Reference: https://www.uniprot.org/help/id_mapping
    """
    out: dict[str, str] = {}
    for i in range(0, len(entrez_ids), chunk_size):
        chunk = entrez_ids[i : i + chunk_size]
        try:
            resp = requests.post(
                "https://rest.uniprot.org/idmapping/run",
                data={"from": "GeneID", "to": "UniProtKB-Swiss-Prot",
                      "ids": ",".join(chunk)},
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"  [warn] idmapping/run returned {resp.status_code}; "
                      f"falling back for chunk {i // chunk_size}")
                continue
            job_id = resp.json().get("jobId")
            if not job_id:
                continue
        except Exception as exc:
            print(f"  [warn] idmapping submit failed: {exc}")
            continue

        # Poll until the job finishes — usually 1-10 s for chunks ≤ 5K.
        for _ in range(60):
            try:
                status = requests.get(
                    f"https://rest.uniprot.org/idmapping/status/{job_id}",
                    timeout=15,
                ).json()
            except Exception:
                time.sleep(2)
                continue
            if status.get("results") is not None or status.get("jobStatus") == "FINISHED":
                break
            time.sleep(2)

        # Fetch results, paged.
        page_url = (
            f"https://rest.uniprot.org/idmapping/results/{job_id}"
            "?format=tsv&fields=from,accession&size=500"
        )
        while page_url:
            try:
                page = requests.get(page_url, timeout=30)
            except Exception:
                break
            if page.status_code != 200:
                break
            lines = page.text.strip().split("\n")
            # First line is header
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1]:
                    out.setdefault(parts[0], parts[1])
            # Pagination via Link header
            page_url = None
            link = page.headers.get("link", "")
            if 'rel="next"' in link:
                # link header looks like: <https://...next>; rel="next"
                import re
                m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
                if m:
                    page_url = m.group(1)

        print(f"  chunk {i // chunk_size + 1}: {len(out)} accessions resolved cumulatively")
    return out


def _batch_fetch_sequences(
    accessions: list[str],
    *,
    chunk_size: int = 100,
    sleep: float = 0.5,
) -> dict[str, str]:
    """Fetch UniProt FASTA sequences for a list of accessions, in chunks.

    Returns ``{accession: sequence}``. Skips entries shorter than 50 aa
    (likely fragments) to match the single-fetch path's behaviour.
    """
    out: dict[str, str] = {}
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i : i + chunk_size]
        url = ("https://rest.uniprot.org/uniprotkb/accessions"
               f"?accessions={','.join(chunk)}&format=fasta")
        try:
            resp = requests.get(url, timeout=60)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        # FASTA blocks separated by `>` headers.
        block_acc = None
        block_lines: list[str] = []
        for line in resp.text.split("\n"):
            if line.startswith(">"):
                if block_acc is not None and block_lines:
                    seq = "".join(block_lines)
                    if len(seq) >= 50:
                        out[block_acc] = seq
                # New header; e.g. ">sp|P04637|P53_HUMAN ..."
                hdr = line.split("|")
                block_acc = hdr[1] if len(hdr) >= 2 else None
                block_lines = []
            elif block_acc is not None:
                block_lines.append(line.strip())
        if block_acc is not None and block_lines:
            seq = "".join(block_lines)
            if len(seq) >= 50:
                out[block_acc] = seq
        time.sleep(sleep)
    return out


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
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only resolve sequences; skip ESM-2 inference.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable UniProt batch ID-mapping; use slow per-gene fetch.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"),
                        default="auto",
                        help="Force a torch device. ESM-2 on Apple MPS hits a "
                             "PyTorch hang in the rotary-embedding kernel; use "
                             "'cpu' for reliable local inference (~4h for 30K "
                             "sequences). Default 'auto' picks the fastest "
                             "available device.")
    args = parser.parse_args()

    model_name, dim = ESM2_VARIANTS[args.variant]
    device = args.device if args.device != "auto" else _default_device()
    if device == "mps":
        print("[warn] ESM-2 on Apple MPS hangs in PyTorch's rotary-embedding "
              "kernel — pass --device cpu if you see no progress within 5 min.")
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

    # Split into cached vs to-fetch.
    resolved: list[tuple[str, str]] = []
    to_fetch: list[str] = []
    for gene in genes:
        if gene in seq_cache:
            resolved.append((gene, seq_cache[gene]))
        else:
            to_fetch.append(gene)
    print(f"Need to fetch: {len(to_fetch)}")

    if to_fetch and not args.no_batch:
        # --- Fast path: bulk ID-mapping + bulk sequence fetch ---------
        # Build the list of unique entrez IDs (the part after 'Gene::').
        entrez_to_genes: dict[str, list[str]] = {}
        for gene in to_fetch:
            entrez = gene.split("::", 1)[1].split(";")[0]
            entrez_to_genes.setdefault(entrez, []).append(gene)
        unique_entrez = list(entrez_to_genes)
        print(f"Submitting {len(unique_entrez)} entrez IDs to UniProt batch ID-mapping...")
        entrez_to_acc = _batch_resolve_entrez_to_uniprot(unique_entrez)
        print(f"  resolved {len(entrez_to_acc)} → UniProt accessions")

        if entrez_to_acc:
            unique_accs = sorted(set(entrez_to_acc.values()))
            print(f"Fetching {len(unique_accs)} sequences in batches of 100...")
            acc_to_seq = _batch_fetch_sequences(unique_accs)
            print(f"  fetched {len(acc_to_seq)} sequences")

            for entrez, acc in entrez_to_acc.items():
                seq = acc_to_seq.get(acc)
                if not seq:
                    continue
                for gene in entrez_to_genes.get(entrez, []):
                    _append_seq_cache(gene, seq)
                    resolved.append((gene, seq))
        # Anything still un-resolved is left for the slow path below
        # (only triggers when batch mapping returns nothing useful).

    elif to_fetch and args.no_batch:
        # --- Slow path (legacy): one HTTP call per gene -----------------
        fetched = 0
        for gene in tqdm(to_fetch, desc="resolve sequences (slow)"):
            seq = _fetch_uniprot_sequence(gene, entrez_to_symbol)
            time.sleep(0.3)  # always sleep, regardless of outcome
            if seq is not None:
                _append_seq_cache(gene, seq)
                resolved.append((gene, seq))
                fetched += 1
        print(f"  fetched {fetched} via slow path")

    print(f"Sequences resolved: {len(resolved)} / {len(genes)}")

    if not resolved:
        sys.exit("No sequences resolved; aborting before embedding step.")

    if args.fetch_only:
        print("--fetch-only set; skipping ESM-2 embedding step.")
        return

    # --- Step 3: ESM-2 inference, in resumable chunks ----------------
    # Each chunk persists immediately so a kill at hour 3 still leaves
    # everything-up-to-now on disk. The next run picks up where this
    # one left off via the chunk-progress sentinel.
    from opencure.scoring.dti_predictor import (
        get_esm2_embeddings, _cache_path_for, save_protein_embeddings,
    )

    gene_ids, sequences = zip(*resolved)
    gene_ids = list(gene_ids)
    sequences = list(sequences)
    print(f"\nComputing ESM-2 {args.variant} embeddings for {len(sequences)} proteins "
          f"(chunked, resumable)...", flush=True)

    final_npz = _cache_path_for(args.variant)
    progress_npz = final_npz.with_suffix(".partial.npz")

    # Resume: if a partial NPZ exists, load it and skip those gene_ids.
    done_genes: set[str] = set()
    chunk_arrays: list[np.ndarray] = []
    chunk_genes: list[str] = []
    if progress_npz.exists():
        try:
            partial = np.load(str(progress_npz), allow_pickle=True)
            chunk_arrays.append(partial["embeddings"])
            chunk_genes.extend(partial["gene_ids"].tolist())
            done_genes = set(chunk_genes)
            print(f"  resuming from {len(done_genes)} previously-embedded proteins",
                  flush=True)
        except Exception as exc:
            print(f"  [warn] could not load partial NPZ ({exc}); starting fresh",
                  flush=True)

    chunk_size = 500  # ~10 batches of 8 worth of work between saves
    total = len(sequences)
    pending = [(g, s) for g, s in zip(gene_ids, sequences) if g not in done_genes]
    print(f"  pending: {len(pending)} (already done: {len(done_genes)})",
          flush=True)

    start = time.time()
    for ci in range(0, len(pending), chunk_size):
        chunk = pending[ci : ci + chunk_size]
        c_genes, c_seqs = zip(*chunk)
        chunk_emb = get_esm2_embeddings(
            list(c_seqs),
            model_name=model_name,
            batch_size=args.batch_size,
            device=device,
        )
        chunk_arrays.append(chunk_emb)
        chunk_genes.extend(c_genes)

        # Persist progress after every chunk.
        merged = np.vstack(chunk_arrays)
        np.savez_compressed(
            str(progress_npz),
            embeddings=merged,
            gene_ids=np.array(chunk_genes),
        )
        elapsed = time.time() - start
        done = len(chunk_genes) - len(done_genes)
        eta_min = (elapsed / max(done, 1)) * (len(pending) - done) / 60
        print(f"  chunk {ci // chunk_size + 1}: total={len(chunk_genes)} "
              f"({100 * len(chunk_genes) / total:.1f}% of {total}); "
              f"ETA {eta_min:.1f}m", flush=True)

    # --- Step 4: finalize -----------------------------------------------
    embeddings = np.vstack(chunk_arrays)
    print(f"  {embeddings.shape} in {time.time() - start:.1f}s", flush=True)
    save_protein_embeddings(embeddings, chunk_genes, variant=args.variant)
    if progress_npz.exists():
        progress_npz.unlink()
    print(f"Saved to {final_npz}", flush=True)


if __name__ == "__main__":
    main()
