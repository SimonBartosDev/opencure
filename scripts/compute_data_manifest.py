"""
Compute a versioned manifest of every data source OpenCure depends on.

Emits data/manifest.json with a SHA-256 hash of each input file + size +
mtime. Systematic_screening writes the manifest's combined hash into
every result JSON as `data_version`, so every prediction is traceable
to the exact data version that produced it.

Regenerate whenever a data file changes:
    python3 scripts/compute_data_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


TRACK = [
    # DRKG
    "data/drkg/drkg.tsv",
    "data/drkg/compound_smiles.tsv",
    "data/drkg/drug_names_cache.tsv",
    "data/drkg/chembl_phase.json",
    "data/drkg/admet_predictions.json",
    "data/drkg/drug_degree.json",
    "data/drkg/drug_target_activities.json",
    # Open Targets
    "data/open_targets/ot_triplets.tsv",
    # PrimeKG
    "data/primekg/kg.csv",
    # 2024 sources
    "data/mappings/hgnc_complete_set.txt",
    "data/sources_2024/cpic_pairs.json",
    "data/sources_2024/pharmgkb/clinical_annotations.tsv",
    "data/sources_2024/gtex/gtex_median_tpm.gct",
    # Unified / clean KG artifacts
    "data/unified_kg/unified.tsv",
    "data/unified_kg/unified_train_clean.tsv",
]

OUT = Path("data/manifest.json")


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA-256 over large files without loading them."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    entries = {}
    for rel in TRACK:
        p = Path(rel)
        if not p.exists():
            entries[rel] = {"present": False}
            continue
        st = p.stat()
        # For very large files (>500MB), hash only mtime + size to stay quick
        if st.st_size > 500 * 1024 * 1024:
            key = f"size={st.st_size};mtime={int(st.st_mtime)}"
            digest = hashlib.sha256(key.encode()).hexdigest()
            entries[rel] = {
                "present": True,
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
                "sha256": digest,
                "note": "fast-hash (size+mtime) due to file size > 500MB",
            }
        else:
            entries[rel] = {
                "present": True,
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
                "sha256": _hash_file(p),
            }

    # Combined manifest hash — stable ordering
    combined = hashlib.sha256()
    for rel in sorted(entries.keys()):
        e = entries[rel]
        if e.get("present") and e.get("sha256"):
            combined.update(rel.encode())
            combined.update(e["sha256"].encode())

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "opencure_version": "v5",
        "manifest_hash": combined.hexdigest()[:16],
        "files": entries,
    }


def main() -> None:
    t0 = time.time()
    manifest = build_manifest()
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUT} in {time.time()-t0:.1f}s")
    print(f"  manifest_hash: {manifest['manifest_hash']}")
    n_present = sum(1 for e in manifest["files"].values() if e.get("present"))
    print(f"  {n_present}/{len(TRACK)} tracked files present")


if __name__ == "__main__":
    main()
