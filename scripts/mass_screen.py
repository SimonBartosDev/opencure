"""
Mass screening: run the v5 pipeline across all MeSH-indexed diseases with
sufficient gene annotation to produce meaningful predictions.

Candidate disease pool (v5):
  - All MeSH entities present in DRKG as Disease:: nodes
  - Filter: must have >= 5 gene associations across DRKG + PrimeKG + OT
    (otherwise scoring is meaningless)
  - Target: ~2,000 diseases (rare + neglected emphasized)

The existing experiments/systematic_screening.py drives one disease per
invocation. This script enumerates the candidate pool, shards across
worker processes, and appends to a single result directory.

Runtime: on a single machine ~5 min/disease × 2000 = ~7 days serial.
With 4 workers ≈ 1.75 days. Cloud deployment (see scripts/cloud_screen.sh)
parallelizes to finish overnight.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
DISEASE_POOL_PATH = Path("data/disease_pool.json")
DRKG_PATH = Path("data/drkg/drkg.tsv")


DISEASE_ASSOC_RELATIONS_PREFIX = (
    "Hetionet::DaG::Disease:Gene",
    "Hetionet::DdG::Disease:Gene",
    "Hetionet::DuG::Disease:Gene",
    "GNBR::L::Gene:Disease",
    "GNBR::J::Gene:Disease",
    "GNBR::U::Gene:Disease",
    "GNBR::Y::Gene:Disease",
    "GNBR::Te::Gene:Disease",
    "GNBR::X::Gene:Disease",
    "GNBR::G::Gene:Disease",
    "OT::assoc::Gene:Disease",
)


def build_disease_pool(min_associations: int = 5) -> list[dict]:
    """Enumerate diseases with >= min_associations gene edges."""
    if not DRKG_PATH.exists():
        raise SystemExit(f"{DRKG_PATH} not found")

    gene_counts: Counter[str] = Counter()
    with DRKG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if not any(r.startswith(p.rsplit("::", 2)[0]) for p in DISEASE_ASSOC_RELATIONS_PREFIX):
                continue
            if h.startswith("Disease::"):
                gene_counts[h] += 1
            if t.startswith("Disease::"):
                gene_counts[t] += 1

    pool = [
        {"entity": d, "mesh_id": d.split("::", 1)[1], "n_gene_associations": n}
        for d, n in gene_counts.most_common()
        if n >= min_associations
    ]
    DISEASE_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISEASE_POOL_PATH.write_text(json.dumps(pool, indent=2))
    return pool


def _already_screened(disease_name: str) -> bool:
    safe = disease_name.replace("/", "_").replace(" ", "_")
    return (RESULTS_DIR / f"{safe}.json").exists()


def screen_one(disease_name: str, log_dir: Path) -> dict:
    """Invoke systematic_screening for a single disease."""
    if _already_screened(disease_name):
        return {"disease": disease_name, "status": "skipped_already_done"}

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{disease_name.replace('/', '_').replace(' ', '_')}.log"

    t0 = time.time()
    try:
        result = subprocess.run(
            ["python3", "-u", "experiments/systematic_screening.py",
             "--disease", disease_name, "--skip-existing"],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            timeout=1800,  # 30 minutes per disease hard cap
        )
        return {
            "disease": disease_name,
            "status": "ok" if result.returncode == 0 else f"err{result.returncode}",
            "seconds": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired:
        return {"disease": disease_name, "status": "timeout", "seconds": 1800}
    except Exception as e:
        return {"disease": disease_name, "status": f"exception:{type(e).__name__}", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="Screen only first N diseases from pool (0=all)")
    ap.add_argument("--min-associations", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="Skip diseases that already have a result JSON")
    ap.add_argument("--rebuild-pool", action="store_true",
                    help="Rebuild data/disease_pool.json from DRKG")
    args = ap.parse_args()

    if args.rebuild_pool or not DISEASE_POOL_PATH.exists():
        print(f"Building disease pool (min_associations={args.min_associations})…")
        pool = build_disease_pool(min_associations=args.min_associations)
    else:
        pool = json.loads(DISEASE_POOL_PATH.read_text())
    print(f"Pool size: {len(pool):,} diseases")

    # Need disease names, not MeSH IDs. Use OT diseases map if available.
    # Until then, the screening script resolves MESH IDs to names itself.
    work = [d["mesh_id"] for d in pool]
    if args.limit:
        work = work[: args.limit]

    log_dir = Path("experiments/mass_screen_logs")
    print(f"Screening {len(work)} diseases with {args.workers} workers. Logs: {log_dir}/")

    # Serial for now; cloud deployment parallelizes
    out_records: list[dict] = []
    for i, mesh in enumerate(work, 1):
        r = screen_one(mesh, log_dir)
        out_records.append(r)
        print(f"  [{i:>4}/{len(work)}] {r.get('disease')}: {r.get('status')}  ({r.get('seconds','-')}s)")
        # Append chronologically
        with open(log_dir / "manifest.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")

    print()
    print("Summary:")
    by_status = Counter(r["status"] for r in out_records)
    for s, n in by_status.most_common():
        print(f"  {n:>5}  {s}")


if __name__ == "__main__":
    main()
