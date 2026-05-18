"""
One-command v5 finalization — canonical post-screen pipeline.

Each step is idempotent and safe to re-run. The ordering below is load-bearing:
earlier steps populate fields later steps read.

    1. manifest       → data/manifest.json (freezes input-file hashes)
    2. known_labels   → is_known_treatment via DRKG treats-edge lookup
    3. ensemble       → ensemble_prob + ensemble_rank per candidate
    4. tissue         → tissue_context (disease-relevant GTEx tissues)
    5. docking        → docking axis (ChEMBL proxy + not_wired fallback)
    6. clusters       → mechanism clusters across diseases
    7. database       → experiments/results/opencure_database.json (dashboard input)
    8. dashboard      → docs/index.html
    9. snapshot       → content-hashed prospective snapshot
   10. scoring        → scripts/honest_scoring.py report
   11. commit         → git commit of artifacts (optional; --no-commit skips)

Usage
-----
    python3 scripts/finalize_v5.py                    # run all steps
    python3 scripts/finalize_v5.py --only ensemble,tissue
    python3 scripts/finalize_v5.py --skip commit
    python3 scripts/finalize_v5.py --dry-run          # list steps, run nothing

Each step's full stdout/stderr is tee'd to ``experiments/finalize_v5.log``.
A one-line status is printed to the terminal.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LOG = Path("experiments/finalize_v5.log")


@dataclass
class Step:
    name: str            # short key for --only / --skip
    label: str           # human-readable
    cmd: list[str]       # subprocess command
    timeout: int = 1800  # seconds


STEPS: list[Step] = [
    Step("manifest",     "Refreshing data manifest",
         ["python3", "scripts/compute_data_manifest.py"]),
    Step("dg_index",     "Building unified disease-gene index (OT+DRKG/GNBR)",
         ["python3", "scripts/build_disease_gene_index.py"], timeout=900),
    Step("known_labels", "Refreshing is_known_treatment labels",
         ["python3", "scripts/refresh_known_treatment_labels.py"], timeout=600),
    Step("shared_tgt",   "Backfilling shared drug-disease targets",
         ["python3", "scripts/backfill_shared_targets.py"], timeout=600),
    Step("ensemble",     "Attaching ensemble_v5 probabilities",
         ["python3", "scripts/score_ensemble_v5.py"], timeout=900),
    Step("novelty_rank", "v7: re-ranking — demote known treatments below repurposing leads",
         ["python3", "scripts/apply_novelty_ranking.py"], timeout=300),
    Step("tissue",       "Populating tissue_context (GTEx)",
         ["python3", "scripts/wire_tissue_context.py"], timeout=600),
    Step("docking",      "Populating docking axis (ChEMBL proxy)",
         ["python3", "scripts/add_docking_proxy_axis.py"], timeout=600),
    Step("pains",        "Annotating structural alerts (PAINS/Brenk/NIH)",
         ["python3", "scripts/annotate_structural_alerts.py"], timeout=300),
    Step("red_team",     "v7: adversarial red-team critique per top-K candidate",
         ["python3", "scripts/red_team_v7.py"], timeout=900),
    Step("clusters",     "Recomputing cross-disease mechanism clusters",
         ["python3", "-m", "opencure.scoring.mechanism_cluster", "experiments/results"]),
    Step("database",     "Generating dashboard database JSON",
         ["python3", "experiments/generate_database.py"]),
    Step("dashboard",    "Building HTML dashboard",
         ["python3", "scripts/build_explorer.py"]),
    Step("snapshot",     "Writing content-hashed prospective snapshot",
         ["python3", "scripts/snapshot_predictions.py"]),
    Step("briefs",       "v7: per-disease wet-lab briefs (top-5 each)",
         ["python3", "scripts/generate_wetlab_briefs.py"], timeout=600),
    Step("scoring",      "Running honest-scoring report",
         ["python3", "scripts/honest_scoring.py"]),
]


def run_step(step: Step, dry: bool) -> bool:
    print(f"\n▶ {step.label}")
    if dry:
        print(f"  (dry-run) {' '.join(step.cmd)}")
        return True
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + "."
    t0 = time.time()
    try:
        r = subprocess.run(step.cmd, env=env, capture_output=True,
                           text=True, timeout=step.timeout)
        ok = r.returncode == 0
        tail = (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
        if not ok and r.stderr.strip():
            tail += " | " + r.stderr.strip().splitlines()[-1]
        print(f"  {'✓' if ok else '✗'} {time.time()-t0:.1f}s  {tail[:110]}")
        with LOG.open("a") as f:
            f.write(f"\n=== [{step.name}] {step.label} "
                    f"({'OK' if ok else 'FAIL'}, {time.time()-t0:.1f}s) ===\n")
            f.write("STDOUT:\n" + (r.stdout or "") + "\n")
            if r.stderr:
                f.write("STDERR:\n" + r.stderr + "\n")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT after {step.timeout}s")
        return False
    except Exception as exc:
        print(f"  ✗ {type(exc).__name__}: {exc}")
        return False


def count_result_jsons() -> int:
    d = Path("experiments/results")
    if not d.exists():
        return 0
    ignore = {"opencure_database", "screening_summary",
              "novel_candidates", "mechanism_clusters"}
    return sum(1 for f in d.glob("*.json")
               if not any(x in f.stem for x in ignore))


def maybe_commit(results: dict[str, bool], n_diseases: int) -> None:
    print("\n▶ Committing finalize artifacts")
    subprocess.run(["git", "add",
                    "experiments/results/", "docs/", "data/manifest.json",
                    "data/prospective/"], capture_output=True)
    msg = (
        f"v5 finalize run: {n_diseases}/61 diseases, "
        f"{sum(results.values())}/{len(results)} steps OK\n\n"
        "Auto-generated by scripts/finalize_v5.py.\n\n"
        "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )
    r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
    print("  ✓ committed" if r.returncode == 0 else "  (no changes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Comma-separated step names to run (others skipped)")
    ap.add_argument("--skip", help="Comma-separated step names to skip")
    ap.add_argument("--dry-run", action="store_true", help="List steps, run nothing")
    ap.add_argument("--no-commit", action="store_true", help="Skip the final git commit")
    args = ap.parse_args()

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    to_run = [s for s in STEPS
              if (not only or s.name in only) and s.name not in skip]

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(
        f"v5 finalize started {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"Steps: {[s.name for s in to_run]}\n"
    )

    n_diseases = count_result_jsons()
    print(f"v5 finalize — {n_diseases} disease JSONs on disk; {len(to_run)} step(s) queued")
    if n_diseases < 30 and not args.dry_run:
        print(f"  ⚠ only {n_diseases}/61 diseases — re-screen may still be running")
        print("     (running anyway on what's present; safe to re-run later)")

    results: dict[str, bool] = {}
    for s in to_run:
        results[s.name] = run_step(s, args.dry_run)

    if not args.dry_run and not args.no_commit and "commit" not in skip:
        maybe_commit(results, n_diseases)

    # Final summary
    print("\n" + "=" * 64)
    print("v5 FINALIZE SUMMARY")
    print("=" * 64)
    print(f"  Disease JSONs: {n_diseases}/61")
    for name, ok in results.items():
        print(f"  {name:<14s} {'✓' if ok else '✗'}")
    print(f"\n  Full log: {LOG}")
    print(f"  Dashboard: docs/index.html")
    print(f"  Latest snapshot: data/prospective/snapshots/")
    if results and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
