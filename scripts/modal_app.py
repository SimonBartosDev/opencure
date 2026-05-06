"""
OpenCure v6 GPU retrain — Modal app.

Wraps the seven-step CUDA chain (preflight → train RotatE → train R-GCN →
eval → ensemble → re-screen → finalize) as Modal functions. Same code
as ``scripts/gpu_full_retrain.sh`` runs unchanged on a serverless A100.

Cost: ~$15-17 on A100 40GB at ~9h wall-clock — fits inside Modal's free
$30 credit with ~$13 headroom.

Quick start
-----------

    # 1. Once per machine
    pip install modal
    modal token new                             # one-time auth

    # 2. Once: upload data to a persistent volume (~10-20 min, ~2.5 GB)
    modal run scripts/modal_app.py::upload_data

    # 3. The retrain — fully managed; detach freely
    modal run --detach scripts/modal_app.py::full_chain

    # 4. Watch progress (optional)
    modal app logs opencure-v6-retrain

    # 5. When the chain finishes, pull the artifacts back
    modal run scripts/modal_app.py::download_artifacts

Resume after a crash
--------------------
The training scripts write checkpoints; re-running ``full_chain`` will
pick up from the latest checkpoint. To redo just one step:

    modal run scripts/modal_app.py::train_kg
    modal run scripts/modal_app.py::train_rgcn
    # etc.

Why this architecture
---------------------
* Volume holds code + data + outputs persistently between functions.
* Image bakes Python deps once; cold-start is fast on subsequent runs.
* Each step is its own Modal function so failures are scoped + retryable.
* CPU-only steps (ensemble, finalize) skip GPU billing entirely.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# App + persistent volume
# ---------------------------------------------------------------------------

app = modal.App("opencure-v6-retrain")
volume = modal.Volume.from_name("opencure-workspace", create_if_missing=True)

VOLUME_ROOT = "/workspace"
WORKDIR = f"{VOLUME_ROOT}/opencure"

# Local repo root (the directory this script lives under)
LOCAL_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Image: CUDA torch + ML deps + repo source baked in
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "rsync", "curl")
    # CUDA-enabled torch (cu121 works on every modern Modal A100/L4/A10G)
    .run_commands(
        "pip install --no-cache-dir torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "pykeen",
        "torch-geometric",
        "xgboost",
        "scikit-learn",
        "pandas",
        "rdkit-pypi",
        "tqdm",
        "numpy",
        "requests",
        "matplotlib",
        "pyyaml",
    )
    # Bake the local repo into the image (everything except heavy data dirs).
    # Re-baked automatically when local code changes.
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path="/repo",
        ignore=[
            "data/",                    # uploaded to Volume separately
            "experiments/results/*",    # produced inside the run
            "experiments/finalize_v5.log",
            "logs/",
            ".git/",
            ".venv*/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            "save_folder/",
            "data/evidence_cache/",
            "data/prospective/snapshots/",
        ],
    )
)


# ---------------------------------------------------------------------------
# Helpers shared across functions
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """First-run setup: copy /repo into the volume so code lives next to data."""
    import shutil

    Path(WORKDIR).mkdir(parents=True, exist_ok=True)
    src = Path("/repo")
    if not (Path(WORKDIR) / "scripts" / "gpu_full_retrain.sh").exists():
        # Fresh volume — copy code in.
        for item in src.iterdir():
            dst = Path(WORKDIR) / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        print(f"[bootstrap] code copied {src} → {WORKDIR}")
    else:
        # Volume already has older code — refresh source files (data is preserved).
        for item in src.iterdir():
            if item.name == "data":
                continue  # data is curated separately
            dst = Path(WORKDIR) / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        print(f"[bootstrap] code refreshed in {WORKDIR}")


def _run(cmd: list[str], step: str) -> int:
    """Run a subprocess in WORKDIR with PYTHONPATH set; tee to a log."""
    log = Path(WORKDIR) / "logs" / f"modal_{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = WORKDIR + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[{step}] $ {' '.join(cmd)}")
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=WORKDIR, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    # Echo last 30 lines for live monitoring
    try:
        tail = log.read_text().splitlines()[-30:]
        print("\n".join(tail))
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"[{step}] failed with exit {proc.returncode}; full log: {log}"
        )
    return proc.returncode


def _commit_volume() -> None:
    """Persist writes so the next function sees them."""
    volume.commit()


# ---------------------------------------------------------------------------
# Per-step functions
# ---------------------------------------------------------------------------

GPU_TRAIN = "A100-40GB"          # full-power steps
GPU_LIGHT = "A10G"               # ~$1.10/h vs $1.81 — for eval / rescreen
TIMEOUT_LONG = 10 * 3600         # 10 h cap on training functions
TIMEOUT_MED = 2 * 3600           # 2 h
TIMEOUT_SHORT = 30 * 60          # 30 min


@app.function(
    image=image, gpu=GPU_TRAIN,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_SHORT,
)
def preflight() -> None:
    _bootstrap()
    _run(["python3", "scripts/preflight_gpu.py"], "preflight")
    _commit_volume()


@app.function(
    image=image, gpu=GPU_TRAIN,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_LONG,
)
def train_kg() -> None:
    _bootstrap()
    _run([
        "python3", "scripts/train_unified_kg.py",
        "--model", "RotatE", "--epochs", "400",
        "--embedding-dim", "400", "--batch-size", "4096",
        "--num-negs-per-pos", "64",
        "--checkpoint-every", "10",
        "--resume",   # idempotent — picks up from checkpoint if present
    ], "train_kg")
    _commit_volume()


@app.function(
    image=image, gpu=GPU_TRAIN,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_LONG,
)
def train_rgcn() -> None:
    _bootstrap()
    _run([
        "python3", "scripts/train_rgcn.py",
        "--embedding_dim", "200", "--epochs", "50",
        "--batch_size", "4096", "--neg_samples", "20",
        "--device", "cuda",
    ], "train_rgcn")
    _commit_volume()


@app.function(
    image=image, gpu=GPU_LIGHT,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_MED,
)
def eval_holdout() -> None:
    _bootstrap()
    _run(["python3", "scripts/run_unified_heldout_eval.py"], "eval")
    _commit_volume()


@app.function(
    image=image,                # CPU only — XGBoost training is fast on CPU
    cpu=8, memory=16384,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_MED,
)
def train_ensemble() -> None:
    _bootstrap()
    _run(["python3", "scripts/phase_c_pipeline.py"], "ensemble")
    _commit_volume()


@app.function(
    image=image, gpu=GPU_LIGHT,  # search.py uses MPS path on Mac, CUDA elsewhere
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_LONG,         # 93 diseases × ~1 min/disease cache-warm
    cpu=8, memory=32768,
)
def rescreen() -> None:
    _bootstrap()
    _run([
        "python3", "experiments/systematic_screening.py", "--no-resume",
    ], "rescreen")
    _commit_volume()


@app.function(
    image=image,                  # CPU only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_MED,
)
def finalize() -> None:
    _bootstrap()
    _run([
        "python3", "scripts/finalize_v5.py", "--no-commit",
    ], "finalize")
    _commit_volume()


# ---------------------------------------------------------------------------
# One-time helpers (data upload + artifact download)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_MED,
)
def _verify_volume_contents() -> dict:
    """Sanity-check the volume after upload."""
    import json
    from pathlib import Path
    out: dict[str, object] = {}
    for p in [
        "data/drkg/drkg.tsv",
        "data/unified_kg/unified.tsv",
        "data/open_targets/ot_triplets.tsv",
        "data/sources_2024/gtex/gtex_median_tpm.gct",
        "data/disease_gene_index.json",
        "data/manifest.json",
        "data/eval/holdout_test.jsonl",
        "data/models/drkg_transE_clean/trained_model.pkl",
        "data/models/primekg",
    ]:
        full = Path(WORKDIR) / p
        out[p] = (full.exists(), round(full.stat().st_size / 1024**2, 1)
                  if full.exists() else 0)
    return out


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def upload_data() -> None:
    """Upload local ``data/`` to the persistent volume.

    Uses ``modal volume put`` under the hood — survives across runs and
    is shared by every function. Run once after the first checkout.

    Files included:
      data/drkg/                  ~750 MB
      data/unified_kg/            ~900 MB
      data/open_targets/          ~5 MB
      data/sources_2024/          ~500 MB (GTEx)
      data/mappings/              ~17 MB (HGNC)
      data/eval/                  ~150 KB
      data/models/drkg_transE_clean/  ~65 MB (existing 2020 baseline)
      data/models/primekg/             ~36 MB
      data/disease_gene_index.json     ~1 MB
      data/disease_pool.json
      data/manifest.json
      data/drkg/admet_predictions.json
      data/drkg/drug_target_activities.json
      data/drkg/chembl_phase.json
    """
    import shutil

    paths_to_upload = [
        "data/drkg",
        "data/unified_kg",
        "data/open_targets",
        "data/sources_2024",
        "data/mappings",
        "data/eval",
        "data/models/drkg_transE_clean",
        "data/models/primekg",
        "data/disease_gene_index.json",
        "data/disease_pool.json",
        "data/manifest.json",
    ]

    print(f"Uploading {len(paths_to_upload)} paths to Modal volume "
          "'opencure-workspace'…")
    print("(This is a one-time ~2.5 GB upload; subsequent runs reuse the volume.)")

    # Use modal CLI for the upload — much faster than function-based copies
    # because Modal streams in parallel with native protocol.
    for p in paths_to_upload:
        if not Path(LOCAL_REPO / p).exists():
            print(f"  ⚠ skip (missing): {p}")
            continue
        remote = f"opencure/{p}"
        print(f"  → {p}  →  volume:/{remote}")
        # `modal volume put` is the right tool here. We invoke the CLI
        # because the Python SDK lacks a high-level "put" helper.
        cmd = ["modal", "volume", "put", "--force",
               "opencure-workspace",
               str(LOCAL_REPO / p), f"/{remote}"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"    ✗ upload failed for {p}; aborting")
            sys.exit(1)

    print("\nVerifying volume contents…")
    summary = _verify_volume_contents.remote()
    for path, (exists, size_mb) in summary.items():
        mark = "✓" if exists else "✗"
        print(f"  {mark} {path:<50s} {size_mb:>8.1f} MB")
    print("\nUpload complete. Next:  modal run --detach scripts/modal_app.py::full_chain")


@app.local_entrypoint()
def download_artifacts(out_dir: str = "modal_v6_artifacts") -> None:
    """Pull v6 artifacts from the volume back to your local machine."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = [
        "opencure/data/models/unified_transE_clean",
        "opencure/data/models/rgcn_v5",
        "opencure/data/models/ensemble_v5.pkl",
        "opencure/data/models/ensemble_v5_report.json",
        "opencure/experiments/results",
        "opencure/data/prospective/snapshots",
        "opencure/data/manifest.json",
        "opencure/docs/index.html",
        "opencure/experiments/eval",
        "opencure/logs",
    ]
    for p in paths:
        local = Path(out_dir) / p.replace("opencure/", "")
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ← {p}  →  {local}")
        subprocess.run([
            "modal", "volume", "get", "--force",
            "opencure-workspace", f"/{p}", str(local),
        ])
    print(f"\nDone. Artifacts in {out_dir}/. Next on your Mac:")
    print(f"  rsync -av {out_dir}/ ./")
    print("  git add -A && git commit -m 'v6: Modal-trained artifacts'")


@app.local_entrypoint()
def full_chain() -> None:
    """Run all 7 chain steps sequentially. Reuses prior checkpoints."""
    import time

    steps: list[tuple[str, modal.Function]] = [
        ("preflight",      preflight),
        ("train_kg",       train_kg),
        ("train_rgcn",     train_rgcn),
        ("eval",           eval_holdout),
        ("ensemble",       train_ensemble),
        ("rescreen",       rescreen),
        ("finalize",       finalize),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}")
        s0 = time.time()
        try:
            fn.remote()
            dt = time.time() - s0
            print(f"  ✓ {name} done in {dt:.0f}s")
        except Exception as exc:
            dt = time.time() - s0
            print(f"  ✗ {name} FAILED after {dt:.0f}s: {exc}")
            print("  Re-run the chain or just this step:")
            print(f"    modal run scripts/modal_app.py::{name}")
            sys.exit(1)
    print(f"\n{'=' * 64}\n✓ Full chain done in {(time.time()-t0)/3600:.1f}h\n"
          f"{'=' * 64}")
    print("Next:  modal run scripts/modal_app.py::download_artifacts")
