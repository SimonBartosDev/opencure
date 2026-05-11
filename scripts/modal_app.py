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
        "rdkit",
        "tqdm",
        "numpy",
        "requests",
        "matplotlib",
        "pyyaml",
        # v7: MoLFormer-XL, ESM-2 (HuggingFace transformers + tokenizers).
        # transformers must be <4.49 — MoLFormer-XL's HuggingFace
        # configuration_molformer.py imports transformers.onnx, which was
        # removed in 4.49.0. ESM-2 is compatible with 4.48.x as well.
        "transformers==4.48.3",
        "tokenizers",
        "sentencepiece",
        # v7: optional acceleration for the MoLFormer-XL precompute
        "accelerate",
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

GPU_TRAIN = "A100-40GB"          # RotatE + preflight + eval (40 GB is plenty)
GPU_RGCN  = "A100-80GB"          # R-GCN full-graph encode needs ~50 GB at 200-dim
GPU_LIGHT = "A10G"               # ~$1.10/h vs $1.81 — rescreen pillars
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
    image=image, gpu=GPU_RGCN,          # 80GB — RGCNConv's per-relation
                                        # autograd graph (162 rel × per-rel
                                        # message tensors) is huge even with
                                        # sampling
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def train_rgcn() -> None:
    _bootstrap()
    # 128-dim is the sweet spot: same quality range as 200-dim per the
    # R-GCN paper (PyG examples use 128), 2.5x less memory across all
    # tensors, comfortable headroom on A100 80GB.
    # 1M edges/epoch × 50 epochs ≈ ~80% coverage with replacement.
    _run([
        "python3", "scripts/train_rgcn.py",
        "--embedding_dim", "128", "--epochs", "50",
        "--neg_samples", "20", "--device", "cuda",
        "--edges_per_epoch", "1000000",
        "--triples_per_epoch", "300000",
        "--checkpoint_every", "10",
        "--resume",
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
    """Clean rescreen with crash-resume semantics.

    First time this function runs after train_kg + train_rgcn complete, it
    needs to use the FRESH models — so old pre-GPU result JSONs must be
    cleared first. We use a sentinel file ``data/v6_screen_started`` to
    distinguish "first call after retrain" (clear old JSONs) from "retry
    after a crash mid-rescreen" (preserve partial progress).

    With the sentinel + default ``resume=True``, you get:
      - first run → empty results dir → screens all 93 from scratch on new models
      - crash mid-run → re-invoking rescreen skips diseases already done
      - second-pass cleanup → can manually delete sentinel + JSONs to redo
    """
    import os
    from pathlib import Path

    _bootstrap()
    sentinel = Path(WORKDIR) / "data" / "v6_screen_started"
    results_dir = Path(WORKDIR) / "experiments" / "results"

    if not sentinel.exists():
        # First post-retrain rescreen — wipe old per-disease JSONs.
        # Aggregates (screening_summary, mechanism_clusters, opencure_database,
        # novel_candidates) are regenerated by finalize.
        n_removed = 0
        if results_dir.exists():
            for p in results_dir.glob("*.json"):
                p.unlink()
                n_removed += 1
        print(f"  cleared {n_removed} pre-GPU result JSONs (first v6 rescreen)")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("v6 rescreen started; delete this file to force a clean re-screen")

    _run([
        "python3", "experiments/systematic_screening.py",  # resume=True default
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

    # Curated upload list — explicitly avoid the 28 GB ChEMBL SQLite tree
    # under data/sources_2024/chembl_34* which is dev-only. The runtime
    # already has the precomputed extracts (drug_target_activities.json,
    # chembl_phase.json) inside data/drkg/.
    paths_to_upload = [
        "data/drkg",                                            # ~750 MB
        "data/unified_kg",                                      # ~900 MB
        "data/open_targets",                                    # ~5 MB
        "data/sources_2024/gtex",                               # ~24 MB
        "data/sources_2024/pharmgkb",                           # ~11 MB
        "data/sources_2024/cpic_pairs.json",                    # 76 KB
        "data/sources_2024/9606.protein.aliases.v12.0.txt.gz",  # 19 MB STRING
        "data/sources_2024/9606.protein.links.v12.0.txt.gz",    # 79 MB STRING
        "data/mappings",                                        # ~17 MB HGNC
        "data/eval",                                            # ~150 KB
        "data/models/drkg_transE_clean",                        # ~65 MB
        "data/models/primekg",                                  # ~36 MB
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


@app.function(
    image=image,                  # CPU only — orchestrator just dispatches
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=12 * 3600,             # 12 h cap on whole chain (incl. waits)
)
def chain_orchestrator() -> None:
    """Server-side orchestrator. Runs on Modal so the chain survives the
    local CLI disconnecting (the warning ``modal run --detach`` prints
    about ``.remote()`` cancellation only applies to local entrypoints).
    """
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
        print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            dt = time.time() - s0
            print(f"  ✓ {name} done in {dt:.0f}s", flush=True)
        except Exception as exc:
            dt = time.time() - s0
            print(f"  ✗ {name} FAILED after {dt:.0f}s: {exc}", flush=True)
            print(f"  Resume just this step: modal run scripts/modal_app.py::{name}",
                  flush=True)
            raise
    print(f"\n{'=' * 64}\n✓ Full chain done in {(time.time()-t0)/3600:.1f}h\n"
          f"{'=' * 64}", flush=True)


@app.function(
    image=image,
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=4 * 3600,
)
def chain_resume_post_rgcn() -> None:
    """Resume the chain from `eval` onward — skips train_rgcn.

    Use this after train_kg succeeds but train_rgcn fails (e.g. OOM on
    A100 40GB at full 200-dim). R-GCN is the scaffolded 12th pillar;
    rgcn_score = 0 across candidates without it, which is the same
    state as before this run. Everything else (RotatE retrain, eval,
    ensemble, 93-disease re-screen, dashboard, snapshot) still ships.
    """
    import time

    steps: list[tuple[str, modal.Function]] = [
        ("eval",       eval_holdout),
        ("ensemble",   train_ensemble),
        ("rescreen",   rescreen),
        ("finalize",   finalize),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            print(f"  ✓ {name} done in {time.time()-s0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  ✗ {name} FAILED after {time.time()-s0:.0f}s: {exc}", flush=True)
            raise
    print(f"\n{'=' * 64}\n✓ Resume chain done in {(time.time()-t0)/3600:.2f}h\n"
          f"{'=' * 64}", flush=True)


@app.local_entrypoint()
def resume_post_rgcn() -> None:
    """Spawn the resume-after-RGCN-OOM chain on Modal."""
    handle = chain_resume_post_rgcn.spawn()
    print(f"✓ Resume chain spawned — function call id: {handle.object_id}")
    print(f"  Watch: modal app logs <app-id>")


@app.function(
    image=image,
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=10 * 3600,                 # 10h cap on whole RGCN-included chain
)
def chain_with_rgcn_first() -> None:
    """train_kg already done on volume → run R-GCN (80 GB) → rest of chain.

    Skips train_kg because the RotatE checkpoint is already on the
    volume from the previous run. R-GCN on A100 80GB should fit and
    finish in ~3-4h with 30 epochs (was OOMing at 50 epochs/40GB).
    Then eval, ensemble, rescreen, finalize all run with the full
    12-pillar model set.
    """
    import time

    steps: list[tuple[str, modal.Function]] = [
        ("train_rgcn", train_rgcn),     # 80 GB GPU, 30 epochs
        ("eval",       eval_holdout),
        ("ensemble",   train_ensemble),
        ("rescreen",   rescreen),
        ("finalize",   finalize),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            print(f"  ✓ {name} done in {time.time()-s0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  ✗ {name} FAILED after {time.time()-s0:.0f}s: {exc}", flush=True)
            raise
    print(f"\n{'=' * 64}\n✓ R-GCN chain done in {(time.time()-t0)/3600:.2f}h\n"
          f"{'=' * 64}", flush=True)


@app.local_entrypoint()
def chain_with_rgcn() -> None:
    """Spawn R-GCN-included chain (assumes RotatE already on volume)."""
    handle = chain_with_rgcn_first.spawn()
    print(f"✓ R-GCN chain spawned — function call id: {handle.object_id}")
    print(f"  Watch: modal app logs <app-id>")
    print(f"  GPU: A100 80GB for R-GCN, A100 40GB for eval, CPU for ensemble/finalize")
    print(f"  Estimated wall-clock: ~4-5h. Estimated cost: ~$10-12.")


@app.function(
    image=image,
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=8 * 3600,
)
def chain_rescreen_finalize() -> None:
    """Resume just the tail (rescreen + finalize). Use when R-GCN, eval,
    and ensemble are already done on the volume but rescreen got
    interrupted or the evidence cache wasn't warm."""
    import time
    steps = [("rescreen", rescreen), ("finalize", finalize)]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            print(f"  ✓ {name} done in {time.time()-s0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  ✗ {name} FAILED: {exc}", flush=True)
            raise
    print(f"\n✓ Tail chain done in {(time.time()-t0)/60:.1f}m\n", flush=True)


@app.local_entrypoint()
def resume_rescreen_finalize() -> None:
    """Spawn just rescreen + finalize on Modal (cheaper resume after cache warmup)."""
    handle = chain_rescreen_finalize.spawn()
    print(f"✓ Tail chain spawned — function call id: {handle.object_id}")


@app.local_entrypoint()
def full_chain() -> None:
    """Spawn the chain on Modal and exit immediately.

    The chain runs entirely on Modal — closing your laptop, the SSH
    session, or this terminal won't affect it. Watch progress with
    ``modal app logs opencure-v6-retrain``.
    """
    handle = chain_orchestrator.spawn()
    print(f"✓ Chain spawned on Modal — function call id: {handle.object_id}")
    print(f"  Watch live:  modal app logs opencure-v6-retrain")
    print(f"  Status:      modal app list")
    print(f"  Cancel:      modal app stop opencure-v6-retrain")
    print(f"  Pull artifacts when done:  "
          f"modal run scripts/modal_app.py::download_artifacts")


# ---------------------------------------------------------------------------
# v7 functions — foundation-model precomputes + post-processors
# ---------------------------------------------------------------------------
# Each function is small, self-contained, and idempotent. The cheap ones
# (CPU-only) total <$1; the foundation-model precomputes (GPU) total ~$13
# and fit comfortably in a single month of Modal's $30 free tier. The
# orchestrator ``v7_precomputes_cheap`` runs the foundation-model precomputes
# in sequence so a single ``modal run`` produces the whole v7 artifact set.


@app.function(
    image=image, gpu=GPU_LIGHT,                     # A10G ~$1.10/h
    volumes={VOLUME_ROOT: volume},
    timeout=3 * 3600,                                # 3h cap; typical 1-2h
)
def precompute_molformer_xl() -> None:
    """v7 Phase A1 — MoLFormer-XL embeddings for ~10K DRKG compounds.

    Output: ``data/drkg/embeddings/molformer_embeddings.npz``.
    Cost: ~$2 on A10G. ~1-2h wall-clock at batch_size 32.
    """
    _bootstrap()
    _run([
        "python3", "scripts/precompute_embeddings.py", "molformer",
    ], "molformer_xl")
    _commit_volume()


@app.function(
    image=image, gpu=GPU_TRAIN,                     # A100 40GB
    volumes={VOLUME_ROOT: volume},
    timeout=8 * 3600,                                # 8h cap; typical 4-6h
)
def precompute_esm2_150m() -> None:
    """v7 Phase A1 — ESM-2 150M protein embeddings for every DRKG Gene::.

    Network-bound first leg (UniProt sequence fetch ~100 min) then
    GPU-bound inference (~4-6h). Saves to
    ``data/drkg/embeddings/protein_embeddings_esm2_150M.npz``.
    Cost: ~$11 on A100 40GB.
    """
    _bootstrap()
    _run([
        "python3", "scripts/precompute_esm2_embeddings.py",
        "--variant", "150M",
    ], "esm2_150m")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def precompute_jump_cp_smoke() -> None:
    """v7 Phase A3 — JUMP-CP smoke artifact (8 synthetic profiles).

    The full ingest needs the consortium's ~10 GB profile parquet;
    this entry-point produces the smoke-test artifact so the rest of
    the v7 pipeline runs end-to-end. Replace by re-running with
    ``--features <real_path>`` once the real download is staged.
    Cost: ~$0.20 (CPU-only).
    """
    _bootstrap()
    _run([
        "python3", "scripts/precompute_jump_cp.py", "--smoke",
    ], "jump_cp_smoke")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def precompute_depmap_smoke() -> None:
    """v7 Phase A4 — DepMap smoke artifact (10 hand-picked genes).

    Production ingest needs the consortium's CRISPRGeneEffect.csv
    (~2 GB). The smoke artifact carries the canonical pan-essential
    (RPL5, POLR2A, RPS6) and safe (EGFR, CFTR, HBB, GBA, NPC1) genes
    so candidate-target lookups light up correctly during testing.
    Cost: ~$0.10 (CPU-only).
    """
    _bootstrap()
    _run([
        "python3", "scripts/precompute_depmap.py", "--smoke",
    ], "depmap_smoke")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def head_to_head_v7(holdout: str = "time_sliced") -> None:
    """v7 Phase B4 — head-to-head benchmark for the methods paper §4.5.

    For each baseline scoring column, re-rank every disease's
    candidates and evaluate against the held-out set. Writes:
        experiments/head_to_head_v7.md  — Markdown table for the paper
        experiments/head_to_head_v7.json — raw per-baseline metrics

    ``holdout`` = ``"time_sliced"`` (210 post-2020 pairs, the tighter
    publication test) or ``"random"`` (993-pair random holdout, denser
    disease coverage that overlaps more of the v6.1 screen results).

    Cost: ~$0.05, ~5 min on 4 CPU.
    """
    _bootstrap()
    _run([
        "python3", "scripts/head_to_head_benchmark.py",
        "--holdout", holdout,
    ], "head_to_head_v7")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=3 * 3600,
)
def score_ensemble_v7_only() -> None:
    """v7 Phase A2 + A5 — attach v7 fields to every existing result JSON.

    Runs ``score_ensemble_v5.py`` against the existing per-disease
    JSONs without running the rest of the 11-step finalize pipeline.
    Picks up the freshly-trained shared + per-class ensemble heads and
    the conformal calibrator from disk, so every candidate gains:

      - refreshed ``ensemble_prob``
      - ``ensemble_head`` tag (which head scored it: per-class or "shared")
      - ``ensemble_prob_lower`` / ``ensemble_prob_upper`` / ``prediction_set_at_90``

    Cheap (~$0.10), idempotent, safe to re-run.
    """
    _bootstrap()
    _run([
        "python3", "scripts/score_ensemble_v5.py",
    ], "score_ensemble_v7_only")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def calibrate_conformal_v7() -> None:
    """v7 Phase A2 — fit conformal calibrator on the held-out set.

    Reads the ensemble + held-out positives/sampled-negatives, computes
    the conformal quantile, and saves to ``data/models/conformal_v7.npz``.
    Required before ``score_ensemble_v5`` will emit
    ``ensemble_prob_lower``/``_upper`` fields.
    """
    _bootstrap()
    _run([
        "python3", "scripts/calibrate_conformal.py",
    ], "calibrate_conformal_v7")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def red_team_v7_pass() -> None:
    """v7 Phase A5 — adversarial critique on top-K of every result JSON.

    Deterministic critic (no LLM dependency); attaches
    ``red_team_assessment`` to each candidate. Re-runnable safely.
    """
    _bootstrap()
    _run([
        "python3", "scripts/red_team_v7.py",
    ], "red_team_v7")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=2 * 3600,
)
def generate_briefs_v7() -> None:
    """v7 Phase A5 — wet-lab briefs for top-5 of every disease.

    Writes ``experiments/results/briefs/<disease>_top5.md`` per disease.
    """
    _bootstrap()
    _run([
        "python3", "scripts/generate_wetlab_briefs.py",
    ], "generate_briefs_v7")
    _commit_volume()


@app.function(
    image=image,                                    # CPU-only (PubMed API)
    cpu=4, memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=4 * 3600,                                # PubMed rate-limited
)
def retrospective_prospective_v7() -> None:
    """v7 Phase B3 — query 2024-2025 PubMed for every top-K prediction.

    Output:
      experiments/prospective_v7_2024_2025.md
      data/prospective/retrospective_v7.jsonl
    """
    _bootstrap()
    _run([
        "python3", "scripts/retrospective_prospective.py",
    ], "retrospective_prospective_v7")
    _commit_volume()


# ---------------------------------------------------------------------------
# v7 orchestrators
# ---------------------------------------------------------------------------

@app.function(
    image=image,                                    # CPU-only orchestrator
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=12 * 3600,
)
def chain_v7_precomputes_cheap() -> None:
    """v7 cheap precomputes — fits in one month of $30 free tier.

    Sequence (rough cost):
      precompute_molformer_xl       (A10G,    ~1-2h, ~$2)
      precompute_esm2_150m          (A100-40, ~4-6h, ~$11)
      precompute_jump_cp_smoke      (CPU,     <1h,   ~$0.20)
      precompute_depmap_smoke       (CPU,     <1h,   ~$0.10)
      calibrate_conformal_v7        (CPU,     <1h,   ~$0.20)

    Total: ~$13. Leaves $17 of the free tier for the 93-disease screen
    and finalize tail in a second invocation (or in month two).
    """
    import time

    steps: list[tuple[str, modal.Function]] = [
        ("molformer_xl",          precompute_molformer_xl),
        ("esm2_150m",             precompute_esm2_150m),
        ("jump_cp_smoke",         precompute_jump_cp_smoke),
        ("depmap_smoke",          precompute_depmap_smoke),
        ("calibrate_conformal",   calibrate_conformal_v7),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ v7 cheap-precompute: {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            print(f"  ✓ {name} done in {time.time() - s0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  ✗ {name} FAILED after {time.time() - s0:.0f}s: {exc}",
                  flush=True)
            print(f"  Resume just this step: "
                  f"modal run scripts/modal_app.py::{name}", flush=True)
            raise
    print(f"\n{'=' * 64}\n✓ v7 cheap-precompute chain done in "
          f"{(time.time()-t0)/3600:.2f}h\n{'=' * 64}", flush=True)


@app.function(
    image=image,                                    # CPU-only orchestrator
    cpu=2, memory=2048,
    volumes={VOLUME_ROOT: volume},
    timeout=12 * 3600,
)
def chain_v7_post_screen() -> None:
    """v7 post-screen tail — runs after the 93-disease screen completes.

    Sequence (rough cost):
      train_ensemble                (CPU,     ~5min, ~$0.10) — refits per-class heads too
      red_team_v7_pass              (CPU,     <1h,   ~$0.20)
      generate_briefs_v7            (CPU,     <1h,   ~$0.10)
      retrospective_prospective_v7  (CPU,     ~3h,   ~$1)
      finalize                      (CPU,     <1h,   ~$0.20)

    Total: ~$2. Run after ``rescreen`` lands the per-disease JSONs.
    """
    import time

    steps: list[tuple[str, modal.Function]] = [
        ("train_ensemble",            train_ensemble),
        ("red_team_v7",               red_team_v7_pass),
        ("generate_briefs_v7",        generate_briefs_v7),
        ("retrospective_prospective", retrospective_prospective_v7),
        ("finalize",                  finalize),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n{'=' * 64}\n▶ v7 post-screen: {name}\n{'=' * 64}", flush=True)
        s0 = time.time()
        try:
            fn.remote()
            print(f"  ✓ {name} done in {time.time() - s0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  ✗ {name} FAILED after {time.time() - s0:.0f}s: {exc}",
                  flush=True)
            raise
    print(f"\n{'=' * 64}\n✓ v7 post-screen chain done in "
          f"{(time.time()-t0)/60:.1f}m\n{'=' * 64}", flush=True)


@app.local_entrypoint()
def v7_precomputes_cheap() -> None:
    """Spawn the cheap v7 precomputes on Modal (~$13, fits in free tier)."""
    handle = chain_v7_precomputes_cheap.spawn()
    print(f"✓ v7 cheap precomputes spawned — call id: {handle.object_id}")
    print(f"  Watch live:  modal app logs opencure-v6-retrain")
    print(f"  Cancel:      modal app stop opencure-v6-retrain")
    print(f"  Estimated cost: ~$13 (MoLFormer + ESM-2 + JUMP-smoke + DepMap-smoke)")
    print(f"  Wall-clock:     ~6-8h end-to-end")


@app.local_entrypoint()
def v7_post_screen_tail() -> None:
    """Spawn the v7 post-screen tail on Modal (~$2, after rescreen lands)."""
    handle = chain_v7_post_screen.spawn()
    print(f"✓ v7 post-screen tail spawned — call id: {handle.object_id}")
    print(f"  Watch live:  modal app logs opencure-v6-retrain")
    print(f"  Estimated cost: ~$2")
    print(f"  Wall-clock:     ~4-5h end-to-end (PubMed-bound)")
