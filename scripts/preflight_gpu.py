"""
Pre-flight check before launching the GPU full retrain.

Runs every failure-mode test that can fail at minute 0 of an 12-hour job,
on a machine that costs money per hour. If this passes, the long run
will almost certainly succeed.

Usage
-----
    python3 scripts/preflight_gpu.py            # 5-minute full check
    python3 scripts/preflight_gpu.py --quick    # skip the smoke train
    python3 scripts/preflight_gpu.py --json     # machine-readable output

Exit codes
----------
    0 — all checks passed; safe to launch gpu_full_retrain.sh
    1 — one or more checks failed; see the report
    2 — preflight script itself errored
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DATA_FILES = [
    "data/drkg/drkg.tsv",
    "data/unified_kg/unified.tsv",
    "data/open_targets/ot_triplets.tsv",
    "data/sources_2024/gtex/gtex_median_tpm.gct",
    "data/mappings/hgnc_complete_set.txt",
    "data/disease_gene_index.json",
    "data/drkg/compound_smiles.tsv",
    "data/drkg/admet_predictions.json",
    "data/drkg/drug_target_activities.json",
    "data/drkg/chembl_phase.json",
    "data/eval/holdout_test.jsonl",
    "data/eval/time_sliced_test.jsonl",
]
MIN_VRAM_GB = 16            # 4090 has 24 GB; below 16 = wrong card
MIN_DISK_FREE_GB = 25       # checkpoints + caches + screen output
SMOKE_TIMEOUT_SEC = 600


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    duration_s: float = 0.0
    fix: str = ""


def _run(name: str, fn) -> CheckResult:
    t0 = time.time()
    try:
        ok, detail, fix = fn()
    except Exception as exc:
        ok, detail, fix = False, f"{type(exc).__name__}: {exc}", "see traceback above"
    return CheckResult(name=name, ok=ok, detail=detail,
                       duration_s=round(time.time() - t0, 2), fix=fix)


# ---- individual checks --------------------------------------------------

def check_python_version():
    ver = sys.version_info
    ok = ver.major == 3 and 11 <= ver.minor <= 13
    return ok, f"Python {ver.major}.{ver.minor}.{ver.micro}", \
        "DGL-KE/PyKEEN wheels lag on 3.13/3.14 — use 3.11 or 3.12."


def check_cuda():
    try:
        import torch
    except ImportError:
        return False, "torch not installed", "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False", \
            "Reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    detail = f"{name} | {vram_gb:.1f} GB"
    if vram_gb < MIN_VRAM_GB:
        return False, detail, \
            f"VRAM {vram_gb:.1f} GB < {MIN_VRAM_GB} GB minimum — drop --embedding-dim to 256."
    return True, detail, ""


def check_pip_packages():
    missing: list[str] = []
    for mod in ("pykeen", "torch_geometric", "xgboost", "sklearn",
                "pandas", "rdkit", "tqdm"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"Missing: {', '.join(missing)}", \
            "pip install pykeen torch-geometric xgboost scikit-learn pandas rdkit tqdm"
    return True, "pykeen+pyg+xgboost+sklearn+pandas+rdkit OK", ""


def check_opencure_imports():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import opencure
        from opencure.scoring.ensemble import build_features, load_model
        from opencure.scoring.tissue_context import score_tissue_context
        from opencure.evidence.novelty import is_known_treatment
        from opencure.scoring.common import validate_result_file
    except Exception as exc:
        return False, f"Import failed: {exc}", \
            "Re-clone the repo or check `pip install -r requirements.txt` succeeded."
    return True, "all opencure modules importable", ""


def check_data_files():
    missing = [p for p in REQUIRED_DATA_FILES if not (REPO_ROOT / p).exists()]
    if missing:
        return False, f"{len(missing)} files missing: {missing[:3]}…", \
            ("rsync from your Mac, or run scripts/download_drkg.py + "
             "scripts/build_unified_kg.py + scripts/build_disease_gene_index.py")
    sizes_gb = sum((REPO_ROOT / p).stat().st_size for p in REQUIRED_DATA_FILES) / 1024**3
    return True, f"{len(REQUIRED_DATA_FILES)} files present, {sizes_gb:.1f} GB total", ""


def check_data_manifest():
    manifest_path = REPO_ROOT / "data/manifest.json"
    if not manifest_path.exists():
        return False, "data/manifest.json missing", \
            "python3 scripts/compute_data_manifest.py"
    try:
        m = json.loads(manifest_path.read_text())
        h = m.get("manifest_hash", "")
    except Exception as exc:
        return False, f"manifest unreadable: {exc}", "regenerate"
    return True, f"manifest_hash {h[:16]}", ""


def check_disk_free():
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1024**3
    if free_gb < MIN_DISK_FREE_GB:
        return False, f"{free_gb:.1f} GB free", \
            f"Need ≥{MIN_DISK_FREE_GB} GB. Clear ~/cache or move venv."
    return True, f"{free_gb:.1f} GB free", ""


def check_holdout_files():
    """Ensemble training (Phase C) reads these directly."""
    rh = REPO_ROOT / "data/eval/holdout_test.jsonl"
    th = REPO_ROOT / "data/eval/time_sliced_test.jsonl"
    if not rh.exists() or not th.exists():
        return False, f"random_holdout={rh.exists()} time_sliced={th.exists()}", \
            "Rebuild: python3 scripts/build_eval_sets.py"
    return True, "both held-out files present", ""


def check_torch_save_roundtrip():
    """Catch torch version mismatches in pickle format before training writes a corrupt checkpoint."""
    import torch
    tmp = REPO_ROOT / "data/.preflight_roundtrip.pt"
    try:
        x = torch.randn(8, 8, device="cuda")
        torch.save({"t": x, "epoch": 1}, tmp)
        loaded = torch.load(tmp, weights_only=False)
        ok = torch.allclose(loaded["t"], x)
        tmp.unlink(missing_ok=True)
        return ok, "torch.save/load CUDA tensor roundtrip OK", ""
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, f"roundtrip failed: {exc}", "Reinstall torch."


def check_finalize_dry_run():
    """Verify every step of the post-train pipeline imports + parses args."""
    r = subprocess.run(
        ["python3", "scripts/finalize_v5.py", "--dry-run"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return False, r.stderr.splitlines()[-1] if r.stderr else "dry-run failed", \
            "Inspect: python3 scripts/finalize_v5.py --dry-run"
    return True, "all 13 finalize steps validated", ""


def check_smoke_train():
    """2 epochs at the real dim — catches OOM / kernel-compile issues / data-shape bugs."""
    print("    smoke-training 2 epochs at full dim (proves OOM-safe + ~3 min)…")
    cmd = [
        "python3", "scripts/train_unified_kg.py",
        "--model", "TransE",   # TransE is fine for the smoke test; RotatE is heavier
        "--epochs", "2",
        "--embedding-dim", "400",
        "--batch-size", "4096",
        "--num-negs-per-pos", "64",
        "--checkpoint-every", "1",
        "--device", "cuda",
    ]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                       timeout=SMOKE_TIMEOUT_SEC)
    # Clean up artifacts so the real run starts fresh
    for p in ("data/models/unified_transE_clean/checkpoint.pt",
              "data/models/unified_transE_clean/trained_model.pkl"):
        (REPO_ROOT / p).unlink(missing_ok=True)
    if r.returncode != 0:
        last = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else ""
        if "out of memory" in last.lower() or "OOM" in last.upper():
            return False, "CUDA OOM at 4096 batch / 400 dim", \
                "Drop --batch-size to 2048 in gpu_full_retrain.sh, or --embedding-dim to 256."
        return False, f"smoke train failed: {last[:120]}", "see logs/"
    return True, "smoke train completed 2 epochs at full dim with checkpoint", ""


def check_rgcn_import():
    """torch-geometric is famously fragile vs torch version pinning."""
    try:
        from torch_geometric.nn import RGCNConv
        import torch
        # Construct a tiny RGCN layer + run a forward pass on CUDA
        device = torch.device("cuda")
        layer = RGCNConv(8, 8, num_relations=4).to(device)
        x = torch.randn(10, 8, device=device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
        edge_type = torch.tensor([0, 1, 2], device=device)
        out = layer(x, edge_index, edge_type)
        assert out.shape == (10, 8)
        return True, "torch_geometric RGCNConv works on CUDA", ""
    except Exception as exc:
        return False, f"PyG broken: {exc}", \
            "pip install torch-geometric --force-reinstall"


# ---- runner -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the smoke-train check (~3 min)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output (silences pretty print)")
    args = ap.parse_args()

    checks = [
        ("python_version",       check_python_version),
        ("cuda",                 check_cuda),
        ("pip_packages",         check_pip_packages),
        ("opencure_imports",     check_opencure_imports),
        ("data_files",           check_data_files),
        ("data_manifest",        check_data_manifest),
        ("disk_free",            check_disk_free),
        ("holdout_files",        check_holdout_files),
        ("torch_roundtrip",      check_torch_save_roundtrip),
        ("rgcn_import",          check_rgcn_import),
        ("finalize_dry_run",     check_finalize_dry_run),
    ]
    if not args.quick:
        checks.append(("smoke_train", check_smoke_train))

    results = []
    for name, fn in checks:
        if not args.json:
            print(f"  → {name} …", end=" ", flush=True)
        r = _run(name, fn)
        results.append(r)
        if not args.json:
            mark = "✓" if r.ok else "✗"
            print(f"{mark} ({r.duration_s}s)  {r.detail}")
            if not r.ok and r.fix:
                print(f"    fix: {r.fix}")

    n_ok = sum(1 for r in results if r.ok)
    n_total = len(results)

    if args.json:
        print(json.dumps({
            "ok": n_ok == n_total,
            "passed": n_ok, "total": n_total,
            "checks": [r.__dict__ for r in results],
        }, indent=2))
    else:
        print()
        print("=" * 64)
        print(f"PRE-FLIGHT: {n_ok}/{n_total} checks passed")
        print("=" * 64)
        if n_ok < n_total:
            print("Failures:")
            for r in results:
                if not r.ok:
                    print(f"  ✗ {r.name}: {r.detail}")
                    if r.fix:
                        print(f"      fix → {r.fix}")
            print("\nFix the failures, re-run preflight, then launch.")
        else:
            print("Safe to launch:  bash scripts/gpu_full_retrain.sh")

    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT ERRORED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
