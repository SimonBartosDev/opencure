#!/usr/bin/env bash
# OpenCure CUDA full retrain — launch once, walk away, return to a fresh v6.
#
# Runs (sequentially, with checkpointing) on a CUDA-enabled host:
#   1. Unified-KG RotatE  @ 400-dim / 400 epochs   (10–16 h on RTX 4070)
#   2. R-GCN 12th pillar  @ 200-dim / 50 epochs    (12–24 h on RTX 4070)
#   3. Held-out eval      → data/eval/holdout_summary.json
#   4. Ensemble retrain   → data/models/ensemble_v5.pkl
#   5. Full 61-disease re-screen with the new models
#   6. finalize_v5.py     → fresh dashboard + snapshot + scoring
#
# Each step writes its own log to logs/. Failures abort the chain (set -e)
# but training resumes from the latest checkpoint on a clean restart.
#
# Total wall-clock: ~30–45 hours (one weekend on a 4070 laptop).
#
# Usage:
#   bash scripts/gpu_full_retrain.sh                        # full chain
#   bash scripts/gpu_full_retrain.sh --skip rgcn,rescreen   # only specific steps
#   bash scripts/gpu_full_retrain.sh --only train_kg        # one step
#   bash scripts/gpu_full_retrain.sh --resume               # resume training

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ---- arg parsing ----
SKIP=""
ONLY=""
RESUME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)   SKIP="$2"; shift 2;;
        --only)   ONLY="$2"; shift 2;;
        --resume) RESUME="--resume"; shift;;
        -h|--help)
            grep -E "^# " "$0" | sed 's/^# //'; exit 0;;
        *) echo "Unknown flag: $1"; exit 2;;
    esac
done

want() {
    local name="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$name,"* ]]
    else
        [[ ",$SKIP," != *",$name,"* ]]
    fi
}

step() {
    local name="$1"; shift
    local log="logs/gpu_${name}.log"
    if ! want "$name"; then
        echo "[skip] $name"
        return 0
    fi
    echo
    echo "================================================================"
    echo "▶ $name  (log: $log)"
    echo "================================================================"
    local t0=$(date +%s)
    if "$@" 2>&1 | tee "$log"; then
        local dt=$(( $(date +%s) - t0 ))
        echo "✓ $name  done in ${dt}s"
    else
        local dt=$(( $(date +%s) - t0 ))
        echo "✗ $name  FAILED after ${dt}s — see $log"
        exit 1
    fi
}

# ---- pre-flight ----
echo "OpenCure GPU full retrain — host: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo:    $REPO_ROOT"

# Run the full preflight unless --skip preflight was passed.
step preflight \
    python3 scripts/preflight_gpu.py

# ---- 1. Unified-KG RotatE ----
step train_kg \
    python3 scripts/train_unified_kg.py \
        --model RotatE --epochs 400 \
        --embedding-dim 400 --batch-size 4096 --num-negs-per-pos 64 \
        --checkpoint-every 10 $RESUME

# ---- 2. R-GCN 12th pillar ----
step train_rgcn \
    python3 scripts/train_rgcn.py \
        --embedding_dim 200 --epochs 50 --batch_size 4096 \
        --neg_samples 20 --device cuda

# ---- 3. Held-out eval ----
step eval \
    python3 scripts/run_unified_heldout_eval.py

# ---- 4. Retrain calibrated ensemble ----
step ensemble \
    python3 scripts/phase_c_pipeline.py

# ---- 5. Re-screen all 61 diseases ----
step rescreen \
    python3 experiments/systematic_screening.py --no-resume

# ---- 6. Finalize ----
step finalize \
    python3 scripts/finalize_v5.py --no-commit

echo
echo "================================================================"
echo "✓ GPU full retrain complete."
echo "  data/models/unified_transE_clean/  (RotatE)"
echo "  data/models/rgcn_v5/                (R-GCN)"
echo "  data/models/ensemble_v5.pkl         (retrained)"
echo "  experiments/results/                 (61 fresh JSONs)"
echo "  docs/index.html                      (rebuilt dashboard)"
echo "  data/prospective/snapshots/<ts>/     (new snapshot)"
echo "================================================================"
