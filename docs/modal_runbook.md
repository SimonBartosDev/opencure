# OpenCure v6 retrain on Modal

Step-by-step for running the GPU retrain via Modal's serverless GPUs
using the $30 free credit. Total wall-clock ~9h, total cost ~$15-17
(comfortably inside the credit).

## Why Modal vs RunPod / Vast

The same `gpu_full_retrain.sh` chain runs on either. Modal trade-offs:

| | Modal A100 40GB | RunPod RTX 4090 |
|---|---|---|
| Cash cost | $0 (within $30 credit) | ~$7 |
| Wall-clock | ~8–9h | ~11–14h |
| Setup | 5 min (Modal CLI auth) | 5 min (SSH) |
| Live monitoring | `modal app logs` | tmux + ssh |
| Mid-run failure | Resume per-step | tmux resume |
| Iteration cost (5 runs) | $0 | ~$35 |

Pick Modal if you want zero cash spend and faster training. Pick RunPod if
you want SSH familiarity and don't mind paying $7.

---

## Step 1 — Install Modal + auth (2 min, one-time)

```bash
# On your Mac
pip install modal
modal token new
# Browser opens. Sign in with the account holding your $30 credit. Done.
```

Verify:
```bash
modal token list      # should show your authenticated token
```

---

## Step 2 — Upload data to a Modal volume (~15 min, one-time)

```bash
cd ~/Ideas/Drug_reuse
modal run scripts/modal_app.py::upload_data
```

What this does:
- Creates the persistent volume `opencure-workspace` if missing
- Uploads ~2.5 GB of data files (DRKG, unified KG, OT, GTEx, HGNC,
  evals, existing models) using `modal volume put`
- Verifies every required file landed correctly

You'll see progress for each path:
```
  → data/drkg                   →  volume:/opencure/data/drkg
  → data/unified_kg             →  volume:/opencure/data/unified_kg
  ...

Verifying volume contents…
  ✓ data/drkg/drkg.tsv                                ~750.0 MB
  ✓ data/unified_kg/unified.tsv                       ~900.0 MB
  ...

Upload complete. Next:  modal run --detach scripts/modal_app.py::full_chain
```

If any file shows ✗, that path was missing on your Mac. Address (rsync from
elsewhere, rebuild via the data-build scripts) and re-run upload_data.

---

## Step 3 — Launch the chain (10 sec, then walk away)

```bash
modal run --detach scripts/modal_app.py::full_chain
```

Expected output:
```
✓ App launched: https://modal.com/apps/<your-username>/opencure-v6-retrain
```

The `--detach` flag means the run continues even if you close your laptop.
The chain runs sequentially:

1. **preflight** (~5 min, A100) — validates everything before the long jobs
2. **train_kg** (~5–6h, A100) — Unified RotatE 400-dim, 400 epochs
3. **train_rgcn** (~3–4h, A100) — R-GCN 12th pillar, 50 epochs
4. **eval** (~10 min, A10G) — held-out Hit@10 metrics
5. **ensemble** (~5 min, CPU) — XGBoost + isotonic calibration
6. **rescreen** (~30 min, A10G) — full 93-disease re-screen
7. **finalize** (~10 min, CPU) — manifest + dashboard + snapshot

**Total: ~9h on A100 40GB. Cost: $15-17 against your $30 credit.**

---

## Step 4 — Monitor (anytime, optional)

### Live logs
```bash
modal app logs opencure-v6-retrain        # tail all functions
```

### Web dashboard
Visit the URL printed when you launched. Per-function GPU utilization,
log streams, cost-per-step.

### From the CLI
```bash
modal app list                            # see status
```

---

## Step 5 — If something fails mid-run

Each step is its own Modal function — so retry just that step:

```bash
# Resume training (picks up from latest checkpoint)
modal run scripts/modal_app.py::train_kg
modal run scripts/modal_app.py::train_rgcn

# Or skip ahead if a downstream step is the only one that broke
modal run scripts/modal_app.py::eval_holdout
modal run scripts/modal_app.py::train_ensemble
modal run scripts/modal_app.py::rescreen
modal run scripts/modal_app.py::finalize
```

The `--resume` flag is baked into `train_kg` so re-running it with an
existing checkpoint loses at most 10 epochs (~10 min).

If you're really stuck, the volume persists across runs; you can
inspect it manually:

```bash
# Grab any single file from the volume to look at
modal volume get opencure-workspace /opencure/logs/modal_train_kg.log .
```

---

## Step 6 — Pull artifacts back when done (~5 min)

```bash
cd ~/Ideas/Drug_reuse
modal run scripts/modal_app.py::download_artifacts
```

This pulls everything to `./modal_v6_artifacts/`:
```
modal_v6_artifacts/
├── data/
│   ├── manifest.json
│   ├── models/
│   │   ├── unified_transE_clean/
│   │   ├── rgcn_v5/
│   │   ├── ensemble_v5.pkl
│   │   └── ensemble_v5_report.json
│   └── prospective/snapshots/<timestamp>/
├── docs/index.html
├── experiments/eval/
├── experiments/results/
└── logs/
```

Then merge into your repo:
```bash
rsync -av modal_v6_artifacts/ ./
git add -A data/models data/prospective experiments/results experiments/eval docs/index.html data/manifest.json logs
git commit -m "v6: Modal-trained artifacts (RotatE 400-dim + R-GCN 12th pillar + retrained ensemble)"
git push
rm -rf modal_v6_artifacts
```

---

## Cost tracking

Modal shows live cost in the dashboard. Check anytime:
```bash
modal app logs opencure-v6-retrain | grep -i "billed\|charged"
```

Or visit https://modal.com/settings/usage for a per-app breakdown.

Worst case — if the run goes long for some reason:
- A100 40GB at $1.81/hr × 11h = $19.91 (still inside $30)
- Add 1h debugging on cheaper functions: ~$1
- **Hard ceiling: ~$25 even in pessimistic scenarios**

---

## Common gotchas

1. **First-run image build is slow (~5 min).** Modal compiles the CUDA
   torch + ML deps image on first invocation. Subsequent runs reuse it.
   Don't panic if `preflight` takes 10 min the first time.
2. **Volume writes need explicit commit.** Each function calls
   `volume.commit()` after writing. If you write to the volume from a
   one-off shell session, run `modal.Volume.from_name(...).commit()`
   yourself.
3. **`--detach` is essential for long runs.** Without it, your laptop
   going to sleep can sever the connection (the run continues on Modal,
   but you lose live log streaming until you reattach).
4. **Free credit doesn't auto-renew.** $30 is one-time. After it's gone
   you'll be billed against the card on file. Set a usage limit in
   Modal settings if you want a hard cap.
5. **Check VRAM if you go off-script.** Default config uses ~14 GB on
   A100 40GB. If you crank `embedding_dim` higher, switch to A100 80GB
   in `modal_app.py` (~33% more cost).

---

## When to come back to this runbook

- After you've set up Modal once, just `modal run --detach
  scripts/modal_app.py::full_chain` — that's the whole loop.
- If you change disease list / add pillars / tweak hyperparameters: edit
  the relevant script, then re-run the chain. Modal re-bakes the image
  automatically.
- If you blow through the $30 credit before finishing: the volume is
  preserved. Add billing, re-run the chain — it picks up from the
  latest checkpoint.
