# OpenCure Roadmap

See `docs/RELEASE_v5.md` for what's currently shipping and
`docs/history/` for earlier milestones.

## Done (v5.1 — current)

- 11 active scoring pillars + 1 scaffolded (R-GCN awaits CUDA training).
- Grouped combiner (RRF over KG, max over structural/network) +
  hub-degree normalization + two-stage ADMET multiplier.
- Clinical guardrails on every top prediction: dose plausibility,
  DDI warnings, pharmacogenomic flags (CPIC + PharmGKB), mechanism
  paths, 4-axis triangulation (KG + docking + Pharos TDL + literature),
  GTEx tissue context.
- Calibrated XGBoost ensemble (AUC-ROC 0.9968 ± 0.0004, KG-memorization
  caveat disclosed) attached at inference by `scripts/score_ensemble_v5.py`.
- Canonical JSON output schema (`opencure/scoring/common.py`) +
  validator + 146 automated tests + GitHub Actions CI.
- Content-hashed prospective-validation snapshots, Zenodo-ready.
- 61-disease systematic screen in progress (~610 top-10 candidates
  with full clinical guardrails).

## Next on local hardware (v5.2 polish)

- Finalize post-screen: run `scripts/finalize_v5.py` to regenerate
  dashboard, clusters, snapshot, and honest-scoring report.
- Populate the 5 lab outreach briefs in `docs/lab_outreach_briefs.md`
  with real v5 top-10 candidates per disease.
- Cross-disease polypharmacology cluster analysis on the full 61.
- Zenodo DOI mint via `scripts/zenodo_upload.py` (once `ZENODO_TOKEN` set).

## Next with cloud GPU (v6)

- Unified-KG RotatE retrain on 14M-triple union at 400-dim/400-epoch
  (unblocks publication-competitive clean Hit@10).
- R-GCN heterogeneous GNN 12th pillar training.
- 2024-native KG re-screen with Open Targets 24.09 as primary (not
  addendum), to unlock non-zero time-sliced Hit@10.
- AutoDock Vina integration to wire the docking triangulation axis.

## Next with wet-lab partnership

- First validated prediction (any disease) → co-authored preprint.
- Rolling prospective precision@10 report every 90 days.
- FDA-referenceable prospective precision@K (12+ months of calendar
  time required).

## Next in peer review

- bioRxiv submission of `docs/methods_paper_draft.md`.
- Target journals: Nature Machine Intelligence, Bioinformatics.

## License + governance

Apache 2.0 with patent grant. Nonprofit / mission-locked; all code,
data, and models open-source.
