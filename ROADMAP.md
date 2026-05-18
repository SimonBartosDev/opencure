# OpenCure Roadmap

See `docs/methods_paper_draft.md` for the full architecture writeup and
`docs/history/` for earlier milestones.

## Done — v7 (architecture complete)

The v7 push (commits on `v5-truly-best`) shipped 13 active pillars plus
five orthogonal honesty/uncertainty layers, all fail-open and
regression-tested (357 tests passing):

- **Foundation-model swap** — MoLFormer-XL replaces ChemBERTa for
  chemistry embeddings; ESM-2 150M replaces the 8M variant for protein
  embeddings (loader auto-prefers the strongest artifact on disk).
- **Conformal prediction** — split-conformal calibrator; every
  candidate ships with a 90 %-coverage interval. Empirical coverage
  measured at 90.1 %.
- **93-disease negative-control suite** — curated clinically-implausible
  compounds per disease + a CI gate asserting they rank below median.
- **Per-disease-class ensemble heads** — six classes (parasitic, viral,
  bacterial, oncology, rare_metabolic, chronic_systemic) with
  shared-head fallback.
- **JUMP Cell Painting** — the 13th pillar; phenotype-space
  morphological similarity to known treatments.
- **Selectivity + DepMap essentiality + mechanism-uncertainty** — three
  surfacing layers that flag promiscuous binders, pan-essential
  targets, and poorly-mapped disease biology.
- **Adversarial red-team agent** — deterministic seven-check critique
  per top-K candidate (optional local-LLM narration).
- **Wet-lab brief generator** — 1-page Markdown brief per disease;
  40 NTD/rare-disease outreach briefs under `docs/outreach/`.
- **Modal orchestration** — `scripts/modal_app.py` wires the full v7
  retrain + precompute + post-screen chain as serverless GPU jobs.

## In progress — v7 rescreen (gated on compute)

The v7 *code* is complete and merged; the v7 *numbers* land once these
GPU jobs run (see `docs/modal_runbook.md`):

- Unified-KG RotatE retrain with the NSSALoss fix (the v6.0 run used a
  silently-degenerate default loss).
- R-GCN retrain.
- Full 93-disease v7 rescreen — produces the candidate density the
  head-to-head benchmark (§5.9) and outreach briefs need.
- Post-screen tail: red-team pass + wet-lab briefs +
  retrospective-prospective validation + finalize.

Status of v7 artifacts already produced: MoLFormer-XL embeddings,
ESM-2 150M embeddings (29,917 proteins), shared + 3/6 per-class
ensemble heads, the conformal calibrator (90.1 % coverage).

## Next — credibility & outreach

- Fill `docs/methods_paper_draft.md` §5 Results with v7 rescreen
  numbers; submit to bioRxiv.
- Deploy the public dashboard to GitHub Pages (`docs/_config.yml`
  + `docs/index.html` are ready).
- Send the four lead-disease outreach emails (Schistosomiasis → DNDi /
  Caffrey lab; Chagas → DNDi / Fiocruz; Sickle Cell → CureSCi;
  Niemann-Pick → APMRF / NPUK).
- Modal + Anthropic nonprofit-credit applications (drafts in
  `docs/nonprofit_credits_email.md`).

## Next — v8 (closes the structural gaps)

Honestly-flagged limitations the v7 architecture does not yet address:

- **JUMP Cell Painting raw-image rerank** — v7 uses the consortium's
  distilled CellProfiler features; an image foundation model
  (DINOv2 / Phenom) over the raw microscopy is the stronger play.
- **Real molecular docking** — replace the ChEMBL bioactivity proxy
  with Boltz-1 / Gnina over AlphaFold-3 structures.
- **Allosteric-pocket prediction** layer.
- **Drug-combination scoring** — every score is currently monotherapy.
- **Active-learning loop** — each wet-lab readout retrains the relevant
  per-class head.
- **Live data refresh** — quarterly rebuild against post-2024
  ChEMBL / DrugBank / OpenTargets.
- **Bayesian mechanism-uncertainty** — replace the gene-count heuristic
  with a proper posterior over OpenTargets evidence categories.

## Next — wet-lab partnership

- First validated prediction (any disease) → co-authored preprint.
- Rolling prospective precision@10 report every 90 days.
- FDA-referenceable prospective precision@K (12+ months of calendar
  time required).

## License + governance

Apache 2.0 with patent grant. Nonprofit / mission-locked; all code,
data, and models open-source.
