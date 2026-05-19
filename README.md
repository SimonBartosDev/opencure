<p align="center">
  <h1 align="center">OpenCure</h1>
  <p align="center">
    <strong>Open-source multi-pillar AI platform for drug repurposing with integrated clinical guardrails</strong>
  </p>
  <p align="center">
    <a href="https://github.com/SimonBartosDev/opencure/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python"></a>
    <a href="https://simonbartosdev.github.io/opencure/"><img src="https://img.shields.io/badge/Explorer-Live_Dashboard-orange.svg" alt="Dashboard"></a>
    <img src="https://img.shields.io/badge/Tests-357%2Fpassing-brightgreen.svg" alt="Tests">
    <img src="https://img.shields.io/badge/Version-v7-purple.svg" alt="Version">
  </p>
</p>

---

## What OpenCure does

OpenCure screens ~10,500 FDA-approved and investigational drugs against any disease using **13 complementary AI scoring pillars**, then layers **clinical guardrails**, **calibrated uncertainty**, and an **adversarial red-team pass** on every top prediction so results are actionable and honest — not just ranked.

Every top-K prediction surfaces:
- **Calibrated uncertainty** (v7) — a 90 %-coverage conformal interval `[ensemble_prob_lower, ensemble_prob_upper]` plus a binary prediction set (`{1}` confident-positive, `{0,1}` genuinely uncertain, `{0}` confident-negative)
- **Adversarial red-team critique** (v7) — seven failure modes checked automatically per candidate (single-pillar artifact, low selectivity, pan-essential target, hub-bias, low mechanism confidence, evidence shortage, failed-trial history)
- **Dose plausibility** — is the drug's clinical plasma Cmax high enough to engage the predicted target? (stage-2 ChEMBL 34 bioactivities)
- **Drug-drug interactions** — top 10 dangerous co-prescriptions from 1.4M DrugBank DDI edges
- **Pharmacogenomic flags** — CPIC/PharmGKB variant warnings (HLA, CYP, VKORC1, etc.)
- **Mechanism path** — natural-language graph path `Drug →[inhibits]→ Target →[linked to]→ Disease`
- **Triangulation score** — agreement across 4 independent axes (KG + docking + Pharos target-development-level + literature)
- **Selectivity + essentiality flags** (v7) — ChEMBL off-target panel + DepMap pan-essentiality warning on the primary target
- **Tissue context** — GTEx tissue-specific expression modifier

Every prediction is **content-fingerprinted** (SHA-256) and traceable to an immutable `data_manifest_hash` covering the source files that produced it — critical for reproducibility in peer review.

**[Browse live predictions →](https://simonbartosdev.github.io/opencure/)**

## What OpenCure is — and is not

OpenCure is a **hypothesis-generation and triage tool**. It systematically
ranks, critiques, documents, and uncertainty-annotates drug-repurposing
hypotheses so a wet-lab scientist can decide what is worth testing.

It is **not a validated predictor.** We publish **no benchmark accuracy
figure**, and we make no claim about how often a top-ranked candidate is
correct. A leak-free retrospective benchmark is not currently possible: the
knowledge graph (DRKG, 2020-vintage) predates the repurposing events that
would be needed to test it, and the only post-2020 repurposing examples
available are too few to constitute a benchmark. The platform's predictive
accuracy is therefore **unestablished** — honestly stated rather than
hidden behind a number.

> An earlier version of this README reported an ensemble "AUC-ROC 0.997".
> That figure was an artefact of **data leakage** — the knowledge-graph
> features were scored from a graph that still contained the test edges —
> and has been **withdrawn**. See [docs/architecture.md](docs/architecture.md)
> for the full honest evaluation discussion.

Treat every output as a structured, transparent, adversarially-critiqued
hypothesis for expert review — not a recommendation.

## Why this matters

Drug development takes 10-15 years and costs >$2B. Repurposing approved drugs skips most of safety testing because the drugs are already proven tolerable in humans. The blocker has been: which drugs to test for which diseases, out of tens of millions of possibilities?

OpenCure's answer: screen them all computationally with 13 complementary methods, surface the candidates where independent methods converge, state the uncertainty honestly, adversarially critique each call, and give clinicians/researchers the clinical context they need to decide whether a hypothesis is worth testing in their lab. OpenCure narrows the search space for human experts — it does not replace experimental validation.

## The 13 scoring pillars

| # | Pillar | Signal | Source |
|---|---|---|---|
| 1 | **TransE** | Knowledge graph embedding | DRKG 5.87M edges |
| 2 | **RotatE** (PyKEEN) | KG embedding with relation rotations | DRKG |
| 3 | **Unified-KG TransE** | KG embedding on the DRKG+PrimeKG+OT union | 14M-edge unified graph |
| 4 | **PrimeKG** | Independent KG scoring | Harvard 8.1M edges |
| 5 | **TxGNN** | Graph neural network (Harvard) | pre-computed; v7 salt-form drug-name matching |
| 6 | **Molecular fingerprints** | Morgan/ECFP structural similarity | RDKit |
| 7 | **MoLFormer-XL** (v7 swap from ChemBERTa) | Transformer embedding of SMILES (IBM, 1.1B-compound pretrain) | HuggingFace |
| 8 | **DeepPurpose DTI** | Drug-target binding affinity; v7 adds ESM-2 150M protein embeddings | BindingDB / ESM-2 |
| 9 | **Network proximity** | Shortest-path on PPI | STRING v12 (473K high-confidence edges) |
| 10 | **Gene signatures** | Disease expression reversal | L1000 + OT × ChEMBL mechanistic reversal |
| 11 | **Mendelian randomization** | Genetic causal evidence | Open Targets GraphQL |
| 12 | **R-GCN** | Heterogeneous GNN with DistMult head | trained on DRKG; v6.1+ |
| 13 | **JUMP Cell Painting** (v7) | Morphological-similarity to known treatments in phenotype space | JUMP-CP consortium |

ADMET (Chemprop drug-likeness / toxicity) runs as a multiplier, not a pillar. Several pillars (TransE, RotatE, Unified-KG, PrimeKG, TxGNN, R-GCN) are knowledge-graph embeddings of largely overlapping graphs and are **correlated, not independent**; they are **grouped before combining** to limit double-counting (KG-group via RRF; structural-group via max — including JUMP morphological similarity; network-group via max), then weighted by an XGBoost ensemble with isotonic-calibrated outputs.

> **No accuracy figure is attached to the ensemble.** A leak-free retrain of
> the ensemble scores far below the withdrawn 0.997, and on a fair temporal
> test it is at chance — the simple ensemble does not predict prospective
> repurposing. It is retained only as one ranking input among many. See
> [docs/architecture.md](docs/architecture.md) for the honest evaluation.

v7 adds **per-disease-class ensemble heads** (six classes) and a **split conformal-prediction wrapper** for 90 %-coverage uncertainty intervals.

## Clinical guardrails

What separates OpenCure from pure-ranking repurposing platforms: every prediction is actionable. Per-candidate fields include:

```json
"dose_plausibility": {
  "plausibility": "yes",
  "confidence": "high",
  "target_affinity": {"median_ic50_nM": 17.76, "cmax_over_ic50_ratio": 56.3}
},
"ddi_warnings": {
  "n_interactions": 1477,
  "top_interactions": [{"drug": "Warfarin", "severity": "high"}, ...]
},
"pharmacogenomics": {
  "highest_risk": "high_risk",
  "summary": "CPIC-A (CYP2D6) • PharmGKB-1A (CYP2D6 CYP2D6*1/*2)"
},
"triangulation": {
  "n_axes_agree": 3,
  "label": "silver-standard",
  "axes": {"kg": true, "docking": false, "pharos": true, "literature": true}
},
"mechanistic_hypothesis": "Donepezil —[treats]→ Alzheimer's disease",
"ensemble_prob": 0.81,
"ensemble_prob_lower": 0.50, "ensemble_prob_upper": 1.00,
"prediction_set_at_90": [1],
"ensemble_head": "chronic_systemic",
"selectivity_score": 0.78, "n_off_targets": 6,
"target_essentiality": 0.04, "essentiality_warning": false,
"red_team_assessment": "No structural red flags detected."
```

## v7 — calibration, honesty, and image-based screening

v7 adds five orthogonal layers on top of the pillar stack, each fail-open
(a missing artifact degrades gracefully, never breaks the pipeline):

- **Conformal prediction** — split-conformal calibrator (`opencure/scoring/conformal.py`); every candidate ships with a 90 %-coverage interval. Empirical coverage measured at 90.1 % on the held-out calibration set.
- **93-disease negative-control suite** — `tests/data/negative_controls.yaml` lists clinically implausible compounds per disease; a CI gate (`scripts/verify_negative_controls.py`) asserts they rank below the per-disease median.
- **Per-disease-class ensemble heads** — six logistic heads (parasitic, viral, bacterial, oncology, rare_metabolic, chronic_systemic) specialise on each class's dominant repurposing mechanism; routing falls back to the shared head when a class has too few training positives.
- **JUMP Cell Painting** — the 13th pillar; morphological-profile similarity in phenotype space, the largest single closure of the gap to closed-platform image-based screening.
- **Adversarial red-team + wet-lab briefs** — every top-K candidate is critiqued by a deterministic adversarial pass (optionally narrated by a local Llama-3.1-8B); every disease gets a 1-page Markdown wet-lab brief with suggested assay, concentration range, and caveats.

## Validation

- **Held-out random split**: 993 DrugBank treats pairs held out, scored against the full 10,551-compound candidate pool
- **Time-sliced benchmark**: 210 post-2020 approved drug-disease pairs (drugs with `yearOfFirstApproval >= 2020` from OT 24.09), for testing generalization beyond the 2020-era knowledge graph
- **Edge-stripped retraining**: `scripts/strip_heldout_edges.py` removes test-set edges from DRKG+PrimeKG+OT before training a clean model
- **Conformal coverage**: empirical 90.1 % at the nominal 90 % target on the held-out calibration set
- **Negative-control suite**: per-disease CI gate that catches hub-bias and hallucinated predictions
- **Head-to-head benchmark**: `scripts/head_to_head_benchmark.py` re-ranks each disease's candidates by every single-pillar baseline vs the v7 ensemble (methods paper §5.9)
- **357 automated tests** across filters, scoring, evidence, conformal, negative-control, per-class, JUMP-CP, selectivity, DepMap, red-team, and regression suites
- **Continuous integration** via GitHub Actions on Python 3.11 and 3.12
- **Prospective validation infrastructure**: `scripts/snapshot_predictions.py` takes timestamped immutable snapshots; `scripts/prospective_monitor.py` re-queries PubMed/CT.gov monthly to compute rolling precision@K on predictions older than 90 days; `scripts/retrospective_prospective.py` scores predictions against 2024-2025 publications the model never saw

Honest disclosure of the KG retrieval numbers is maintained in the eval reports — we report both the training-contaminated upper bound and the clean edge-stripped number so reviewers can assess both.

## Quick start

```bash
pip install -r requirements.txt
bash setup_data.sh                          # Downloads DRKG, STRING, embeddings (~3GB)

# Single disease search
python -m opencure.cli "Alzheimer's disease"

# Full 93-disease screening (~6 hours cold, ~3-4 hours with evidence cache warm)
python experiments/systematic_screening.py

# After screening: ensemble + conformal + red-team + briefs + dashboard + snapshot
python scripts/finalize_v5.py

# Honest status report anytime
python scripts/honest_scoring.py

# v7 negative-control CI gate
python scripts/verify_negative_controls.py
```

GPU-heavy retraining (KG embeddings, R-GCN, foundation-model precomputes,
93-disease rescreen) runs on Modal — see `docs/modal_runbook.md`.

## Data integrations (2024)

- **Open Targets 24.09** — 83K derived triplets covering drug-target mechanism, gene-disease association, clinical indication
- **ChEMBL 34** (Nov 2024) — 94,717 DrugBank-mapped drug-target bioactivities (median IC50/Ki in nM)
- **CPIC** + **PharmGKB** (2025-07) — pharmacogenomic annotations
- **GTEx v8** — median TPM expression for 54 tissues × ~56K genes
- **STRING v12** — 473K high-confidence protein-protein interactions
- **HGNC complete set** — Ensembl↔Entrez↔symbol mapping for 41K+ genes

## Disease coverage

93 curated diseases currently — 22 neglected tropical (Chagas, Dengue, HIV, Malaria, Leishmaniasis, Schistosomiasis, TB, Hepatitis C, African trypanosomiasis, Onchocerciasis, Lymphatic filariasis, Leprosy, Buruli ulcer, and more), 19 rare (Sickle Cell, Gaucher, Fabry, Duchenne MD, Niemann-Pick, Pompe, Hunter syndrome, Spinal muscular atrophy, and more), plus neurodegenerative, cancer (incl. pediatric), cardiovascular, metabolic, autoimmune, respiratory, and neuropsychiatric.

Pool of **2,507 MeSH-indexed diseases** with ≥5 gene associations available for cloud-scale screening via `scripts/mass_screen.py`.

40 of the NTD + rare diseases carry partnership-ready outreach briefs at `docs/outreach/` — the four lead diseases (Schistosomiasis, Chagas, Sickle Cell, Niemann-Pick) get deep curation with named labs and suggested assays.

## Architecture

```
Disease name
  ▼ Find disease entities (DRKG + PrimeKG + OT alias resolution)
  ▼ 13 scoring pillars run in parallel
  ▼ Hard filters
      SMILES rules → metabolite blacklist → IUPAC heuristic
      → ChEMBL phase → critical ADMET (FDA-bypass per stage)
  ▼ Pillar grouping + hub-degree normalization
      KG group (RRF of TransE + RotatE + PrimeKG + unified + R-GCN)
      Structural group (max of fingerprints + MoLFormer-XL + DTI + JUMP-CP)
      Network group (max of proximity + gene signatures)
  ▼ Grouped combiner + calibrated XGBoost ensemble
      per-disease-class head routing (6 classes) → shared-head fallback
  ▼ Conformal-prediction wrapper (90%-coverage interval + prediction set)
  ▼ Evidence gathering (cached, 4,000× speedup on repeat queries)
      PubMed + ClinicalTrials.gov + FAERS + Semantic Scholar
  ▼ Clinical guardrails layer
      Dose plausibility + DDI + pharmacogenomics + triangulation + tissue
  ▼ v7 surfacing layer
      selectivity panel + DepMap essentiality + mechanism-uncertainty
  ▼ Mechanism path resolution (bounded BFS on filtered DRKG subgraph)
  ▼ Adversarial red-team critique per top-K candidate
  ▼ Confidence + novelty assessment
  ▼ Wet-lab brief generation (1-page Markdown per disease)
  ▼ Dashboard + JSON + CSV exports
  ▼ Immutable snapshot with content hash + manifest_hash provenance
```

## Repository structure

```
opencure/
  eval/                  Held-out + time-sliced benchmarks, negative-control suite, disease classes
  filters/               Hard filters (metabolite blacklist, name heuristics)
  scoring/               13 pillars + common types + grouped combiner + hub normalize +
                         conformal + per-class ensemble + selectivity + DepMap + JUMP-CP + red-team + wet-lab brief
  evidence/              PubMed/CT.gov/FAERS/Semantic Scholar + DDI + PGx + dose + triangulation +
                         tissue + mechanism-uncertainty + cache
  data/                  DRKG + PrimeKG + Open Targets loaders
  web/                   FastAPI app + crowd validation endpoint
  log_setup.py           Structured logging + timing metrics
tests/                   357 tests (unit + integration + regression + schema + v7 suites)
experiments/
  systematic_screening.py  Full 93-disease pipeline
  eval/                    Held-out metric reports
  results/                 Per-disease JSON + briefs/ + aggregated database
scripts/
  finalize_v5.py           Post-rescreen regeneration (ensemble + conformal + red-team + briefs + dashboard + snapshot)
  phase_c_pipeline.py      XGBoost ensemble + per-class head training
  calibrate_conformal.py   Fit the conformal calibrator
  verify_negative_controls.py  Negative-control CI gate
  head_to_head_benchmark.py    Single-pillar vs ensemble benchmark
  precompute_embeddings.py / precompute_esm2_embeddings.py / precompute_jump_cp.py / precompute_depmap.py
  red_team_v7.py / generate_wetlab_briefs.py / generate_outreach_briefs.py
  retrospective_prospective.py  2024-2025 publication validation
  modal_app.py             Modal serverless GPU orchestration
  snapshot_predictions.py / zenodo_upload.py / honest_scoring.py / compute_data_manifest.py / mass_screen.py
data/
  manifest.json            Provenance hash of every tracked source
  prospective/snapshots/   Content-fingerprinted prediction archives
docs/
  index.html               Live dashboard
  methods_paper_draft.md   Peer-review-ready methods writeup
  about.md                 Mission, ethics, current state
  modal_runbook.md         Modal GPU retrain runbook
  output_schema.md         Canonical result-JSON schema
  lab_outreach_briefs.md   Index of the 40 per-disease outreach briefs
  outreach/                40 per-disease partnership briefs
.github/workflows/
  tests.yml                CI on every push (Python 3.11 + 3.12)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact contributions: (1) wet-lab validation of top-ranked predictions, (2) JUMP Cell Painting raw-image foundation-model rerank, (3) allosteric-pocket prediction over AlphaFold-3 structures, (4) cell-type-resolved expression integration.

## License

Apache 2.0 — free for academic, commercial, and nonprofit use. Patent grant applies for pharmaceutical and biotech applications.

## Citation

```bibtex
@misc{bartos2026opencure,
  title  = {OpenCure: An Open 13-Pillar Drug Repurposing Platform with
            Calibrated Uncertainty, Adversarial Red-Teaming, and
            Prospective Validation},
  author = {Bartos, Simon},
  year   = {2026},
  version = {v7},
  url    = {https://github.com/SimonBartosDev/opencure},
  note   = {Zenodo DOI pending; snapshot fingerprints available at data/prospective/snapshots/}
}
```
