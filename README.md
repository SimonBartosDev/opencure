<p align="center">
  <h1 align="center">OpenCure</h1>
  <p align="center">
    <strong>Open-source multi-pillar AI platform for drug repurposing with integrated clinical guardrails</strong>
  </p>
  <p align="center">
    <a href="https://github.com/SimonBartosDev/opencure/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python"></a>
    <a href="https://simonbartosdev.github.io/opencure/"><img src="https://img.shields.io/badge/Explorer-Live_Dashboard-orange.svg" alt="Dashboard"></a>
    <img src="https://img.shields.io/badge/Tests-79%2Fpassing-brightgreen.svg" alt="Tests">
    <img src="https://img.shields.io/badge/Version-v5-purple.svg" alt="Version">
  </p>
</p>

---

## What OpenCure does

OpenCure screens ~10,500 FDA-approved and investigational drugs against any disease using **11 independent AI scoring pillars**, then layers **clinical guardrails** on every top prediction so results are actionable, not just ranked.

Every top-10 prediction surfaces:
- **Dose plausibility** — is the drug's clinical plasma Cmax high enough to engage the predicted target? (stage-2 ChEMBL 34 bioactivities)
- **Drug-drug interactions** — top 10 dangerous co-prescriptions from 1.4M DrugBank DDI edges
- **Pharmacogenomic flags** — CPIC/PharmGKB variant warnings (HLA, CYP, VKORC1, etc.)
- **Mechanism path** — natural-language graph path `Drug →[inhibits]→ Target →[linked to]→ Disease`
- **Triangulation score** — agreement across 4 independent axes (KG + docking + Pharos target-development-level + literature)
- **Tissue context** — GTEx tissue-specific expression modifier

Every prediction is **content-fingerprinted** (SHA-256) and traceable to an immutable `data_manifest_hash` covering the 15 source files that produced it — critical for reproducibility in peer review.

**[Browse live predictions →](https://simonbartosdev.github.io/opencure/)**

## Why this matters

Drug development takes 10-15 years and costs >$2B. Repurposing approved drugs skips most of safety testing because the drugs are already proven tolerable in humans. The blocker has been: which drugs to test for which diseases, out of tens of millions of possibilities?

OpenCure's answer: screen them all computationally with 11 orthogonal methods, surface the predictions where methods converge, and give clinicians/researchers the clinical context they need to decide whether a prediction is worth testing in their lab.

## The 11 scoring pillars

| # | Pillar | Signal | Source |
|---|---|---|---|
| 1 | **TransE** | Knowledge graph embedding | DRKG 5.87M edges |
| 2 | **RotatE** (PyKEEN) | KG embedding with relation rotations | DRKG |
| 3 | **TxGNN** | Graph neural network (Harvard) | 60 pre-computed diseases |
| 4 | **Molecular fingerprints** | Morgan/ECFP structural similarity | RDKit |
| 5 | **ChemBERTa** | Transformer embedding of SMILES | HuggingFace |
| 6 | **Gene signatures** | Disease expression reversal | L1000CDS2 + OT × ChEMBL mechanistic reversal |
| 7 | **Network proximity** | Shortest-path on PPI | STRING v12 (473K high-confidence edges) |
| 8 | **Mendelian randomization** | Genetic causal evidence | Open Targets GraphQL |
| 9 | **ADMET** | Toxicity + drug-likeness (77 endpoints) | Chemprop |
| 10 | **PrimeKG** | Independent KG scoring | Harvard 8.1M edges |
| 11 | **DeepPurpose DTI** | Drug-target binding affinity | Pre-trained on BindingDB |

Pillars are **grouped before combining** to avoid double-counting (KG-group via RRF; structural-group via max; network-group via max), then weighted by learned importances from a calibrated XGBoost ensemble (AUC-ROC 0.997 on held-out pairs; isotonic-calibrated so `score=0.7` ≈ 70% precision).

## Clinical guardrails (v5)

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
"mechanistic_hypothesis": "Donepezil —[treats]→ Alzheimer's disease"
```

## Validation

- **Held-out random split**: 993 DrugBank treats pairs held out, scored against the full 10,551-compound candidate pool
- **Time-sliced benchmark**: 210 post-2020 approved drug-disease pairs (drugs with `yearOfFirstApproval >= 2020` from OT 24.09), for testing generalization beyond the 2020-era knowledge graph
- **Edge-stripped retraining**: `scripts/strip_heldout_edges.py` removes 1,200 test-set edges from DRKG+PrimeKG+OT before training a clean model
- **79 automated tests** across filters, scoring, evidence, and regression suites (catches the 3-pillar silent-zero bug class)
- **Continuous integration** via GitHub Actions on Python 3.11 and 3.12
- **Prospective validation infrastructure**: `scripts/snapshot_predictions.py` takes timestamped immutable snapshots; `scripts/prospective_monitor.py` re-queries PubMed/CT.gov monthly to compute rolling precision@K on predictions older than 90 days

Honest disclosure of the KG retrieval numbers is maintained in `experiments/eval/v5_honest_score.txt` — we report both the training-contaminated upper bound and the clean edge-stripped number so reviewers can assess both.

## Quick start

```bash
pip install -r requirements.txt
bash setup_data.sh                          # Downloads DRKG, STRING, embeddings (~3GB)

# Single disease search
python -m opencure.cli "Alzheimer's disease"

# Full 61-disease screening (~6 hours cold, ~1 hour with evidence cache warm)
python experiments/systematic_screening.py

# After screening: regenerate dashboard + Zenodo snapshot + honest-scoring report
python scripts/finalize_v5.py

# Honest status report anytime
python scripts/honest_scoring.py
```

## Data integrations (2024)

- **Open Targets 24.09** — 83K derived triplets covering drug-target mechanism, gene-disease association, clinical indication
- **ChEMBL 34** (Nov 2024) — 94,717 DrugBank-mapped drug-target bioactivities (median IC50/Ki in nM)
- **CPIC** + **PharmGKB** (2025-07) — pharmacogenomic annotations
- **GTEx v8** — median TPM expression for 54 tissues × ~56K genes
- **STRING v12** — 473K high-confidence protein-protein interactions
- **HGNC complete set** — Ensembl↔Entrez↔symbol mapping for 41K+ genes

## Disease coverage

61 curated diseases currently — neglected tropical (Chagas, Dengue, HIV, Malaria, Leishmaniasis, Schistosomiasis, TB, Hepatitis C), rare (Sickle Cell, Gaucher, Fabry, Duchenne MD, Ehlers-Danlos, Fragile X, Marfan, Neurofibromatosis), neurodegenerative, cancer, cardiovascular, metabolic, autoimmune, respiratory, neuropsychiatric.

Pool of **2,507 MeSH-indexed diseases** with ≥5 gene associations available for cloud-scale screening via `scripts/mass_screen.py`.

## Architecture

```
Disease name
  ▼ Find disease entities (DRKG + PrimeKG + OT alias resolution)
  ▼ 11 scoring pillars run in parallel
  ▼ Hard filters
      SMILES rules → metabolite blacklist → IUPAC heuristic
      → ChEMBL phase → critical ADMET (FDA-bypass per stage)
  ▼ Pillar grouping + hub-degree normalization
      KG group (RRF of TransE + RotatE + PrimeKG + unified)
      Structural group (max of fingerprints + ChemBERTa + DTI)
      Network group (max of proximity + gene signatures)
  ▼ Grouped combiner or learned XGBoost ensemble
  ▼ Evidence gathering (cached, 4,000× speedup on repeat queries)
      PubMed + ClinicalTrials.gov + FAERS + Semantic Scholar
  ▼ Clinical guardrails layer
      Dose plausibility + DDI + pharmacogenomics + triangulation + tissue
  ▼ Mechanism path resolution (bounded BFS on filtered DRKG subgraph)
  ▼ Confidence + novelty assessment
  ▼ Dashboard + PDF reports + JSON + CSV exports
  ▼ Immutable snapshot with content hash + manifest_hash provenance
```

## Repository structure

```
opencure/
  eval/                  Held-out + time-sliced benchmarks, ground truth
  filters/               Hard filters (metabolite blacklist, name heuristics)
  scoring/               11 pillars + common types + grouped combiner + hub normalize + mechanistic reversal
  evidence/              PubMed/CT.gov/FAERS/Semantic Scholar + DDI + PGx + dose + triangulation + tissue + cache
  data/                  DRKG + PrimeKG + Open Targets loaders
  web/                   FastAPI app + crowd validation endpoint
  log_setup.py           Structured logging + timing metrics
tests/                   79 tests (unit + integration + regression)
experiments/
  systematic_screening.py  Full 61-disease pipeline
  eval/                    Held-out metric reports
  results/                 Per-disease JSON + aggregated database
scripts/
  finalize_v5.py           Post-rescreen regeneration (dashboard + snapshot + scoring)
  phase_c_pipeline.py      XGBoost ensemble training
  snapshot_predictions.py  Immutable prospective snapshots
  zenodo_upload.py         DOI minting for snapshots
  honest_scoring.py        Full audit report
  compute_data_manifest.py Data provenance tracking
  mass_screen.py           Scale to 2,507 diseases
data/
  manifest.json            Provenance hash of every tracked source
  prospective/snapshots/   Content-fingerprinted prediction archives
docs/
  index.html               Live dashboard
  methods_paper_draft.md   Peer-review-ready methods writeup
  lab_outreach_briefs.md   Per-disease briefs for PI contact
.github/workflows/
  tests.yml                CI on every push (Python 3.11 + 3.12)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact contributions: (1) wet-lab validation of top-ranked predictions, (2) 2024-native KG retrain on CUDA GPU, (3) cell-type-resolved expression integration.

## License

Apache 2.0 — free for academic, commercial, and nonprofit use. Patent grant applies for pharmaceutical and biotech applications.

## Citation

```bibtex
@misc{bartos2026opencure,
  title  = {OpenCure: An Open Multi-Pillar Drug Repurposing Platform with
            Integrated Clinical Guardrails and Prospective Validation},
  author = {Bartos, Simon},
  year   = {2026},
  url    = {https://github.com/SimonBartosDev/opencure},
  note   = {Zenodo DOI pending; snapshot fingerprints available at data/prospective/snapshots/}
}
```
