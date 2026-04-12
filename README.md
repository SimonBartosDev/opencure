<p align="center">
  <h1 align="center">OpenCure</h1>
  <p align="center">
    <strong>Open-source AI drug repurposing platform for neglected, rare, and underserved diseases</strong>
  </p>
  <p align="center">
    <a href="https://github.com/SimonBartosDev/opencure/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-green.svg" alt="Python"></a>
    <a href="https://simonbartosdev.github.io/opencure/"><img src="https://img.shields.io/badge/Explorer-Live_Dashboard-orange.svg" alt="Explorer"></a>
  </p>
</p>

---

OpenCure combines **8 independent AI scoring methods** and **6 evidence sources** to identify new therapeutic uses for existing FDA-approved drugs. All predictions, code, and data are freely available.

> **342 candidates** across 36 diseases | **34 breakthrough predictions** with zero published literature | **AUC-ROC 0.998** ensemble validation

## Scoring Pillars

| # | Pillar | Method | What It Measures |
|---|--------|--------|-----------------|
| 1 | TransE | Knowledge graph embeddings | Statistical patterns in 5.87M biomedical relationships |
| 2 | RotatE | Complex rotation embeddings | Symmetric/asymmetric relation patterns |
| 3 | TxGNN | Graph neural network | Zero-shot drug-disease predictions |
| 4 | Fingerprints | Morgan/ECFP molecular similarity | Structural similarity to known treatments |
| 5 | ChemBERTa | Transformer embeddings | Learned functional drug similarity |
| 6 | Gene Signatures | L1000CDS2 reversal | Drugs that reverse disease gene expression |
| 7 | Network Proximity | STRING PPI shortest paths | Drug target to disease gene distance |
| 8 | Mendelian Randomization | Open Targets causal evidence | Genetic support for drug targets |

Each drug receives a **dynamic weighted score** across applicable pillars, with convergence bonuses when multiple methods agree. Evidence is gathered from PubMed, ClinicalTrials.gov, Semantic Scholar, FDA FAERS, Open Targets, and L1000CDS2.

## Results

Screening ~10,500 FDA-approved drugs (active screening expanding to 61 diseases):

| Category | Count |
|----------|-------|
| Total candidates | 342 |
| BREAKTHROUGH (no literature) | 34 |
| NOVEL (minimal literature) | 119 |
| HIGH confidence | 168 |
| Diseases screened | 36+ |

Novelty scoring uses **synonym-expanded PubMed search** (55 disease synonym mappings) to avoid false positives. Non-therapeutic compounds (toxic chemicals, pesticides, solvents) are filtered from results.

**Cross-disease discovery**: 19 drugs are predicted across multiple diseases, suggesting shared mechanisms. Doxorubicin appears in 8 diseases.

Browse all predictions interactively: **[OpenCure Explorer](https://simonbartosdev.github.io/opencure/)**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download data (DRKG, STRING, embeddings)
bash setup_data.sh

# Web interface
python -m opencure.web.run

# Command line
python -m opencure.cli "Alzheimer's disease"

# Full screening (61 diseases)
python experiments/systematic_screening.py
```

## Architecture

```
Disease name
  -> 8 AI pillars (parallel scoring)
  -> Per-drug dynamic weighting + convergence bonus
  -> MR causal bonus
  -> Ranked candidates
  -> Evidence reports (PubMed, ClinicalTrials.gov, FAERS, Semantic Scholar)
  -> Confidence + Novelty scoring (synonym-expanded)
```

## Repository Structure

```
opencure/               Core library
  scoring/              8 scoring pillar implementations
  evidence/             Evidence gathering (PubMed, FAERS, etc.)
  web/                  FastAPI web application
experiments/            Screening scripts and results
  results/              JSON/CSV outputs for all diseases
scripts/                Data preparation and report generation
reports/                PDF reports per disease (researcher-ready)
docs/                   Interactive Explorer dashboard (GitHub Pages)
agents/                 Autonomous agent system (literature monitoring, outreach, etc.)
```

## Deployment

```bash
# Docker
docker build -t opencure .
docker run -p 8000:8000 -v opencure_data:/app/data opencure

# Or directly
python -m opencure.web.run
```

## Diseases Screened

Alzheimer's, ALS, Breast Cancer, Chagas, Colorectal Cancer, Cystic Fibrosis, Dengue, Duchenne Muscular Dystrophy, Ehlers-Danlos Syndrome, Fabry Disease, Fragile X Syndrome, Gaucher Disease, Glioblastoma, Hepatitis C, HIV, Huntington's, Idiopathic Pulmonary Fibrosis, Leishmaniasis, Leukemia, Lung Cancer, Lymphoma, Malaria, Marfan Syndrome, Melanoma, Multiple Myeloma, Multiple Sclerosis, Neurofibromatosis, Ovarian Cancer, Pancreatic Cancer, Parkinson's, Prostate Cancer, Pulmonary Hypertension, Schistosomiasis, Sepsis, Sickle Cell Disease, Tuberculosis

*Screening actively expanding to 61 diseases including cardiovascular, autoimmune, metabolic, respiratory, and neuropsychiatric conditions.*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute predictions, code, or validation data.

## License

Apache License 2.0 — free for academic, commercial, and nonprofit use.

## Citation

```bibtex
@article{bartos2025opencure,
  title={OpenCure: An Open-Source Multi-Pillar AI Platform for Systematic
         Drug Repurposing in Neglected and Rare Diseases},
  author={Bartos, Simon},
  year={2025},
  url={https://github.com/SimonBartosDev/opencure}
}
```
