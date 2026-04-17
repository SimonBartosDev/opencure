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

## The Problem

Developing a new drug takes 10-15 years and costs over $2 billion. Meanwhile, thousands of FDA-approved drugs sit on shelves with untapped therapeutic potential. A drug approved for one disease often works for others because most drugs bind to multiple protein targets. This is drug repurposing — and it can skip years of safety testing because the drugs are already proven safe in humans.

The challenge: there are ~10,500 approved drugs and thousands of diseases. Testing every combination in a lab is impossible. That's where computational prediction comes in.

## What OpenCure Does

OpenCure screens all ~10,500 FDA-approved drugs against any disease using **11 independent AI methods**, each capturing a different biological signal. When multiple methods independently agree that a drug should work for a disease, the prediction is likely real.

After scoring, the platform gathers real-world evidence from 6 biomedical databases to assess how confident and how novel each prediction is — separating genuine new discoveries from drugs that are already known to work.

> **588 candidates** across 61 diseases | **59 breakthrough predictions** with zero published literature | **AUC-ROC 0.998** ensemble validation

**[Browse all predictions interactively](https://simonbartosdev.github.io/opencure/)**

## How It Works

### Step 1: Multi-Pillar AI Scoring

Each drug-disease pair is scored by 11 independent methods. No single method is reliable alone — but when 5+ agree, the signal is strong.

| # | Pillar | What It Does | Why It Matters |
|---|--------|-------------|----------------|
| 1 | **TransE** | Embeds drugs and diseases as vectors in the same space using 5.87M biological relationships (DRKG knowledge graph). Drugs "close to" a disease in this space are predicted to treat it. | Captures statistical patterns across all of biomedical knowledge |
| 2 | **RotatE** | A more advanced graph embedding that handles complex relationship types (symmetric, asymmetric, compositional) that TransE misses. | Catches drug-disease links that simpler models overlook |
| 3 | **TxGNN** | A graph neural network from Harvard designed specifically for therapeutic use prediction. Passes messages through the biological network to predict which drugs treat which diseases. | State-of-the-art GNN, 49% improvement over baselines in zero-shot prediction |
| 4 | **Molecular Fingerprints** | Encodes each drug's molecular structure as a bit vector (Morgan/ECFP fingerprints) and compares it to drugs already approved for the target disease. | If a drug looks structurally similar to a known treatment, it likely has a similar mechanism |
| 5 | **ChemBERTa** | A transformer model trained on molecular SMILES strings. Learns functional similarity beyond structural fingerprints — drugs that look different but act the same. | Catches "hidden" similarities that fingerprints miss |
| 6 | **Gene Signatures** | Queries L1000CDS2 for drugs that reverse the disease's gene expression signature. If a disease upregulates gene X, finds drugs that downregulate it. | Directly addresses the molecular root cause |
| 7 | **Network Proximity** | Measures shortest path distance between drug targets and disease genes in the STRING protein-protein interaction network (16,201 proteins, 236,930 interactions). | Closer in the network = more likely to have therapeutic effect |
| 8 | **Mendelian Randomization** | Uses human genetic data (GWAS) as natural experiments. If a genetic variant that mimics a drug's effect also reduces disease risk, that's causal evidence the drug should work. | The strongest form of drug target validation — causal, not just correlational |
| 9 | **ADMET Filtering** | Predicts 77 ADMET endpoints (toxicity, drug-likeness, pharmacokinetics) for each compound using Chemprop models. Filters toxic compounds and scores drug-likeness. | Removes non-viable candidates before expensive downstream analysis |
| 10 | **PrimeKG** | Independent knowledge graph from Harvard with 8.1M relationships across 17,080 diseases. Scored with TransE embeddings, providing orthogonal evidence to DRKG. | Two independent KGs agreeing = much higher confidence |
| 11 | **DeepPurpose DTI** | Deep learning drug-target interaction prediction using molecular graphs + protein sequences. Pre-trained on BindingDB binding affinity data. | Predicts physical drug-protein binding without 3D structures |

Each drug receives a **dynamic weighted score** — pillars with stronger signal for a given drug get higher weight. When multiple pillars converge on the same prediction, a **convergence bonus** increases the score further.

### Step 2: Evidence Gathering

For each top candidate, the platform queries 6 external databases:

- **PubMed** — existing research papers (with synonym-expanded disease queries to avoid false novelty)
- **ClinicalTrials.gov** — active or completed clinical trials
- **FDA FAERS** — real-world adverse event reports that may signal drug-disease co-occurrence
- **Semantic Scholar** — papers with citation counts to identify high-impact evidence
- **Open Targets** — genetic associations and causal evidence
- **L1000CDS2** — transcriptomic data for gene expression reversal

### Step 3: Confidence and Novelty Scoring

Each prediction gets two independent assessments:

**Confidence** (can we trust this prediction?):
- HIGH = multiple evidence types agree (PubMed articles + clinical trials + genetic evidence)
- MEDIUM = some supporting evidence
- LOW = computational prediction only, no external validation yet

**Novelty** (is this actually new?):
- BREAKTHROUGH = zero published literature linking this drug to this disease
- NOVEL = minimal literature (1-10 papers)
- EMERGING = some evidence exists, not yet in trials
- KNOWN = moderate evidence, already being studied
- ESTABLISHED = well-known treatment (serves as positive control to validate the pipeline)

Novelty scoring uses **synonym-expanded PubMed search** across 55 disease name mappings to avoid false breakthroughs. Non-therapeutic compounds (toxic chemicals, pesticides, solvents) are filtered from results.

### Step 4: Cross-Disease Analysis

32 drugs appear as novel predictions across multiple diseases, suggesting shared biological mechanisms. When independent AI methods predict the same drug for different diseases via different pathways, confidence in the prediction increases substantially.

## Results

| Category | Count |
|----------|-------|
| Total candidates | 588 |
| BREAKTHROUGH (zero literature) | 59 |
| NOVEL (minimal literature) | 163 |
| HIGH confidence | 365 |
| Diseases screened | 61 |
| Cross-disease drugs | 32 |

### Validation

A GradientBoosting ensemble trained on 54,775 known drug-disease pairs achieved **AUC-ROC 0.998** on held-out test data, confirming the multi-pillar scoring reliably distinguishes true drug-disease relationships from random pairs.

### Independent Discovery Validation

Several OpenCure predictions independently rediscovered drug-disease relationships that were later confirmed by published wet-lab research — without the system having access to those papers. For example, the platform predicted melatonin for pulmonary fibrosis using only computational methods, matching a decade of published animal model research it had never seen.

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
  -> Evidence gathering (PubMed, ClinicalTrials.gov, FAERS, Semantic Scholar)
  -> Confidence + Novelty scoring (synonym-expanded)
  -> Interactive dashboard + PDF reports + CSV export
```

## Repository Structure

```
opencure/                Core library
  scoring/               8 scoring pillar implementations
  evidence/              Evidence gathering, novelty scoring, confidence assessment
  data/                  DRKG knowledge graph interface, drug name resolution
  web/                   FastAPI web application with search and report APIs
experiments/             Screening pipeline and results
  results/               JSON/CSV outputs for all 61 diseases
  systematic_screening.py  Full multi-disease screening pipeline
scripts/                 Data preparation, report generation, dashboard builder
reports/                 61 PDF reports (one per disease, researcher-ready)
docs/                    Interactive Explorer dashboard (GitHub Pages)
agents/                  Autonomous agent system (literature monitoring, outreach, grants)
```

## Deployment

```bash
# Docker
docker build -t opencure .
docker run -p 8000:8000 -v opencure_data:/app/data opencure

# Or directly
python -m opencure.web.run
```

## Diseases Covered

**Neglected Tropical:** Chagas, Dengue, HIV, Hepatitis C, Leishmaniasis, Malaria, Schistosomiasis, Tuberculosis

**Rare:** Duchenne Muscular Dystrophy, Ehlers-Danlos Syndrome, Fabry Disease, Fragile X Syndrome, Gaucher Disease, Marfan Syndrome, Neurofibromatosis, Sickle Cell Disease

**Neurodegenerative:** Alzheimer's, ALS, Huntington's, Multiple Sclerosis, Parkinson's

**Cancer:** Breast, Colorectal, Glioblastoma, Leukemia, Lung, Lymphoma, Melanoma, Multiple Myeloma, Ovarian, Pancreatic, Prostate

**Cardiovascular & Metabolic:** Atherosclerosis, Atrial Fibrillation, Chronic Kidney Disease, Coronary Artery Disease, Heart Failure, Hypertension, Liver Cirrhosis, Obesity, Osteoporosis, Type 2 Diabetes

**Autoimmune & Inflammatory:** Crohn's, IBD, Lupus, Psoriasis, Rheumatoid Arthritis, Ulcerative Colitis

**Respiratory:** Asthma, COPD, COVID-19, Cystic Fibrosis, Idiopathic Pulmonary Fibrosis, Pulmonary Hypertension, Sepsis

**Neuropsychiatric:** Anxiety, Bipolar Disorder, Depression, Endometriosis, Epilepsy, Schizophrenia

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute predictions, code, or validation data. The highest-impact contribution is experimentally validating a prediction.

## License

Apache License 2.0 — free for academic, commercial, and nonprofit use. Chosen specifically for its patent grant, which is relevant for pharmaceutical and biotech applications.

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
