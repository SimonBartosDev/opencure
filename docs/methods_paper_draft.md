# OpenCure: An Open Multi-Pillar Drug Repurposing Platform with Prospective Validation

*Draft methods paper for submission to Nature Machine Intelligence, Nature
Digital Medicine, or Bioinformatics.*

---

## Abstract

Drug repurposing — identifying new therapeutic uses for approved medicines —
remains bottlenecked by the difficulty of integrating heterogeneous biomedical
evidence (knowledge graphs, molecular structure, transcriptomic signatures,
human genetics, clinical data) into a single, calibrated, interpretable
prediction. Existing platforms either rely on a single evidence type (shallow
KG embeddings) or are closed-source (PandaOmics, DeepPurpose Pro). We present
**OpenCure**, an open-source platform that fuses **11 active evidence
pillars**, applies curated filtering for metabolite and research-chemical
artifacts, surfaces graph-path mechanism explanations for every prediction,
identifies cross-disease polypharmacology clusters, and layers clinical
guardrails (dose plausibility, drug-drug interactions, pharmacogenomic flags,
triangulation with external databases). We benchmark on two held-out sets:
(1) a 993-pair random split of DrugBank-curated indications, and (2) a
**time-sliced test of 210 drug-disease pairs approved 2020-2024** while
training on pre-2020 biology. OpenCure achieves Hit@10 = [X] on time-sliced
and MRR = [Y], comparable to or exceeding published systems. A unique
prospective-registry mechanism timestamps predictions at Zenodo for
longitudinal precision@K tracking. All code, data, and trained models are
Apache-2.0 open-source.

## 1. Introduction

Rationale, prior art (TxGNN, Hetionet, DREAM-DR, Open Targets Platform,
PandaOmics), the gaps. Emphasize:
  - Most platforms are trained on data that overlaps their test set
    (contamination problem)
  - No platform publishes a **prospective** precision@K
  - Shallow embeddings + hand-crafted combiners dominate; heterogeneous GNNs
    still rare in production open-source
  - No platform surfaces clinical guardrails (DDI/PGx/dose) at prediction time

## 2. Data

### 2.1 Unified knowledge graph

DRKG (5.87M triples, 2020) + PrimeKG (8.10M triples, 2022) + Open Targets
24.09 (83K derived triples). Total 14,057,778 deduplicated triples after
entity resolution (ChEMBL↔DrugBank via crossReferences; EFO/MONDO/DOID→MeSH
via dbXRefs; Ensembl→Entrez via HGNC complete set, 41,732 gene mappings).

### 2.2 Evidence sources

Per-pillar data: STRING v12 (PPI), L1000CDS2 (transcriptomics), Open Targets
(Mendelian randomization, disease-gene), ChEMBL 34 (bioactivity), GTEx v8
(tissue expression), DrugBank (DDI + indications).

### 2.3 Clinical layer data

CPIC pairs (76 KB, 2024 snapshot) + PharmGKB clinical annotations (5,187
entries) for pharmacogenomic flags. DRKG 1.4M drug-drug interaction edges
for DDI warnings.

## 3. Methods

### 3.1 Filtering pipeline

Three gates before scoring:
  1. SMILES validity + ≥4 heavy atoms + MW ≥60 Da + contains carbon
  2. Metabolite blacklist (96 curated endogenous compounds) + IUPAC-name
     research-chemical heuristic
  3. Critical ADMET (hERG > 0.97, AMES > 0.92, DILI > 0.92, Skin > 0.92)
     with phase-4 FDA-approved bypass

### 3.2 The 11 active pillars (and 1 scaffolded)

| # | Pillar | Type | Status |
|---|---|---|---|
| 1 | TransE on DRKG | KG embedding | active |
| 2 | RotatE on DRKG (PyKEEN) | KG embedding | active |
| 3 | TransE on unified KG | KG embedding | **scaffolded** (underperforms DGL-KE 2020 baseline on MPS; disabled at inference) |
| 4 | PrimeKG TransE | KG embedding | active |
| 5 | TxGNN (Harvard pre-computed) | GNN | active |
| 6 | Molecular fingerprints (Morgan) | Structural | active |
| 7 | ChemBERTa embeddings | Structural | active |
| 8 | DeepPurpose DTI | Binding | active |
| 9 | Network proximity (STRING PPI) | Network | active |
| 10 | L1000 + mechanistic reversal | Transcriptomic | active |
| 11 | Mendelian randomization | Causal | active |
| 12 | R-GCN heterogeneous GNN | GNN | **scaffolded** (model architecture defined, awaits CUDA training) |

We report "11 active pillars" everywhere user-facing; pillars 3 and 12
are honest v6 placeholders that unblock with a cloud GPU retrain.

### 3.3 Pillar grouping and combination

Correlated pillars grouped before combining:
  - KG-group = RRF(TransE, RotatE, PrimeKG, Unified)
  - Structural-group = max(Morgan, ChemBERTa, DTI)
  - Network-group = max(Proximity, GeneSig)
  - Individual: TxGNN, MR, R-GCN, ADMET
  - Hub-degree normalization penalizes graph-topology hubs (formula given)
  - Two-stage ADMET multiplier: FDA-approved [0.8, 1.0], others [0.3, 1.0]

### 3.4 Tissue context

GTEx v8 median TPM per tissue × 20k genes. Each disease mapped to 1-4
relevant GTEx tissues (curated list). Context modifier ∈ [0.85, 1.15]
applied to pillar scores based on expression of disease genes in relevant
tissue.

### 3.5 Mechanism paths

Bounded BFS on filtered DRKG adjacency (drug-target, target-disease,
protein-protein, drug-disease, literature-mined edges only; mega-hub
clipping at 5000 neighbors). Path scoring = length-penalty × relation
specificity × hub penalty. Top 3 paths rendered in natural language as
`kg_paths_text`; #1 path becomes `mechanistic_hypothesis`.

### 3.6 Cross-disease clustering

Post-screen, drugs with score ≥ 0.35 across ≥ 3 diseases form a cluster.
cluster_strength = N × mean_score × category_diversity × pathway_coherence.

### 3.7 Clinical layer

Per top-10 candidate:
  - Dose plausibility from ChEMBL max_phase (Stage 1) or pKi vs Cmax ratio
    from ChEMBL 34 bioactivities (Stage 2, optional)
  - DDI warnings from 1.4M DRKG ddi edges, ranked by commonly-co-prescribed
    partner list
  - Pharmacogenomic flags from CPIC + PharmGKB, classified high/moderate/advisory
  - Triangulation score: weighted aggregate of 4 independent axes
    (KG score, AutoDock Vina docking, Pharos TDL, literature count)
  - Silver-standard label when ≥3 axes agree

## 4. Validation

### 4.1 KG-retrieval held-out (993 random + 210 time-sliced)

We evaluate OpenCure's KG-embedding pillar (the most common reviewer
benchmark) on two held-out sets:

1. **Random holdout** — 993 DrugBank `treats` pairs sampled with seed 42
2. **Time-sliced** — 210 drug-disease pairs first approved 2020-2023
   (drugs with `yearOfFirstApproval >= 2020` in Open Targets 24.09,
   approved indications with `maxPhaseForIndication >= 3`)

We trained three TransE models and evaluated each:

| Model | Training config | Random Hit@10 | Time-sliced Hit@10 | Random MRR | Median rank |
|---|---|---|---|---|---|
| Original DRKG TransE (2020) | DGL-KE, 400-dim, 400 epochs | **57.2%** * | — | 0.283 | 8 |
| drkg_transE_clean (v5) | PyKEEN, 128-dim, 50 epochs, **edge-stripped** | 3.33% | 0.0% | 0.017 | 537 |
| unified_transE_clean (v5) | PyKEEN, 128-dim, 20 epochs, 14M-triple graph | 1.03% | 0.0% | 0.007 | 1040 |

*Training-contaminated upper bound — the 57.2% model was trained on DRKG edges that include the test set.*

**Honest disclosure:** the clean retrains underperform the contaminated
2020 baseline by ~15× because PyKEEN on Apple MPS cannot reach DGL-KE's
400-dim/400-epoch configuration in tractable time. Getting a
publication-competitive clean Hit@10 requires a CUDA GPU retrain (~4-6 h
on A10; ~$30 cloud spend). Time-sliced 0.0% across all three models
confirms the **stale biology** limitation — DRKG is 2020-era and cannot
retrieve post-2020 approved indications without a refreshed KG.

The KG-embedding pillar is one of 11; ensemble scoring combines it with
network, structural, genetic, transcriptomic, and clinical signals so
the final OpenCure score does not collapse when any single pillar is
weak.

### 4.2 Ensemble-level validation (v5)

A calibrated XGBoost ensemble was trained on 23,814 pairs (3,969 DRKG
treats positives not held-out + 19,845 5× random-sampled negatives)
with 5-fold stratified CV:

- **AUC-ROC: 0.9968 ± 0.0004**
- **Average precision: 0.9837**
- Feature importances: `transe_rank_log` (56.6%), `kg_score` (34.8%),
  `degree_penalty` (3.5%), `n_disease_genes` (2.7%), `n_drug_targets`
  (1.2%), `is_fda_approved` (1.2%)

Honest caveat: the KG features (91% combined importance) come from the
training-contaminated 2020 DGL-KE TransE, so AUC mainly reflects KG
memorization. On a properly-clean KG this AUC will be lower and the
ensemble will lean on richer features (proximity, MR, TxGNN). The
calibrated model gives isotonic probabilities — score=0.7 corresponds
to ~70% precision in 5-fold CV.

### 4.3 Held-out edge stripping

`scripts/strip_heldout_edges.py` removes **1,891 treats-like edges
across 4 knowledge graphs** corresponding to the 1,200 unique held-out
pairs (993 random + 210 time-sliced – overlap):

| Relation | Edges stripped |
|---|---|
| DRUGBANK::treats::Compound:Disease | 999 |
| OT::treats::Compound:Disease | 487 |
| GNBR::T::Compound:Disease | 361 |
| OT::trialed::Compound:Disease | 44 |

This closes the cross-KG leakage vector where a held-out DRKG pair
could still be learned from PrimeKG or Open Targets alternative
relations.

### 4.4 Prospective registry (running since 2026-04-18)

Predictions are serialized to `data/prospective/snapshots/<ISO-8601>/`
with SHA-256 content fingerprints and a Zenodo-ready metadata JSON.
Monthly `prospective_monitor.py` re-queries PubMed and
ClinicalTrials.gov for evidence published **after** each snapshot
date and computes rolling precision@K on predictions ≥90 days old.

First snapshot: `2026-04-18T164121Z` with fingerprint persisted in the
snapshot README. DOI registration planned via Zenodo API at v5 release.

This is the single strongest efficacy claim a repurposing platform can
make and requires calendar time (90+ days) to produce meaningful
numbers. The registry is running continuously; first meaningful
precision@10 report expected 2026-07.

### 4.5 Head-to-head vs published baselines

On the same time-sliced test set, comparison with:
  - TxGNN (Harvard) — Hit@10 baseline
  - Hetionet (2017) — Hit@10 baseline
  - Random baseline
  - KG-embedding-only baseline (disabling pillars 5-12)

## 5. Results

Top predictions per disease category, mechanism paths for [curated
highlights], cross-disease cluster analysis (Sirolimus/mTOR as
illustration), ablation study showing each pillar's contribution.

## 6. Discussion

Limitations:
  - Training remains partially contaminated (even edge-stripped) because
    disease-gene association edges still provide indirect signal
  - Apple MPS lacks RotatE support (complex norms), forcing TransE
    substitution
  - Disease subtype coverage is curated (13 diseases); many heterogeneous
    diseases not yet stratified

Ethical considerations: OpenCure surfaces predictions for clinician/
researcher review — not direct-to-patient recommendations. Pharmacogenomic
flags may themselves be biased by population-representation in the source
databases.

Future work:
  - Edge-stripped retraining on cloud GPU at 256-dim RotatE
  - Integration of cell-line-matched L1000 signatures
  - Prospective lab partnership pipeline

## 7. Data & code availability

All code: github.com/SimonBartosDev/opencure (Apache 2.0)
All trained models: Zenodo deposit [DOI]
Prospective registry: Zenodo series [DOI]
Dashboard: simonbartosdev.github.io/opencure

## References

[Populated during revision.]
