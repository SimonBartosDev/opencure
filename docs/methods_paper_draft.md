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
**OpenCure**, an open-source platform that fuses **12 independent evidence
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

### 3.2 The 12 pillars

| # | Pillar | Type |
|---|---|---|
| 1 | TransE on DRKG | KG embedding |
| 2 | RotatE on DRKG (PyKEEN) | KG embedding |
| 3 | TransE on unified KG | KG embedding |
| 4 | PrimeKG TransE | KG embedding |
| 5 | TxGNN (Harvard pre-computed) | GNN |
| 6 | Molecular fingerprints (Morgan) | Structural |
| 7 | ChemBERTa embeddings | Structural |
| 8 | DeepPurpose DTI | Binding |
| 9 | Network proximity (STRING PPI) | Network |
| 10 | L1000 gene signatures | Transcriptomic |
| 11 | Mendelian randomization | Causal |
| 12 | R-GCN heterogeneous GNN | GNN |

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

### 4.1 Random-split held-out (993 pairs)

80/20 split of DRKG DRUGBANK::treats edges. Hit@10 = [X] (training-
contaminated upper bound) / [Y] (edge-stripped clean). MRR = [Z].

### 4.2 Time-sliced benchmark (210 pairs)

Drugs with OT yearOfFirstApproval ≥ 2020 × approved indications (phase ≥ 3).
Edge-stripped training ensures no leakage. Hit@10 = [X], MRR = [Y].

| Year | Pairs |
|---|---|
| 2020 | 91 |
| 2021 | 49 |
| 2022 | 45 |
| 2023 | 25 |

### 4.3 Prospective registry

Each OpenCure release is timestamped and content-hashed; predictions
deposited at Zenodo for immutable record. Monthly re-query of PubMed
and ClinicalTrials.gov for evidence appearing AFTER the snapshot date
yields a rolling precision@K that cannot be retrospectively data-mined.
First snapshot (2026-04): [fingerprint X].

### 4.4 Head-to-head vs published baselines

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
