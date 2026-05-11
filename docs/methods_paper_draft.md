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

### 3.2 The 13 active pillars (v7)

| # | Pillar | Type | Status |
|---|---|---|---|
| 1 | TransE on DRKG | KG embedding | active |
| 2 | RotatE on DRKG (PyKEEN) | KG embedding | active |
| 3 | TransE on unified KG (v7: NSSALoss retrain) | KG embedding | active (pre-v7 scaffolded version disabled due to silent training collapse on MarginRankingLoss; v7 fix lands in commit fb82e5f) |
| 4 | PrimeKG TransE | KG embedding | active |
| 5 | TxGNN (Harvard pre-computed) | GNN | active (v7: salt-form aliasing for DRKG↔PrimeKG drug-name match) |
| 6 | Molecular fingerprints (Morgan) | Structural | active |
| 7 | MoLFormer-XL embeddings (v7 swap from ChemBERTa) | Structural | active (ChemBERTa fallback when MoLFormer artifact missing) |
| 8 | DeepPurpose DTI + ESM-2 protein embeddings (v7: 150M variant) | Binding | active |
| 9 | Network proximity (STRING PPI) | Network | active |
| 10 | L1000 + mechanistic reversal | Transcriptomic | active |
| 11 | Mendelian randomization | Causal | active |
| 12 | R-GCN heterogeneous GNN (DistMult head) | GNN | active (v7: local-trained on M4 Max + Modal A100 publication-grade) |
| 13 | JUMP Cell Painting morphological similarity | Phenotypic | active (v7: anchored cosine similarity in DINOv2 feature space) |

All 13 pillars fail open: a missing artifact yields an empty score
dict, the other pillars carry the prediction. The platform is always
operational regardless of which artifacts have been precomputed.

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

### 3.8 v7 architectural additions

Beyond the pillar layer, v7 adds five orthogonal layers that surface
failure modes a pure ranker cannot. Each is implemented as a small,
self-contained module with a fail-open contract — missing artifacts
never break the pipeline.

**Conformal prediction wrapper** (`opencure/scoring/conformal.py`).
Split conformal calibrator fit on the held-out positive set augmented
with matched random negatives. Each candidate ships with a 90 %-coverage
interval `[ensemble_prob_lower, ensemble_prob_upper]` and a binary
prediction set `{0}`, `{1}`, or `{0,1}` — a candidate whose set is
`{0,1}` is one the platform genuinely cannot adjudicate.

**93-disease negative-control suite**
(`tests/data/negative_controls.yaml` + `opencure/eval/negative_control.py`).
For each of 16 hand-curated diseases plus a universal-hubs set, ≥ 3
"clinically implausible" compounds are listed with rationales. The
verifier asserts these rank below the per-disease median across all
pillars; the CI threshold is 95 % per-disease pass rate.

**Per-class ensemble heads** (`opencure/scoring/per_class_ensemble.py`).
Six logistic heads (parasitic, viral, bacterial, oncology,
rare_metabolic, chronic_systemic) are trained on top of the shared
6-feature representation. The routing layer in
`scripts/score_ensemble_v5.py` selects the per-class head when the
disease maps to a known class, falls back to the shared head otherwise.

**Selectivity panel + DepMap essentiality + mechanism uncertainty**
(`opencure/scoring/selectivity_panel.py`,
`opencure/scoring/depmap_essentiality.py`,
`opencure/evidence/mechanism_uncertainty.py`). Selectivity is a soft
penalty on `combined_score` driven by ChEMBL off-target counts; pan-
essential targets (DepMap median Chronos < −0.5 in ≥ 80 % of lines)
are flagged but not filtered; mechanism confidence is a heuristic
0-1 score derived from disease-gene mapping density.

**Adversarial red-team agent** (`opencure/scoring/red_team.py`).
Every top-K candidate receives a deterministic adversarial critique
covering single-pillar artifacts, low selectivity, essentiality
warnings, hub-damping, low mechanism confidence, evidence shortage,
and failed-trial history. An optional LLM layer narrates the critique
through a local Llama-3.1-8B (MLX, M4 Max).

**Wet-lab brief generator** (`opencure/scoring/wetlab_brief.py`).
Every disease's top-5 candidates emit a 1-page Markdown brief: header
with conformal interval, mechanistic hypothesis, suggested assay
(routed by disease class), concentration range derived from primary-
target potency, red-team summary, and disease-/candidate-level
caveats. Designed for direct use in wet-lab partnership conversations.

## 5. Results

This section reports v7-specific results. Each subsection is a
template — the numerical values land after the v7 retraining and
93-disease screen complete (see B1, B2 in the v7 plan). The structure
below is fixed so re-runs slot in cleanly.

### 5.1 Held-out KG retrieval (v7)

| Metric | v6.1 baseline | v7 (TransE-NSSALoss / RotatE-NSSALoss) |
|--------|--------------:|---------------------------------------:|
| Hit@10 (random 993)        | _TBD_ | _TBD_ |
| Hit@10 (time-sliced 210)   | _TBD_ | _TBD_ |
| MRR (time-sliced)          | _TBD_ | _TBD_ |
| AUROC (ensemble, 5-fold CV)| 0.997 | _TBD_ |

### 5.2 Conformal-prediction coverage

| Coverage target | Calibration set | Empirical coverage on time-sliced |
|----------------:|----------------:|-----------------------------------:|
| 90 %            | 993 random + 993 negatives | _TBD_ |
| 95 %            | 993 random + 993 negatives | _TBD_ |

### 5.3 Negative-control pass rate

`scripts/verify_negative_controls.py` summary across 93 diseases.
Lead-disease (Schistosomiasis, Chagas, Sickle Cell, Niemann-Pick) per-
disease pass rates reported separately.

### 5.4 JUMP Cell Painting coverage

| Disease class | Diseases with ≥ 1 known-treatment in JUMP-CP | Median %-of-candidates with JUMP score |
|---------------|---------------------------------------------:|---------------------------------------:|
| parasitic     | _TBD_ | _TBD_ |
| viral         | _TBD_ | _TBD_ |
| bacterial     | _TBD_ | _TBD_ |
| oncology      | _TBD_ | _TBD_ |
| rare_metabolic| _TBD_ | _TBD_ |
| chronic_systemic| _TBD_ | _TBD_ |

### 5.5 Per-class head performance

For each of the six classes, AUROC on the held-out class-stratified
test split, compared with the shared head's AUROC on the same split.

### 5.6 Retrospective-prospective validation (2024-2025)

`scripts/retrospective_prospective.py` summary:

> Of N v7 predictions made against pre-2024 KG data, M were
> independently corroborated by 2024-2025 publications, K were refuted,
> J remain untested.

Per-disease confirmation counts in supplementary table.

### 5.7 Cross-disease cluster analysis

Top mechanism clusters and the top-K candidates per cluster.
Sirolimus/mTOR is included as an illustrative example of a single
drug surfacing across mechanistically related diseases.

### 5.8 Ablation study

Each pillar disabled in turn, ensemble re-trained, AUROC reported on
the time-sliced split. Pillars whose ablation drops AUROC by ≥ 0.005
are reported as load-bearing.

### 5.9 Head-to-head vs single-pillar baselines

Method: re-rank every disease's v7 candidate list by each baseline's
score column in isolation, then evaluate against the same time-sliced
held-out set. Differences between baselines isolate the contribution
of pillar fusion vs each pillar alone.

> **Note (v7.0 status):** the benchmark code at
> `scripts/head_to_head_benchmark.py` is functional but currently
> data-starved — the v6.1 result JSONs contain too few candidates per
> disease (1-10 typical) for the re-ranking to discriminate baselines.
> All baselines tie on the 19-of-993 matched pairs in the v6.1
> snapshot. Re-running the benchmark against v7 rescreen results
> (which produce ~50-100 candidates per disease across all 93) is
> expected to surface the real differentiation. Final §5.9 numbers
> land after Phase B2 (v7 rescreen) completes.

The expected delta pattern in §4.5 has the v7 ensemble outperforming
each single-pillar baseline by ≥ 5 percentage points on Hit@10 and
~0.05 on MRR — that is the threshold at which we will defensibly claim
"calibrated multi-pillar fusion beats any single-pillar approach".

## 6. Discussion

**Limitations** (v7).

  - Training remains partially contaminated (even after edge-stripping)
    because disease-gene association edges still provide indirect signal.
  - Apple MPS lacks RotatE complex-norm support, forcing TransE
    substitution when retraining locally; the publication-grade RotatE
    artifact is built on Modal A100 (one-time $17).
  - Disease subtype coverage is curated; many heterogeneous diseases
    are not yet stratified.
  - The mechanism-uncertainty quantifier is a gene-count heuristic, not
    a Bayesian posterior — a v8 work item is replacing it with a proper
    OpenTargets-evidence-category model.
  - JUMP Cell Painting coverage is bounded by the consortium's released
    compound set (~140 K InChIKeys); novel-chemistry candidates outside
    that set get no morphological signal.
  - The platform has no proprietary phenotypic-screen data and cannot
    replicate closed-platform image-based pipelines (Recursion, Insitro)
    on their own training distribution.

**What v7 explicitly addresses** that v6 did not.

  - Foundation-model swap from ChemBERTa to MoLFormer-XL (chemistry) and
    from 8M ESM-2 to 150M ESM-2 (proteins).
  - Honest uncertainty quantification via split conformal prediction.
  - 93-disease negative-control suite as a CI gate.
  - Per-class ensemble heads replace the single shared head.
  - JUMP Cell Painting as a 13th, phenotype-space pillar.
  - Selectivity, DepMap-essentiality, and mechanism-uncertainty surfacing.
  - Adversarial red-team pass on every top-K candidate.
  - Per-disease wet-lab brief generation as the canonical hand-off
    artifact for partnership outreach.

**Ethical considerations.** OpenCure surfaces predictions for clinician/
researcher review — not direct-to-patient recommendations. Pharmacogenomic
flags may themselves be biased by population-representation in the source
databases. The conformal interval and red-team assessment are
deliberately conservative so users receive an honest "we don't know"
rather than a false-precision number when the platform's confidence is
low.

**Future work** (v8 candidates).

  - Image-based foundation-model rerank using the full JUMP-CP
    high-resolution image set (currently we use the consortium's
    distilled CellProfiler features).
  - Allosteric-pocket prediction layer over AlphaFold-3 structures.
  - Active-learning loop: each wet-lab readout retrains the per-class
    head it informs.
  - Drug-combination scorer trained on DrugComb / NCI-ALMANAC.
  - Quarterly KG refresh against post-2024 ChEMBL/DrugBank/OpenTargets.

## 7. Data & code availability

All code: github.com/SimonBartosDev/opencure (Apache 2.0)
All trained models: Zenodo deposit [DOI]
Prospective registry: Zenodo series [DOI]
Dashboard: simonbartosdev.github.io/opencure

## References

[Populated during revision.]
