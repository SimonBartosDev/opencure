---
title: OpenCure v7 — Architecture
description: How the OpenCure drug-repurposing platform works, end to end.
---

# OpenCure v7 — How it works

🌐 **English** · [Slovenčina](architecture.sk.html)

**[← Live dashboard](index.html)** · [About & mission](about.html) · [Methods paper](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md) · [GitHub](https://github.com/SimonBartosDev/opencure)

OpenCure is an open-source, mission-locked platform that ranks existing
FDA-approved and clinically-staged drugs against neglected tropical
diseases, rare genetic disorders, and other under-served indications.
This page explains the whole architecture — every pillar, every fusion
step, and every honesty layer — in one place.

The design principle throughout: **no single method is trusted alone,
every prediction ships with calibrated uncertainty, and every output is
adversarially critiqued before a human sees it.**

---

## 1. The problem

Drug development takes 10–15 years and costs over $2 billion. Repurposing
an already-approved drug skips most of the safety pipeline — the drug is
already known to be tolerable in humans. The bottleneck is not *ideas*;
it is *credible* ideas: which of tens of millions of drug–disease pairs
are worth a wet-lab experiment?

OpenCure's answer is to score every pair with 13 independent methods,
fuse them, calibrate the uncertainty honestly, adversarially critique
each surviving candidate, and hand a wet-lab scientist a one-page brief
they can act on.

---

## 2. The pipeline at a glance

```
Disease name
  │
  ▼  Resolve to knowledge-graph entities (DRKG + PrimeKG + OpenTargets aliases)
  │
  ▼  13 scoring pillars run in parallel
  │
  ▼  Hard filters  (SMILES sanity → metabolite blacklist → IUPAC heuristic
  │                 → ChEMBL phase → critical-ADMET, with an FDA-approved bypass)
  │
  ▼  Pillar grouping + hub-degree damping
  │     KG group        = Reciprocal Rank Fusion(TransE, RotatE, PrimeKG, Unified, R-GCN)
  │     Structural group = max(fingerprints, MoLFormer-XL, DTI, JUMP Cell Painting)
  │     Network group    = max(STRING proximity, gene-signature reversal)
  │
  ▼  Calibrated ensemble  (XGBoost + isotonic; per-disease-class head routing)
  │
  ▼  Conformal-prediction wrapper  (90%-coverage interval + binary prediction set)
  │
  ▼  Evidence gathering  (PubMed + ClinicalTrials.gov + FAERS + Semantic Scholar; cached)
  │
  ▼  Clinical guardrails  (dose plausibility, DDI, pharmacogenomics, triangulation, tissue)
  │
  ▼  v7 surfacing layers  (selectivity panel, DepMap essentiality, mechanism uncertainty)
  │
  ▼  Adversarial red-team critique  (seven failure modes checked per candidate)
  │
  ▼  Wet-lab brief generation  (one-page Markdown per disease)
  │
  ▼  Dashboard + JSON + CSV + content-hashed prospective snapshot
```

Every stage is **fail-open**: if an artifact is missing (a model not yet
trained, a dataset not yet downloaded), that pillar contributes an empty
result and the rest of the pipeline carries on. The platform is never
all-or-nothing.

---

## 3. The 13 scoring pillars

Each pillar is an independent estimate of "does this drug treat this
disease?", built on a different kind of evidence. They are deliberately
*orthogonal* — knowledge-graph topology, chemical structure, protein
binding, network biology, genetics, transcriptomics, and cell
morphology are different windows onto the same question.

| # | Pillar | What signal it captures | Data source |
|---|--------|-------------------------|-------------|
| 1 | **TransE** | Knowledge-graph embedding — translational geometry of drug→disease edges | DRKG (5.87M edges) |
| 2 | **RotatE** | KG embedding with relation-as-rotation; captures relation patterns TransE cannot | DRKG, via PyKEEN |
| 3 | **Unified-KG TransE** | KG embedding on the DRKG + PrimeKG + OpenTargets *union* | 14M-edge unified graph |
| 4 | **PrimeKG** | Independent KG embedding on Harvard's precision-medicine graph | PrimeKG (8.1M edges) |
| 5 | **TxGNN** | Graph neural network designed for zero-shot drug repurposing | Harvard TxGNN, pre-computed |
| 6 | **Molecular fingerprints** | 2D structural similarity to known treatments (Morgan / ECFP) | RDKit |
| 7 | **MoLFormer-XL** | Learned chemistry embedding — a transformer pre-trained on 1.1B molecules | IBM MoLFormer-XL |
| 8 | **DeepPurpose DTI** | Predicted drug–target binding affinity; v7 adds ESM-2 150M protein embeddings | BindingDB / ESM-2 |
| 9 | **Network proximity** | Shortest-path distance between drug targets and disease genes on the PPI graph | STRING v12 (473K edges) |
| 10 | **Gene-signature reversal** | Does the drug reverse the disease's transcriptomic signature? | L1000 + OpenTargets × ChEMBL |
| 11 | **Mendelian randomization** | Genetic causal evidence — is the target causally linked to the disease? | OpenTargets GraphQL |
| 12 | **R-GCN** | Heterogeneous graph neural network with a DistMult scoring head | trained on DRKG |
| 13 | **JUMP Cell Painting** | *Phenotypic* similarity — does the drug produce the same cell-morphology change as a known treatment? | JUMP-CP consortium |

**ADMET** (Chemprop drug-likeness and toxicity, 77 endpoints) runs
alongside the pillars but as an orthogonal *multiplier* on the final
score, not as a pillar — a toxic drug should be damped, not averaged.

A few pillars deserve a note:

- **Why four knowledge-graph embeddings (1–4)?** Each KG is built
  differently and has different blind spots. DRKG is broad but
  2020-vintage; PrimeKG is precision-medicine-focused; the unified
  graph merges everything. Fusing them via Reciprocal Rank Fusion is
  more robust than trusting any one.
- **Pillar 13, JUMP Cell Painting**, is the v7 flagship addition. The
  JUMP consortium released ~140K compound *morphological profiles* —
  five-channel fluorescent images of cells perturbed by each compound,
  distilled to a feature vector. OpenCure scores a candidate by how
  closely its morphological profile matches the *centroid* of known
  treatments for the disease. A drug that is structurally novel but
  produces the same cellular phenotype as a known treatment is exactly
  the high-value signal repurposing wants — and it is the single
  largest closure of the gap to closed-platform image-based screening.

---

## 4. Pillar grouping and fusion

Several pillars capture overlapping information — four KG embeddings,
say, are highly correlated. Feeding all 13 raw into the ensemble would
let correlated signals dominate by sheer count. So pillars are
**grouped before combining**:

- **KG group** — TransE, RotatE, PrimeKG, Unified, R-GCN are fused with
  **Reciprocal Rank Fusion** (RRF). RRF combines rankings, not raw
  scores, so it is immune to the wildly different score scales of the
  five embeddings.
- **Structural group** — fingerprints, MoLFormer-XL, DTI, and JUMP Cell
  Painting are combined by taking the **maximum** per compound (the
  most-optimistic structural/phenotypic signal).
- **Network group** — STRING proximity and gene-signature reversal are
  combined by **maximum**.
- **Ungrouped** — TxGNN, Mendelian randomization, and ADMET stay
  separate; they are mechanistically distinct enough that grouping
  would lose information.

**Hub-degree damping.** Some drugs (Cimetidine, Dexamethasone, Calcium,
Glutathione) are connected to almost everything in the knowledge graph,
so they score mechanically high for *every* disease. OpenCure applies a
multiplicative penalty to the KG and network groups based on a drug's
graph degree, calibrated against the median degree of phase-≥1 ChEMBL
drugs. The honest accounting of what this fixes — and what bias still
persists — is in [`hub_bias_analysis.md`](https://github.com/SimonBartosDev/opencure/blob/main/docs/hub_bias_analysis.md).

---

## 5. The ensemble — and per-disease-class heads

The grouped scores feed a **calibrated gradient-boosted ensemble**
(XGBoost + isotonic calibration). It is trained on 23,814 drug–disease
pairs and reaches AUC-ROC ≈ 0.997 in 5-fold cross-validation. Isotonic
calibration means a reported `score = 0.7` corresponds to roughly 70 %
empirical precision.

v7 adds **per-disease-class ensemble heads**. The 93 diseases are
grouped into six therapeutic classes — *parasitic, viral, bacterial,
oncology, rare-metabolic, chronic-systemic* — by dominant repurposing
mechanism. Each class with enough training data gets its own logistic
head on top of the shared feature representation, because the signal
that predicts a good anti-helminthic is not the signal that predicts a
good kinase inhibitor. A disease whose class has too few training
positives **falls back to the shared head** — the routing is
fail-safe, never fail-closed.

---

## 6. Conformal prediction — honest uncertainty

A calibrated probability tells you that, *across all* predictions of
0.7, about 70 % are correct. It does *not* tell you how sure the
platform is about *this specific* 0.7 — it could secretly be a 0.5.

v7 closes that gap with **split conformal prediction**. A held-out
calibration set yields an empirical residual quantile; every prediction
then ships with:

- a **distribution-free interval** `[ensemble_prob_lower,
  ensemble_prob_upper]` that contains the true label with ≥ 90 %
  probability, and
- a **binary prediction set**: `{1}` (confidently positive), `{0}`
  (confidently negative), or `{0, 1}` (the platform genuinely cannot
  tell).

Measured empirical coverage is **90.1 %** against the nominal 90 %
target. A wet-lab partner reading `prob 0.7 [0.39–1.00], set {0,1}`
knows the platform is saying "probably, but I am not certain" — which
is the truthful answer, and far more useful than false precision.

---

## 7. Clinical guardrails

What separates OpenCure from a pure ranking engine: every top
prediction is *actionable*. Each carries:

- **Dose plausibility** — is the drug's clinical plasma Cmax high enough
  to engage the predicted target, given ChEMBL bioactivity data?
- **Drug–drug interactions** — the most dangerous co-prescriptions,
  drawn from 1.4M DrugBank DDI edges.
- **Pharmacogenomic flags** — CPIC and PharmGKB variant warnings
  (HLA, CYP, VKORC1, …).
- **Mechanism path** — a natural-language graph path,
  `Drug →[inhibits]→ Target →[linked to]→ Disease`, resolved by bounded
  breadth-first search on the filtered knowledge graph.
- **Triangulation** — agreement across four independent axes
  (knowledge graph, docking, Pharos target-development level,
  literature); ≥ 3 agreeing earns a "silver-standard" label.
- **Tissue context** — a GTEx expression modifier that down-weights a
  prediction when the disease genes are not expressed in the relevant
  tissue.

---

## 8. The v7 honesty layers

v7's theme is *honesty*. Five layers exist specifically to catch the
platform's own failure modes before a human is misled.

- **93-disease negative-control suite.** For each disease, a curated
  list of clinically-implausible compounds (`tests/data/negative_controls.yaml`).
  A continuous-integration gate asserts these rank *below* the
  per-disease median. If a hallucinated prediction creeps in, CI fails.
- **Selectivity panel.** A drug that binds 50 targets at sub-micromolar
  affinity is a toxicity problem, not a clean lead. The selectivity
  score (from ChEMBL off-target counts) damps promiscuous binders.
- **DepMap essentiality flag.** If a drug's primary target is
  *pan-essential* — required for survival in ≥ 80 % of DepMap cell
  lines — the candidate is flagged. Pan-essential targets are druggable
  in oncology but a systemic-toxicity risk elsewhere.
- **Mechanism-uncertainty score.** For many rare diseases the molecular
  mechanism is poorly mapped. A 0–1 confidence score (derived from
  disease-gene mapping density) is attached per disease; below 0.4,
  every prediction is flagged as speculative.
- **Adversarial red-team agent.** Every top-K candidate is critiqued by
  a deterministic adversarial pass that checks seven failure modes:
  single-pillar artifacts, low selectivity, pan-essential targets,
  hub-bias, low mechanism confidence, evidence shortage, and
  failed-trial history. An optional local LLM narrates the critique
  into prose.

---

## 9. The output — wet-lab briefs

The platform's final artifact is not a leaderboard; it is a **one-page
wet-lab brief** per disease. For each top-5 candidate the brief states
the mechanistic hypothesis (cite-grounded), a suggested assay matched
to the disease class, a concentration range derived from the primary
target's potency, the red-team critique, and explicit caveats. Forty
NTD and rare-disease briefs are published under
[`docs/outreach/`](https://github.com/SimonBartosDev/opencure/tree/main/docs/outreach);
the four lead diseases — Schistosomiasis, Chagas, Sickle Cell, and
Niemann-Pick — carry deep curation with named target labs.

---

## 10. Data sources

| Source | What it provides |
|--------|------------------|
| **DRKG** | Drug Repurposing Knowledge Graph — 5.87M edges, the primary KG |
| **PrimeKG** | Harvard precision-medicine knowledge graph — 8.1M edges |
| **Open Targets 24.09** | Gene–disease association, drug-target mechanism, clinical indication |
| **ChEMBL 34** | 94,717 DrugBank-mapped drug–target bioactivities (median IC50/Ki) |
| **STRING v12** | 473K high-confidence protein–protein interactions |
| **GTEx v8** | Median expression for 54 tissues × ~56K genes |
| **L1000** | Transcriptomic perturbation signatures |
| **JUMP Cell Painting** | ~140K compound morphological profiles |
| **DepMap** | CRISPR gene-essentiality across 1000+ cell lines |
| **CPIC + PharmGKB** | Pharmacogenomic annotations |
| **HGNC** | Gene-identifier mapping for 41K+ genes |
| **MoLFormer-XL, ESM-2** | Foundation-model embeddings (chemistry, protein) |

---

## 11. Validation strategy

- **Held-out random split** — 993 DrugBank treats-pairs held out,
  scored against the full 10,551-compound pool.
- **Time-sliced benchmark** — 210 drug–disease pairs approved *after*
  2020, to test generalisation beyond the 2020-vintage knowledge graph.
- **Edge-stripped retraining** — test-set edges are removed from
  DRKG + PrimeKG + OpenTargets before a clean model is trained, so
  retrieval numbers are not inflated by memorisation.
- **Conformal coverage** — empirical 90.1 % at the 90 % nominal target.
- **Negative-control suite** — the CI gate described in §8.
- **Head-to-head benchmark** — each disease's candidates re-ranked by
  every single-pillar baseline versus the fused ensemble.
- **Retrospective-prospective validation** — predictions made against
  pre-2024 data are checked against 2024–2025 publications the model
  never saw.
- **357 automated tests** across filters, scoring, evidence, conformal,
  negative-control, per-class, JUMP-CP, selectivity, DepMap, red-team,
  and regression suites, run on every commit via GitHub Actions.

---

## 12. Reproducibility

Every result file carries a `data_manifest_hash` — a SHA-256
fingerprint of every input-data file that produced it. Every model
checkpoint is content-hashed. Predictions are written to immutable,
timestamped **prospective snapshots** with Zenodo DOI registration, so
a claim made today can be verified against future literature. The
pipeline version is stamped on every output.

---

## 13. Honest limitations

OpenCure is, by design, candid about what it cannot do:

- **No proprietary phenotypic-screen data.** Closed platforms
  (Recursion, Insitro) train on billions of in-house cell images.
  OpenCure cannot replicate that and does not claim to.
- **No prospective wet-lab validation yet.** Until a partner lab
  confirms a prediction, the platform's prospective predictive power is
  unproven — retrospective metrics could be measuring leakage.
- **The knowledge graph is 2020-vintage.** A quarterly refresh against
  current ChEMBL / DrugBank / OpenTargets is a v8 work item.
- **Mechanism-uncertainty is a heuristic**, not a Bayesian posterior.
- **The DTI pillar** still uses DeepPurpose's own protein encoder; the
  ESM-2 150M embeddings are staged for a future ESM-2-native DTI head.

The roadmap that addresses these is in
[`ROADMAP.md`](https://github.com/SimonBartosDev/opencure/blob/main/ROADMAP.md).

---

## 14. Where to go next

- **[Live dashboard](index.html)** — browse predictions across 93 diseases.
- **[About & mission](about.html)** — why OpenCure is non-profit and mission-locked.
- **[Methods paper draft](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md)** — the peer-review-grade writeup.
- **[Lab outreach briefs](https://github.com/SimonBartosDev/opencure/blob/main/docs/lab_outreach_briefs.md)** — 40 partnership-ready disease briefs.
- **[GitHub repository](https://github.com/SimonBartosDev/opencure)** — all code, Apache 2.0.

*OpenCure surfaces predictions for clinician and researcher review — not
direct-to-patient recommendations. Every prediction is a computational
hypothesis awaiting experimental test.*
