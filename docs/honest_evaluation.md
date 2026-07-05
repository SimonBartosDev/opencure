---
title: "OpenCure: An Honest, Leak-Controlled Evaluation of Multi-Pillar Drug Repurposing"
description: What survives a rigorous, leakage-controlled evaluation of a 13-pillar computational drug-repurposing platform — and what does not.
---

# An Honest, Leak-Controlled Evaluation of Multi-Pillar Drug Repurposing

*OpenCure findings report. All code, data, and evaluation artifacts are open
(Apache-2.0); every number below is reproducible from the repository.*

## Summary

We built a 13-pillar computational drug-repurposing platform and then
subjected it to a strict, leakage-controlled evaluation. The findings are
sobering and, we believe, useful to a field in which inflated retrospective
benchmarks are common:

1. The platform's original headline metric — an ensemble **AUC-ROC of 0.997** —
   was **data leakage**, not performance.
2. Under leak-free evaluation, **three independent signal families — knowledge-
   graph embeddings, chemical-structure similarity, and cell-morphology
   (Cell Painting) similarity — all fail to beat a trivial popularity
   baseline.** The shared "similarity-to-known-treatments" paradigm collapses
   to "rank popular drugs."
3. A **genetics-anchored, target-based** approach is the one method that beats
   the baseline — **~5× on the genetics-covered subset, leak-free, replicated** —
   but it covers only part of the disease space and **mostly rediscovers known
   drug–target pairs** rather than producing novel repurposing leads.
4. The complete repurposing query (genetically-causal gene + correct direction
   + approved-for-another-disease + never-tried-here) surfaces 894 candidates
   but **none with verifiable direction of effect** from the standard release.
5. Building a **clean, leak-free direction-of-effect layer** (ClinGen dosage
   sensitivity + gnomAD constraint) resurrected **40 direction-concordant
   candidates from zero** — but **every one collapses on inspection**, for two
   structural reasons: (a) a gene's Mendelian/constraint direction is often the
   *opposite* of its role in the complex disease queried (dangerous
   inversions), and (b) constraint evidences almost only **loss-of-function**,
   so concordant drugs must be **activators** — a class that barely exists
   among approved small molecules. A clean direction layer is **necessary but
   not sufficient.**
6. A **non-anchored, target-based** pillar — which asks not "is this drug
   *similar* to known treatments" but "does this drug's **measured** ChEMBL
   bioactivity pharmacologically *hit the disease's causal genes*" — had never
   been benchmarked. Measured leak-free, it **also loses to popularity** (Hit@10
   0.6% vs 3.3%; median rank 12,211 vs 909), for a coverage reason: for ~99% of
   held-out pairs the true drug has no high-affinity measured activity on its
   disease's causal genes.
7. Replacing the constraint proxy with **Open Targets' own disease-specific
   direction of effect** (fully populating its cache) breaks the loss-of-function
   asymmetry — **80 direction-concordant leads, 48 of them inhibitor-class**, and
   it *fixes* the showcase inversion (prostate cancer's AR lead flips from the
   constraint layer's harmful **testosterone** to the correct **anti-androgen**).
   But a datasource audit dissolves the optimism: the leak-free direction that
   exists at scale is **cancer-gene catalogs + Mendelian/mouse knockouts**;
   `gwas_credible_sets` — the disease-specific common-variant source — supplies
   **zero** directional votes. So the method is direction-correct only for
   **oncology** (where it merely **rediscovers** precision-oncology matches) and
   **inverts** outside it (a TLR7 *activator* for lupus, a KCNJ11 *opener* for
   diabetes). **Barrier (a) is relocated, not dissolved.**

**Conclusion: every honest discovery angle was exhausted, each with a clear,
understood reason for failing. In computational drug repurposing, honest
evaluation — not model sophistication — is the binding constraint. We release
the leak-free instrument so others can hold their own tools to the same
standard.**

## 1. Background

Computational drug-repurposing tools routinely report retrospective AUROCs of
0.95–0.99. Such numbers are frequently artifacts of evaluation leakage: the
features encode, directly or indirectly, the very drug–disease links being
predicted. OpenCure began as one such tool and reported AUC-ROC 0.997. This
report documents what happened when we tried to verify that number honestly.

## 2. The leakage case study

The 0.997 came from a gradient-boosted ensemble whose two dominant features
(`transe_rank_log`, `kg_score`; ~90% of feature importance) were derived from
a TransE knowledge-graph embedding trained on **the same DRKG `treats` edges
used as evaluation labels**. The model was, in effect, graded on recalling its
own training graph. A retrained, leak-free ensemble (KG features scored from an
edge-stripped model, trained and tested only on held-out pairs the model never
saw) scored **CV AUROC ≈ 0.72 with hard negatives**, and was **at chance on a
fair temporal test**. The 0.997 has been withdrawn from all materials.

## 3. The leak-free instrument

`scripts/leakfree_benchmark.py` measures each *timeless* pillar (one whose data
is a fixed physical/measured property, not literature-derived) against a 993-
pair held-out set. The single possible leak path — a held-out drug appearing in
its disease's "known-treatment" anchor set — is closed by stripping held-out
treatment edges from the anchors. Rankings use tie-aware mid-ranks against the
full candidate pool. The honest comparator is a **popularity baseline**
(knowledge-graph node degree): a pillar earns its place only by beating it.

## 4. Results — what fails

| Pillar (leak-free) | Hit@10 | Hit@100 | median rank | vs popularity |
|---|---|---|---|---|
| Chemical structure (ChemBERTa) | 4.4% | 21.7% | 473 | ties / no gain |
| **Popularity baseline** | 3.8% | 23.1% | 407 | — |
| Cell-morphology (JUMP Cell Painting) | 2.8% | 8.9% | 1293 | **loses** |
| Popularity baseline (same pool) | 3.0% | 22.4% | 397 | — |
| Knowledge-graph (edge-stripped) | — | — | — | at/below baseline |
| Target-based reversal (genetics-filtered, leak-clean) | 0.6% | 6.7% | 12,211 | **loses** |
| Popularity baseline (same pool) | 3.3% | 14.7% | 909 | — |

Three independent signal families — graph, chemistry, phenotype — none beats
popularity. The diagnosis: all three score a drug by *similarity to a disease's
known treatments*, and a disease's treatments are mechanistically heterogeneous
(hypertension is treated by beta-blockers, diuretics, ACE inhibitors — unalike
by every measure). "Be similar to that set" is an incoherent target that
rewards only being a well-connected, popular drug.

We then tested the obvious alternative, which the instrument had never measured:
a **non-anchored, target-based** pillar that asks "does this drug's *measured*
ChEMBL bioactivity (IC50/Ki) pharmacologically hit the disease's causal genes?"
— a coherent target, unlike similarity. Leak-free (disease genes restricted to
genetics datasources; no drug-derived edge ever read), it loses to popularity
even more decisively (last two rows above). The cause is **coverage**: for
roughly 99% of held-out pairs the true drug has no high-affinity measured
activity against its disease's *genetically-causal* genes — most repurposing
acts through mechanisms not captured as a curated potency on the causal gene —
so the true drug scores zero and sinks into the pool. (A predicted-binding
variant via a DTI model was dropped before benchmarking: the repository's DTI
network has no trained weights, and the only leak-clean alternative is a
*predicted* proxy of the *measured* signal that had just failed.)

## 5. Results — what works, with a caveat

A **genetics-anchored** ranker (disease → human-genetics causal gene → drug
with a curated ChEMBL mechanism on that gene) is the one approach that beats
the baseline. On the subset of held-out pairs where the disease has genetic
evidence (n≈53–62, two independent runs):

| | Hit@10 | Hit@100 | median rank |
|---|---|---|---|
| **Genetics-anchored** | **20.8%** | **69.8%** | **64** |
| Popularity baseline (same) | 3.8% | 28.3% | 499 |

This is consistent with the established finding that genetically-supported drug
targets succeed ~2.6× more often in clinical trials (Minikel et al., *Nature*
2024). The signal is genuine and leak-free. Two caveats bound it:

- **Coverage.** Of 93 screened diseases, 69 are genetics-covered; the rest —
  notably parasitic neglected tropical diseases (Chagas, leishmaniasis,
  schistosomiasis) — have no human-genetic causal architecture and are
  honestly returned as *not assessed*. Genetics-anchored repurposing
  structurally cannot serve pathogen-driven disease.
- **Rediscovery.** For well-studied diseases, strong genetics has *already*
  driven drug development against the implicated target, so the top
  genetics-anchored "lead" is typically the disease's existing drug, not a
  novel repurposing.

## 6. The complete repurposing query, and the binding constraint

To isolate genuinely novel leads we ran the full four-filter query: a drug that
(1) hits a genetically-causal gene for the disease, (2) in the **correct
direction**, (3) is approved for a **different** disease, and (4) has **never
been tried** for this one. Across the 69 covered diseases this surfaced 894
candidates passing filters 1, 3, and 4 — but **0 with verifiable direction
concordance.** The reason is a data gap: in the available Open Targets release,
genetic datasources (GWAS, gene burden) carry no direction-of-effect; only
clinical-precedence evidence does, and that is itself derived from known drugs
(leak-prone). Without direction, a drug that targets a disease's causal gene is
a coin-flip between corrective and harmful — not a credible lead.

## 7. Closing the direction gap — necessary, but not sufficient

We then built the missing layer. From two **genetics/constraint** sources with
no drug-derived signal — **ClinGen dosage sensitivity** (haploinsufficiency →
loss-of-function; triplosensitivity → gain-of-function) and **gnomAD
constraint** (LoF-intolerance) — we assigned a clean LoF/GoF mechanism to 3,299
genes, and 86 of the 236 genes implicated across the covered diseases. Re-running
the four-filter query with real direction concordance, **the count of credible
candidates rose from 0 to 40** across 14 diseases. The concept works
mechanically.

Yet on inspection **all 40 collapse**, for two structural reasons that are
themselves the finding:

1. **Gene-level direction ≠ disease-specific direction.** Constraint/ClinGen
   report a gene's *canonical Mendelian* mechanism, frequently the opposite of
   its role in the GWAS-associated complex disease. *AR* is loss-of-function in
   androgen-insensitivity syndrome but a **gain-of-function driver** in prostate
   cancer (treated with **anti**-androgens) — so the "concordant" lead is
   testosterone, which would be **harmful**. The same inversion recurs for
   *KCNQ1*, *ESR1*, *RARB*. These are dangerous false positives, not leads.
2. **A loss-of-function asymmetry.** All 86 directed genes resolve to LoF,
   because gene-level constraint almost only ever evidences LoF-intolerance
   (only ~2 human genes carry a strong gain-of-function call). LoF → the
   corrective drug must be an **activator/agonist** — a class that barely
   exists among approved small molecules. The single genuinely
   right-direction case, *BMPR2* → pulmonary hypertension, matches only
   **bone-graft proteins** (rhBMP-2/7), not viable systemic therapeutics.

**The clean direction layer is necessary but not sufficient.** It is a valid
*safety filter* — it correctly removes drugs that hit the right gene in the
harmful direction — but it cannot, on its own, certify a credible novel lead,
because it cannot distinguish a gene's Mendelian dosage mechanism from its
complex-disease role.

## 7b. Real disease-specific direction — the barrier is relocated, not dissolved

The constraint layer's two failures are artifacts of its *source*, so we
replaced it with **Open Targets' own disease-specific direction of effect**
(`directionOnTarget` / `directionOnTrait`), and fully populated its evidence
cache for every implicated gene–disease pair (124 → 366 cached queries; the
original §6 "0 concordant" was an *under-populated cache*, not a data void).

The corrected four-filter run looks, at first, like a breakthrough. Against the
constraint layer's "40 leads, all activators," it returns **80 direction-
concordant leads, 48 of them inhibitor-class** — the abundant approved class the
constraint layer structurally could never reach (barrier (b) broken). It even
*fixes* the showcase inversion: prostate cancer's *AR* lead flips from the
constraint layer's harmful **testosterone** (activator) to the correct
**anti-androgen** (inhibitor, *clascoterone*).

Then a datasource audit of *what justified each concordant call* dissolves the
optimism (`experiments/eval/cross_indication_triage.json`). The leak-free
direction that exists at scale is **cancer-gene catalogs** (`cancer_gene_census`
45×, `intogen` 15×) and **Mendelian / mouse-knockout** sources (ClinVar/`eva`
17×, `impc` 29×). The disease-specific common-variant source,
`gwas_credible_sets`, supplies **zero** directional votes — every GWAS row is
direction-free. The 80 leads therefore split along a hard line:

- **39 are oncology + inhibitor.** Direction-correct, because an oncogene's sense
  is unambiguous and well-catalogued (gain-of-function → inhibitor) — but these
  are **precision-oncology rediscoveries or already in basket trials** (EGFR →
  lung, BRAF → melanoma/lymphoma, FGFR → several carcinomas, KRAS → myeloma,
  ALK → neuroblastoma, AR → prostate). Not novel.
- **30 are non-oncology + activator**, resting on Mendelian/mouse direction that
  **inverts** against the complex-disease direction — dangerous false positives:
  a *TLR7 activator* (imiquimod) for lupus, where TLR7 **gain**-of-function
  *drives* lupus; a *KCNJ11 opener* (minoxidil/pinacidil) for type-2 diabetes,
  which would *suppress* insulin. Eleven of the 80 calls rested on conflicting
  votes (AR's correct call was a 1-vs-1 tie resolved by chance). Only **five**
  leads were backed by a genuinely disease-specific source (rare-variant
  `gene_burden`) — all activator rediscoveries (PPARG, MC4R).

So even *real* disease-specific direction is **necessary but not sufficient**,
and for a sharper reason than the constraint layer's: the leak-free direction
that is actually *available* is cancer-driver and Mendelian/mouse — not the
disease-specific common-variant signal a complex disease needs. **Barrier (a) is
relocated, not dissolved.** The method works only where a gene's direction is
unambiguous (oncology), and there it merely rediscovers; everywhere a novel
common-disease lead would live, the direction either does not exist (GWAS) or
inverts (Mendelian). The genuinely missing instrument is **directional GWAS** —
GWAS-eQTL colocalization with sign — which, as of this release, exists neither
in Open Targets nor as a clean bulk download (the published Genetic Priority
Score's directional layer is portal-only, and its score is itself drug-trained,
hence leak-prone).

## 8. Honest conclusions

- Most of a multi-pillar architecture's apparent power can be evaluation
  leakage. We strongly recommend leak-controlled, popularity-baselined
  evaluation as a default.
- Of the signal families we tested, only human genetics carries leak-free
  predictive signal — and only where genetic evidence exists, and largely for
  target *prioritization* (which is already known) rather than novel-lead
  *discovery*.
- Direction of effect, the prerequisite for credible target-based repurposing,
  cannot be supplied by gene-level genetic constraint alone: constraint encodes
  a gene's Mendelian mechanism, not its (often opposite) role in a complex
  disease. Disease-specific direction would require GWAS-eQTL colocalization
  with directional integration — a substantially harder data problem.
- The loss-of-function asymmetry of human constraint means genetics-anchored
  repurposing is structurally biased toward needing *activator* drugs, which
  are scarce — a fundamental, not incidental, limit. Real disease-specific
  direction (Open Targets `directionOnTrait`) lifts this asymmetry — 48 of 80
  concordant leads become inhibitor-class — confirming the limit was the
  *source*, not the concept.
- But real direction only *relocates* the disease-specificity gap. The leak-free
  direction available at scale is cancer-gene-catalog and Mendelian/mouse;
  `gwas_credible_sets` carries none. So target-based repurposing is
  direction-trustworthy only in **oncology** — where it rediscovers known
  precision matches — and inverts on common disease. The decisive missing
  instrument is **directional GWAS** (GWAS-eQTL colocalization with sign), which
  is not yet available leak-free in bulk.
- A *non-anchored* target pillar (does a drug's measured bioactivity hit the
  disease's causal genes?) is a coherent question but fails on **coverage**: the
  true repurposing drug almost never has a curated high-affinity potency on its
  disease's genetically-causal gene. Measured pharmacology is too sparse, and a
  predicted-binding proxy is both weaker and unavailable (no trained DTI model).
- We did not find a novel, credible, wet-lab-ready repurposing lead, across
  every angle attempted — now including non-anchored target scoring and real
  disease-specific direction of effect. We report this plainly rather than dress
  curation up as discovery.

## 9. Limitations

Single knowledge graph (DRKG, 2020) for the held-out construction; modest
genetics-covered subset sizes (n≈53–62); no per-indication approval dates
(precluding a fully temporal benchmark); chemical and morphology pillars
limited to compounds with available embeddings/profiles. None of these
weaknesses inflate the headline conclusions; several make the leak-free numbers
*optimistic*.

## 10. Availability

All code, the leak-free instrument (now scoring non-anchored target pillars as
well as anchored-similarity ones), per-pillar scorecards, the genetics-anchored
scorer, the constraint and Open Targets direction-of-effect layers, the
concordant-lead triage (`scripts/triage_cross_indication.py` →
`experiments/eval/cross_indication_triage.json`), and the per-disease results
are in the repository under `scripts/`, `opencure/scoring/`, `data/genetics/`,
and `experiments/eval/`. Nothing in this report depends on a closed model or
private data.

## 11. Addendum (June 2026): conditional-lift and temporal validation

The one signal that beats baseline — genetics-anchored target prioritization —
was subsequently tested for *incremental* lift over popularity, not just
side-by-side ranking, and against a leak-clean popularity baseline (degree from
`drkg_stripped.tsv`, not the full graph that still held the held-out edges). The
genetics signal is **independent of popularity** — it wins in the *low*-degree
quartiles where popularity scores 0% and loses only among high-degree hubs — and
it **survives a temporal post-2020 holdout** — and, decisively, survives it
using genetics frozen at **Feb 2020** (Open Targets 20.02), which provably
predates every test approval: the ranker still beats popularity on 38/40 pairs
(90% CI [0.90, 1.0]) where popularity scores 0%. The headline temporal Hit@10 is
honestly *deflated* by removing posterior-contamination (32.6% with current
genetics → **10.0%** with provably-pre-2020 genetics), but the prospective
*lift over popularity* is genuine. This confirms the genetics-anchored ranker as
a genuine **prioritizer** in the genetics-covered regime, not a popularity
artifact — while leaving every novel-discovery conclusion above intact (a
directioned-survivor audit found no novel, non-oncology, credibly-directioned
lead). Full numbers and method: [`docs/conditional_lift_validation.md`](conditional_lift_validation.md);
instrument: `scripts/popularity_residualized_lift.py`.
