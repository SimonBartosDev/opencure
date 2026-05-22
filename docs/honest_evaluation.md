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

Three independent signal families — graph, chemistry, phenotype — none beats
popularity. The diagnosis: all three score a drug by *similarity to a disease's
known treatments*, and a disease's treatments are mechanistically heterogeneous
(hypertension is treated by beta-blockers, diuretics, ACE inhibitors — unalike
by every measure). "Be similar to that set" is an incoherent target that
rewards only being a well-connected, popular drug.

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
  are scarce — a fundamental, not incidental, limit.
- We did not find a novel, credible, wet-lab-ready repurposing lead, across
  every angle attempted. We report this plainly rather than dress curation up
  as discovery.

## 9. Limitations

Single knowledge graph (DRKG, 2020) for the held-out construction; modest
genetics-covered subset sizes (n≈53–62); no per-indication approval dates
(precluding a fully temporal benchmark); chemical and morphology pillars
limited to compounds with available embeddings/profiles. None of these
weaknesses inflate the headline conclusions; several make the leak-free numbers
*optimistic*.

## 10. Availability

All code, the leak-free instrument, per-pillar scorecards, the genetics-
anchored scorer, the clean direction-of-effect layer, and the per-disease
results are in the repository under `scripts/`, `opencure/scoring/`,
`data/genetics/`, and `experiments/eval/`. Nothing in this report depends on a
closed model or private data.
