---
title: About OpenCure
description: Mission, ethics, and current state of the OpenCure platform.
---

<div style="position:fixed;top:14px;right:14px;z-index:9999;display:flex;font:700 13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.2)">
<a href="about.html" style="padding:7px 14px;text-decoration:none;background:#2563eb;color:#fff" aria-current="page">EN</a>
<a href="about.sk.html" style="padding:7px 14px;text-decoration:none;background:#fff;color:#57606a">SK</a>
</div>

# About OpenCure

**[← Home](index.html)** · [How it works (architecture)](architecture.html) · [Methods paper](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md) · [GitHub](https://github.com/SimonBartosDev/opencure)

## Mission

OpenCure is an open, mission-locked drug-repurposing platform. We rank
existing FDA-approved and clinically-staged compounds against neglected
tropical diseases, rare diseases, and other under-served indications,
producing triage hypotheses for expert review. Zero predictions are
wet-lab confirmed, and no novel credible lead has been found.

The platform exists to **save lives by collapsing the gap between
computational prediction and laboratory test**. Every prediction
ships with a conformal interval (nominal coverage on the calibration
split — not a validated probability that a candidate is correct), an
adversarial critique, and a 1-page brief a PI can review in under 10
minutes.

We are **non-profit and open source**. All code is Apache 2.0. All
trained models are deposited to Zenodo with a content hash and DOI.
All predictions live on a public website with a citation widget. We
will not pivot to monetisation, will not gate the data behind an
API key, and will not sell predictions to pharma. The platform is
built so that a graduate student in Nairobi or São Paulo can cite,
audit, and extend it as easily as a research group at MIT.

## What OpenCure is — and is not

OpenCure is a **hypothesis-generation and triage tool**. It ranks,
adversarially critiques, and documents drug-repurposing hypotheses, each
with an explicit uncertainty interval, so a wet-lab scientist can decide
what is worth testing.

It is **not a validated predictor.** We publish **no benchmark accuracy
figure** and make no claim about how often a top-ranked candidate is
correct. A leak-free retrospective benchmark is not currently possible —
the knowledge graph predates the repurposing events that would be needed
to test it — so the platform's predictive accuracy is **unestablished**,
honestly stated rather than hidden behind a number. An earlier reported
ensemble figure ("AUC-ROC 0.997") was an artefact of data leakage and has
been withdrawn.

One component does beat baseline: **genetics-anchored target
prioritization beats a popularity baseline ~5× on the genetics-covered
subset** (leak-free, temporally validated, honest temporal Hit@10 ~10 %).
It is rediscovery-leaning — it mostly re-finds a disease's existing drug —
and it covers only part of the diseases screened.

Every OpenCure output is a structured, transparent hypothesis for expert
review — never a recommendation, and never a substitute for experimental
validation.

## What's in v7

Thirteen scoring pillars combined into three groups (knowledge graph,
structural / phenotype, network) plus six ungrouped per-disease-class
signals. The knowledge-graph pillars are correlated embeddings of
overlapping graphs, not independent signals. Under leak-free,
popularity-baselined evaluation the KG-embedding, chemical-structure and
cell-morphology pillars do **not** beat a trivial popularity baseline, and
the fused multi-pillar score does not either; the only component that beats
baseline is genetics-anchored target prioritization. Every top-K candidate
ships with:

- **Conformal interval.** A conformal interval with nominal coverage on
  the calibration split — **not** a validated probability that a candidate
  is correct — and a binary prediction set (`{0}`, `{1}`, or `{0,1}`).
- **Adversarial critique.** Seven failure modes are checked
  automatically for every prediction; an optional local LLM narrates
  the deterministic critique into prose.
- **Triage brief.** A 1-page Markdown summary — a triage hypothesis for
  expert review, not a wet-lab-confirmed lead — including suggested assay,
  concentration range, mechanistic hypothesis, and caveats.
- **Off-target & essentiality flags.** Selectivity score from ChEMBL,
  pan-essentiality flag from DepMap, mechanism-confidence score from
  the OpenTargets gene-association density.

See the [methodology paper draft](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md) for the
full architecture description and validation strategy.

## Current state

- **Architecture:** v7 (13 active pillars, conformal-interval layer, per-
  disease-class ensemble heads, image-based phenotypic similarity,
  selectivity / essentiality / mechanism-uncertainty layers,
  adversarial red-team agent, triage-brief generator). Under leak-free
  evaluation only the genetics-anchored target-prioritization component
  beats a popularity baseline; the KG-embedding, chemical-structure and
  cell-morphology pillars do not.
- **Diseases screened:** 93 (22 NTDs, 19 rare diseases, 18 cancers,
  9 cardiovascular/metabolic, 6 autoimmune, 5 respiratory,
  5 neuropsychiatric, 5 neurodegenerative, 4 other under-served).
- **Test coverage:** 357+ regression tests across 13 test files.
- **Reproducibility:** every result JSON carries the data manifest
  hash and pipeline version that produced it; every model checkpoint
  is content-hashed and shipped to Zenodo.

## Lead diseases (priority outreach)

For four diseases we have written deep partnership briefs with
mechanistic narratives, suggested assays, and target lab affiliations:

- **Schistosomiasis** — DNDi, SCI Foundation, KEMRI, Imperial-Wellcome.
- **Chagas disease** — DNDi, Mundo Sano.
- **Sickle cell disease** — CureSCi consortium, Doris Duke Foundation.
- **Niemann-Pick disease** — Ara Parseghian Medical Research
  Foundation, NPUK.

See [`lab_outreach_briefs.md`](https://github.com/SimonBartosDev/opencure/blob/main/docs/lab_outreach_briefs.md) for the per-disease
briefs (41 NTD + rare-disease briefs in total).

## Team & contact

OpenCure is currently a single-developer project, with the explicit
goal of remaining mission-locked through any future growth. The
governance structure prevents transfer of the platform to a for-profit
entity without a community vote.

For wet-lab partnership inquiries: `imon.bartos@gmail.com`.

## How to cite

When v1 of the methodology paper is published on bioRxiv, the
recommended citation will be linked from the homepage and embedded
on every disease-specific brief.

## License & ethics

Code: Apache 2.0. Data deposits: CC-BY 4.0 where the upstream license
permits. Training data sources retain their original licenses.

OpenCure surfaces predictions for clinician/researcher review — not
direct-to-patient recommendations. We make no claim that any specific
prediction will benefit a specific patient. Pharmacogenomic flags may
themselves be biased by population-representation in the source
databases; this is documented per-prediction.
