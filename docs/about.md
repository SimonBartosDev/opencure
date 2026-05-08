# About OpenCure

## Mission

OpenCure is an open, mission-locked drug-repurposing platform. We rank
existing FDA-approved and clinically-staged compounds against neglected
tropical diseases, rare diseases, and other under-served indications,
producing predictions a wet lab can act on.

The platform exists to **save lives by collapsing the gap between
computational prediction and laboratory test**. Every prediction
ships with calibrated uncertainty, an adversarial critique, and a
1-page brief a PI can review in under 10 minutes.

We are **non-profit and open source**. All code is Apache 2.0. All
trained models are deposited to Zenodo with a content hash and DOI.
All predictions live on a public website with a citation widget. We
will not pivot to monetisation, will not gate the data behind an
API key, and will not sell predictions to pharma. The platform is
built so that a graduate student in Nairobi or São Paulo can cite,
audit, and extend it as easily as a research group at MIT.

## What's in v7

Thirteen orthogonal scoring pillars combined into three orthogonal
groups (knowledge graph, structural / phenotype, network) plus six
ungrouped per-disease-class signals. Every top-K candidate ships with:

- **Calibrated uncertainty.** A 90 %-coverage conformal interval and
  a binary prediction set (`{0}`, `{1}`, or `{0,1}`).
- **Adversarial critique.** Seven failure modes are checked
  automatically for every prediction; an optional local LLM narrates
  the deterministic critique into prose.
- **Wet-lab brief.** A 1-page Markdown summary including suggested
  assay, concentration range, mechanistic hypothesis, and caveats.
- **Off-target & essentiality flags.** Selectivity score from ChEMBL,
  pan-essentiality flag from DepMap, mechanism-confidence score from
  the OpenTargets gene-association density.

See the [methodology paper draft](methods_paper_draft.md) for the
full architecture description and validation strategy.

## Current state

- **Architecture:** v7 (13 active pillars, calibrated uncertainty, per-
  disease-class ensemble heads, image-based phenotypic similarity,
  selectivity / essentiality / mechanism-uncertainty layers,
  adversarial red-team agent, wet-lab brief generator).
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

See [`lab_outreach_briefs.md`](lab_outreach_briefs.md) for the per-disease
briefs (40 NTD + rare-disease briefs in total).

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
