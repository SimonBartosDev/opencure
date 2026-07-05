# OpenCure — Partnership pitch for biotech & pharma repurposing teams

*10-slide pitch deck in markdown. Convert to Google Slides / Keynote /
Powerpoint via any markdown-to-slides tool; every slide is already
self-contained with a headline + 3-5 bullets + one data point.*

---

## Slide 1 — The problem you already know

**Drug repurposing's bottleneck is not candidates — it's credible
candidates.**

- Every biotech repurposing team sees 10-100× more candidate hypotheses
  than they can test
- The expensive step is the go / no-go call, not the list generation
- Internal pipelines (PandaOmics, proprietary KGs) are closed,
  non-auditable, and hard to cross-check

**Ask:** how much of your repurposing team's time goes into
triangulating single candidates across external evidence sources?
*OpenCure automates that.*

---

## Slide 2 — What OpenCure is

**A fully open-source drug repurposing platform: an honest,
leak-controlled evaluation instrument plus a narrow genetics-anchored
triage tool, with integrated clinical guardrails.**

- 13 independent scoring pillars (KG embeddings ×4, R-GCN, TxGNN,
  structural ×2, DTI, network, genetic, transcriptomic, and JUMP Cell
  Painting image-based phenotypic screening). **Honest caveat:** under
  leak-free, popularity-baselined evaluation the KG-embedding,
  chemical-structure and cell-morphology pillars do **not** beat a
  trivial popularity baseline; the only component that does is
  genetics-anchored target prioritization.
- Screens ~10,500 approved + investigational compounds against any
  disease
- Every top-K prediction auto-populated with:
  - A conformal interval with nominal 90 % coverage on the calibration
    split — **not** a validated probability that a candidate is correct
  - Adversarial red-team critique (seven failure modes checked)
  - Dose plausibility (Cmax/IC50 against predicted target)
  - DDI warnings (from 1.4M DrugBank edges)
  - Pharmacogenomic flags (CPIC + PharmGKB)
  - Mechanism path (graph-native natural-language explanation)
  - 4-axis triangulation score (KG + docking + Pharos TDL + literature)
  - Selectivity panel + DepMap pan-essentiality flag

**Apache 2.0**. No rate limits. No vendor lock-in. Patent grant
applies to pharmaceutical and biotech uses.

---

## Slide 3 — How it's different from the closed tools

| | PandaOmics | BenevolentAI | TxGNN (academic) | **OpenCure v7** |
|---|---|---|---|---|
| Open-source | ✗ | ✗ | ✓ (code) | ✓ (code + data + models) |
| Scoring pillars | proprietary | proprietary | 1 (GNN) | 13 fused (leak-free: only the genetics-anchored one beats a popularity baseline) |
| Conformal interval | ✗ | ✗ | ✗ | ✓ (nominal 90 % coverage on the calibration split — not a validated correctness probability) |
| Clinical guardrails bundled | partial | partial | ✗ | ✓ (dose/DDI/PGx/triangulation) |
| Adversarial red-team pass | ✗ | ✗ | ✗ | ✓ (per top-K candidate) |
| Prospective timestamping registry | ✗ | ✗ | ✗ | ✓ (content-hashed, Zenodo DOI; records predictions for future checking — no validated outcome yet) |
| Time-sliced benchmark | ✗ | ✗ | ✓ | ✓ (210 post-2020 indications) |
| Mechanism-path explanations | partial | ✓ | ✗ | ✓ (graph-native, every prediction) |
| Cost to test | $$ enterprise | $$$ enterprise | free | free |

---

## Slide 4 — What's already built (technical proof)

**14M-triple unified knowledge graph** (DRKG + PrimeKG + Open Targets 24.09)

**357 automated tests** across 13 test modules + GitHub Actions CI
→ the class of bugs you'd find in internal tools is gone

**Foundation models** — MoLFormer-XL (chemistry) + ESM-2 150M
(proteins) + JUMP Cell Painting morphological profiles
→ current-generation embeddings, not 2021-era ones

**Conformal interval** — empirical 90.1 % coverage at the nominal
90 % target on the calibration split → every score ships with an
interval of nominal coverage, **not** a validated probability that the
candidate is correct

**Evidence cache** with 4,174× verified speedup on repeat queries
→ screen a new disease in minutes, not hours

**Data manifest provenance** — every prediction carries the SHA-256
hash of every source file that produced it
→ reproducible 5 years from now, even if we update DRKG

**Immutable prospective timestamping** with Zenodo DOI registration
→ records predictions so they can be checked against future literature.
This is timestamping, not validation: it has produced no validated
outcome and is not evidence of accuracy.

**Current status:** 93-disease systematic screen; v7 architecture
complete (13 pillars + conformal intervals + red-team + per-class
heads), v7 rescreen gated on a GPU retrain cycle. **What actually
beats baseline:** under leak-free evaluation the KG-embedding,
chemical-structure and cell-morphology pillars do not beat a
popularity baseline; the one component that does is genetics-anchored
target prioritization — ~5× popularity on the genetics-covered subset
(Hit@10 20.8 % vs 3.8 %, median rank 64), leak-free and temporally
validated (honest post-2020 Hit@10 ~10 %). It is rediscovery-leaning
and covers only part of diseases (~69 of 93; pathogen-driven NTDs have
no human genetics and are not assessed).

---

## Slide 5 — Honest scoring

We don't oversell. OpenCure is a **hypothesis-generation and triage tool**,
not a validated predictor. Known limits disclosed publicly:

**What we genuinely offer**
- A systematic, reproducible pipeline that ranks, critiques, and documents
  **triage hypotheses for expert review** across 13 scoring methods.
  Under leak-free, popularity-baselined evaluation, the KG-embedding,
  chemical-structure and cell-morphology pillars do **not** beat a
  popularity baseline; the one component that does is genetics-anchored
  target prioritization (~5× popularity on the genetics-covered subset:
  Hit@10 20.8 % vs 3.8 %, median rank 64), leak-free and temporally
  validated (honest post-2020 Hit@10 ~10 %) — but it is rediscovery-leaning
  and covers only part of diseases (~69 of 93; pathogen-driven NTDs have
  no human genetics and are not assessed)
- A conformal interval with nominal 90 % coverage on the calibration split
  on every top candidate — **not** a validated probability that the
  candidate is correct
- Evidence-triangulated, clinically-annotated output — dose, DDI, PGx,
  mechanism path, and an adversarial red-team critique on every candidate
- Full reproducibility — content-hashed manifests, open Apache-2.0 code,
  357 automated tests

**What we do NOT claim — and why**
- **No benchmark accuracy figure.** An earlier "AUC-ROC 0.997" was found to
  be data leakage (KG features scored from a graph containing the test
  edges) and has been withdrawn. A leak-free retrain scores far lower, and
  on a fair temporal test the ensemble is at chance.
- **Predictive accuracy is unestablished.** A leak-free retrospective
  benchmark is not currently possible — the 2020-vintage knowledge graph
  predates the repurposing events needed to test it. We do not know, and do
  not claim, how often a top candidate is correct.
- **Zero wet-lab-confirmed predictions.** Experimental validation is exactly
  what a lab partnership provides — and the honest reason to collaborate.

Full disclosure: `docs/architecture.md` (honest evaluation section) and the
eval reports under `experiments/eval/`.

---

## Slide 6 — What a partnership looks like

**Tier 1 — Use the platform** (free, immediate)

- Pull predictions for your indications from the dashboard or API
- Cite OpenCure in internal decision docs; request source-data
  lineage for any specific prediction
- No obligation; no cost

**Tier 2 — Contribute data** (revenue-neutral, mutual)

- Supply anonymized assay results on predictions you tested
- OpenCure credits your contribution in the public prospective
  registry (your brand gets a published precision@K — a
  co-occurrence / later-mention rate, not a validated accuracy figure)
- You get early access to the re-calibrated model trained on your
  real-world data

**Tier 3 — Funded collaboration** (structured, scoped)

- Joint screening of a named disease / compound library
- Custom clinical-layer configuration (your dose / indication / PGx
  library)
- Named co-authorship on methods or results paper
- Typical engagement: 3-6 months, scoped milestones, IP-clean work
  product under Apache 2.0

---

## Slide 7 — The specific ask for neglected-tropical partners

We have **40 lab outreach briefs** ready for NTD + rare diseases
(`docs/lab_outreach_briefs.md` + `docs/outreach/`), with deep curation
for four lead diseases:

- **Schistosomiasis** — DNDi / Conor Caffrey lab (UC San Diego) /
  Imperial-Wellcome SCI Foundation
- **Chagas** — DNDi Chagas cluster / Fundação Oswaldo Cruz
- **Sickle Cell** — CureSCi consortium / Doris Duke Foundation
- **Niemann-Pick** — Ara Parseghian Medical Research Foundation / NPUK

Each brief carries top predictions, a suggested assay matched to the
disease class, a concentration range, and named target labs.

**Ask for these partners:** screen 1-5 compounds from our top-10
triage hypotheses in your validated assay. We pay compound cost. You
publish the outcome regardless of result. OpenCure timestamps it in
the prospective registry (recorded for future checking — not itself
evidence of accuracy). *These are triage hypotheses for expert review;
zero OpenCure predictions are wet-lab confirmed and no novel credible
lead has been found — experimental validation is exactly what this
partnership provides.*

---

## Slide 8 — Who benefits and how

**For biotech repurposing teams** — faster triage of candidate list,
with attached DDI + PGx context your clinicians were going to ask
about anyway.

**For pharma affiliates** — a defensible "we checked open platforms"
step in your due-diligence trail.

**For contract research orgs** — a structured input for hypothesis
generation that cites itself.

**For academic labs** — free compute scored against the same KG that
your grant application cites.

**For patients** — more predictions tested per dollar of research
funding, especially in neglected-disease space where commercial
incentives are weakest.

---

## Slide 9 — What we need

**Immediate (next 90 days)**

- 3 wet-lab partnerships from Tier 2 or Tier 3 track for
  neglected-tropical indications
- GPU compute for the v7 retrain cycle (~$40-55 one-time, or
  research-credit grant)
- One pharma citation of OpenCure in a preprint or patent filing

**Medium-term (6-12 months)**

- v7 93-disease rescreen + bioRxiv methods paper (target: Nature
  Machine Intelligence or Bioinformatics)
- First wet-lab confirmed prediction published as co-authored preprint
- Zenodo DOI series with published rolling precision@K (a
  co-occurrence / later-mention rate, not a validated accuracy figure)

**Long-term (2+ years) — the v8 roadmap**

- JUMP Cell Painting raw-image foundation-model rerank
- Real molecular docking (Boltz-1 / Gnina over AlphaFold-3 structures)
- Drug-combination scoring + active-learning loop
- Cell-type-resolved target scoring (CellxGene / Tabula Sapiens)
- Prospective precision@K accumulated over 12+ months of registry
  calendar time (a co-occurrence / later-mention rate, not a validated
  accuracy figure)

---

## Slide 10 — Contact

**GitHub:** github.com/SimonBartosDev/opencure
**Dashboard:** simonbartosdev.github.io/opencure
**Email:** imon.bartos@gmail.com
**Zenodo DOI series:** content-hashed snapshots at `data/prospective/snapshots/`

**Repo status:** v7 — 13 pillars + conformal intervals + red-team, 357
tests passing, CI green. Under leak-free evaluation only the
genetics-anchored pillar beats a popularity baseline.

**License:** Apache 2.0 with patent grant.

---

## Appendix A — Selected predictions (examples for credibility)

*Note: these are computational predictions, not wet-lab validated, and
are v5-vintage — they will be refreshed with v7 conformal intervals
after the v7 rescreen. Triangulation score indicates
independent-evidence agreement.*

| Disease | Drug | Combined score | Triangulation axes | Status |
|---|---|---|---|---|
| Schistosomiasis | Oxamniquine | 0.62 | 3 (KG + Pharos + lit) | Known treatment (positive control) |
| Sickle Cell | Senicapoc | 0.35 | 2 (KG + lit) | Failed Phase 3 2011 — correct disease-specificity |
| Malaria | Erythromycin | 0.46 | 3 (KG + Pharos + lit 55 PMIDs + 10 trials) | Documented repurposing candidate |
| Malaria | Clarithromycin | 0.53 | 3 (KG + Pharos + lit 13 PMIDs) | Published repurposing |
| Tuberculosis | Dexamethasone | 0.52 | 3 (KG + lit + trials) | Standard adjunct for TB meningitis |

*Full candidate lists with clinical-layer attachments: see dashboard.*

## Appendix B — Data sources with citations

All sources with their academic citations available at
`docs/methods_paper_draft.md` section "Data".
