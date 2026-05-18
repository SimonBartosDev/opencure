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

**The only fully open-source drug repurposing platform with integrated
clinical guardrails, calibrated uncertainty, and prospective
validation.**

- 13 independent scoring pillars (KG embeddings ×4, R-GCN, TxGNN,
  structural ×2, DTI, network, genetic, transcriptomic, and JUMP Cell
  Painting image-based phenotypic screening)
- Screens ~10,500 approved + investigational compounds against any
  disease
- Every top-K prediction auto-populated with:
  - Calibrated 90 %-coverage conformal interval + prediction set
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
| Scoring pillars | proprietary | proprietary | 1 (GNN) | 13 fused + calibrated ensemble |
| Calibrated uncertainty | ✗ | ✗ | ✗ | ✓ (90 %-coverage conformal interval) |
| Clinical guardrails bundled | partial | partial | ✗ | ✓ (dose/DDI/PGx/triangulation) |
| Adversarial red-team pass | ✗ | ✗ | ✗ | ✓ (per top-K candidate) |
| Prospective validation registry | ✗ | ✗ | ✗ | ✓ (content-hashed, Zenodo DOI) |
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

**Conformal calibration** — empirical 90.1 % coverage at the nominal
90 % target → every score ships with an honest uncertainty interval

**Evidence cache** with 4,174× verified speedup on repeat queries
→ screen a new disease in minutes, not hours

**Data manifest provenance** — every prediction carries the SHA-256
hash of every source file that produced it
→ reproducible 5 years from now, even if we update DRKG

**Immutable prospective snapshots** with Zenodo DOI registration
→ the only repurposing platform that makes claims you can verify
against future literature

**Current status:** 93-disease systematic screen; v7 architecture
complete (13 pillars + calibration + red-team + per-class heads),
v7 rescreen gated on a GPU retrain cycle.

---

## Slide 5 — Honest scoring

We don't oversell. Known limits disclosed publicly:

**What we're good at**
- Multi-pillar ensemble AUC-ROC 0.997 on 23,814 held-out pairs
- Conformal calibration: empirical 90.1 % coverage at the 90 % target
- Evidence-triangulated, clinically-annotated predictions — dose, DDI,
  PGx, mechanism path, conformal interval, and red-team critique on
  every top candidate
- 357 automated tests, GitHub Actions CI green

**What we're not (yet) good at**
- v7 rescreen numbers: the v7 *architecture* is complete and merged,
  but the v7 93-disease *numbers* await a GPU retrain cycle. The
  head-to-head benchmark (§5.9) and final methods-paper Results land
  with that rescreen.
- Clean time-sliced Hit@10 on 2020-era KG retrieval: KG-memorization-
  dependent benchmarks are marked as such.
- Wet-lab validation — zero confirmed predictions yet. This is where
  partnership with your lab adds value.
- Peer review — bioRxiv pending; methods paper drafted.

Full honesty disclosure: `docs/methods_paper_draft.md` §6 and the
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
  registry (your brand gets published precision@K)
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

**Ask for these partners:** screen 1-5 compounds from our top-10 in
your validated assay. We pay compound cost. You publish the outcome
regardless of result. OpenCure logs it as a prospective-validation
data point.

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
- Zenodo DOI series with published rolling precision@K

**Long-term (2+ years) — the v8 roadmap**

- JUMP Cell Painting raw-image foundation-model rerank
- Real molecular docking (Boltz-1 / Gnina over AlphaFold-3 structures)
- Drug-combination scoring + active-learning loop
- Cell-type-resolved target scoring (CellxGene / Tabula Sapiens)
- FDA-referenceable prospective precision@K (requires 12+ months of
  prospective-registry calendar time)

---

## Slide 10 — Contact

**GitHub:** github.com/SimonBartosDev/opencure
**Dashboard:** simonbartosdev.github.io/opencure
**Email:** imon.bartos@gmail.com
**Zenodo DOI series:** content-hashed snapshots at `data/prospective/snapshots/`

**Repo status:** v7 — 13 pillars + calibration + red-team, 357 tests
passing, CI green.

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
