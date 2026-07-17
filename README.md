<p align="center">
  <h1 align="center">OpenCure</h1>
  <p align="center">
    <strong>An open, leak-controlled evaluation instrument for computational drug repurposing — and the one narrow triage tool that survived it</strong>
  </p>
  <p align="center">
    <a href="https://github.com/SimonBartosDev/opencure/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python"></a>
    <a href="https://simonbartosdev.github.io/opencure/"><img src="https://img.shields.io/badge/Docs-Findings_%26_methods-blue.svg" alt="Docs"></a>
    <img src="https://img.shields.io/badge/Regression_tests-357_passing-brightgreen.svg" alt="Regression tests">
    <img src="https://img.shields.io/badge/Wet--lab_confirmed_predictions-0-lightgrey.svg" alt="Wet-lab confirmed">
  </p>
</p>

---

We built a multi-pillar repurposing platform, then evaluated it under strict
leakage control. Most of its apparent power was leakage or popularity. One
signal — human genetics — carried real, leak-free lift. This README reports
both, and ships the tool that works.

**[Read the honest findings →](https://simonbartosdev.github.io/opencure/)**

## What works: genetics-anchored triage

The one validated tool. It asks a single, coherent question:

> disease → human-genetics causal gene → drug with a curated ChEMBL mechanism on that gene

On the genetics-covered subset of held-out pairs (n=53), leak-free, it ranks the
true drug at **Hit@10 20.8% vs a popularity baseline's 3.8%** (~5×), median rank
**64**. Its lift over popularity is CI-backed (paired win-fraction 0.698, 90% CI
[0.589, 0.804]), and it **survives a temporal holdout** with genetics frozen at
Feb 2020 (Open Targets 20.02, provably predating the 2020–2023 test approvals):
still better than popularity on **38 / 40** post-2020 pairs.

```bash
python scripts/screen_genetics_anchored.py "type 2 diabetes mellitus" --top 6 --offline
python scripts/screen_genetics_anchored.py Disease::MESH:D003924
python scripts/screen_genetics_anchored.py "Crohn disease" --json
```

Accepts a disease name, DRKG entity, MeSH id, or OT id. Flags: `--top N`
(default 20), `--offline` (caches only, no OT API), `--json`. Python API:
`from opencure.scoring.genetics_anchored import score_disease`.

**It refuses when it has no signal.** This is the property we care about most.
Real output for Chagas disease:

```
Disease query : Chagas disease
Verdict       : NOT_ASSESSED

  NOT ASSESSED — no genetics-datasource associations for this disease (it maps
  to Open Targets but has no GWAS / rare-variant / clinical-genetics gene evidence)

  OpenCure has no genetic signal for this disease and will not
  emit a ranking it cannot support.
```

A repurposing tool that declines to answer when it doesn't know is the
differentiator. It never fabricates a ranking to fill the space.

**[Genetics triage dashboard →](https://simonbartosdev.github.io/opencure/genetics_dashboard.html)**

## What we found

Leak-controlled evaluation against a held-out set, with a trivial popularity
(knowledge-graph node-degree) baseline as the comparator. A method earns its
place only by beating that baseline.

| Signal family | Hit@10 | median rank | Popularity baseline | Verdict |
|---|---|---|---|---|
| **Genetics-anchored** (covered subset, n=53) | **20.8%** | **64** | 3.8% / rank 499 | **~5× — works** |
| Knowledge-graph embedding, leak-free TransE (n=960) | 3.1% | 538 | 2.7% / rank 871 | no useful lift |
| Chemical structure (ChemBERTa) | 4.4% | 473 | 3.8% / rank 407 | ties / no gain |
| Cell morphology (JUMP Cell Painting) | 2.8% | 1293 | 3.0% / rank 397 | loses |
| Fused multi-pillar score | *no accuracy figure* | — | — | does not beat popularity |

Three independent similarity families — graph, chemistry, phenotype — and none
beats popularity. The diagnosis: each scores a drug by *similarity to a
disease's known treatments*, but a disease's treatments are mechanistically
unalike (hypertension is treated by beta-blockers, diuretics, ACE inhibitors).
"Be similar to that set" is an incoherent target that rewards only being a
well-connected, popular drug.

### The KG contamination result (n=960)

The project's best-evidenced result. At **fixed pool** (24,313), **fixed degree
baseline** (from the stripped graph), **fixed tie-aware mid-rank**, **fixed
scorer** and **fixed query relations** — so the only difference between arms is
whether the embeddings saw the held-out edges in training:

| Arm | Hit@10 | median rank |
|---|---|---|
| TransE trained on the full graph (contaminated) | **52.8%** | 8 |
| The SAME model retrained leak-free | **3.1%** | 538 |
| Popularity (node-degree) baseline | 2.7% | 871 |

**Contamination = 49.7pp.** The original 57.2% headline was overwhelmingly
*leakage*, not a small-pool artifact — re-measured on the honest 24,313 pool it
is still 52.8%. Clean TransE does not usefully beat degree: its edge is 4 pairs
in the top 10 out of 960, and it scores 0.0% under a different but equally
defensible relation config. Artifact:
[`experiments/eval/leakfree_kg_scorecard.json`](experiments/eval/leakfree_kg_scorecard.json).

> An earlier version of this README reported an ensemble **"AUC-ROC 0.997"**.
> That figure was an artefact of **data leakage** — the knowledge-graph features
> were scored from a graph that still contained the test edges — and has been
> **withdrawn**. There is no validated benchmark accuracy figure for the
> multi-pillar score.

### And it rediscovers

Where genetics is strong, drug development usually already happened, so the top
genetics-anchored lead is typically the disease's existing drug. **Across every
angle tested, no novel, credible, wet-lab-ready lead was found. Zero predictions
are wet-lab confirmed.** The honest prospective Hit@10 under the temporal
holdout is a modest **~10%** (an uncorrected 32.6% was ~2/3 posterior-inflated).
See [docs/conditional_lift_validation.md](docs/conditional_lift_validation.md).

## What OpenCure is — and is not

**It is:**
- An open, leak-honest **evaluation instrument** that measures how much apparent
  repurposing signal is really leakage or popularity.
- A narrow **genetics-anchored triage tool** for diseases where human genetics
  implicates a causal target — which refuses when it has none.
- A generator of transparent, reproducible hypotheses for expert review.

**It is not:**
- **Not a validated predictor.** No trustworthy benchmark accuracy figure exists;
  the fused multi-pillar score does not beat a trivial popularity baseline.
- **Not a discovery engine.** Where the genetics signal works, it largely
  rediscovers a disease's existing drug.
- **Not wet-lab validated.** Zero predictions have experimental confirmation.

Treat every output as a structured, transparent hypothesis for expert review —
not a recommendation.

## Reproduce the results

Every number above is reproducible from this repository.

```bash
# Leak-free per-pillar benchmark (similarity families vs popularity)
python scripts/leakfree_benchmark.py

# KG contamination result (contaminated vs leak-free TransE vs popularity)
python scripts/leakfree_kg_benchmark.py

# Genetics conditional lift over popularity
python scripts/popularity_residualized_lift.py

# Temporal holdout with genetics frozen at Feb 2020
python scripts/popularity_residualized_lift.py \
  data/eval/time_sliced_test.jsonl _temporal_pre2020 \
  data/open_targets/genetics_pre2020_efo.json
```

Artifacts land in `experiments/eval/` — `leakfree_kg_scorecard.json`,
`leakfree_pillar_scorecard.json`, `conditional_lift_report*.json`.

## Coverage and scope

**69 of 93** screened diseases are genetics-covered. The other 24 are returned
as `not_assessed`: 14 with no genetics associations, 8 with no mechanism drug,
2 with no EFO mapping.

Pathogen-driven neglected tropical diseases (Chagas, leishmaniasis,
schistosomiasis) have **no human-genetic causal architecture** and are
structurally out of scope for this approach. OpenCure says so rather than
guessing.

## Install

```bash
pip install -r requirements.txt
bash setup_data.sh    # Downloads DRKG, STRING, embeddings (~3GB)
```

The genetics-anchored CLI above is the shipped entry point. The multi-pillar
pipeline is retained for transparency — it is what the evaluation instrument
measured, and it is **not a validated ranker**:

```bash
python -m opencure.cli "Alzheimer's disease"      # multi-pillar; unvalidated
python experiments/systematic_screening.py         # full 93-disease screen
```

GPU-heavy retraining runs on Modal — see [docs/modal_runbook.md](docs/modal_runbook.md).

## Repository structure

```
opencure/
  scoring/genetics_anchored.py   The shipped genetics-anchored scorer
  scoring/                       Multi-pillar scorers (retained, unvalidated)
  eval/                          Held-out + time-sliced benchmarks, disease classes
  filters/                       Hard filters (metabolite blacklist, name heuristics)
  evidence/                      PubMed/CT.gov/FAERS + DDI + PGx + dose + cache
  data/                          DRKG + PrimeKG + Open Targets loaders
  web/                           FastAPI app
tests/                           357 software regression tests — they check
                                 pipeline correctness, NOT predictive accuracy
scripts/
  screen_genetics_anchored.py    CLI for the shipped tool
  leakfree_kg_benchmark.py       KG contamination scorecard
  leakfree_benchmark.py          Leak-free per-pillar benchmark
  popularity_residualized_lift.py  Conditional lift + temporal holdout
  build_genetics_dashboard.py    Genetics triage dashboard
  build_pre2020_genetics.py      Freezes genetics at OT 20.02 for the temporal test
experiments/
  eval/                          Scorecards and lift reports (the artifacts above)
  results/                       Per-disease JSON + aggregated database
docs/
  index.html                     Honest landing page
  genetics_dashboard.html        Genetics-anchored triage dashboard
  honest_evaluation.md           Full findings: what fails, what works, why
  conditional_lift_validation.md Conditional-lift + temporal validation
  architecture.md                Pillars, fusion, and the evaluation instrument
  about.md                       Mission, ethics, current state
.github/workflows/
  tests.yml                      CI on every push (Python 3.11 + 3.12)
```

## Data integrations

- **Open Targets 24.09** — drug-target mechanism, gene-disease association,
  clinical indication (plus **OT 20.02** frozen for the temporal holdout)
- **ChEMBL 34** — DrugBank-mapped drug-target bioactivities
- **DRKG** — 5.87M-edge knowledge graph, plus `drkg_stripped.tsv` with all
  held-out edges removed for leak-free training
- **STRING v12**, **GTEx v8**, **CPIC** + **PharmGKB**, **HGNC**

## Mission

OpenCure is an open-source, mission-locked, **nonprofit** drug-repurposing
project. We publish negative results as plainly as positive ones: most of a
multi-pillar architecture's apparent power was evaluation leakage, only human
genetics carried leak-free signal, and we found no novel wet-lab-ready lead. We
release the leak-free instrument so others can hold their own tools to the same
standard.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact contributions:
(1) closing the direction-of-effect data gap with drug-independent sources,
(2) genetics coverage for diseases currently returned `not_assessed`,
(3) independent replication of the leak-free benchmarks.

## License

Apache 2.0 — free for academic, commercial, and nonprofit use. Patent grant
applies for pharmaceutical and biotech applications.

## Citation

```bibtex
@misc{bartos2026opencure,
  title  = {OpenCure: A Leak-Controlled Evaluation of Computational Drug
            Repurposing, and a Genetics-Anchored Triage Tool},
  author = {Bartos, Simon},
  year   = {2026},
  url    = {https://github.com/SimonBartosDev/opencure},
  note   = {Reports negative results; no wet-lab-confirmed predictions.}
}
```
</content>
