---
title: "Leakage, not learning: a controlled ablation of knowledge-graph drug repurposing and the popularity baseline the field omits"
description: A single-variable ablation measuring how much retrospective drug-repurposing performance is test-set leakage (49.7 pp), why a node-degree baseline is the control the field omits, and what survives — genetics-anchored target prioritization under a frozen pre-2020 evidence snapshot.
---

# Leakage, not learning: a controlled ablation of knowledge-graph drug repurposing and the popularity baseline the field omits

*Preprint draft. All code, data and evaluation artifacts open under Apache-2.0.*

## Abstract

Computational drug-repurposing methods routinely report retrospective ranking
performance in the AUROC 0.95–0.99 range. Leakage of the held-out treatment
edges into training is widely suspected as the cause but rarely *measured*,
because measuring it requires retraining the model rather than re-analysing its
output. We measure it. Holding the candidate pool (24,313 DRKG compounds), the
degree baseline, the tie-aware rank convention, the scoring function and the
query relations fixed, and varying **only** whether the knowledge-graph
embedding saw the held-out `treats` edges during training, Hit@10 falls from
**52.8% to 3.1%** (n=960): contamination accounts for **49.7 percentage
points**. The leak-free model lands next to a trivial node-degree ("popularity")
baseline at 2.7%. No similarity family we tested — knowledge graph, chemical
structure, cell morphology — demonstrates a gain over that baseline, which we
argue is structural: each scores a drug by similarity to a disease's known
treatments, and treatment sets are mechanistically heterogeneous. One approach
survives: genetics-anchored target prioritization shows CI-backed conditional
lift over degree, concentrated in the low-degree drugs where degree is useless,
and it survives a temporal control with genetics frozen at Open Targets release
20.02 (February 2020) — though freezing the snapshot deflates its prospective
Hit@10 from 32.6% to 10.0%. We recommend leak-controlled evaluation with an
explicit degree baseline as the field's default, and reporting conditional lift
over degree rather than side-by-side rankings.

## 1. Introduction

The retrospective evaluation protocol in computational drug repurposing is
close to standardized: hold out a set of known drug–disease treatment pairs,
rank candidate compounds for each held-out disease, report Hit@k, MRR or AUROC.
Reported numbers are excellent and have been for years.

The protocol has a known weakness. When the ranker is a knowledge-graph
embedding trained on the same graph the labels were drawn from, the held-out
`treats` edge is not merely a label — it is a training example, recoverable by
memorization. This is discussed as a caveat but seldom quantified, because
quantifying it means retraining on an edge-stripped graph, and because a
correctly-run ablation typically destroys the headline result.

A second weakness is subtler and more damaging: the omitted control. In a
knowledge graph, a well-studied drug has high node degree, and a well-studied
drug is disproportionately likely to be the answer. Any scorer correlating with
degree inherits that. Without a degree baseline reported *on the same pool*, a
repurposing ranker cannot be distinguished from "name a famous drug." Papers
report their method against other methods; the baseline that matters is one line
of code.

This paper runs both controls. The contribution is not a better ranker — it is a
measurement of how much reported performance the two omitted controls account
for, and a demonstration of the one signal in our stack that survives them.

## 2. Methods

**Data.** DRKG (97,238 entities). The candidate pool is all 24,313
`Compound::` nodes, defined once as `sorted(c for c in degree if
c.startswith('Compound::'))` and shared byte-for-byte across every arm.
Held-out set: 993 drug–disease `treats` pairs over 539 diseases.

**The strip and the leak audit.** We removed every held-out `treats` edge from
the training graph and retrained TransE from scratch on the result
(`data/drkg/drkg_stripped.tsv`). The strip removed all 993 held-out
`DRUGBANK::treats` edges and all 329 held-out `GNBR::T` edges (1,360 edges total
including the time-sliced set). We then audited the stripped graph directly for
survivors under *any* relation name, because an edge deleted under one relation
may persist under a synonym. Exactly **1 of 993** held-out pairs survives, under
`GNBR::Pa` (palliates). The clean arm is therefore genuinely leak-free rather
than silently recovering the same fact under another name.

**The instrument.** Both TransE arms are scored by the same function
(`opencure/scoring/transe.py::score_drugs_for_disease_vectorized`) with the same
query relations. The primary relation config is `{DRUGBANK::treats,
Hetionet::CtD}` — the natural query for "drug treats disease", and the config
*most favourable to the clean model*. Because this choice turns out to straddle
the baseline, we also report the project's canonical config (`GNBR::T`,
`Hetionet::CtD`, `GNBR::C`, `GNBR::Pa`, `GNBR::J`, `GNBR::Mp`) rather than
silently selecting the favourable one. The clean PyKEEN model is TransE with
p=2 / `power_norm=False`, so its native `score_hrt` is exactly the −‖h+r−t‖₂ the
vectorized scorer computes; agreement was verified numerically to <1e-4.

**Popularity baseline.** Node degree computed from the *stripped* graph. This
matters: the full graph still contains each held-out `treats` edge, and each
such edge adds +1 degree to exactly the drug being predicted — the baseline
would otherwise be inflated by the answers it is being asked to find.

**Rank convention.** Tie-aware mid-rank, identical across arms. Ranks are
against the full pool; unscored candidates are not silently dropped, so each arm
pays for its own coverage gap (pool coverage was 100% for every arm reported
here).

**The sign trap.** TransE scores are negative (−‖h+r−t‖₂; measured range −17.6
to −9.9), while the harness fills unscored candidates with 0.0. Used raw, every
out-of-vocabulary compound would rank above every scored one and invert the
result. Scores were mapped `sim = 1/(1 − transe_score)`: strictly monotone
increasing, so the in-vocabulary ranking is preserved exactly, and bounded to
(0,1], so the 0.0 fill lands last. We flag this because it is a silent,
result-inverting trap in any pipeline that mixes signed and unsigned scorers
behind a common interface.

**Evaluability asymmetry.** The clean model has 97,206 entities against the full
graph's 97,238. The 32 missing nodes are disease nodes whose *only* edges were
the held-out `treats` edges — stripping the edge deleted the node, taking 33
held-out pairs with it. Those diseases are unevaluable for the clean arm but
evaluable for the contaminated arm, which can score them *only because of the
leak*. All headline metrics are reported on the common set (n=960).

**Conditional lift.** Paired win-fraction (fraction of pairs where method A
ranks the true drug strictly above method B), with 90% CIs from a bootstrap
resampling **disease clusters**, not pairs (2,000 iterations over 507 clusters
for the KG arm), since pairs from one disease are not independent. We also fit
`is_true ~ log_degree + genetics_score` and report standardized partial
coefficients with cluster-bootstrap CIs.

**Temporal freeze.** For the prospective test, disease→gene genetics is taken
from Open Targets release **20.02 (February 2020)**, which provably predates
every approval in the 210-pair post-2020 first-approval test set. Genetics
sources are restricted to human-genetics datasources only (`clingen`, `eva`,
`eva_somatic`, `gene_burden`, `genomics_england`, `gwas_credible_sets`,
`orphanet`, `ot_genetics_portal`, `uniprot_variants`); ChEMBL/known-drug,
literature, expression and animal-model sources are excluded. The drug→gene side
uses only curated ChEMBL mechanisms. No `treats` or `indication` edge is read on
either side.

## 3. Results

### 3.1 The leakage ablation

Everything is held fixed except training-time exposure to the held-out edges.

| Arm (n=960, common set) | Hit@10 | Hit@30 | Hit@100 | MRR | Median rank |
|---|---|---|---|---|---|
| TransE, **contaminated** | **52.8%** | 74.2% | 88.4% | 0.2046 | **8** |
| TransE, **retrained leak-free** | 3.1% | 7.6% | 17.2% | 0.0149 | 538 |
| Popularity (node degree) | 2.7% | 5.8% | 15.9% | 0.0133 | 871 |

**Contamination = 49.7 percentage points.** The leak-free model sits beside the
trivial baseline; the contaminated model looks like a solved problem.

This is not a small-pool artifact. A previously reported 57.2% for this model
used a DrugBank-only 10,551-compound pool; re-measured on the honest 24,313-pool
under identical relations it is 52.8%. **Widening the pool costs ~4 pp;
leakage costs ~50 pp.** Pool size is the explanation usually offered for
implausible retrospective numbers. In this system it is a rounding error next to
contamination.

Two honesty notes. First, clean TransE's 3.1% is nominally above popularity's
2.7%, and its paired win-fraction is 0.581 (90% CI [0.549, 0.615]) — but the
top-10 edge is **4 pairs out of 960** (30 vs 26 hits; post-hoc difference CI
[−0.9, +1.8] pp, straddling zero), and it collapses to **0.0%** under the
equally defensible canonical relation set. We report this as **no useful lift**.
Second, our pre-registered criterion technically passed, and it should not have
been able to: it was **under-specified**, because it did not fix the query
relation set, and that free choice straddles the baseline. The lesson is in
§4 — pre-registration must fix *every* free choice, not just the metric.

What clean TransE does retain is a weak ordering edge over degree (median rank
538 vs 871), concentrated in the low-degree quartiles where popularity scores
0.0% (q1: 6.2% vs 0.0%). That is structurally the same pattern as the genetics
signal in §3.3 — but unlike it, it never converts into top-10 retrieval, which
is what a shortlist requires.

### 3.2 No similarity family beats popularity

Each pillar is scored against its own candidate pool with its own matched degree
baseline on the same 993 held-out pairs (pillars differ in coverage, hence in
evaluable n).

| Pillar | n | Hit@10 | Hit@100 | Median rank | Matched popularity Hit@10 / Hit@100 / median | Verdict |
|---|---|---|---|---|---|---|
| Chemical structure (ChemBERTa) | 631 | 4.4% | 21.7% | 473 | 3.8% / 23.1% / 407 | ties — no gain |
| Cell morphology (JUMP Cell Painting) | 496 | 2.8% | 8.9% | 1293 | 3.0% / 22.4% / 397 | strictly worse |
| Knowledge graph (clean TransE) | 960 | 3.1% | 17.2% | 538 | 2.7% / 15.9% / 871 | ties — no gain |

ChemBERTa's nominally higher Hit@10 is not a gain: it is *worse* than the
baseline on both median rank and Hit@100. Cell morphology loses outright.

**No similarity family demonstrates a gain over popularity.** We think the
diagnosis is structural rather than a matter of tuning. All three score a
candidate by similarity to a disease's *known treatments*. But a disease's
treatments are mechanistically heterogeneous: first-line hypertension therapy
includes beta-blockers, thiazide diuretics and ACE inhibitors, which are unalike
by chemical structure, by cellular morphology, and by graph neighbourhood.
"Be similar to that set" is an incoherent target. What it actually rewards is
being a well-connected, well-studied drug — which is degree.

### 3.3 The one survivor: genetics-anchored target prioritization

The surviving signal does not use similarity at all. It anchors on causal
biology: disease → human-genetics causal gene → drug with a curated ChEMBL
mechanism against that gene.

On the genetics-covered subset of the random split (n=53):

| Metric | Genetics | Clean popularity |
|---|---|---|
| Hit@10 | **20.8%** | 3.8% |
| Hit@100 | 69.8% | 28.3% |
| Median rank | **64** | 499 |

The ~5× gap is not the evidence — side-by-side rankings are exactly what we
criticize in §1. The evidence is conditional: paired win-fraction **0.698**, 90%
CI **[0.589, 0.804]**; logistic partial coefficient of genetics controlling for
log-degree **+0.22**, 90% CI [0.19, 0.27]. Both CIs exclude the null.

The degree-stratified breakdown is the sharpest evidence, because it shows the
two signals are complementary rather than competing:

| Degree quartile (n=53) | n | Genetics Hit@10 | Popularity Hit@10 |
|---|---|---|---|
| Q1 (deg 254–1095) | 13 | **38.5%** | 0.0% |
| Q2 (deg 1098–1490) | 13 | **30.8%** | 0.0% |
| Q3 (deg 1518–2258) | 13 | **15.4%** | 0.0% |
| Q4 (deg 2315–2958) | 14 | 0.0% | **14.3%** |

Genetics wins precisely where popularity is useless — the low-degree, non-hub
drugs where a novel repurposing lead would live. Popularity wins only among
hubs: the rediscovery-of-famous-drugs regime. This is what a genuinely
residualized signal looks like, and it is why an unstratified comparison would
have hidden it.

### 3.4 Temporal freeze, and an honest deflation

A concern remains: current genetics may itself be a posterior. Evidence accrues
around genes that turned out to be drug targets, so "2026 genetics predicts a
2021 approval" may be reverse causation. We tested this directly by re-running
on 210 post-2020 first-approval pairs with genetics frozen at Open Targets
release 20.02 (February 2020), which predates every test approval.

| Temporal split, genetics vintage | Hit@10 (median) | Popularity Hit@10 | Paired win frac (90% CI) | Logistic coef (90% CI) |
|---|---|---|---|---|
| Current (live API, posterior-suspect) | 32.6% (20) | 0.0% | 1.00 [1.0, 1.0] | +0.41 [0.33, 0.44] |
| **Feb 2020 (provably pre-approval)** | **10.0% (117)** | 0.0% | **0.95 [0.90, 1.0]** | **+0.52 [0.47, 0.68]** |

Two findings, both worth reporting:

1. **The prospective claim survives.** With genetics frozen before the
   approvals, the ranker still beats popularity on **38/40** covered pairs (CI
   excludes 0.5), and the logistic lift is *larger* (+0.52). The log-degree
   coefficient goes **negative** (−0.25): popularity is not merely uninformative
   for new drugs but *anti*-predictive, since a newly approved drug is by
   construction sparsely connected in a 2020 graph. This is foresight, not a
   split artifact.
2. **Two-thirds of the absolute number was posterior inflation.** Hit@10 falls
   from 32.6% to **10.0%** once the snapshot is frozen. The honest prospective
   number is the modest one. We report the deflation itself as a finding: any
   temporal-validation claim built on a live evidence API is inflated by an
   amount of this order, and the inflation is invisible unless the snapshot is
   frozen.

The direction of this result is consistent with Minikel et al., *Nature* 2024:
genetically-supported targets succeed ~2.6× more often in clinical trials.

### 3.5 Rediscovery, and no novel lead

The positive must be bounded honestly.

**Coverage.** 69 of 93 screened diseases are genetics-covered. Pathogen-driven
neglected tropical diseases (Chagas, leishmaniasis, schistosomiasis) have no
human-genetic causal architecture and are structurally out of scope; the system
returns `not_assessed` rather than a guess.

**Rediscovery-leaning.** Where genetics is strong, drug development has already
happened against that target, so the top genetics-anchored lead is typically the
disease's *existing* drug. A directioned-survivor audit of 80 direction-
concordant cross-indication leads found the single promising class — the 9
non-oncology inhibitors — to be entirely **JAK2 inhibitors for ulcerative
colitis** (tofacitinib, upadacitinib, baricitinib, filgotinib and others),
several already approved for that indication. They passed the novelty filter
only because the 2020-vintage knowledge graph predates those approvals. Of the
rest, 39 oncology inhibitors are precision-oncology rediscoveries or already in
basket trials, and 30 non-oncology activators rest on Mendelian/mouse direction
that inverts against the complex-disease direction — dangerous false positives,
not leads.

**Zero wet-lab-confirmed predictions.** Across every angle tested, no novel,
credible, wet-lab-ready lead was found. The validated object is a *prioritizer*
in the genetics-covered regime, not a discovery engine.

## 4. Discussion

The 49.7 pp figure is specific to this graph, model and split, and we do not
claim it transfers numerically. We claim it is *measurable at all* — one
retraining run — and that the field reports it approximately never. When a model
trained on a graph containing its own evaluation edges reports Hit@10 above 50%,
the hypothesis that ~50 pp is memorization is now empirically grounded rather
than a rhetorical worry, and the burden of excluding it is a single ablation.

The degree baseline deserves equal emphasis, because it is cheaper still: the
degree ranker took 1.9 seconds, and it ties or beats every similarity family we
tested. If a new method cannot be shown to beat it on the same pool, the method
has not been shown to do anything — regardless of how it compares to other
published methods, which may share the same failure.

We recommend, concretely:

1. **Leak-controlled evaluation by default.** Retrain on the edge-stripped
   graph and *audit the strip* against the actual training file for survivors
   under synonymous relations. Report the contaminated and clean arms side by
   side; the gap is the paper's most informative number.
2. **An explicit popularity/degree baseline on the same pool.** Build degree
   from the stripped graph, or the baseline is inflated by the answers.
3. **Report conditional lift over degree, not side-by-side rankings.** Paired
   win-fraction with disease-clustered bootstrap CIs, plus a degree-stratified
   breakdown. A method whose lift lives only in the top-degree quartile is
   rediscovering famous drugs.
4. **Freeze the evidence snapshot when claiming temporal validity.** A live
   evidence API is a posterior over the very approvals under test. Our own
   deflation was ~2/3 of the headline.
5. **Pre-register every free choice**, not just the metric: query relation set,
   candidate pool, rank convention, tie handling. Our criterion passed on a
   choice we had left free, and inverted under an equally defensible one. A
   pre-registration that leaves a degree of freedom open is not a
   pre-registration.

The positive finding points the same way. Genetics-anchored prioritization works
*because* it does not ask "what resembles the known treatments" — it asks "what
does the causal biology implicate", a question whose answer is independent of
how well-studied the drug is. That independence is exactly what the
degree-stratified table measures, and it is the property a repurposing signal
must have to be worth anything.

## 5. Limitations

- n=53 for the genetics-covered random-split subset and n=40 for the frozen
  temporal subset are small; disease-clustered bootstrap CIs over few binary
  outcomes are correspondingly wide.
- The ablation covers one embedding family (TransE) on one graph (DRKG). A
  capacity control — a higher-dimensional, longer-trained clean model — has not
  been completed, so "KG embedding does not beat degree once leakage is removed"
  is stated for this configuration, not as a universal claim.
- The 993-pair held-out set is drawn from DRUGBANK-derived `treats` edges and
  inherits that resource's indication biases.
- The genetics result is bounded to the covered regime (69/93 diseases) and is
  rediscovery-leaning within it.
- No prediction has wet-lab confirmation. Every number here is retrospective or
  simulated-prospective.

## 6. Data and code availability

All code, data manifests and evaluation artifacts are open under **Apache-2.0**
at **github.com/SimonBartosDev/opencure**.

| Artifact | Path |
|---|---|
| KG leakage ablation (headline) | `scripts/leakfree_kg_benchmark.py` → `experiments/eval/leakfree_kg_scorecard.json` |
| Per-pillar leak-free benchmark | `scripts/leakfree_benchmark.py` → `experiments/eval/leakfree_pillar_scorecard.json` |
| Conditional lift, degree-stratified, temporal | `scripts/popularity_residualized_lift.py` → `experiments/eval/conditional_lift_report*.json` |
| Frozen Feb-2020 genetics snapshot | `scripts/build_pre2020_genetics.py` |
| Scoring function under test | `opencure/scoring/transe.py::score_drugs_for_disease_vectorized` |
| Stripped training graph | `data/drkg/drkg_stripped.tsv` |

Reproduce the headline ablation and the temporal freeze:

```bash
python3 scripts/leakfree_kg_benchmark.py
python3 scripts/build_pre2020_genetics.py
python3 scripts/popularity_residualized_lift.py data/eval/time_sliced_test.jsonl \
        _temporal_pre2020 data/open_targets/genetics_pre2020_efo.json
```

## References

Minikel EV, Painter JL, Dong CC, Nelson MR. Refining the impact of genetic
evidence on clinical success. *Nature* 2024.
