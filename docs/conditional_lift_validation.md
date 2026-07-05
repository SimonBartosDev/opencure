---
title: "Conditional-lift and temporal validation of the genetics-anchored ranker"
description: A leak-clean test of whether OpenCure's genetics-anchored signal adds rank lift ABOVE a popularity baseline, and whether it survives a temporal (post-2020) holdout.
---

# Conditional-lift and temporal validation of the genetics-anchored ranker

*New empirical result (June 2026). Reproducible from
`scripts/popularity_residualized_lift.py`; outputs in `experiments/eval/`.*

## Summary

The genetics-anchored ranker was previously reported to beat a popularity
baseline ~5× on the genetics-covered subset (Hit@10 20.8% vs 3.8%). Two things
were missing from that headline, and this analysis supplies both:

1. **A clean baseline.** The earlier popularity comparator built node degree
   from the *full* DRKG graph (`drkg.tsv`), which still contains the 999
   held-out `treats` edges — inflating the comparator by the very answers being
   predicted. Corrected here using `drkg_stripped.tsv`.
2. **A conditional-lift number.** The earlier scorecard reported genetics and
   popularity as two side-by-side rankings, never the *incremental* lift of
   genetics **conditional on** popularity, and it defined its "genetic-signal
   subset" by `pool_score(true_drug) > 0` (conditioning the eval subset on the
   feature value at the label — selection bias).

**Result: the genetics signal is real, leak-free, and demonstrably independent
of popularity. It is strongest exactly where popularity is useless (low-degree,
non-hub drugs), and it survives a temporal post-2020 holdout where the
popularity baseline collapses to zero.** This validates the genetics-anchored
ranker as a *prioritizer* in the genetics-covered regime — not as a novel-lead
discovery engine (see Caveats).

## Method

Leak control is **inherited unchanged** from
`scripts/genetics_anchored_benchmark_v3.py`: the disease→gene side uses only
human-genetics datasources (chembl/known-drug, literature, expression and
animal-model sources excluded); the drug→gene side uses only curated ChEMBL
mechanism targets; no `treats`/`indication` edge is read. The only changes are
(a) a leak-clean popularity comparator (degree from `drkg_stripped.tsv`) and
(b) conditional metrics. Three independent analyses:

- **(A) Corrected ranking** — re-rank with the clean popularity baseline.
- **(B) Degree-stratified Hit@10** — bin held-out positives by their true
  drug's (stripped) degree quartile; within each quartile compare genetics vs
  popularity. A signal that wins in the *low*-degree quartiles is genuinely
  residualized (independent of popularity).
- **(C) Conditional lift** — the fraction of covered positives where genetics
  ranks the true drug strictly better than the clean popularity baseline
  (paired, bootstrap CI over diseases), plus the partial coefficient of
  genetics_score in `is_true ~ log_degree + genetics_score`.

Two held-out sets: the random split (`holdout_test.jsonl`, 993 pairs) and a
temporal split (`time_sliced_test.jsonl`, 210 post-2020 first-approval pairs).

## Results — (A) corrected ranking, genetics-covered subset

| Split | Genetics Hit@10 (median) | Clean popularity Hit@10 (median) | Inflated-baseline impact |
|---|---|---|---|
| Random (n=53) | **20.8%** (64) | 3.8% (499) | none (clean == inflated) |
| Temporal post-2020 (n=43) | **32.6%** (20) | **0.0%** (5531) | none |

The self-inflated baseline was a real methodological flaw but is **empirically
immaterial**: the 24k-compound pool swamps 999 edges, so the clean and inflated
baselines are identical in both splits. Honest correction, conclusion unchanged.

## Results — (B) degree-stratified Hit@10 (the key evidence)

Genetics wins **exactly where popularity is useless** — the low-degree
(non-hub) drugs where a novel repurposing lead would live:

**Random split (covered, n=53):**

| Degree quartile | n | Genetics Hit@10 | Popularity Hit@10 |
|---|---|---|---|
| Q1 (deg 254–1095) | 13 | **38.5%** | 0.0% |
| Q2 (deg 1098–1490) | 13 | **30.8%** | 0.0% |
| Q3 (deg 1518–2258) | 13 | **15.4%** | 0.0% |
| Q4 (deg 2315–2958) | 14 | 0.0% | 14.3% |

**Temporal split (covered, n=43):**

| Degree quartile | n | Genetics Hit@10 | Popularity Hit@10 |
|---|---|---|---|
| Q1 (deg 1–3) | 10 | 0.0% | 0.0% |
| Q2 (deg 4–9) | 11 | **45.5%** | 0.0% |
| Q3 (deg 13–19) | 11 | **54.5%** | 0.0% |
| Q4 (deg 21–1043) | 11 | **27.3%** | 0.0% |

Popularity wins only in the high-degree hub quartile of the random split — the
rediscovery-of-famous-drugs regime. On the temporal split, post-2020 drugs are
barely connected in the 2020 graph, so "rank popular drugs" scores **0%**.

## Results — (C) conditional lift

| Split | Paired win fraction (genetics > clean popularity) | Logistic std-coef (genetics \| log_degree) |
|---|---|---|
| Random (n=53) | **0.698**, 90% CI [0.589, 0.804] | **+0.22**, 90% CI [0.19, 0.27] |
| Temporal (n=43) | **1.00**, 90% CI [1.0, 1.0] | **+0.41**, 90% CI [0.33, 0.44] |

Both CIs exclude the null (0.5 for the paired test, 0 for the coefficient), so
the genetics signal carries lift **beyond** popularity. On the temporal split
the log-degree coefficient is **negative** (−0.20) — popularity is actively
*anti*-predictive for new drugs, while genetics is positive.

## Results — temporal hardening with a provably pre-2020 genetics snapshot

To remove the posterior-leak concern directly, the temporal test was re-run
with disease→gene genetics taken from **Open Targets release 20.02
(February 2020)** — the `genetic_association` datatype score, which provably
predates every 2020–2023 test approval (`scripts/build_pre2020_genetics.py`;
89/150 temporal diseases covered, 40 evaluable genetics-covered positives).

| Temporal split, genetics vintage | Genetics Hit@10 (median) | Clean popularity Hit@10 | Paired win frac (90% CI) | Logistic std-coef (90% CI) |
|---|---|---|---|---|
| ~2024–26 (live API, posterior-suspect) | 32.6% (20) | 0.0% (5531) | 1.00 [1.0, 1.0] | +0.41 [0.33, 0.44] |
| **Feb 2020 (provably pre-approval)** | **10.0% (117)** | **0.0% (5531)** | **0.95 [0.90, 1.0]** | **+0.52 [0.47, 0.68]** |

**Two findings, both honest:**

1. **The prospective claim is clean and survives.** Using genetics frozen
   *before* the approvals, the ranker still beats the popularity baseline
   decisively — it ranks post-2020 true drugs strictly above popularity on
   38/40 pairs (CI excludes 0.5), the logistic lift is *larger* (+0.52), and
   popularity remains at 0% (log-degree coefficient −0.25, anti-predictive).
   This is no longer "not a split artifact" — it is genuine foresight: human
   genetics known by Feb 2020 prioritizes drugs that were only later approved
   for those diseases, where popularity is useless.
2. **The absolute level was partly posterior-inflated.** The temporal Hit@10
   falls from 32.6% (contaminated) to **10.0%** (clean) — roughly two-thirds of
   the headline temporal hit rate came from genetics that accrued *after* the
   approvals. The honest temporal number is the modest one.

A random-split consistency check with the same Feb-2020 genetics reproduces the
pattern (genetics Hit@10 12.5% vs clean popularity 2.5%; paired win 0.75
[0.62, 0.86]; logistic +0.26 [0.18, 0.35]) — genetics wins in the low-degree
quartiles, loses only among high-degree hubs, exactly as with current genetics
but at a lower absolute level reflecting the sparser 2020 evidence.

## Caveats (these do not go away)

- **Coverage is the binding constraint.** Only 53/576 (random) and 43/143
  (temporal) evaluable held-out positives are genetics-covered. Outside that
  subset the ranker has no signal and must not be used.
- **The covered wins are rediscovery-leaning.** For well-studied diseases,
  strong genetics has already driven drug development, so the top
  genetics-anchored "lead" is typically the disease's existing drug. A
  directioned-survivor audit of the 80 cross-indication concordant leads found
  no genuinely novel, non-oncology, credibly-directioned lead (the most
  promising class was JAK inhibitors for ulcerative colitis — several already
  approved, direction sourced from a cancer-gene catalog). This validates a
  *prioritizer*, not a novel-discovery engine.
- **Temporal posterior-leak — resolved, with an honest deflation.** The
  pre-2020-snapshot test (above) confirms the *relative* dominance over
  popularity is genuine foresight (paired 0.95, CI excludes the null, popularity
  still 0%), but shows the *absolute* temporal Hit@10 was ~2/3 posterior-inflated
  (32.6% → 10.0% once genetics is frozen pre-approval). The honest temporal
  number is 10.0%.
- Modest subset sizes (n≈43–53); disease-grouped bootstrap CIs over few binary
  outcomes are wide.

## Reproduce

```bash
# random split (default)
python3 scripts/popularity_residualized_lift.py
# temporal post-2020 holdout
python3 scripts/popularity_residualized_lift.py data/eval/time_sliced_test.jsonl _temporal
# temporal with provably-pre-2020 (OT 20.02) genetics
python3 scripts/build_pre2020_genetics.py
python3 scripts/popularity_residualized_lift.py data/eval/time_sliced_test.jsonl \
        _temporal_pre2020 data/open_targets/genetics_pre2020_efo.json
```

Outputs: `experiments/eval/conditional_lift_report*.json`,
`experiments/eval/conditional_lift_pairs*.jsonl`.

## Conclusion

In the genetics-covered regime, OpenCure's genetics-anchored ranker is a
**leak-clean, popularity-residualized, temporally-validated prioritizer**: it
adds rank lift over popularity (CI-backed), concentrated in the low-popularity
drugs where it matters, and it predicts post-2020 indications that a popularity
baseline cannot. This is the honest sense in which the predictions work. It is
**not** a solution to novel-lead discovery, which remains unsolved across every
angle tested.
