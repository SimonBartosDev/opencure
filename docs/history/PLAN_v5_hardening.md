# OpenCure v5 "Best We Can Do Right Now" Hardening Plan

**Goal:** move the code from 7/10 to 8.5/10 — the ceiling reachable by a
single developer in this repository, with current compute, this week.

**After this plan lands, the public phase starts:** Zenodo DOI, bioRxiv,
lab outreach, pharma pitches.

## Scope: what's IN

Five phases, ~5 working days of focused work, all code-only, no external
blockers:

| # | Phase | Effort | Ships |
|---|---|---|---|
| A | Search refactor + field unification | 1 day | 7 → 7.5 |
| B | Pytest regression suite | 1 day | 7.5 → 8 |
| C | Learned ensemble weights (after training) | 0.5 day | 8 → 8.2 |
| D | Async evidence + structured logging | 1.5 days | 8.2 → 8.4 |
| E | AutoDock Vina wiring + data provenance | 1 day | 8.4 → 8.5 |

## What's explicitly OUT (and why)

- **R-GCN training** — needs CUDA. Scaffold exists in `rgcn_scorer.py`.
  Will train on cloud once lab partnerships motivate the compute spend.
- **Ontology-based auto-curation** — too much scope; current manual curation
  (13 subtypes, 55 tissues) is good enough for v5.
- **Single-cell atlas integration** — 3-4 weeks, too much scope.
- **Real 2024-native KG retraining from scratch** — covered by `v4-breakthrough`
  scaffolding; OT 24.09 already merged; full native retrain deferred.

---

## Phase A — Search module refactor + field unification (1 day)

**Problem.** `opencure/search.py` has v2 legacy code (`_combine_scores_v2`,
lines 669–917) coexisting with v3 grouped combiner. The recent 3-pillar
silent-zero bug was a field-name mismatch between these paths. Further
bugs likely lurk.

**Actions.**
1. **Delete `_combine_scores_v2`** entirely (it's the except-branch fallback;
   with grouped_combiner stable we don't need it).
2. **Canonicalize score-dict keys**: exactly one name per concept.
   - Drop every `*_raw_score` variant. Replace with a single `*_score`.
   - Drop every `*_rank` that duplicates a normalized score.
3. **Introduce `PillarScore` TypedDict** in
   `opencure/scoring/common.py` so field names are static-checkable.
4. **EvidenceReport field cleanup**: remove fields duplicated between
   dataclass and to_dict (e.g. `mr_score` vs `mr_group_score` — keep
   only the pillar-level one, drop the group-level echo).
5. **One regression check** — re-run Malaria search, verify all 11
   pillars still fire with the new canonical field names.

**Acceptance.** `search.py` drops from 1,000 → ~600 lines. Zero fields
have two names. All 11 pillars still fire on a live Malaria search.

## Phase B — Pytest regression suite (1 day)

**Problem.** No test files in the repo. The 3-pillar silent bug survived
from v3 → v4 → v5 because we had no automated regression check.

**Actions.**
1. `tests/` directory + pytest config
2. `tests/test_filters.py`: metabolite blacklist, IUPAC heuristic, drug_filter
   — all already unit-testable in isolation
3. `tests/test_scoring.py`: hub_normalize calibration, grouped_combiner
   fixed-weight math, mechanistic_reversal on synthetic OT/ChEMBL inputs
4. `tests/test_evidence.py`: pharmacogenomics, DDI, dose_plausibility,
   triangulation — all synthesizable without external API calls
5. `tests/test_integration.py`: minimal disease search on a frozen 3-drug
   mock knowledge graph. Verify all 11 pillars surface non-zero.
6. `.github/workflows/tests.yml`: run pytest on every push to main +
   pull requests
7. Target: ≥60% coverage on `opencure/{filters,scoring,evidence,eval}/`

**Acceptance.** `pytest` passes from a fresh clone (needing only cached
data files). CI green. 3-pillar regression check exists.

## Phase C — Learned ensemble weights (0.5 day)

**Problem.** `EFFICACY_GROUPS` weights in `grouped_combiner.py` are
hand-guessed. A best-in-class version learns them from data.

**Requires.** Clean edge-stripped training must finish first (background
job at ~2h remaining as of writing).

**Actions.**
1. After clean training produces `data/models/unified_transE_clean/`, run
   `scripts/run_heldout_eval.py` to get a dataset of
   (features, outcome) pairs: features = 12 pillar scores per candidate,
   outcome = whether it was a held-out true treats edge
2. Train XGBoost meta-learner via `scripts/train_ensemble_v5.py` using
   sklearn with 5-fold CV on time-sliced + random held-out
3. Save learned weights + model to `data/models/ensemble_v5.pkl`
4. Wire into `opencure/scoring/ensemble.py::score_ensemble()` with fallback
   to hand-weighted grouped_combiner if model load fails
5. Calibrate via sklearn's `CalibratedClassifierCV` (isotonic) so a
   score of 0.7 actually means ~70% precision

**Acceptance.** A combined_score of 0.X correlates with P(held-out hit) = 0.X
within ±0.05 on the time-sliced test set.

## Phase D — Async evidence + structured logging (1.5 days)

**Problem.** Evidence gathering is sequential: PubMed → CT.gov → FAERS →
Semantic Scholar → L1000CDS2 → Pharos. ~30-60s per candidate × 10 = 5-10 min
just for evidence per disease. With 2,507 diseases queued this is the main
bottleneck.

**Actions.**
1. Convert `opencure/evidence/*.py` functions to `async def` with
   `aiohttp.ClientSession`
2. In `report.py`, run the independent fetches with `asyncio.gather(...)`
3. Per-call TTL-disk cache at `data/evidence_cache/` keyed by (drug,
   disease) hash
4. Replace every `print()` in hot paths with `logging.getLogger(__name__)`
5. JSON-format log handler when writing to `experiments/*.log`
6. Per-pillar timing metrics → `data/metrics/screening_timings.csv`

**Acceptance.** Evidence gathering per candidate drops from 30-60s to
5-10s. Logs parseable as JSON. Timing CSV lets us spot which pillar is
slowest per disease.

## Phase E — AutoDock Vina wiring + data provenance (1 day)

**Problem.** `opencure/scoring/structure_docking.py` exists but isn't
wired into the pipeline. No audit trail for "which data version produced
this prediction."

**Actions.**

**E1 — Docking integration.**
1. Fix `structure_docking.py::score_drugs_by_docking` if broken
2. Gate it on a `--use-docking` flag (expensive: ~30s/drug)
3. Feed docking score into `triangulation.compute_triangulation_score`
   as the real 'docking' axis (currently always None)

**E2 — Data provenance.**
1. `scripts/compute_data_hashes.py`: hashes every source file in
   `data/drkg/`, `data/open_targets/`, `data/sources_2024/`, emits a
   versioned `data_manifest.json`
2. `experiments/systematic_screening.py` now records the manifest hash
   in every result JSON as `data_version`
3. Snapshot script captures manifest in each snapshot folder

**Acceptance.** Every saved prediction JSON has a `data_version` field
pointing at an immutable manifest. Docking score surfaces for the top-10
of one pilot disease.

---

## Dependency graph

```
A (refactor) ────┬──► B (tests) ────┬──► v5 RE-SCREEN ──► PUBLIC PHASE
                 │                  │
                 └──► D (async+log) ┘
                                    │
C (learned weights) ──────────────┤  (parallel, waits on training)
                                    │
E (docking + provenance) ─────────┘
```

A → B in order (tests assume clean field names).
D + E can run in parallel with B.
C waits on background training (~2h remaining).

## After this plan

Code is 8.5/10. Ready for:
- bioRxiv submission with honest numbers
- Zenodo DOI registration
- Lab outreach (5 briefs ready)
- Pharma pitch (deck ready)
- Peer review submission

The remaining 1.5 points to reach 10 are human-only:
lab partnerships, pharma deals, peer review, Nature MI acceptance,
12-month prospective precision data.
