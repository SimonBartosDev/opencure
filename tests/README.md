# OpenCure test suite

## Running

```bash
# Full suite (requires local data: DRKG, ChEMBL, GTEx, etc.)
PYTHONPATH=. pytest -v

# Unit tests only (no data required — run in CI)
PYTHONPATH=. pytest -v -m "not integration"

# Integration tests only
PYTHONPATH=. pytest -v -m "integration"

# Regression tests (catches specific shipped-bugs)
PYTHONPATH=. pytest -v tests/test_regression.py
```

## Markers

- `integration` — requires local data files (DRKG, ChEMBL cache, GTEx, etc.)
- `slow` — tests taking >30 seconds
- `network` — hits external APIs

## The canonical regressions

- **3-pillar silent-zero bug** (v3–v5.0) — `test_regression.py::TestPillarFieldNameRegression`.
  Proximity/DTI/Gene-Sig pillars were computed at runtime but never reached
  saved output due to field-name mismatch.
- **Metabolite leakers** — `test_regression.py::TestFilterRegression`.
  Drugs like Uric Acid, Glutathione, Cordycepin Triphosphate must stay rejected.

## When adding a scoring/filtering/evidence module

Add a test class to the relevant file:
- `test_scoring.py` for anything in `opencure/scoring/`
- `test_filters.py` for `opencure/filters/`
- `test_evidence.py` for `opencure/evidence/`
- `test_eval.py` for `opencure/eval/`

If you ship a fix for a bug, add a regression test in `test_regression.py`.
