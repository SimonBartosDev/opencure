# OpenCure v5 Prospective Prediction Snapshot

**Timestamp (UTC):** 2026-04-20T063801Z
**Git commit:** a824c6488942
**Predictions:** 610 total across 61 diseases
**Content fingerprint:** `5d9257095bdc3276`

## Why this snapshot exists

Retrospective validation (checking which of a model's predictions *happen to*
match known treatments already in the training data) is the weakest form of
evidence in drug repurposing. Predictions can be memorized.

A *prospective* validation is stronger: freeze the predictions now, then wait.
If literature or trials published AFTER this date confirm any of them, the
confirmation is genuinely independent.

This snapshot is immutable (see fingerprint). `scripts/prospective_monitor.py`
runs monthly to re-query PubMed and ClinicalTrials.gov for evidence published
after this timestamp and computes rolling precision@5 and precision@10.

## Zenodo deposition

To mint a DOI:

```
pip install zenodo-client
python -m zenodo_client upload \
    data/prospective/snapshots/2026-04-20T063801Z/ \
    --metadata data/prospective/snapshots/2026-04-20T063801Z/zenodo_metadata.json
```

The resulting DOI becomes the permanent, citeable reference for this
snapshot.

## Reproduce

```
git checkout a824c6488942
bash setup_data.sh
python experiments/systematic_screening.py
```
