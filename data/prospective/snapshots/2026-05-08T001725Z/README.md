# OpenCure v5 Prospective Prediction Snapshot

**Timestamp (UTC):** 2026-05-08T001725Z
**Git commit:** 7dd52b3ce878
**Predictions:** 710 total across 84 diseases
**Content fingerprint:** `ac428fc4e622e3d7`

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
    data/prospective/snapshots/2026-05-08T001725Z/ \
    --metadata data/prospective/snapshots/2026-05-08T001725Z/zenodo_metadata.json
```

The resulting DOI becomes the permanent, citeable reference for this
snapshot.

## Reproduce

```
git checkout 7dd52b3ce878
bash setup_data.sh
python experiments/systematic_screening.py
```
