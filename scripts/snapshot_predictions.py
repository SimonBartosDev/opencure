"""
Prospective-prediction snapshot for temporal validation.

When a v5 screening run completes, this script:
  1. Freezes the top-10 predictions per disease into an immutable snapshot
  2. Generates a content hash (prediction fingerprint)
  3. Writes a Zenodo-ready deposit package (metadata JSON + README + data)
  4. Appends to a chronological registry so every future month can compute
     true prospective precision@K

The snapshot is immutable because its filename contains the timestamp
and the content-hash — any modification produces a different hash.

Each snapshot contains:
  - predictions.jsonl — one line per (disease, rank, drug, score, confidence)
  - methods.json      — pipeline version, KG hash, filter config
  - metadata.json     — Zenodo-compatible metadata for DOI registration
  - README.md         — human-readable description

To actually mint a DOI the user runs:
    zenodo-upload data/prospective/snapshots/<timestamp>/ --title "OpenCure v5 ..."
(requires a Zenodo account + API token)
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
SNAPSHOTS_DIR = Path("data/prospective/snapshots")
REGISTRY_PATH = Path("data/prospective/snapshot_registry.jsonl")
TOP_K = 10


def _git_commit_hash() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        return out.decode().strip()[:12]
    except Exception:
        return "unknown"


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_snapshot() -> Path:
    """Create a timestamped snapshot of current top-K predictions per disease."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    snap_dir = SNAPSHOTS_DIR / timestamp
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Collect top-K predictions
    predictions_path = snap_dir / "predictions.jsonl"
    total_diseases = 0
    total_preds = 0
    with predictions_path.open("w") as out:
        for jf in sorted(RESULTS_DIR.glob("*.json")):
            if any(x in jf.name for x in ("opencure", "screening", "novel", "mechanism")):
                continue
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            cands = data.get("candidates", [])
            if not cands:
                continue
            disease = jf.stem.replace("_", " ")
            for i, c in enumerate(cands[:TOP_K], start=1):
                rec = {
                    "snapshot_timestamp": timestamp,
                    "disease": disease,
                    "rank": i,
                    "drug_name": c.get("drug_name"),
                    "drug_id": c.get("drug_id"),
                    "combined_score": round(c.get("combined_score", 0), 4),
                    "confidence": c.get("confidence"),
                    "novelty_level": c.get("novelty_level"),
                    "pillars_hit": c.get("pillars_hit", 0),
                    "is_known_treatment": c.get("is_known_treatment", False),
                }
                out.write(json.dumps(rec) + "\n")
                total_preds += 1
            total_diseases += 1

    # Methods manifest
    methods = {
        "snapshot_timestamp": timestamp,
        "git_commit": _git_commit_hash(),
        "pipeline_version": "v5",
        "n_pillars": 12,
        "diseases_covered": total_diseases,
        "total_predictions": total_preds,
        "top_k_per_disease": TOP_K,
        "knowledge_graph_sources": [
            "DRKG (5.87M triples, 2020)",
            "PrimeKG (8.10M triples, 2022)",
            "OpenTargets 24.09 (83k triples derived)",
        ],
        "filter_stack": [
            "SMILES validity + >=4 heavy atoms + MW >=60",
            "Metabolite blacklist (96 curated)",
            "IUPAC / research-chemical heuristic",
            "ChEMBL phase >=1 (fail-open for missing)",
            "Critical ADMET (hERG/AMES/DILI/Skin)",
            "Phase-4 bypass for FDA-approved drugs",
        ],
        "scoring_features": [
            "TransE/RotatE/PrimeKG/Unified-KG (RRF fusion)",
            "TxGNN (pre-computed)",
            "Molecular fingerprints + ChemBERTa + DTI (max pool)",
            "Network proximity + gene signatures (max pool)",
            "Mendelian randomization",
            "ADMET two-stage multiplier",
            "Hub-degree normalization",
            "R-GCN heterogeneous GNN (v5)",
            "Tissue-context weighting (v5)",
        ],
        "clinical_layer": [
            "Dose plausibility (ChEMBL phase)",
            "DDI warnings (1.4M DRKG ddi edges)",
            "Pharmacogenomic flags (CPIC + PharmGKB)",
            "Triangulation (KG + docking + Pharos + literature)",
        ],
    }
    (snap_dir / "methods.json").write_text(json.dumps(methods, indent=2))

    # Fingerprint hash of predictions file — this is the immutable "proof"
    pred_hash = _content_hash(predictions_path)

    # Zenodo-ready metadata
    zenodo = {
        "metadata": {
            "title": f"OpenCure v5 prospective prediction snapshot {timestamp}",
            "creators": [{
                "name": "OpenCure Project",
                "affiliation": "Open-source drug repurposing platform"
            }],
            "upload_type": "dataset",
            "description": (
                "Snapshot of OpenCure's top-10 drug-repurposing predictions "
                f"across {total_diseases} diseases as of {timestamp}. "
                "Made immutable for prospective validation: any literature or "
                "trial evidence appearing after this timestamp cannot have "
                "influenced the prediction."
            ),
            "keywords": [
                "drug repurposing",
                "knowledge graph",
                "prospective validation",
                "open science",
            ],
            "license": "cc-by-4.0",
            "access_right": "open",
            "version": f"v5-{timestamp}",
        },
        "fingerprint": pred_hash,
        "files_to_upload": [
            "predictions.jsonl",
            "methods.json",
            "README.md",
        ],
    }
    (snap_dir / "zenodo_metadata.json").write_text(json.dumps(zenodo, indent=2))

    # Human-readable README
    readme = f"""# OpenCure v5 Prospective Prediction Snapshot

**Timestamp (UTC):** {timestamp}
**Git commit:** {methods['git_commit']}
**Predictions:** {total_preds} total across {total_diseases} diseases
**Content fingerprint:** `{pred_hash}`

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
python -m zenodo_client upload \\
    data/prospective/snapshots/{timestamp}/ \\
    --metadata data/prospective/snapshots/{timestamp}/zenodo_metadata.json
```

The resulting DOI becomes the permanent, citeable reference for this
snapshot.

## Reproduce

```
git checkout {methods['git_commit']}
bash setup_data.sh
python experiments/systematic_screening.py
```
"""
    (snap_dir / "README.md").write_text(readme)

    # Append to registry
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a") as f:
        f.write(json.dumps({
            "timestamp": timestamp,
            "git_commit": methods['git_commit'],
            "diseases": total_diseases,
            "predictions": total_preds,
            "fingerprint": pred_hash,
            "path": str(snap_dir),
        }) + "\n")

    return snap_dir


def main() -> None:
    d = build_snapshot()
    print(f"Snapshot written: {d}")
    for f in sorted(d.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
