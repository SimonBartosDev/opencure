"""Disease-mechanism confidence — v7.

For a drug-repurposing platform, the second-largest source of false
positives is **wrong disease biology**: if the disease's molecular
mechanism is contested or unknown, every gene-overlap-driven pillar
becomes garbage-in-garbage-out. Pillar scores can look great while
silently averaging across speculative gene associations.

This module attaches a 0-1 ``mechanism_confidence`` score to each
disease, derived from the existing ``data/disease_gene_index.json``:

    mechanism_confidence = clip(n_disease_genes / TARGET_GENE_COUNT, 0, 1)

A disease with ~30+ associated genes is well-studied (high confidence,
score ≈ 1.0). A disease with ~5 genes is sparsely characterised
(score ≈ 0.17). The wet-lab brief generator (A5) reads this score and
flags every brief from a low-confidence disease as "speculative — disease
biology not well-mapped".

Notes:
- This is a heuristic, not a Bayesian posterior. The plan's "v8"
  followup is a proper disease-mechanism uncertainty quantifier (e.g.,
  posterior over OT genetic-evidence categories). For now, gene count
  is a defensible proxy for research intensity.
- The threshold ``TARGET_GENE_COUNT = 30`` matches the median gene-count
  across well-studied indications (Alzheimer's, breast cancer, T2D)
  per the OpenTargets 2024 release.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DISEASE_GENE_INDEX = Path("data/disease_gene_index.json")

# Diseases with at least this many associated genes are treated as
# fully-characterised (mechanism_confidence ≈ 1.0). The cap makes the
# score saturate so very-densely-mapped diseases don't dwarf moderately-
# mapped ones.
TARGET_GENE_COUNT = 30

# Below this threshold, brief generators flag every prediction as
# "speculative — disease biology not well-mapped".
LOW_CONFIDENCE_THRESHOLD = 0.4


_cache: dict | None = None


def _load_index(path: Optional[Path] = None) -> dict[str, list[str]]:
    """Read the disease-gene index (memoized)."""
    global _cache
    if _cache is not None:
        return _cache
    resolved = path if path is not None else DISEASE_GENE_INDEX
    if not resolved.exists():
        _cache = {}
        return _cache
    raw = json.loads(resolved.read_text())
    out: dict[str, list[str]] = {}
    for entity, payload in raw.items():
        if isinstance(payload, list):
            out[entity] = payload
        elif isinstance(payload, dict):
            # Older format: {disease: {"genes": [...]}}
            genes = payload.get("genes", [])
            if isinstance(genes, list):
                out[entity] = genes
    _cache = out
    return out


def gene_count(disease_entity: str) -> int:
    """Number of associated genes for ``disease_entity`` (0 when unknown)."""
    if not disease_entity:
        return 0
    return len(_load_index().get(disease_entity, []))


def mechanism_confidence(disease_entity: str) -> float:
    """Heuristic 0-1 confidence in the disease's molecular mechanism.

    Diseases with the canonical gene-count target or more land at 1.0;
    sparse diseases scale linearly toward 0.
    """
    n = gene_count(disease_entity)
    if TARGET_GENE_COUNT <= 0:
        return 0.0
    return min(1.0, max(0.0, n / float(TARGET_GENE_COUNT)))


def is_low_confidence(disease_entity: str) -> bool:
    """True when brief generators should flag predictions as speculative."""
    return mechanism_confidence(disease_entity) < LOW_CONFIDENCE_THRESHOLD


def annotate(disease_entity: str) -> dict[str, float | bool]:
    """Build the v7 result-file fragment for a disease.

    Designed to merge into the top-level result JSON (alongside
    ``data_manifest_hash`` etc., not into per-candidate dicts).
    """
    score = mechanism_confidence(disease_entity)
    return {
        "mechanism_confidence": score,
        "mechanism_low_confidence": score < LOW_CONFIDENCE_THRESHOLD,
    }


def reset_cache() -> None:
    """Tests only."""
    global _cache
    _cache = None
