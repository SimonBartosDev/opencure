"""DepMap essentiality flag — v7.

The Broad's DepMap project provides gene-level CRISPR essentiality
scores (Chronos) across 1000+ cancer cell lines. Negative scores =
essential (knockout kills the cell); strongly negative across the
majority of lines = pan-essential (housekeeping). Targeting a
pan-essential gene may work in cancer (the cancer cells die), but in
non-oncology indications it's a recipe for systemic toxicity.

v7 attaches essentiality records per candidate as a *flag*, not a hard
filter — the brief generator and the wet-lab partner decide whether the
target's essentiality is acceptable in their disease context.

Source: ``data/depmap/gene_essentiality.tsv`` produced by
``scripts/precompute_depmap.py``. Schema:

    gene_symbol \t median_chronos \t fraction_essential_lines

Where ``fraction_essential_lines`` = fraction of cell lines in which
Chronos < -0.5. Both metrics are surfaced because the median can hide
bimodality (the "essential in half the lines" pattern).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ESSENTIALITY_PATH = Path("data/depmap/gene_essentiality.tsv")

# Chronos threshold below which a gene counts as essential in a cell line.
ESSENTIAL_THRESHOLD = -0.5

# A gene is "broadly essential" (warning-worthy) when ≥ 80% of cell
# lines flag it as essential.
WARNING_FRACTION = 0.80


@dataclass
class EssentialityRecord:
    gene: str
    median_chronos: float
    fraction_essential_lines: float

    @property
    def warning(self) -> bool:
        """True when targeting this gene risks broad toxicity."""
        return self.fraction_essential_lines >= WARNING_FRACTION


_cache: dict[str, EssentialityRecord] | None = None


def load_essentiality_table(
    path: Optional[Path] = None,
) -> dict[str, EssentialityRecord]:
    """Read the TSV into a per-gene essentiality dict (memoized).

    Empty dict when the artifact is absent — the flag pillar fails open
    like every other v7 layer.

    ``path`` defaults to the module-level ``ESSENTIALITY_PATH`` looked up
    at call-time (so tests that monkeypatch the constant take effect).
    """
    global _cache
    if _cache is not None:
        return _cache
    resolved = path if path is not None else ESSENTIALITY_PATH
    if not resolved.exists():
        _cache = {}
        return _cache

    out: dict[str, EssentialityRecord] = {}
    with resolved.open() as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene = parts[0].strip()
            try:
                median = float(parts[1])
                frac = float(parts[2])
            except ValueError:
                continue
            out[gene] = EssentialityRecord(gene, median, frac)
    _cache = out
    return out


def lookup(gene: str) -> Optional[EssentialityRecord]:
    """Per-gene record, or ``None`` when DepMap has no data."""
    return load_essentiality_table().get(gene)


def annotate(primary_target: str) -> dict[str, float | bool | None]:
    """Build the v7 candidate-record fragment from a primary-target gene.

    Returns ``{"target_essentiality": float|None,
              "essentiality_warning": bool}``.
    Designed to merge directly into a candidate dict.
    """
    if not primary_target:
        return {"target_essentiality": None, "essentiality_warning": False}
    rec = lookup(primary_target)
    if rec is None:
        return {"target_essentiality": None, "essentiality_warning": False}
    return {
        "target_essentiality": float(rec.median_chronos),
        "essentiality_warning": bool(rec.warning),
    }


def reset_cache() -> None:
    """Tests only — clear the memoized table."""
    global _cache
    _cache = None
