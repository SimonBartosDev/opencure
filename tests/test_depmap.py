"""Regression tests for the v7 DepMap essentiality flag."""
from __future__ import annotations

from pathlib import Path

import pytest

from opencure.scoring.depmap_essentiality import (
    EssentialityRecord,
    WARNING_FRACTION,
    annotate,
    load_essentiality_table,
    lookup,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_cache()
    yield
    reset_cache()


def _write_table(path: Path, rows: list[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("gene_symbol\tmedian_chronos\tfraction_essential_lines\n")
        for gene, med, frac in rows:
            fh.write(f"{gene}\t{med}\t{frac}\n")


# ---- Schema -----------------------------------------------------------

def test_essentiality_fields_in_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS
    assert "target_essentiality" in CANDIDATE_FIELDS
    assert "essentiality_warning" in CANDIDATE_FIELDS


# ---- Loader fail-open -------------------------------------------------

def test_load_returns_empty_when_artifact_missing(tmp_path, monkeypatch) -> None:
    from opencure.scoring import depmap_essentiality as de
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", tmp_path / "missing.tsv")
    reset_cache()
    assert load_essentiality_table() == {}


def test_load_parses_three_columns(tmp_path, monkeypatch) -> None:
    from opencure.scoring import depmap_essentiality as de

    p = tmp_path / "essentiality.tsv"
    _write_table(p, [
        ("EGFR", 0.05, 0.05),
        ("RPL5", -1.5, 0.98),
    ])
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", p)
    reset_cache()

    table = load_essentiality_table()
    assert "EGFR" in table
    assert isinstance(table["EGFR"], EssentialityRecord)
    assert table["EGFR"].median_chronos == pytest.approx(0.05)
    assert table["RPL5"].fraction_essential_lines == pytest.approx(0.98)


# ---- Warning logic ---------------------------------------------------

def test_pan_essential_gene_flags_warning(tmp_path, monkeypatch) -> None:
    from opencure.scoring import depmap_essentiality as de

    p = tmp_path / "essentiality.tsv"
    _write_table(p, [("RPL5", -1.5, 0.98)])
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", p)
    reset_cache()

    rec = lookup("RPL5")
    assert rec is not None
    assert rec.warning is True


def test_non_essential_gene_no_warning(tmp_path, monkeypatch) -> None:
    from opencure.scoring import depmap_essentiality as de

    p = tmp_path / "essentiality.tsv"
    _write_table(p, [("EGFR", 0.05, 0.05)])
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", p)
    reset_cache()

    rec = lookup("EGFR")
    assert rec is not None
    assert rec.warning is False


def test_warning_threshold_exact_boundary(tmp_path, monkeypatch) -> None:
    """A gene at exactly WARNING_FRACTION should trigger the warning."""
    from opencure.scoring import depmap_essentiality as de

    p = tmp_path / "essentiality.tsv"
    _write_table(p, [("X1", -0.5, WARNING_FRACTION),
                     ("X2", -0.5, WARNING_FRACTION - 0.01)])
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", p)
    reset_cache()

    assert lookup("X1").warning is True
    assert lookup("X2").warning is False


# ---- Annotate API for candidate records -------------------------------

def test_annotate_returns_two_canonical_keys(tmp_path, monkeypatch) -> None:
    from opencure.scoring import depmap_essentiality as de

    p = tmp_path / "essentiality.tsv"
    _write_table(p, [("RPL5", -1.5, 0.98)])
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", p)
    reset_cache()

    out = annotate("RPL5")
    assert set(out) == {"target_essentiality", "essentiality_warning"}
    assert out["target_essentiality"] == pytest.approx(-1.5)
    assert out["essentiality_warning"] is True


def test_annotate_handles_missing_target(tmp_path, monkeypatch) -> None:
    """Gene not in DepMap → warning False, score None."""
    from opencure.scoring import depmap_essentiality as de
    monkeypatch.setattr(de, "ESSENTIALITY_PATH", tmp_path / "empty.tsv")
    reset_cache()
    out = annotate("UNKNOWN_GENE")
    assert out["target_essentiality"] is None
    assert out["essentiality_warning"] is False


def test_annotate_with_empty_target_string() -> None:
    """When the candidate has no primary target → safe defaults."""
    out = annotate("")
    assert out["target_essentiality"] is None
    assert out["essentiality_warning"] is False
