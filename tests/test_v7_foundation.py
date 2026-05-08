"""Regression tests for the v7 foundation-model swap.

Locks in the loader-fallback contract that lets the codebase always
pick up the strongest chemistry embedding currently on disk, without
crashing when the artifact is missing or upgrading silently when
MoLFormer-XL lands next to an older ChemBERTa cache.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---- Loader fallback chain (MoLFormer-XL > ChemBERTa > None) ------------

def _stub_npz(path: Path, n: int = 4, dim: int = 8) -> None:
    """Write a tiny .npz that load_cached_embeddings can read."""
    rng = np.random.default_rng(0)
    np.savez_compressed(
        str(path),
        embeddings=rng.standard_normal((n, dim)).astype(np.float32),
        entities=np.array([f"Compound::DB{1000+i}" for i in range(n)]),
    )


def test_load_best_prefers_molformer(tmp_path, monkeypatch) -> None:
    """When both caches exist, MoLFormer-XL wins."""
    from opencure.scoring import molecular_embeddings as me

    chemberta = tmp_path / "chemberta_embeddings.npz"
    molformer = tmp_path / "molformer_embeddings.npz"
    _stub_npz(chemberta, n=3, dim=8)
    _stub_npz(molformer, n=3, dim=16)  # different dim → distinguishable

    monkeypatch.setattr(me, "CHEMBERTA_CACHE", chemberta)
    monkeypatch.setattr(me, "MOLFORMER_CACHE", molformer)

    emb, entities, tag = me.load_best_molecular_embeddings()
    assert tag == "molformer"
    assert emb.shape[1] == 16, "should have loaded MoLFormer (the 16-dim cache)"
    assert len(entities) == 3


def test_load_best_falls_back_to_chemberta(tmp_path, monkeypatch) -> None:
    """When only ChemBERTa is cached, that's what we get."""
    from opencure.scoring import molecular_embeddings as me

    chemberta = tmp_path / "chemberta_embeddings.npz"
    molformer = tmp_path / "molformer_embeddings.npz"  # absent
    _stub_npz(chemberta, n=3, dim=8)

    monkeypatch.setattr(me, "CHEMBERTA_CACHE", chemberta)
    monkeypatch.setattr(me, "MOLFORMER_CACHE", molformer)

    emb, entities, tag = me.load_best_molecular_embeddings()
    assert tag == "chemberta"
    assert emb.shape == (3, 8)


def test_load_best_returns_none_triple_when_no_cache(tmp_path, monkeypatch) -> None:
    """No cache on disk → triple of Nones, never raises."""
    from opencure.scoring import molecular_embeddings as me

    monkeypatch.setattr(me, "CHEMBERTA_CACHE", tmp_path / "missing_chemberta.npz")
    monkeypatch.setattr(me, "MOLFORMER_CACHE", tmp_path / "missing_molformer.npz")

    emb, entities, tag = me.load_best_molecular_embeddings()
    assert emb is None
    assert entities is None
    assert tag is None


# ---- Device auto-detection ----------------------------------------------

def test_default_device_returns_valid_choice() -> None:
    """The auto-detect helper returns one of cpu/mps/cuda — never crashes."""
    from opencure.scoring.molecular_embeddings import _default_device

    device = _default_device()
    assert device in {"cpu", "mps", "cuda"}


def test_dti_default_device_returns_valid_choice() -> None:
    """DTI module has its own copy of the helper for symmetry; same contract."""
    from opencure.scoring.dti_predictor import _default_device

    assert _default_device() in {"cpu", "mps", "cuda"}


# ---- ESM-2 protein-embedding loader -------------------------------------

def _stub_protein_npz(path: Path, n: int = 4, dim: int = 8) -> None:
    """Write a tiny .npz that load_protein_embeddings can read."""
    rng = np.random.default_rng(1)
    np.savez_compressed(
        str(path),
        embeddings=rng.standard_normal((n, dim)).astype(np.float32),
        gene_ids=np.array([f"Gene::{1000+i}" for i in range(n)]),
    )


def test_esm2_load_best_prefers_650M_then_150M_then_8M(tmp_path, monkeypatch) -> None:
    """v7: 650M > 150M > 8M when multiple variants are cached."""
    from opencure.scoring import dti_predictor as dp

    p_8m = tmp_path / "p_8m.npz"
    p_150m = tmp_path / "p_150m.npz"
    p_650m = tmp_path / "p_650m.npz"
    _stub_protein_npz(p_8m, n=2, dim=320)
    _stub_protein_npz(p_150m, n=2, dim=640)
    _stub_protein_npz(p_650m, n=2, dim=1280)

    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE", p_8m)
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_150M", p_150m)
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_650M", p_650m)

    emb, genes, variant = dp.load_best_protein_embeddings()
    assert variant == "650M"
    assert emb.shape[1] == 1280


def test_esm2_load_best_falls_back_to_150M_when_650M_absent(tmp_path, monkeypatch) -> None:
    from opencure.scoring import dti_predictor as dp

    p_8m = tmp_path / "p_8m.npz"
    p_150m = tmp_path / "p_150m.npz"
    p_650m = tmp_path / "p_650m_missing.npz"  # absent
    _stub_protein_npz(p_8m, n=2, dim=320)
    _stub_protein_npz(p_150m, n=2, dim=640)

    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE", p_8m)
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_150M", p_150m)
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_650M", p_650m)

    _, _, variant = dp.load_best_protein_embeddings()
    assert variant == "150M"


def test_esm2_load_best_returns_triple_none_when_no_cache(tmp_path, monkeypatch) -> None:
    from opencure.scoring import dti_predictor as dp

    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE", tmp_path / "missing_8m.npz")
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_150M", tmp_path / "missing_150m.npz")
    monkeypatch.setattr(dp, "PROTEIN_EMB_CACHE_650M", tmp_path / "missing_650m.npz")

    emb, genes, variant = dp.load_best_protein_embeddings()
    assert emb is None and genes is None and variant is None


def test_dti_predictor_default_protein_dim_is_640() -> None:
    """v7: DTIPredictor matches ESM-2 150M output dim by default."""
    from opencure.scoring.dti_predictor import DTIPredictor

    model = DTIPredictor()
    assert model.protein_dim == 640
    assert model.drug_dim == 768  # MoLFormer-XL & ChemBERTa both 768


def test_esm2_variants_table_exposes_three_tiers() -> None:
    """Module-level constant must list the three supported variants."""
    from opencure.scoring.dti_predictor import ESM2_VARIANTS

    assert set(ESM2_VARIANTS) == {"8M", "150M", "650M"}
    # Sanity: dims grow with model size.
    dims = [ESM2_VARIANTS[v][1] for v in ("8M", "150M", "650M")]
    assert dims == sorted(dims)


# ---- search.py wiring ---------------------------------------------------

def test_search_uses_generic_mol_emb_keys() -> None:
    """The search-data dict now uses ``mol_emb`` / ``mol_emb_entities``
    / ``mol_emb_model`` instead of the v6.x ``chemberta_emb`` /
    ``chemberta_entities``. Catches anyone re-introducing the old keys
    in a refactor (search.py is loaded lazily on first call, but the
    helper that populates the cache is callable in isolation)."""
    from opencure import search as s

    src = Path(s.__file__).read_text()
    assert "chemberta_emb" not in src, \
        "v7 swapped chemberta_emb → mol_emb; reintroducing the old key " \
        "splits the chemistry-embedding pillar across two code paths"
    assert "_load_chemberta(" not in src, \
        "v7 swapped _load_chemberta → _load_mol_embeddings"
    assert "mol_emb" in src and "mol_emb_model" in src
