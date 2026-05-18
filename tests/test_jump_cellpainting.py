"""Regression tests for the v7 JUMP Cell Painting pillar.

Locks in:
- The pillar fails open when the data artifact is absent (no JUMP-CP
  download → empty dict, search.py keeps running with the other 12
  pillars).
- ``load_jump_features`` re-keys profiles by Compound:: entity using
  the InChIKey → DrugBank map, and drops profiles with no DrugBank match.
- The scorer returns the canonical ``(similarity, similar_to)`` shape
  and applies the min-similarity threshold.
- Schema fields land in CANDIDATE_FIELDS so finalize_v5 doesn't flag
  them as unknown.
- ``group_structural_scores`` accepts JUMP scores and the structural
  group prefers the morphological signal when it's the strongest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest


# ---- Schema integration -------------------------------------------------

def test_jump_fields_registered_in_pillar_schema() -> None:
    from opencure.scoring.common import CANDIDATE_FIELDS, PILLAR_FIELDS
    for field in ("jump_score", "jump_rank", "jump_similar_to"):
        assert field in PILLAR_FIELDS
        assert field in CANDIDATE_FIELDS


def test_validate_candidate_accepts_jump_fields() -> None:
    from opencure.scoring.common import validate_candidate
    cand = {
        "drug_id": "DB00001", "drug_name": "Test",
        "disease_name": "TestDisease", "combined_score": 0.5,
        "pillars_hit": 3, "confidence": "MEDIUM",
        "jump_score": 0.82, "jump_rank": 7, "jump_similar_to": "Compound::DB00945",
    }
    warnings = validate_candidate(cand)
    for w in warnings:
        assert "jump_score" not in w
        assert "jump_rank" not in w
        assert "jump_similar_to" not in w


# ---- Loader fail-open --------------------------------------------------

def test_load_jump_features_returns_none_when_artifact_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Fresh checkout → no JUMP data → pillar is silent, search continues."""
    from opencure.scoring import jump_cell_painting as jcp

    monkeypatch.setattr(jcp, "FEATURES_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(jcp, "INCHIKEY_MAP_PATH", tmp_path / "missing.tsv")
    jcp.reset_cache()

    emb, entities = jcp.load_jump_features()
    assert emb is None and entities is None


def _write_jump_artifacts(
    dir: Path,
    profiles: list[tuple[str, str, np.ndarray]],  # (inchikey, drugbank_id, vec)
) -> tuple[Path, Path]:
    feat_path = dir / "compound_features.npz"
    map_path = dir / "inchikey_to_drugbank.tsv"
    embeddings = np.stack([v for _, _, v in profiles]).astype(np.float32)
    inchikeys = np.array([ik for ik, _, _ in profiles])
    np.savez_compressed(str(feat_path),
                        embeddings=embeddings, inchikeys=inchikeys)
    with map_path.open("w") as fh:
        fh.write("inchikey\tdrugbank_id\n")
        for ik, db, _ in profiles:
            if db:  # blank db → drop from map
                fh.write(f"{ik}\t{db}\n")
    return feat_path, map_path


def test_load_jump_features_rekeys_to_compound_entities(
    tmp_path: Path, monkeypatch
) -> None:
    from opencure.scoring import jump_cell_painting as jcp

    profiles = [
        ("INCHIKEY-A", "DB00001", np.full(8, 1.0)),
        ("INCHIKEY-B", "DB00002", np.full(8, 2.0)),
        ("INCHIKEY-C", "", np.full(8, 3.0)),  # no DrugBank match → dropped
    ]
    feat, mp = _write_jump_artifacts(tmp_path, profiles)
    monkeypatch.setattr(jcp, "FEATURES_PATH", feat)
    monkeypatch.setattr(jcp, "INCHIKEY_MAP_PATH", mp)
    jcp.reset_cache()

    emb, entities = jcp.load_jump_features()
    assert emb is not None
    assert entities == ["Compound::DB00001", "Compound::DB00002"]
    assert emb.shape == (2, 8)


# ---- Scorer behavior ---------------------------------------------------

def test_scorer_returns_empty_when_disease_has_no_known_treatment(
    monkeypatch
) -> None:
    """No known treatment in the JUMP coverage → scorer returns empty."""
    from opencure.scoring import jump_cell_painting as jcp

    embeddings = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    entities = [f"Compound::DB0000{i}" for i in range(4)]
    # Empty triplets — no known treatments resolvable.
    fake_triplets = pd.DataFrame({
        "head": pd.Series([], dtype="object"),
        "relation": pd.Series([], dtype="object"),
        "tail": pd.Series([], dtype="object"),
    })

    out = jcp.score_drugs_for_disease_jump(
        "Disease::MESH:Dxxxxx",
        triplets=fake_triplets,
        compound_set=entities,
        embeddings=embeddings,
        embedding_entities=entities,
    )
    assert out == {}


def test_scorer_returns_canonical_shape() -> None:
    """When a known treatment IS in JUMP, the scorer returns
    ``{compound_entity: (similarity_float, similar_to_str)}``."""
    from opencure.scoring import jump_cell_painting as jcp

    # Build an embedding where DB00001 is the known treatment, and
    # DB00002 is a near-clone of it (cosine ≈ 1) → should score high.
    base = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.array([
        base,                                # DB00001 (known)
        base + 1e-3,                          # DB00002 (very similar — should hit)
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # DB00003 (orthogonal)
    ])
    entities = ["Compound::DB00001", "Compound::DB00002", "Compound::DB00003"]

    # Triplets carrying a treats edge from DB00001 → the test disease.
    # Use Hetionet::CtD which IS in TREATMENT_RELATIONS (DRUGBANK::treats is
    # only in KNOWN_TREATMENT_RELATIONS for labelling, not pillar scoring).
    triplets = pd.DataFrame({
        "head": ["Compound::DB00001"],
        "relation": ["Hetionet::CtD::Compound:Disease"],
        "tail": ["Disease::MESH:DTEST"],
    })

    out = jcp.score_drugs_for_disease_jump(
        "Disease::MESH:DTEST",
        triplets=triplets,
        compound_set=entities,
        embeddings=embeddings,
        embedding_entities=entities,
        min_similarity=0.5,
    )

    # DB00002 should be returned (very similar to DB00001), DB00003 dropped.
    assert "Compound::DB00002" in out
    sim, similar_to = out["Compound::DB00002"]
    assert 0.5 <= sim <= 1.0
    assert similar_to == "Compound::DB00001"
    # Known treatment itself is excluded.
    assert "Compound::DB00001" not in out


# ---- Structural-group integration --------------------------------------

def test_group_structural_includes_jump() -> None:
    """JUMP scores feed group_structural_scores and win when strongest."""
    from opencure.scoring.pillar_groups import group_structural_scores

    mol_fp = {"Compound::DB00001": (0.4, "X")}
    jump = {"Compound::DB00001": (0.9, "Compound::DBKNOWN")}

    out = group_structural_scores(mol_fp_scores=mol_fp, jump_scores=jump)
    assert "Compound::DB00001" in out
    score, best_pillar, group_tag = out["Compound::DB00001"]
    assert score == pytest.approx(0.9)
    assert best_pillar == "jump"
    assert group_tag == "structural_group"


def test_group_structural_jump_only_still_works() -> None:
    """JUMP can be the only structural pillar present (e.g. SMILES cache absent)."""
    from opencure.scoring.pillar_groups import group_structural_scores

    out = group_structural_scores(jump_scores={"Compound::DB00001": (0.7, "ref")})
    assert "Compound::DB00001" in out
    assert out["Compound::DB00001"][1] == "jump"
