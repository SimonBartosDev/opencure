"""JUMP Cell Painting scoring — v7 Pillar 13.

The JUMP-CP consortium (broad public release, 2024) provides ~140K
compound morphological profiles in U2OS cells: 5-channel fluorescent
images of cells perturbed with each compound, distilled to a feature
vector per compound. We use the published profile-level embeddings
(no GPU work required on our side) and score candidates by morphological
similarity to known treatments — the same anchored-similarity logic the
chemistry-embedding pillar uses, but in *phenotype space* rather than
chemistry space. This is the single largest closure of the gap to
closed-platform image-based screening.

Data layout (downloaded by ``scripts/precompute_jump_cp.py``):

    data/jump_cp/
        manifest.json                 — provenance, version, entry counts
        compound_features.npz         — embeddings + InChIKey index
        inchikey_to_drugbank.tsv      — InChIKey → DrugBank ID map
                                        (with salt-stripped aliases)

Salt-form handling mirrors ``txgnn_scorer.py``: JUMP indexes by parent-
compound InChIKey, while DRKG carries salt-form drug entities. We
augment the lookup with stripped aliases so e.g. metformin hydrochloride
in DRKG matches the metformin parent profile in JUMP-CP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from opencure.config import DATA_DIR

JUMP_DIR = DATA_DIR.parent / "jump_cp"
FEATURES_PATH = JUMP_DIR / "compound_features.npz"
INCHIKEY_MAP_PATH = JUMP_DIR / "inchikey_to_drugbank.tsv"

_cache: dict = {}


def load_jump_features() -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Return ``(embeddings, drug_entities)`` aligned by row, or ``(None, None)``.

    Fail-open: missing artifact → no JUMP pillar contribution. The
    Compound:: prefix is added so downstream consumers can use the same
    string convention as the rest of search.py.
    """
    if "features" in _cache:
        return _cache["features"], _cache["entities"]
    if not FEATURES_PATH.exists() or not INCHIKEY_MAP_PATH.exists():
        _cache["features"] = None
        _cache["entities"] = None
        return None, None

    data = np.load(str(FEATURES_PATH), allow_pickle=True)
    embeddings = data["embeddings"]
    inchikeys = data["inchikeys"].tolist()

    inchikey_to_drugbank: dict[str, str] = {}
    with INCHIKEY_MAP_PATH.open() as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                inchikey_to_drugbank[parts[0]] = parts[1]

    # Re-key by Compound:: entity for direct comparison with DRKG.
    entities: list[str] = []
    keep_idx: list[int] = []
    for i, ik in enumerate(inchikeys):
        db_id = inchikey_to_drugbank.get(ik)
        if db_id:
            entities.append(f"Compound::{db_id}")
            keep_idx.append(i)

    if not entities:
        _cache["features"] = None
        _cache["entities"] = None
        return None, None

    embeddings = embeddings[np.array(keep_idx)]
    _cache["features"] = embeddings
    _cache["entities"] = entities
    return embeddings, entities


def score_drugs_for_disease_jump(
    disease_entity: str,
    *,
    triplets,
    compound_set: list[str],
    embeddings: np.ndarray,
    embedding_entities: list[str],
    top_k: int = 200,
    min_similarity: float = 0.6,
) -> dict[str, tuple[float, str]]:
    """Cosine similarity to morphological centroid of known treatments.

    Mirrors the anchored-similarity pattern in
    ``molecular_embeddings.score_by_learned_similarity`` but operates
    over phenotype-space embeddings instead of chemistry-space
    embeddings — so it captures *what the drug does to a cell*, not just
    what it looks like structurally. A compound that's structurally
    novel but produces the same morphological response as a known
    treatment is exactly the high-value signal we want.

    Returns ``dict[compound_entity] -> (similarity, similar_to_drug)``.
    Empty dict when JUMP coverage of the disease's known-treatment set
    is too thin (fewer than 1 known with morphological profile).
    """
    if embeddings is None or not embedding_entities:
        return {}

    from opencure.config import TREATMENT_RELATIONS
    from opencure.scoring.molecular import get_known_treatments
    from opencure.scoring.molecular_embeddings import compute_cosine_similarity

    entity_to_idx = {e: i for i, e in enumerate(embedding_entities)}
    known = get_known_treatments(disease_entity, triplets)
    known_with_morph = [c for c in known if c in entity_to_idx]
    if not known_with_morph:
        return {}

    known_indices = [entity_to_idx[c] for c in known_with_morph]
    known_embs = embeddings[known_indices]

    candidate_set = [c for c in compound_set
                     if c in entity_to_idx and c not in set(known)]
    if not candidate_set:
        return {}

    candidate_indices = [entity_to_idx[c] for c in candidate_set]
    candidate_embs = embeddings[candidate_indices]

    # (N_cand, N_known) similarity matrix; per candidate, take max sim.
    sim_matrix = compute_cosine_similarity(candidate_embs, known_embs)
    max_sims = sim_matrix.max(axis=1)
    best_known = sim_matrix.argmax(axis=1)

    results: dict[str, tuple[float, str]] = {}
    for i, (compound, sim) in enumerate(zip(candidate_set, max_sims)):
        if sim >= min_similarity:
            results[compound] = (float(sim), known_with_morph[best_known[i]])

    # Top-K by similarity
    if len(results) > top_k:
        sorted_items = sorted(results.items(), key=lambda kv: -kv[1][0])
        results = dict(sorted_items[:top_k])

    return results


def reset_cache() -> None:
    """Clear memoized JUMP features — for tests only."""
    _cache.clear()
