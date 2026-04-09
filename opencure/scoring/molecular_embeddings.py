"""
Molecular embeddings using transformer models (ChemBERTa, MoLFormer).

These learned representations capture molecular properties that traditional
fingerprints miss: functional group interactions, 3D conformational effects,
and subtle structure-activity relationships learned from millions of molecules.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Optional

from opencure.config import DATA_DIR

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHEMBERTA_CACHE = EMBEDDINGS_DIR / "chemberta_embeddings.npz"
MOLFORMER_CACHE = EMBEDDINGS_DIR / "molformer_embeddings.npz"


def get_chemberta_embeddings(
    smiles_list: list[str],
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute molecular embeddings using ChemBERTa.

    Args:
        smiles_list: List of SMILES strings
        model_name: HuggingFace model identifier
        batch_size: Batch size for inference
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray of shape (len(smiles_list), embedding_dim)
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding as molecular representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def get_molformer_embeddings(
    smiles_list: list[str],
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute molecular embeddings using MoLFormer (IBM).
    Trained on 1.1 billion molecules - captures broad chemical space.

    Args:
        smiles_list: List of SMILES strings
        model_name: HuggingFace model identifier
        batch_size: Batch size for inference
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray of shape (len(smiles_list), embedding_dim)
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def compute_cosine_similarity(query_embs: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and candidate embeddings.

    Args:
        query_embs: (N, D) embeddings of query molecules
        candidate_embs: (M, D) embeddings of candidate molecules

    Returns:
        (N, M) similarity matrix
    """
    # Normalize and compute cosine similarity safely
    q = np.nan_to_num(query_embs.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    c = np.nan_to_num(candidate_embs.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    c_norms[c_norms == 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = (q / q_norms) @ (c / c_norms).T
    return np.clip(np.nan_to_num(sim, nan=0.0), -1.0, 1.0).astype(np.float32)


def load_cached_embeddings(model_type: str = "chemberta") -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """
    Load pre-computed embeddings from cache.

    Returns:
        (embeddings_array, entity_ids) or (None, None) if not cached
    """
    cache_path = CHEMBERTA_CACHE if model_type == "chemberta" else MOLFORMER_CACHE

    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        return data["embeddings"], data["entities"].tolist()
    return None, None


def save_cached_embeddings(
    embeddings: np.ndarray,
    entity_ids: list[str],
    model_type: str = "chemberta",
):
    """Save computed embeddings to cache."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CHEMBERTA_CACHE if model_type == "chemberta" else MOLFORMER_CACHE
    np.savez_compressed(
        str(cache_path),
        embeddings=embeddings,
        entities=np.array(entity_ids),
    )


def score_by_learned_similarity(
    disease_entity: str,
    triplets,
    all_compounds: list[str],
    embeddings: np.ndarray,
    embedding_entities: list[str],
    top_k: int = 200,
) -> list[tuple[str, float, str]]:
    """
    Score drugs by learned molecular similarity to known treatments.

    Same logic as fingerprint similarity but using ChemBERTa/MoLFormer embeddings
    and cosine similarity instead of Tanimoto.
    """
    from opencure.scoring.molecular import get_known_treatments
    from opencure.config import TREATMENT_RELATIONS

    # Build entity → embedding index mapping
    entity_to_idx = {e: i for i, e in enumerate(embedding_entities)}

    # Find known treatments that have embeddings
    known_treatments = get_known_treatments(disease_entity, triplets)
    known_with_emb = [c for c in known_treatments if c in entity_to_idx]

    if not known_with_emb:
        return []

    # Get embeddings for known treatments
    known_indices = [entity_to_idx[c] for c in known_with_emb]
    known_embs = embeddings[known_indices]

    # Get embeddings for all candidates
    candidate_compounds = [c for c in all_compounds if c in entity_to_idx and c not in set(known_treatments)]
    if not candidate_compounds:
        return []

    candidate_indices = [entity_to_idx[c] for c in candidate_compounds]
    candidate_embs = embeddings[candidate_indices]

    # Compute cosine similarity: each candidate vs all known treatments
    sim_matrix = compute_cosine_similarity(candidate_embs, known_embs)  # (N_cand, N_known)

    # For each candidate, take the max similarity across known treatments
    max_sims = sim_matrix.max(axis=1)
    best_known_idx = sim_matrix.argmax(axis=1)

    # Build results
    results = []
    for i, (compound, sim) in enumerate(zip(candidate_compounds, max_sims)):
        if sim > 0.5:  # Higher threshold for learned embeddings (more meaningful)
            similar_to = known_with_emb[best_known_idx[i]]
            results.append((compound, float(sim), similar_to))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]
