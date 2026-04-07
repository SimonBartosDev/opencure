"""
Drug-Target Interaction prediction using ESM-2 protein embeddings
+ ChemBERTa drug embeddings.

ESM-2 (Meta): Protein language model that generates embeddings from
amino acid sequences. These embeddings encode protein structure,
function, and evolutionary relationships.

Combined with ChemBERTa drug embeddings, we can predict whether a drug
interacts with a protein target - without needing 3D structures.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from opencure.config import DATA_DIR

MODELS_DIR = DATA_DIR.parent / "models"
DTI_MODEL_PATH = MODELS_DIR / "dti_predictor.pt"
PROTEIN_EMB_CACHE = DATA_DIR / "embeddings" / "protein_embeddings.npz"


class DTIPredictor(nn.Module):
    """
    Simple MLP that predicts drug-target interaction from
    concatenated drug + protein embeddings.
    """

    def __init__(self, drug_dim: int = 768, protein_dim: int = 320, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drug_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, drug_emb, protein_emb):
        x = torch.cat([drug_emb, protein_emb], dim=-1)
        return self.net(x)


def get_esm2_embeddings(
    sequences: list[str],
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute protein embeddings using ESM-2.

    Uses the small 8M parameter model by default (fast, CPU-friendly).
    For better quality, use 'facebook/esm2_t33_650M_UR50D' (needs GPU).

    Args:
        sequences: List of amino acid sequences
        model_name: ESM-2 model variant
        batch_size: Batch size
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray of shape (len(sequences), embedding_dim)
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        # Truncate long sequences
        batch = [s[:1022] for s in batch]

        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over sequence length (excluding special tokens)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            mean_emb = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(mean_emb.cpu().numpy())

    return np.vstack(all_embeddings)


def load_protein_embeddings() -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Load cached protein embeddings."""
    if PROTEIN_EMB_CACHE.exists():
        data = np.load(str(PROTEIN_EMB_CACHE), allow_pickle=True)
        return data["embeddings"], data["gene_ids"].tolist()
    return None, None


def save_protein_embeddings(embeddings: np.ndarray, gene_ids: list[str]):
    """Save protein embeddings to cache."""
    PROTEIN_EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(PROTEIN_EMB_CACHE),
        embeddings=embeddings,
        gene_ids=np.array(gene_ids),
    )


def load_dti_model() -> Optional[DTIPredictor]:
    """Load trained DTI predictor model."""
    if DTI_MODEL_PATH.exists():
        model = DTIPredictor()
        model.load_state_dict(torch.load(DTI_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        return model
    return None


def predict_interactions(
    drug_embeddings: np.ndarray,
    protein_embeddings: np.ndarray,
    model: DTIPredictor,
) -> np.ndarray:
    """
    Predict drug-target interaction probabilities.

    Args:
        drug_embeddings: (N_drugs, drug_dim)
        protein_embeddings: (N_proteins, protein_dim)
        model: Trained DTI predictor

    Returns:
        (N_drugs, N_proteins) interaction probability matrix
    """
    model.eval()
    n_drugs = drug_embeddings.shape[0]
    n_proteins = protein_embeddings.shape[0]

    # Compute all pairs
    scores = np.zeros((n_drugs, n_proteins))

    with torch.no_grad():
        drug_tensor = torch.tensor(drug_embeddings, dtype=torch.float32)
        prot_tensor = torch.tensor(protein_embeddings, dtype=torch.float32)

        for i in range(n_drugs):
            drug_repeated = drug_tensor[i].unsqueeze(0).repeat(n_proteins, 1)
            pred = model(drug_repeated, prot_tensor).squeeze(-1).numpy()
            scores[i] = pred

    return scores
