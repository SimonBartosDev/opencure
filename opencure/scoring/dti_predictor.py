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


# ---- DeepPurpose Integration ----

_deeppurpose_model = None


def _load_deeppurpose_model():
    """Load a pre-trained DeepPurpose DTI model."""
    global _deeppurpose_model
    if _deeppurpose_model is not None:
        return _deeppurpose_model

    try:
        from DeepPurpose import utils as dp_utils
        from DeepPurpose import DTI as dp_models

        # Use pre-trained model (Morgan fingerprint drug encoder + CNN protein encoder)
        _deeppurpose_model = dp_models.model_pretrained(model="MPNN_CNN_BindingDB")
        return _deeppurpose_model
    except (ImportError, Exception):
        return None


def predict_dti_deeppurpose(
    drug_smiles: list[str],
    target_sequences: list[str],
    target_names: list[str] | None = None,
) -> np.ndarray:
    """Predict binding affinity using DeepPurpose for all drug-target pairs.

    Returns (N_drugs, N_targets) matrix of predicted binding scores (0-1).
    """
    model = _load_deeppurpose_model()
    if model is None:
        return np.zeros((len(drug_smiles), len(target_sequences)))

    try:
        from DeepPurpose import utils as dp_utils

        scores = np.zeros((len(drug_smiles), len(target_sequences)))

        for j, (seq, name) in enumerate(zip(target_sequences, target_names or [""] * len(target_sequences))):
            # Create drug-target pairs for this target
            X_drug = drug_smiles
            X_target = [seq] * len(drug_smiles)
            drug_names = [f"drug_{i}" for i in range(len(drug_smiles))]
            target_names_list = [name] * len(drug_smiles)

            try:
                X_pred = dp_utils.data_process(
                    X_drug=X_drug,
                    X_target=X_target,
                    y=[0] * len(drug_smiles),
                    drug_encoding="MPNN",
                    target_encoding="CNN",
                    split_method='no_split',  # Don't split for inference
                )
                preds = model.predict(X_pred)
                # DeepPurpose returns pKd predictions (higher = stronger binding)
                # Range typically 2-10. Convert to 0-1: (pKd - 3) / 7 clipped
                import numpy as _np
                for i, pkd in enumerate(preds):
                    scores[i, j] = _np.clip((pkd - 3.0) / 7.0, 0.0, 1.0)
            except Exception:
                pass

        return scores

    except Exception:
        return np.zeros((len(drug_smiles), len(target_sequences)))


def get_disease_target_sequences(
    disease_name: str,
    triplets,
) -> list[tuple[str, str, str]]:
    """Get protein sequences for disease-relevant targets.

    Returns list of (gene_symbol, uniprot_id, sequence) tuples.
    """
    import requests
    import time as _time

    # Resolve disease name to DRKG entity IDs (uses name-to-MESH/DOID map)
    try:
        from opencure.data.drkg import find_disease_entities, load_embeddings
        _, _, entity_to_id, _, _ = load_embeddings()
        disease_matches = find_disease_entities(entity_to_id, disease_name)
        disease_entities = {m[0] for m in disease_matches}
    except Exception:
        disease_entities = set()

    if not disease_entities:
        return []

    # Get disease genes from DRKG by querying triplets where head or tail IS a disease entity
    disease_genes = set()
    mask1 = triplets["head"].isin(disease_entities) & triplets["tail"].str.startswith("Gene::")
    mask2 = triplets["tail"].isin(disease_entities) & triplets["head"].str.startswith("Gene::")

    for gene_entity in triplets[mask1]["tail"].unique():
        gene_id = str(gene_entity).split("::")[1].split(";")[0]
        disease_genes.add(gene_id)
    for gene_entity in triplets[mask2]["head"].unique():
        gene_id = str(gene_entity).split("::")[1].split(";")[0]
        disease_genes.add(gene_id)

    # Map to UniProt sequences (top 10 targets)
    targets = []
    try:
        from opencure.scoring.mendelian_randomization import _load_entrez_to_symbol
        entrez_map = _load_entrez_to_symbol()
    except Exception:
        entrez_map = {}

    for gene_id in list(disease_genes)[:15]:
        symbol = entrez_map.get(gene_id, gene_id)
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
                    "format": "fasta",
                    "size": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200 and ">" in resp.text:
                lines = resp.text.strip().split("\n")
                header = lines[0]
                sequence = "".join(lines[1:])
                if len(sequence) > 50:
                    uniprot_id = header.split("|")[1] if "|" in header else ""
                    targets.append((symbol, uniprot_id, sequence))
                    if len(targets) >= 10:
                        break
            _time.sleep(0.3)
        except Exception:
            pass

    return targets


def score_drugs_for_disease_dti(
    disease_name: str,
    compound_entities: list[str],
    smiles_map: dict,
    triplets,
) -> dict:
    """Score compounds by predicted drug-target interaction affinity.

    Uses DeepPurpose (if available) to predict binding affinity between
    each candidate drug and the disease's protein targets.

    Returns dict[compound_entity] -> (max_dti_score, best_target, "dti")
    """
    # Get disease target sequences
    targets = get_disease_target_sequences(disease_name, triplets)
    if not targets:
        return {}

    # Collect SMILES for candidates (smiles_map is keyed by full entity "Compound::DB00001")
    compounds_with_smiles = []
    smiles_list = []
    for compound in compound_entities:
        # Try entity key first, then DB id as fallback
        smiles = smiles_map.get(compound)
        if smiles is None:
            db_id = compound.split("::")[1] if "::" in compound else compound
            smiles = smiles_map.get(db_id)
        if smiles:
            compounds_with_smiles.append(compound)
            smiles_list.append(smiles)

    if not smiles_list:
        return {}

    target_sequences = [t[2] for t in targets]
    target_names = [t[0] for t in targets]

    # Predict DTI
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores_matrix = predict_dti_deeppurpose(smiles_list, target_sequences, target_names)

    # Aggregate: max score across targets for each drug
    results = {}
    for i, compound in enumerate(compounds_with_smiles):
        max_score = float(scores_matrix[i].max())
        best_target_idx = int(scores_matrix[i].argmax())
        best_target = target_names[best_target_idx] if best_target_idx < len(target_names) else ""

        if max_score > 0.1:  # Minimum threshold
            results[compound] = (max_score, best_target, "dti")

    return results
