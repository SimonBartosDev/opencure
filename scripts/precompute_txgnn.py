"""
Pre-compute TxGNN predictions using retrieve_embedding + dot product scoring.

Usage:
    source data/txgnn_env/bin/activate
    python scripts/precompute_txgnn.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

OUTPUT_FILE = "data/txgnn_predictions.tsv"
MODEL_DIR = "data/txgnn_model"
EMBEDDING_DIR = "data/txgnn_embeddings"


def main():
    print("=" * 60)
    print("  TxGNN Pre-computation")
    print("=" * 60)

    from txgnn import TxData, TxGNN

    # Load data
    print("\nLoading PrimeKG...")
    txdata = TxData(data_folder_path="data/txgnn_data")
    txdata.prepare_split(split="complex_disease", seed=42)

    n_diseases = txdata.G.num_nodes("disease")
    n_drugs = txdata.G.num_nodes("drug")
    print(f"  {n_diseases} diseases, {n_drugs} drugs")

    # Load node names
    nodes = pd.read_csv("data/txgnn_data/node.csv", sep="\t")
    disease_df = nodes[nodes["node_type"] == "disease"].sort_values("node_index").reset_index(drop=True)
    drug_df = nodes[nodes["node_type"] == "drug"].sort_values("node_index").reset_index(drop=True)

    disease_names = dict(enumerate(disease_df["node_name"]))
    drug_names = dict(enumerate(drug_df["node_name"]))

    # Initialize model
    print("\nInitializing TxGNN...")
    model = TxGNN(
        data=txdata, weight_bias_track=False,
        proj_name="OpenCure", exp_name="precompute", device="cpu",
    )
    model.model_initialize(
        n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3,
        attention=False, sim_measure="all_nodes_profile", agg_measure="rarity",
    )

    # Train or load
    if os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
        print(f"Loading pretrained model...")
        model.load_pretrained(MODEL_DIR)
    else:
        print("Training from scratch (2 pretrain + 50 finetune epochs)...")
        model.pretrain(n_epoch=2, learning_rate=1e-3, batch_size=1024, train_print_per_n=20)
        model.finetune(n_epoch=50, learning_rate=5e-4, train_print_per_n=5, valid_per_n=10)
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")

    # Retrieve embeddings
    print("\nRetrieving embeddings...")
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    emb_pkl = os.path.join(EMBEDDING_DIR, "node_emb.pkl")

    if not os.path.exists(emb_pkl):
        model.retrieve_embedding(path=EMBEDDING_DIR)

    # Load from pickle (dict of torch tensors by node type)
    import pickle
    with open(emb_pkl, "rb") as f:
        node_embs = pickle.load(f)

    # Extract drug and disease embeddings
    # node_embs keys are typically indices; we need to identify drug vs disease
    # The node.csv maps node_type + node_index + node_name
    node_df = pd.read_csv("data/txgnn_data/node.csv", sep="\t" if "\t" in open("data/txgnn_data/node.csv").readline() else ",")

    # Get drug and disease rows
    drug_nodes = node_df[node_df.iloc[:, 0] == "drug"]
    disease_nodes = node_df[node_df.iloc[:, 0] == "disease"]

    print(f"  Drug nodes: {len(drug_nodes)}, Disease nodes: {len(disease_nodes)}")

    # The embeddings are stored as a tensor; extract by index
    if isinstance(node_embs, dict):
        # Try different formats
        if "drug" in node_embs:
            drug_emb = node_embs["drug"].cpu().numpy() if torch.is_tensor(node_embs["drug"]) else np.array(node_embs["drug"])
            disease_emb = node_embs["disease"].cpu().numpy() if torch.is_tensor(node_embs["disease"]) else np.array(node_embs["disease"])
        else:
            # Single tensor indexed by node type
            all_emb = list(node_embs.values())[0]
            if torch.is_tensor(all_emb):
                all_emb = all_emb.cpu().numpy()
            drug_indices = drug_nodes.iloc[:, 1].values.astype(int)
            disease_indices = disease_nodes.iloc[:, 1].values.astype(int)
            drug_emb = all_emb[drug_indices]
            disease_emb = all_emb[disease_indices]
    elif torch.is_tensor(node_embs):
        all_emb = node_embs.cpu().numpy()
        drug_indices = drug_nodes.iloc[:, 1].values.astype(int)
        disease_indices = disease_nodes.iloc[:, 1].values.astype(int)
        drug_emb = all_emb[drug_indices]
        disease_emb = all_emb[disease_indices]
    else:
        raise ValueError(f"Unexpected embedding format: {type(node_embs)}")

    print(f"  Drug embeddings: {drug_emb.shape}")
    print(f"  Disease embeddings: {disease_emb.shape}")

    # Score all pairs via dot product
    print("  Computing scores...")
    scores = drug_emb @ disease_emb.T  # (n_drugs, n_diseases)

    # Target diseases
    TARGETS = [
        # Neurodegenerative (5)
        "alzheimer", "parkinson", "huntington", "amyotrophic lateral sclerosis",
        "multiple sclerosis",
        # Cancer (11)
        "breast cancer", "lung cancer", "colorectal", "prostate cancer",
        "pancreatic", "leukemia", "lymphoma", "multiple myeloma",
        "melanoma", "glioblastoma", "ovarian",
        # Cardiovascular (6)
        "hypertension", "heart failure", "atrial fibrillation",
        "atherosclerosis", "coronary artery", "pulmonary hypertension",
        # Metabolic (3)
        "diabetes", "type 2 diabetes", "obesity",
        # Infectious (8)
        "malaria", "tuberculosis", "dengue", "chagas",
        "leishmaniasis", "schistosomiasis", "hiv", "hepatitis",
        # Autoimmune/Inflammatory (7)
        "rheumatoid arthritis", "lupus", "crohn", "ulcerative colitis",
        "inflammatory bowel", "psoriasis", "asthma",
        # Respiratory (3)
        "copd", "pulmonary fibrosis", "cystic fibrosis",
        # Rare Diseases (8)
        "sickle cell", "fragile x", "duchenne", "neurofibromatosis",
        "marfan", "ehlers-danlos", "fabry", "gaucher",
        # Psychiatric/Neurological (5)
        "depress", "schizophrenia", "epilepsy", "bipolar", "anxiety",
        # Other (5)
        "chronic kidney", "liver cirrhosis", "osteoporosis",
        "endometriosis", "sepsis",
    ]

    results = []
    for target in TARGETS:
        matches = [(i, n) for i, n in disease_names.items() if target.lower() in n.lower()]
        if not matches:
            print(f"  {target}: no match")
            continue

        disease_idx, disease_name = min(matches, key=lambda x: len(x[1]))
        disease_scores = scores[:, disease_idx]
        top_indices = np.argsort(-disease_scores)[:100]

        for rank, drug_idx in enumerate(top_indices, 1):
            results.append({
                "disease": disease_name,
                "drug": drug_names.get(int(drug_idx), f"drug_{drug_idx}"),
                "score": float(disease_scores[drug_idx]),
                "rank": rank,
            })

        print(f"  {disease_name}: top={drug_names.get(int(top_indices[0]),'?')} (score={disease_scores[top_indices[0]]:.4f})")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        f.write("disease\tdrug\tscore\trank\n")
        for r in results:
            f.write(f"{r['disease']}\t{r['drug']}\t{r['score']:.6f}\t{r['rank']}\n")
    print(f"\nSaved {len(results)} predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
