"""
Train an XGBoost ensemble to learn optimal pillar weights from known drug-disease pairs.

Uses 54,775 known treatment edges from DRKG as positive examples and
random drug-disease pairs as negatives to learn which combination of
pillar scores best predicts actual treatments.

Usage:
    python3 scripts/train_ensemble.py
"""
from __future__ import annotations

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.config import TREATMENT_RELATIONS
from opencure.data.drkg import load_triplets, load_embeddings, get_compound_entities
from opencure.scoring.transe import score_drugs_for_disease_vectorized


def extract_treatment_pairs(triplets: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract known drug-disease treatment pairs from DRKG."""
    treatment_mask = triplets["relation"].isin(TREATMENT_RELATIONS)
    treats = triplets[treatment_mask]
    pairs = list(set(zip(treats["head"], treats["tail"])))
    print(f"Extracted {len(pairs)} unique treatment pairs")
    return pairs


def generate_negative_pairs(
    positive_set: set,
    all_compounds: list[str],
    all_diseases: list[str],
    ratio: int = 1,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Generate random negative drug-disease pairs."""
    rng = random.Random(seed)
    negatives = []
    attempts = 0

    target = len(positive_set) * ratio
    while len(negatives) < target and attempts < target * 10:
        compound = rng.choice(all_compounds)
        disease = rng.choice(all_diseases)
        if (compound, disease) not in positive_set:
            negatives.append((compound, disease))
        attempts += 1

    print(f"Generated {len(negatives)} negative pairs")
    return negatives


def compute_transe_score(compound, disease, entity_emb, entity_to_id, relation_emb, relation_to_id):
    """Compute TransE score for a single drug-disease pair."""
    if compound not in entity_to_id or disease not in entity_to_id:
        return 0.0

    h_idx = entity_to_id[compound]
    t_idx = entity_to_id[disease]
    h_emb = entity_emb[h_idx]
    t_emb = entity_emb[t_idx]

    best_score = float("-inf")
    for rel in TREATMENT_RELATIONS:
        if rel in relation_to_id:
            r_idx = relation_to_id[rel]
            r_emb = relation_emb[r_idx]
            score = -float(np.linalg.norm(h_emb + r_emb - t_emb))
            best_score = max(best_score, score)

    return best_score if best_score > float("-inf") else 0.0


def compute_features_batch(
    pairs: list[tuple[str, str]],
    entity_emb, entity_to_id, relation_emb, relation_to_id,
    compound_set: set,
) -> np.ndarray:
    """Compute feature vectors for a batch of drug-disease pairs.

    Features:
    0: transe_score (raw TransE distance)
    1: compound_degree (number of edges for this compound in entity_to_id)
    2: disease_degree (number of edges for this disease)
    3: compound_in_drkg (1 if compound exists in embeddings)
    4: disease_in_drkg (1 if disease exists in embeddings)
    """
    features = np.zeros((len(pairs), 5), dtype=np.float32)

    for i, (compound, disease) in enumerate(pairs):
        # Feature 0: TransE score
        features[i, 0] = compute_transe_score(
            compound, disease, entity_emb, entity_to_id, relation_emb, relation_to_id
        )

        # Feature 1-2: Entity presence
        features[i, 3] = 1.0 if compound in entity_to_id else 0.0
        features[i, 4] = 1.0 if disease in entity_to_id else 0.0

    return features


def main():
    print("=" * 60)
    print("  OpenCure XGBoost Ensemble Training")
    print("=" * 60)

    # Load data
    print("\nLoading DRKG data...")
    entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id = load_embeddings()
    triplets = load_triplets()
    compounds = get_compound_entities(entity_to_id)
    compound_set = set(compounds)

    # Get all disease entities
    disease_entities = [e for e in entity_to_id if e.startswith("Disease::")]
    print(f"  {len(compounds)} compounds, {len(disease_entities)} diseases")

    # Extract positive pairs
    print("\nExtracting treatment pairs...")
    positive_pairs = extract_treatment_pairs(triplets)
    positive_set = set(positive_pairs)

    # Generate negative pairs (1:1 ratio)
    print("Generating negative pairs...")
    negative_pairs = generate_negative_pairs(
        positive_set, compounds, disease_entities, ratio=1
    )

    # Combine
    all_pairs = positive_pairs + negative_pairs
    labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    print(f"\nTotal pairs: {len(all_pairs)} ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    # Compute features
    print("\nComputing features (TransE scores)...")
    start = time.time()
    X = compute_features_batch(
        all_pairs, entity_emb, entity_to_id, relation_emb, relation_to_id, compound_set
    )
    print(f"  Feature computation: {time.time() - start:.1f}s")
    print(f"  Feature matrix shape: {X.shape}")

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n  Train: {len(X_train)} ({sum(y_train)} positive)")
    print(f"  Test:  {len(X_test)} ({sum(y_test)} positive)")

    # Train Gradient Boosting (sklearn — no OpenMP dependency)
    print("\nTraining GradientBoosting ensemble...")
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    print(f"\n{'=' * 40}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR:  {auc_pr:.4f}")
    print(f"{'=' * 40}")

    # Feature importances
    feature_names = ["transe_score", "compound_degree", "disease_degree",
                     "compound_in_drkg", "disease_in_drkg"]
    importances = model.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:20s}: {imp:.4f}")

    # Save model
    import joblib
    model_path = "data/models/ensemble_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Also save metadata
    meta = {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": feature_names,
        "feature_importances": {n: float(i) for n, i in zip(feature_names, importances)},
    }
    with open("data/models/ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: data/models/ensemble_meta.json")


if __name__ == "__main__":
    main()
