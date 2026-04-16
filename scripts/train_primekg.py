#!/usr/bin/env python3
"""Train TransE embeddings on PrimeKG knowledge graph.

Reads PrimeKG CSV, converts to triplets, trains TransE using PyKEEN,
and saves embeddings for use by the PrimeKG scorer.

Usage:
    python3 scripts/train_primekg.py                # Full training
    python3 scripts/train_primekg.py --epochs 50    # Fewer epochs (faster)
    python3 scripts/train_primekg.py --test          # Quick test (10 epochs)

Output:
    data/models/primekg/entity_embeddings.npy
    data/models/primekg/relation_embeddings.npy
    data/models/primekg/entity_to_id.json
    data/models/primekg/relation_to_id.json
"""

import sys
import os
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRIMEKG_PATH = Path("data/primekg/kg.csv")
MODEL_DIR = Path("data/models/primekg")


# Relations most relevant for drug repurposing — skip anatomy/bioprocess bulk
KEEP_RELATIONS = {
    "indication",          # drug treats disease (18K)
    "contraindication",    # drug contraindicated for disease (61K)
    "drug_protein",        # drug targets protein (51K)
    "disease_protein",     # disease involves protein (161K)
    "drug_effect",         # drug has effect (130K)
    "drug_drug",           # drug-drug interactions (2.7M) — keep subset
    "protein_protein",     # PPI network (642K)
    "off-label use",       # off-label drug-disease (5K)
    "exposure_disease",    # exposure-disease link (5K)
    "disease_disease",     # disease similarity (64K)
}


def load_triplets(max_per_relation: int = 500_000):
    """Load PrimeKG and convert to (head, relation, tail) triplets.

    Filters to drug-repurposing-relevant relations and caps large
    relation types to keep memory manageable.
    """
    print("Loading PrimeKG (filtered to repurposing-relevant relations)...")
    triplets = []
    entity_set = set()
    relation_set = set()
    relation_counts = {}

    with open(PRIMEKG_PATH) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            relation = row["relation"]
            if relation not in KEEP_RELATIONS:
                continue

            # Cap large relations
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            if relation_counts[relation] > max_per_relation:
                continue

            head = f"{row['x_type']}_{row['x_id']}"
            tail = f"{row['y_type']}_{row['y_id']}"

            triplets.append((head, relation, tail))
            entity_set.add(head)
            entity_set.add(tail)
            relation_set.add(relation)

            if len(triplets) % 500_000 == 0:
                print(f"  Loaded {len(triplets)/1e6:.1f}M triplets...")

    print(f"  Total: {len(triplets)} triplets, {len(entity_set)} entities, {len(relation_set)} relations")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        actual = min(count, max_per_relation)
        print(f"    {rel}: {actual:,}")
    return triplets, sorted(entity_set), sorted(relation_set)


def train_transe(triplets, entities, relations, embedding_dim=200, epochs=100, batch_size=8192, lr=0.01):
    """Train TransE embeddings using PyKEEN."""
    try:
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
    except ImportError:
        print("PyKEEN not installed. Install with: pip install pykeen")
        sys.exit(1)

    print(f"\nPreparing training data ({len(triplets)} triplets)...")

    # Create entity and relation maps
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    # Convert to numpy array of IDs
    triples_array = np.array([
        [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
        for h, r, t in triplets
    ], dtype=np.int64)

    # Create TriplesFactory
    tf = TriplesFactory.from_labeled_triples(
        np.array([[h, r, t] for h, r, t in triplets]),
        create_inverse_triples=False,
    )

    print(f"Training TransE (dim={embedding_dim}, epochs={epochs}, batch={batch_size})...")
    print(f"  This will take several hours on CPU...")
    start = time.time()

    # Split for PyKEEN (needs train/test)
    training_tf, testing_tf = tf.split([0.95, 0.05], random_state=42)

    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        model="TransE",
        model_kwargs={"embedding_dim": embedding_dim},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
        },
        optimizer_kwargs={"lr": lr},
        random_seed=42,
    )

    elapsed = time.time() - start
    print(f"  Training completed in {elapsed/3600:.1f} hours")

    # Extract embeddings
    model = result.model
    entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
    relation_embeddings = model.relation_representations[0]().detach().cpu().numpy()

    # Map PyKEEN IDs back to our entity names
    pykeen_entity_to_id = tf.entity_to_id
    pykeen_relation_to_id = tf.relation_to_id

    # Reorder embeddings to match our entity_to_id
    final_entity_emb = np.zeros((len(entities), embedding_dim), dtype=np.float32)
    our_entity_to_id = {}
    for entity_name in entities:
        if entity_name in pykeen_entity_to_id:
            pykeen_idx = pykeen_entity_to_id[entity_name]
            our_idx = len(our_entity_to_id)
            our_entity_to_id[entity_name] = our_idx
            final_entity_emb[our_idx] = entity_embeddings[pykeen_idx]

    final_rel_emb = np.zeros((len(relations), embedding_dim), dtype=np.float32)
    our_relation_to_id = {}
    for rel_name in relations:
        if rel_name in pykeen_relation_to_id:
            pykeen_idx = pykeen_relation_to_id[rel_name]
            our_idx = len(our_relation_to_id)
            our_relation_to_id[rel_name] = our_idx
            final_rel_emb[our_idx] = relation_embeddings[pykeen_idx]

    return final_entity_emb[:len(our_entity_to_id)], final_rel_emb[:len(our_relation_to_id)], our_entity_to_id, our_relation_to_id


def save_model(entity_emb, relation_emb, entity_to_id, relation_to_id):
    """Save trained embeddings and ID maps."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.save(MODEL_DIR / "entity_embeddings.npy", entity_emb)
    np.save(MODEL_DIR / "relation_embeddings.npy", relation_emb)
    (MODEL_DIR / "entity_to_id.json").write_text(json.dumps(entity_to_id))
    (MODEL_DIR / "relation_to_id.json").write_text(json.dumps(relation_to_id))

    print(f"\nSaved to {MODEL_DIR}:")
    print(f"  entity_embeddings.npy: {entity_emb.shape}")
    print(f"  relation_embeddings.npy: {relation_emb.shape}")
    print(f"  entity_to_id.json: {len(entity_to_id)} entities")
    print(f"  relation_to_id.json: {len(relation_to_id)} relations")


def build_alignment():
    """Build entity alignment between DRKG and PrimeKG."""
    print("\nBuilding entity alignment...")
    from opencure.scoring.primekg_scorer import build_entity_alignment
    alignment = build_entity_alignment()
    comp_count = len(alignment.get("compounds", {}))
    disease_count = len(alignment.get("primekg_diseases", {}))
    print(f"  Aligned {comp_count} compounds, {disease_count} diseases mapped")


def main():
    parser = argparse.ArgumentParser(description="Train TransE on PrimeKG")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--test", action="store_true", help="Quick test (10 epochs)")
    args = parser.parse_args()

    if args.test:
        args.epochs = 10
        print("[TEST MODE] Running 10 epochs only")

    if not PRIMEKG_PATH.exists():
        print(f"PrimeKG not found at {PRIMEKG_PATH}")
        print("Download with: scripts/download_primekg.py")
        sys.exit(1)

    triplets, entities, relations = load_triplets()

    entity_emb, relation_emb, entity_to_id, relation_to_id = train_transe(
        triplets, entities, relations,
        embedding_dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    save_model(entity_emb, relation_emb, entity_to_id, relation_to_id)
    build_alignment()

    print("\nDone! PrimeKG embeddings ready for use.")


if __name__ == "__main__":
    main()
