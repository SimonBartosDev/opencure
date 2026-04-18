"""
Ground-truth (drug, disease) treatment pairs for held-out evaluation.

Primary source: DRKG's 4,968 `DRUGBANK::treats::Compound:Disease` edges.
These are derived from DrugBank indication annotations and are the cleanest
positive labels available without external downloads.

Future sources (Phase 5+): DrugCentral, RepoDB, ChEMBL Indications.
This module is designed so those loaders can be plugged in and merged with
deduplication.

Usage:
    from opencure.eval.ground_truth import load_drkg_treats, split_holdout
    pairs = load_drkg_treats()
    train, test = split_holdout(pairs, frac=0.2, seed=42)
"""

from __future__ import annotations

import json
import random
from pathlib import Path


DRKG_PATH = Path("data/drkg/drkg.tsv")
GROUND_TRUTH_OUT = Path("data/eval/ground_truth.jsonl")
HOLDOUT_OUT = Path("data/eval/holdout_test.jsonl")
HOLDOUT_TRAIN_OUT = Path("data/eval/holdout_train.jsonl")


def load_drkg_treats() -> list[tuple[str, str]]:
    """
    Load DRUGBANK::treats edges from DRKG.

    Returns list of (compound_entity, disease_entity) tuples, deduped.
    Entity form is kept (e.g. "Compound::DB00945", "Disease::MESH:D003550").
    """
    if not DRKG_PATH.exists():
        raise FileNotFoundError(f"{DRKG_PATH} not found")

    pairs: set[tuple[str, str]] = set()
    with DRKG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r != "DRUGBANK::treats::Compound:Disease":
                continue
            if not (h.startswith("Compound::") and t.startswith("Disease::")):
                continue
            pairs.add((h, t))
    return sorted(pairs)


def split_holdout(
    pairs: list[tuple[str, str]],
    frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministically split pairs into (train, test)."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * frac)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def save_splits(
    train: list[tuple[str, str]],
    test: list[tuple[str, str]],
) -> None:
    GROUND_TRUTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_OUT.open("w") as f:
        for d, s in train + test:
            f.write(json.dumps({"compound": d, "disease": s}) + "\n")
    with HOLDOUT_TRAIN_OUT.open("w") as f:
        for d, s in train:
            f.write(json.dumps({"compound": d, "disease": s}) + "\n")
    with HOLDOUT_OUT.open("w") as f:
        for d, s in test:
            f.write(json.dumps({"compound": d, "disease": s}) + "\n")


def load_holdout() -> list[tuple[str, str]]:
    """Load the persisted held-out test set."""
    if not HOLDOUT_OUT.exists():
        raise FileNotFoundError(f"{HOLDOUT_OUT} not found — run `python -m opencure.eval.ground_truth` first")
    pairs: list[tuple[str, str]] = []
    with HOLDOUT_OUT.open() as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["compound"], d["disease"]))
    return pairs


def main() -> None:
    pairs = load_drkg_treats()
    print(f"Loaded {len(pairs)} (drug, disease) treats pairs from DRKG")
    drugs = {p[0] for p in pairs}
    dis = {p[1] for p in pairs}
    print(f"  Unique drugs:    {len(drugs)}")
    print(f"  Unique diseases: {len(dis)}")

    train, test = split_holdout(pairs, frac=0.2, seed=42)
    print(f"Split: train={len(train)}  test={len(test)}")
    save_splits(train, test)
    print(f"Wrote {GROUND_TRUTH_OUT}")
    print(f"Wrote {HOLDOUT_TRAIN_OUT}")
    print(f"Wrote {HOLDOUT_OUT}")


if __name__ == "__main__":
    main()
