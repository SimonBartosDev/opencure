"""
Build per-DRKG-compound JUMP Cell Painting morphological profiles.

Input  (downloaded separately):
  data/jump_cp/compound_profiles.parquet   well-level JUMP-CP compound
                                           profiles (803k wells x 737
                                           harmony-corrected features,
                                           keyed by Metadata_JCP2022)
  data/jump_cp/compound_meta.csv.gz         JCP2022 -> InChIKey / SMILES
  data/jump_cp/drkg_inchikeys.tsv           DrugBank ID -> salt-stripped
                                           InChIKey (built upstream)

Output:
  data/jump_cp/drkg_jump_profiles.npz       entities (Compound::DBxxxx) +
                                           embeddings (N x 737 float32)

Pipeline:
  1. Aggregate well-level profiles to a per-compound consensus (median
     over wells per JCP2022).
  2. Map JCP2022 -> salt-stripped InChIKey connectivity block (first 14
     chars) from the JUMP SMILES.
  3. Match that block to DRKG compounds (also salt-stripped) -> DrugBank.
  4. Persist the matched compound -> morphology matrix.

This is the genuinely orthogonal pillar: morphology is a measured
physical observable, not derived from the literature/knowledge graph.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

JUMP_DIR = Path("data/jump_cp")
PROFILES = JUMP_DIR / "compound_profiles.parquet"
META = JUMP_DIR / "compound_meta.csv.gz"
DRKG_IK = JUMP_DIR / "drkg_inchikeys.tsv"
OUT = JUMP_DIR / "drkg_jump_profiles.npz"


def inchikey14(smiles: str) -> str | None:
    """Salt-stripped InChIKey connectivity block (first 14 chars)."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    big = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    try:
        k = Chem.MolToInchiKey(big)
    except Exception:
        return None
    return k.split("-")[0] if k else None


def main() -> None:
    # 1. well-level -> per-compound consensus (median)
    print("Reading well-level profiles...")
    df = pd.read_parquet(PROFILES)
    feat_cols = [c for c in df.columns if c.startswith("X_")]
    print(f"  {len(df)} wells x {len(feat_cols)} features")
    consensus = df.groupby("Metadata_JCP2022")[feat_cols].median()
    print(f"  -> {len(consensus)} per-compound consensus profiles")
    del df

    # 2. JCP2022 -> salt-stripped InChIKey-14
    print("Mapping JCP2022 -> InChIKey (salt-stripped)...")
    meta = pd.read_csv(META)
    jcp_to_ik14: dict[str, str] = {}
    for jcp, smi in zip(meta["Metadata_JCP2022"], meta["Metadata_SMILES"]):
        if pd.isna(smi):
            continue
        ik = inchikey14(smi)
        if ik:
            jcp_to_ik14[jcp] = ik
    print(f"  {len(jcp_to_ik14)} JCP2022 ids mapped to an InChIKey block")

    # 3. DRKG DrugBank -> InChIKey-14  (first DrugBank id wins on collision)
    drkg = pd.read_csv(DRKG_IK, sep="\t")
    ik14_to_db: dict[str, str] = {}
    for db, ik in zip(drkg["drugbank_id"], drkg["inchikey14"]):
        ik14_to_db.setdefault(ik, db)

    # 4. match
    entities: list[str] = []
    vecs: list[np.ndarray] = []
    seen: set[str] = set()
    for jcp, row in consensus.iterrows():
        ik = jcp_to_ik14.get(jcp)
        if ik is None:
            continue
        db = ik14_to_db.get(ik)
        if db is None or db in seen:
            continue
        seen.add(db)
        entities.append(f"Compound::{db}")
        vecs.append(row.to_numpy(dtype=np.float32))

    embeddings = np.array(vecs, dtype=np.float32)
    np.savez_compressed(str(OUT), entities=np.array(entities), embeddings=embeddings)
    print(f"\nMatched {len(entities)} DRKG compounds to JUMP-CP morphology")
    print(f"Saved {OUT}  ({embeddings.shape})")


if __name__ == "__main__":
    main()
