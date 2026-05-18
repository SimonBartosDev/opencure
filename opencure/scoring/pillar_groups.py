"""
Pillar grouping: combine correlated pillars into orthogonal signals.

Rationale: Several pillars capture overlapping biological information.
Combining them into groups reduces double-counting and makes the ensemble
input cleaner (6 features vs 11 correlated features).

Groups:
  - KG Group: TransE + RotatE + PrimeKG (all knowledge-graph embeddings)
       → Combined via Reciprocal Rank Fusion
  - Structural Group: MolFP + ChemBERTa + DTI (molecular similarity / binding)
       → Take max score per compound
  - Network Group: Proximity + GeneSig (protein network / transcriptomic)
       → Take max score per compound

Left ungrouped (orthogonal signals):
  - TxGNN (state-of-the-art GNN, different architecture)
  - MR (causal genetic evidence, unique method)
  - ADMET (toxicity/drug-likeness, orthogonal to efficacy)
"""

from __future__ import annotations


def group_kg_scores(
    transe_scores: dict | None = None,
    pykeen_scores: dict | None = None,
    primekg_scores: dict | None = None,
    unified_scores: dict | None = None,
    rgcn_scores: dict | None = None,
) -> dict:
    """Fuse knowledge-graph pillars via Reciprocal Rank Fusion.

    unified_scores (v4 Phase 5) is the RotatE embedding trained on the
    DRKG+PrimeKG+OpenTargets unified graph.
    rgcn_scores (v6.1) is the heterogeneous-GNN R-GCN 12th pillar.
    Both are optional; the function passes through with whatever is given.

    Returns dict[compound] -> (fused_score, num_kgs_contributing, "kg_group").
    """
    from opencure.scoring.kg_fusion import fuse_kg_scores
    return fuse_kg_scores(transe_scores, pykeen_scores, primekg_scores,
                          unified_scores, rgcn_scores)


def group_structural_scores(
    mol_fp_scores: dict | None = None,
    mol_emb_scores: dict | None = None,
    dti_scores: dict | None = None,
    jump_scores: dict | None = None,
) -> dict:
    """Combine molecular-similarity + drug-target-interaction + morphology pillars.

    v7 adds ``jump_scores`` (JUMP Cell Painting morphological similarity)
    to the structural group: a compound that produces the same cellular
    phenotype as a known treatment carries structural-mechanism signal
    even when its 2D fingerprint is dissimilar.

    Takes max score across pillars per compound (most optimistic signal).
    Returns dict[compound] -> (max_score, best_pillar, "structural_group").
    """
    dicts = {}
    if mol_fp_scores:
        dicts["mol_fp"] = mol_fp_scores
    if mol_emb_scores:
        dicts["mol_emb"] = mol_emb_scores
    if dti_scores:
        dicts["dti"] = dti_scores
    if jump_scores:
        dicts["jump"] = jump_scores

    if not dicts:
        return {}

    # Collect all compounds
    all_compounds = set()
    for d in dicts.values():
        all_compounds |= set(d.keys())

    result = {}
    for compound in all_compounds:
        best_score = 0.0
        best_pillar = ""
        for pillar_name, scores in dicts.items():
            if compound in scores:
                score = scores[compound][0]  # First element is always score
                if score > best_score:
                    best_score = score
                    best_pillar = pillar_name
        if best_score > 0:
            result[compound] = (best_score, best_pillar, "structural_group")

    return result


def group_network_scores(
    proximity_scores: dict | None = None,
    gene_sig_scores: dict | None = None,
) -> dict:
    """Combine network-proximity + gene-signature pillars.

    Proximity gives 0-1 scores directly. Gene signature gives ranks (1-best).
    Normalize gene_sig rank to 0-1, then take max.

    Returns dict[compound] -> (max_score, best_pillar, "network_group").
    """
    dicts = {}
    if proximity_scores:
        dicts["proximity"] = proximity_scores
    if gene_sig_scores:
        # Convert ranks to 0-1 scores: rank 1 = 1.0, rank 50+ = 0.0
        gene_sig_normalized = {}
        for c, (score, rank) in gene_sig_scores.items():
            normalized = max(0.0, 1.0 - (rank / 50.0)) if rank > 0 else 0
            gene_sig_normalized[c] = (normalized, rank)
        dicts["gene_sig"] = gene_sig_normalized

    if not dicts:
        return {}

    all_compounds = set()
    for d in dicts.values():
        all_compounds |= set(d.keys())

    result = {}
    for compound in all_compounds:
        best_score = 0.0
        best_pillar = ""
        for pillar_name, scores in dicts.items():
            if compound in scores:
                score = scores[compound][0]
                if score > best_score:
                    best_score = score
                    best_pillar = pillar_name
        if best_score > 0:
            result[compound] = (best_score, best_pillar, "network_group")

    return result


def normalize_txgnn(txgnn_scores: dict) -> dict:
    """Normalize TxGNN rank-based scores to 0-1."""
    if not txgnn_scores:
        return {}
    result = {}
    for compound, (score, rank) in txgnn_scores.items():
        # rank 1 = 1.0, rank 100 = 0.0
        normalized = max(0.0, 1.0 - (rank / 100.0)) if rank > 0 else 0
        result[compound] = (normalized, rank, "txgnn")
    return result


def build_feature_matrix(
    kg_group: dict,
    structural_group: dict,
    network_group: dict,
    txgnn_scores: dict,
    mr_scores: dict,
    admet_scores: dict,
    all_compounds: set,
) -> dict:
    """
    Build per-compound feature vectors for ensemble input.

    Returns dict[compound] -> {
        "kg_score": float,
        "structural_score": float,
        "network_score": float,
        "txgnn_score": float,
        "mr_score": float,
        "admet_score": float,
        "groups_hit": int,  # how many of the 6 groups have non-zero score
    }
    """
    txgnn_normalized = normalize_txgnn(txgnn_scores)
    features = {}

    from opencure.scoring.hub_normalize import degree_penalty

    for compound in all_compounds:
        kg = kg_group.get(compound, (0, 0, ""))[0]
        structural = structural_group.get(compound, (0, "", ""))[0]
        network = network_group.get(compound, (0, "", ""))[0]
        txgnn = txgnn_normalized.get(compound, (0, 0, ""))[0]
        mr = mr_scores.get(compound, (0, 0))[0] if compound in mr_scores else 0
        admet = admet_scores.get(compound, (0, "", ""))[0] if compound in admet_scores else 0

        # Hub-degree damping: drugs connected to ~everything in DRKG (Dex, Cimetidine,
        # Calcium, Glutathione) get mechanically high KG/network scores regardless of
        # disease-specific biology. Damp these topology-driven pillars. Structural,
        # MR, TxGNN, ADMET are chemistry/genetics/GNN-based and stay un-damped.
        penalty = degree_penalty(compound)
        kg = kg * penalty
        network = network * penalty

        groups_hit = sum(1 for x in [kg, structural, network, txgnn, mr, admet] if x > 0)

        features[compound] = {
            "kg_score": kg,
            "structural_score": structural,
            "network_score": network,
            "txgnn_score": txgnn,
            "mr_score": mr,
            "admet_score": admet,
            "groups_hit": groups_hit,
            "degree_penalty": penalty,
        }

    return features
