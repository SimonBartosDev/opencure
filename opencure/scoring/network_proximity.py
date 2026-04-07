"""
Network proximity scoring for drug repurposing.

Measures the shortest-path distance between drug target proteins and
disease gene proteins in the STRING protein-protein interaction (PPI) network.

Based on the approach validated by the Barabási lab for COVID-19 drug
repurposing (PNAS 2021). If a drug's targets are topologically close
to disease genes in the interactome, it may modulate the disease pathway.

This is fundamentally different from knowledge graph embeddings - it measures
physical/functional proximity in the protein interaction network.
"""

import gzip
import numpy as np
import networkx as nx
from pathlib import Path
from functools import lru_cache

from opencure.config import DATA_DIR

STRING_DIR = DATA_DIR.parent / "string"
STRING_LINKS = STRING_DIR / "9606.protein.links.txt.gz"
STRING_ALIASES = STRING_DIR / "9606.protein.aliases.txt.gz"

# Module-level cache
_ppi_cache = {}


def load_ppi_network(min_score: int = 700) -> nx.Graph:
    """
    Load STRING PPI network as a NetworkX graph.

    Args:
        min_score: Minimum combined score (0-1000). 700 = high confidence.

    Returns:
        NetworkX Graph with Ensembl protein IDs as nodes.
    """
    if "graph" in _ppi_cache:
        return _ppi_cache["graph"]

    if not STRING_LINKS.exists():
        print("  [WARN] STRING PPI network not found. Run: bash scripts/download_string.sh")
        return nx.Graph()

    print("  Loading STRING PPI network...")
    G = nx.Graph()

    with gzip.open(str(STRING_LINKS), "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                p1, p2, score = parts[0], parts[1], int(parts[2])
                if score >= min_score:
                    # Remove species prefix "9606."
                    p1 = p1.replace("9606.", "")
                    p2 = p2.replace("9606.", "")
                    G.add_edge(p1, p2)

    print(f"  PPI network: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")
    _ppi_cache["graph"] = G
    return G


def load_gene_to_protein_map() -> dict:
    """
    Load mapping from gene symbols/IDs to STRING protein IDs.

    Returns dict: gene_symbol_or_id → ENSP protein ID
    """
    if "gene_map" in _ppi_cache:
        return _ppi_cache["gene_map"]

    if not STRING_ALIASES.exists():
        return {}

    gene_map = {}
    with gzip.open(str(STRING_ALIASES), "rt") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                protein_id = parts[0].replace("9606.", "")
                alias = parts[1]
                source = parts[2]

                # Map gene symbols and Entrez IDs
                if "Ensembl_HGNC" in source or "BioMart_HUGO" in source:
                    gene_map[alias] = protein_id
                elif "Ensembl_EntrezGene" in source or "Ensembl_NCBI" in source:
                    gene_map[alias] = protein_id

    print(f"  Gene-to-protein mapping: {len(gene_map)} entries")
    _ppi_cache["gene_map"] = gene_map
    return gene_map


def get_drug_target_genes(compound_entity: str, triplets) -> list[str]:
    """Get gene targets for a drug from DRKG triplets."""
    drug_gene_rels = triplets[
        (triplets["head"] == compound_entity)
        & (triplets["tail"].str.startswith("Gene::"))
    ]
    return [g.split("::")[1] for g in drug_gene_rels["tail"].unique()]


def get_disease_genes_from_drkg(disease_entity: str, triplets) -> list[str]:
    """Get genes associated with a disease from DRKG triplets."""
    # Disease → Gene edges
    d2g = triplets[
        (triplets["head"] == disease_entity)
        & (triplets["tail"].str.startswith("Gene::"))
    ]["tail"].unique()

    # Gene → Disease edges (reverse)
    g2d = triplets[
        (triplets["tail"] == disease_entity)
        & (triplets["head"].str.startswith("Gene::"))
    ]["head"].unique()

    all_genes = set(g.split("::")[1] for g in d2g) | set(g.split("::")[1] for g in g2d)
    return list(all_genes)


def compute_network_proximity(
    drug_genes: list[str],
    disease_genes: list[str],
    ppi: nx.Graph,
    gene_map: dict,
) -> dict:
    """
    Compute network proximity between drug targets and disease genes.

    Uses the "closest" measure: average minimum distance from each
    drug target to the nearest disease gene.

    Returns dict with:
        distance: float (average shortest path)
        mapped_drug_genes: int (how many drug genes mapped to PPI)
        mapped_disease_genes: int
        proximity_score: float (0-1, higher = closer/better)
    """
    # Map gene IDs to STRING protein IDs
    drug_proteins = set()
    for g in drug_genes:
        if g in gene_map and gene_map[g] in ppi:
            drug_proteins.add(gene_map[g])

    disease_proteins = set()
    for g in disease_genes:
        if g in gene_map and gene_map[g] in ppi:
            disease_proteins.add(gene_map[g])

    if not drug_proteins or not disease_proteins:
        return {
            "distance": float("inf"),
            "mapped_drug_genes": len(drug_proteins),
            "mapped_disease_genes": len(disease_proteins),
            "proximity_score": 0.0,
        }

    # Compute closest distance using BFS with cutoff (MUCH faster)
    # For each drug target, find nearest disease gene within 4 hops
    distances = []
    for dp in list(drug_proteins)[:10]:  # Limit to 10 drug targets for speed
        try:
            # BFS with cutoff is O(branching_factor^cutoff) instead of O(V+E)
            lengths = nx.single_source_shortest_path_length(ppi, dp, cutoff=4)
            min_dist = float("inf")
            for dis_p in disease_proteins:
                if dis_p in lengths:
                    min_dist = min(min_dist, lengths[dis_p])
            if min_dist < float("inf"):
                distances.append(min_dist)
        except Exception:
            continue

    if not distances:
        return {
            "distance": float("inf"),
            "mapped_drug_genes": len(drug_proteins),
            "mapped_disease_genes": len(disease_proteins),
            "proximity_score": 0.0,
        }

    avg_distance = np.mean(distances)

    # Convert to proximity score (0-1, higher = better)
    # Distance 0 → score 1.0, distance 4+ → score 0.0
    proximity_score = max(0.0, 1.0 - avg_distance / 4.0)

    return {
        "distance": round(avg_distance, 2),
        "mapped_drug_genes": len(drug_proteins),
        "mapped_disease_genes": len(disease_proteins),
        "proximity_score": round(proximity_score, 3),
    }


def _precompute_sparse_distances():
    """
    Pre-compute distance matrix using scipy sparse BFS (100x faster than NetworkX).

    Converts the NetworkX PPI graph to a scipy sparse matrix and computes
    all-pairs shortest paths with cutoff. Cached after first computation.

    Returns: (distance_matrix, node_list, node_to_idx) or (None, None, None)
    """
    if "dist_matrix" in _ppi_cache:
        return _ppi_cache["dist_matrix"], _ppi_cache["node_list"], _ppi_cache["node_to_idx"]

    ppi = load_ppi_network()
    if ppi.number_of_nodes() == 0:
        return None, None, None

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    print("  Pre-computing distance matrix (scipy sparse)...")
    node_list = list(ppi.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    # Build sparse adjacency matrix
    rows, cols = [], []
    for u, v in ppi.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))

    # All-pairs shortest paths (unweighted BFS, limit=4)
    # Returns (n, n) matrix with inf for unreachable pairs
    dist_matrix = shortest_path(adj, method='D', directed=False, unweighted=True, limit=4)
    dist_matrix = dist_matrix.astype(np.float32)

    print(f"  Distance matrix: {n}x{n} ({dist_matrix.nbytes / 1e9:.1f} GB)")

    _ppi_cache["dist_matrix"] = dist_matrix
    _ppi_cache["node_list"] = node_list
    _ppi_cache["node_to_idx"] = node_to_idx
    return dist_matrix, node_list, node_to_idx


def score_drugs_by_proximity(
    disease_entity: str,
    compound_entities: list[str],
    triplets,
    top_k: int = None,
) -> dict:
    """
    Score all drugs by network proximity to disease genes.

    Uses pre-computed scipy sparse distance matrix for O(1) lookups.
    No more per-drug BFS — scores ALL drugs in seconds.

    Returns dict: compound_entity → (proximity_score, distance)
    """
    gene_map = load_gene_to_protein_map()
    if not gene_map:
        return {}

    # Try fast scipy path first
    dist_matrix, node_list, node_to_idx = _precompute_sparse_distances()

    if dist_matrix is None:
        # Fallback to old NetworkX BFS
        return _score_drugs_by_proximity_slow(
            disease_entity, compound_entities, triplets, top_k or 200
        )

    # Get disease genes → protein indices
    disease_genes = get_disease_genes_from_drkg(disease_entity, triplets)
    if not disease_genes:
        return {}

    disease_protein_idxs = []
    for g in disease_genes:
        if g in gene_map and gene_map[g] in node_to_idx:
            disease_protein_idxs.append(node_to_idx[gene_map[g]])

    if not disease_protein_idxs:
        return {}

    disease_idxs = np.array(disease_protein_idxs)

    # Pre-build drug → gene targets map (vectorized)
    compound_set = set(compound_entities)
    cg_mask = (
        triplets["head"].isin(compound_set) &
        triplets["tail"].str.startswith("Gene::", na=False)
    )
    drug_gene_pairs = triplets[cg_mask][["head", "tail"]].values

    drug_to_protein_idxs = {}
    for compound, gene_entity in drug_gene_pairs:
        gene_id = gene_entity.split("::")[1]
        # Handle composite IDs
        for sub_id in gene_id.split(";"):
            sub_id = sub_id.strip()
            if sub_id in gene_map and gene_map[sub_id] in node_to_idx:
                drug_to_protein_idxs.setdefault(compound, []).append(
                    node_to_idx[gene_map[sub_id]]
                )

    results = {}
    for compound, protein_idxs in drug_to_protein_idxs.items():
        # Get distances from all drug targets to all disease genes
        drug_idxs = np.array(protein_idxs[:20])  # Max 20 targets

        # Sub-matrix: (n_drug_targets, n_disease_genes)
        sub_dists = dist_matrix[np.ix_(drug_idxs, disease_idxs)]

        # For each drug target: minimum distance to any disease gene
        min_per_target = np.min(sub_dists, axis=1)

        # Filter out inf (unreachable)
        reachable = min_per_target[min_per_target < np.inf]
        if len(reachable) == 0:
            continue

        avg_distance = float(np.mean(reachable))
        proximity_score = max(0.0, 1.0 - avg_distance / 4.0)

        if proximity_score > 0:
            results[compound] = (round(proximity_score, 3), round(avg_distance, 2))

    return results


def _score_drugs_by_proximity_slow(
    disease_entity, compound_entities, triplets, top_k=200
):
    """Fallback: old NetworkX BFS method (slow but works without scipy)."""
    ppi = load_ppi_network()
    if ppi.number_of_nodes() == 0:
        return {}

    gene_map = load_gene_to_protein_map()
    disease_genes = get_disease_genes_from_drkg(disease_entity, triplets)
    if not disease_genes:
        return {}

    results = {}
    scored = 0
    for compound in compound_entities:
        drug_genes = get_drug_target_genes(compound, triplets)
        if not drug_genes:
            continue
        prox = compute_network_proximity(drug_genes, disease_genes, ppi, gene_map)
        if prox["proximity_score"] > 0:
            results[compound] = (prox["proximity_score"], prox["distance"])
            scored += 1
        if scored >= top_k:
            break
    return results
