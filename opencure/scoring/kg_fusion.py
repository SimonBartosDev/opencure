"""
Multi-Knowledge Graph Fusion.

Combines scores from DRKG TransE, DRKG RotatE, and PrimeKG into a single
fused knowledge graph score using Reciprocal Rank Fusion (RRF).

RRF is parameter-free, proven in information retrieval, and handles
missing scores gracefully. When a drug is scored by multiple KGs,
confidence increases.
"""

from __future__ import annotations

RRF_K = 60  # Standard RRF parameter (controls rank sensitivity)


def fuse_kg_scores(
    transe_scores: dict | None = None,
    pykeen_scores: dict | None = None,
    primekg_scores: dict | None = None,
) -> dict:
    """Combine multiple KG score dicts using Reciprocal Rank Fusion.

    Each input is dict[compound] -> (score, metadata...).
    Returns dict[compound] -> (fused_score, num_kgs, "kg_fused").

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each KG i that scored d
    """
    # Collect all KG score dicts that are non-empty
    kg_dicts = {}
    if transe_scores:
        kg_dicts["transe"] = transe_scores
    if pykeen_scores:
        kg_dicts["pykeen"] = pykeen_scores
    if primekg_scores:
        kg_dicts["primekg"] = primekg_scores

    if not kg_dicts:
        return {}

    # If only one KG, pass through (no fusion needed)
    if len(kg_dicts) == 1:
        name, scores = next(iter(kg_dicts.items()))
        return {
            compound: (score_tuple[0], 1, "kg_fused")
            for compound, score_tuple in scores.items()
        }

    # Convert scores to ranks per KG
    kg_ranks = {}
    for kg_name, scores in kg_dicts.items():
        sorted_compounds = sorted(scores.keys(), key=lambda c: -scores[c][0])
        kg_ranks[kg_name] = {comp: rank + 1 for rank, comp in enumerate(sorted_compounds)}

    # Compute RRF scores
    all_compounds = set()
    for scores in kg_dicts.values():
        all_compounds |= set(scores.keys())

    fused = {}
    for compound in all_compounds:
        rrf_score = 0.0
        num_kgs = 0
        for kg_name, ranks in kg_ranks.items():
            if compound in ranks:
                rrf_score += 1.0 / (RRF_K + ranks[compound])
                num_kgs += 1

        # Normalize by max possible score (if ranked #1 in all KGs)
        max_score = len(kg_dicts) * (1.0 / (RRF_K + 1))
        normalized = rrf_score / max_score if max_score > 0 else 0

        fused[compound] = (normalized, num_kgs, "kg_fused")

    return fused
