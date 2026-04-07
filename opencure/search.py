"""
OpenCure v2: Multi-pillar drug repurposing search pipeline.

Combines up to 5 AI pillars:
  - Pillar 1: Molecular Intelligence (fingerprints + ChemBERTa/MoLFormer embeddings)
  - Pillar 2: Protein Docking (Boltz-1 + GNINA) [optional, GPU]
  - Pillar 3: Knowledge Graph (TransE + RotatE/PyKEEN + TxGNN)
  - Pillar 4: Literature AI (BioGPT + PubMedBERT) [on-demand]
  - Pillar 5: Protein Language Models (ESM-2 DTI) [optional]

Each pillar independently scores drug-disease pairs. The ensemble scorer
combines them with learned or heuristic weights, applying convergence
bonuses when multiple pillars agree.
"""
from __future__ import annotations

import numpy as np

from opencure.data.drkg import (
    load_triplets,
    load_embeddings,
    get_compound_entities,
    find_disease_entities,
)
from opencure.data.drugnames import load_drug_names
from opencure.scoring.transe import score_drugs_for_disease_vectorized


# Module-level cache for loaded data
_cache = {}


def _get_data():
    """Load and cache DRKG data (expensive, only do once)."""
    if "loaded" not in _cache:
        print("Loading DRKG data...")
        entity_emb, relation_emb, entity_to_id, id_to_entity, relation_to_id = (
            load_embeddings()
        )
        print(f"  {len(entity_to_id):,} entities, {len(relation_to_id)} relations")

        compounds = get_compound_entities(entity_to_id)
        print(f"  {len(compounds):,} compound entities (DrugBank)")

        # Load triplets for graph-based evidence
        triplets = load_triplets()
        print(f"  {len(triplets):,} triplets loaded")

        # Build a DrugBank ID → name mapping
        drug_names = _build_entity_name_map(entity_to_id)

        # Load SMILES cache for molecular similarity (Pillar 1)
        smiles_map = _load_smiles()

        # Try loading PyKEEN model (Pillar 3 upgrade)
        pykeen_model, pykeen_tf = _load_pykeen()

        # Try loading ChemBERTa embeddings (Pillar 1 upgrade)
        chemberta_emb, chemberta_entities = _load_chemberta()

        _cache.update(
            {
                "loaded": True,
                "entity_emb": entity_emb,
                "relation_emb": relation_emb,
                "entity_to_id": entity_to_id,
                "id_to_entity": id_to_entity,
                "relation_to_id": relation_to_id,
                "compounds": compounds,
                "drug_names": drug_names,
                "triplets": triplets,
                "smiles_map": smiles_map,
                "pykeen_model": pykeen_model,
                "pykeen_tf": pykeen_tf,
                "chemberta_emb": chemberta_emb,
                "chemberta_entities": chemberta_entities,
            }
        )
    return _cache


def _build_entity_name_map(entity_to_id: dict) -> dict:
    """Build compound entity → human-readable name mapping."""
    name_map = load_drug_names()
    names = {}
    for entity in entity_to_id:
        if entity.startswith("Compound::"):
            db_id = entity.split("::")[1]
            if db_id in name_map:
                names[entity] = name_map[db_id]
            elif db_id in DRUGBANK_NAMES:
                names[entity] = DRUGBANK_NAMES[db_id]
            else:
                names[entity] = db_id
    return names


def _load_smiles() -> dict[str, str]:
    """Load SMILES cache if available."""
    from opencure.scoring.molecular import load_smiles_cache
    smiles = load_smiles_cache()
    if smiles:
        print(f"  {len(smiles):,} SMILES structures loaded")
    return smiles


def _load_pykeen():
    """Load PyKEEN RotatE model if trained."""
    try:
        from opencure.scoring.pykeen_scorer import load_pykeen_model
        model, tf = load_pykeen_model("rotate")
        return model, tf
    except Exception:
        return None, None


def _load_chemberta():
    """Load precomputed ChemBERTa embeddings if available."""
    try:
        from opencure.scoring.molecular_embeddings import load_cached_embeddings
        emb, entities = load_cached_embeddings("chemberta")
        if emb is not None:
            print(f"  {len(entities):,} ChemBERTa embeddings loaded")
        return emb, entities
    except Exception:
        return None, None


def search(
    disease_name: str,
    top_k: int = 20,
    use_molecular_similarity: bool = True,
    use_evidence: bool = True,
) -> list[dict]:
    """
    Multi-pillar search for drug repurposing candidates.

    Combines:
      - Pillar 3: Knowledge graph TransE scoring (always on)
      - Pillar 1: Molecular similarity to known treatments (if SMILES available)
      - Evidence: DRKG graph-based evidence (existing relationships)

    Args:
        disease_name: Human-readable disease name or MESH/DOID ID
        top_k: Number of top candidates to return
        use_molecular_similarity: Enable Pillar 1 scoring
        use_evidence: Gather evidence from DRKG graph

    Returns:
        List of dicts with multi-pillar scores and evidence
    """
    data = _get_data()

    # Step 1: Find disease entities in DRKG
    disease_matches = find_disease_entities(data["entity_to_id"], disease_name)
    if not disease_matches:
        print(f"No disease entities found for '{disease_name}'")
        print("Try using a MESH ID directly, e.g., 'MESH:D000544' for Alzheimer's")
        return []

    print(f"\nFound {len(disease_matches)} disease entity match(es):")
    for entity_id, score in disease_matches:
        print(f"  {entity_id} (match score: {score:.2f})")

    # Step 2: Pillar 3a - Knowledge graph TransE scoring (always available)
    print("\n[Pillar 3a] TransE knowledge graph scoring...")
    transe_scores = {}

    for disease_entity, _match_score in disease_matches:
        scored = score_drugs_for_disease_vectorized(
            disease_entity=disease_entity,
            entity_emb=data["entity_emb"],
            relation_emb=data["relation_emb"],
            entity_to_id=data["entity_to_id"],
            relation_to_id=data["relation_to_id"],
            compound_entities=data["compounds"],
        )
        for compound, score, relation in scored:
            if compound not in transe_scores or score > transe_scores[compound][0]:
                transe_scores[compound] = (score, relation, disease_entity)

    # Step 2b: Pillar 3b - PyKEEN RotatE scoring (if trained)
    pykeen_scores = {}
    if data.get("pykeen_model") is not None:
        print("[Pillar 3b] RotatE knowledge graph scoring...")
        from opencure.scoring.pykeen_scorer import score_drugs_for_disease_pykeen

        for disease_entity, _ in disease_matches:
            scored = score_drugs_for_disease_pykeen(
                disease_entity=disease_entity,
                model=data["pykeen_model"],
                triples_factory=data["pykeen_tf"],
                compound_entities=data["compounds"],
                top_k=min(top_k * 2, 500),
            )
            for compound, score, relation in scored:
                if compound not in pykeen_scores or score > pykeen_scores[compound][0]:
                    pykeen_scores[compound] = (score, relation, disease_entity)

        if pykeen_scores:
            # Cap RotatE: keep scores for TransE compounds + only 100 supplementary
            # This prevents RotatE from flooding the candidate pool and diluting TransE
            transe_set = set(transe_scores.keys())
            in_transe = {k: v for k, v in pykeen_scores.items() if k in transe_set}
            supplementary = {k: v for k, v in pykeen_scores.items() if k not in transe_set}
            supplementary = dict(sorted(supplementary.items(), key=lambda x: -x[1][0])[:100])
            pykeen_scores = {**in_transe, **supplementary}
            print(f"  Scored {len(pykeen_scores)} compounds with RotatE ({len(supplementary)} supplementary)")

    # Step 3a: Pillar 1a - Fingerprint molecular similarity
    mol_sim_scores = {}
    if use_molecular_similarity and data["smiles_map"]:
        print("[Pillar 1a] Fingerprint molecular similarity...")
        from opencure.scoring.molecular import score_by_molecular_similarity

        for disease_entity, _ in disease_matches:
            sim_results = score_by_molecular_similarity(
                disease_entity=disease_entity,
                triplets=data["triplets"],
                all_compounds=data["compounds"],
                smiles_map=data["smiles_map"],
                top_k=top_k * 5,
            )
            for compound, sim, similar_to in sim_results:
                if compound not in mol_sim_scores or sim > mol_sim_scores[compound][0]:
                    mol_sim_scores[compound] = (sim, similar_to)

        if mol_sim_scores:
            print(f"  Found {len(mol_sim_scores)} compounds with fingerprint similarity")

    # Step 3b: Pillar 1b - ChemBERTa learned molecular similarity
    mol_emb_scores = {}
    if use_molecular_similarity and data.get("chemberta_emb") is not None:
        print("[Pillar 1b] ChemBERTa learned molecular similarity...")
        from opencure.scoring.molecular_embeddings import score_by_learned_similarity

        for disease_entity, _ in disease_matches:
            sim_results = score_by_learned_similarity(
                disease_entity=disease_entity,
                triplets=data["triplets"],
                all_compounds=data["compounds"],
                embeddings=data["chemberta_emb"],
                embedding_entities=data["chemberta_entities"],
                top_k=top_k * 5,
            )
            for compound, sim, similar_to in sim_results:
                if compound not in mol_emb_scores or sim > mol_emb_scores[compound][0]:
                    mol_emb_scores[compound] = (sim, similar_to)

        if mol_emb_scores:
            print(f"  Found {len(mol_emb_scores)} compounds with learned similarity")

    # Step 2c: TxGNN scoring (if pre-computed predictions available)
    txgnn_scores = {}
    try:
        from opencure.scoring.txgnn_scorer import score_drugs_for_disease_txgnn, TXGNN_PREDICTIONS
        if TXGNN_PREDICTIONS.exists():
            print("[Pillar 6] TxGNN (state-of-the-art GNN)...")
            txgnn_scores = score_drugs_for_disease_txgnn(
                disease_name, data["compounds"], data["drug_names"]
            )
            if txgnn_scores:
                print(f"  Matched {len(txgnn_scores)} drugs from TxGNN predictions")
    except Exception:
        pass

    # Step 2d: Mendelian Randomization (causal evidence)
    mr_scores = {}
    try:
            from opencure.scoring.mendelian_randomization import score_drugs_for_disease_mr
            print("[Pillar 7] Mendelian Randomization (causal)...")
            # Score top TransE candidates only (MR involves API calls)
            top_compounds = sorted(transe_scores.keys(), key=lambda c: -transe_scores[c][0])[:200]
            mr_scores = score_drugs_for_disease_mr(
                disease_name, top_compounds, data["triplets"], data["drug_names"]
            )
            if mr_scores:
                print(f"  {len(mr_scores)} drugs with MR causal evidence")
    except Exception as e:
        print(f"  [WARN] MR failed: {e}")

    # Step 4: Combine all pillar scores
    active_pillars = ["TransE"]
    if pykeen_scores:
        active_pillars.append("RotatE")
    if txgnn_scores:
        active_pillars.append("TxGNN")
    if mol_sim_scores:
        active_pillars.append("Fingerprints")
    if mol_emb_scores:
        active_pillars.append("ChemBERTa")
    if mr_scores:
        active_pillars.append("Mendelian Randomization")

    # Step 4b: Gene signature reversal pillar (if enabled)
    gene_sig_scores = {}
    if use_molecular_similarity:
        try:
            from opencure.evidence.gene_signatures import get_disease_genes, query_l1000cds2_reversal
            up, dn = get_disease_genes(disease_name)
            if len(up) >= 5 and len(dn) >= 5:
                print("[Pillar 4] Gene signature reversal (L1000CDS2)...")
                reversers = query_l1000cds2_reversal(up, dn)
                if reversers:
                    # Map L1000CDS2 drug names to our compounds via fuzzy matching
                    drug_names = data["drug_names"]
                    name_to_entity = {v.lower(): k for k, v in drug_names.items()}

                    # Also build alias map: strip salt forms for better matching
                    salt_suffixes = [" hydrochloride", " sulfate", " sodium", " maleate",
                                     " citrate", " tartrate", " fumarate", " acetate",
                                     " phosphate", " chloride", " bromide", " succinate",
                                     " mesylate", " besylate", " tosylate", " nitrate"]
                    alias_to_entity = dict(name_to_entity)
                    for name, entity in name_to_entity.items():
                        for suffix in salt_suffixes:
                            if name.endswith(suffix):
                                alias_to_entity[name[:-len(suffix)]] = entity

                    for r in reversers:
                        l1000_name = r["drug_name"].lower().strip()
                        rank = reversers.index(r) + 1
                        entity = None

                        # Step 1: Exact match (including aliases)
                        if l1000_name in alias_to_entity:
                            entity = alias_to_entity[l1000_name]
                        else:
                            # Step 2: Fuzzy match with rapidfuzz
                            try:
                                from rapidfuzz import process, fuzz
                                match = process.extractOne(
                                    l1000_name, alias_to_entity.keys(),
                                    scorer=fuzz.ratio, score_cutoff=85
                                )
                                if match:
                                    entity = alias_to_entity[match[0]]
                            except ImportError:
                                pass

                        if entity:
                            gene_sig_scores[entity] = (r["score"], rank)

                    if gene_sig_scores:
                        active_pillars.append("Gene Signatures")
                        print(f"  Matched {len(gene_sig_scores)} drugs to L1000CDS2 reversers")
        except Exception:
            pass

    # Step 4c: Network proximity pillar (if STRING data available)
    proximity_scores = {}
    if use_molecular_similarity:
        try:
            from opencure.scoring.network_proximity import score_drugs_by_proximity, STRING_LINKS
            if STRING_LINKS.exists():
                print("[Pillar 5] Network proximity (STRING PPI)...")
                for disease_entity, _ in disease_matches:
                    # Score top TransE candidates by proximity (not all 10K drugs)
                    prox = score_drugs_by_proximity(
                        disease_entity, data["compounds"], data["triplets"]
                    )
                    for compound, (score, dist) in prox.items():
                        if compound not in proximity_scores or score > proximity_scores[compound][0]:
                            proximity_scores[compound] = (score, dist)
                if proximity_scores:
                    active_pillars.append("Network Proximity")
                    print(f"  Scored {len(proximity_scores)} drugs by PPI proximity")
        except Exception as e:
            print(f"  [WARN] Network proximity failed: {e}")

    print(f"\nCombining {len(active_pillars)} active pillars: {', '.join(active_pillars)}")
    combined = _combine_scores_v2(transe_scores, pykeen_scores, mol_sim_scores, mol_emb_scores, data["compounds"], gene_sig_scores, proximity_scores, txgnn_scores, mr_scores)

    # Step 5: Build results with evidence
    ranked = sorted(combined.items(), key=lambda x: -x[1]["combined_score"])[:top_k]

    results = []
    for rank, (compound, scores) in enumerate(ranked, 1):
        db_id = compound.split("::")[1] if "::" in compound else compound
        drug_name = data["drug_names"].get(compound, db_id)
        rel_type = _parse_relation(scores.get("transe_relation", ""))

        result = {
            "rank": rank,
            "drug_entity": compound,
            "drug_id": db_id,
            "drug_name": drug_name,
            "combined_score": round(scores["combined_score"], 4),
            "transe_score": round(scores.get("transe_score", 0), 4),
            "transe_rank": scores.get("transe_rank"),
            "relation_type": rel_type,
            "disease_entity": scores.get("disease_entity", ""),
            "pillars_hit": scores.get("pillars_hit", 1),
        }

        # Add molecular similarity info if available
        if "mol_similarity" in scores:
            result["mol_similarity"] = round(scores["mol_similarity"], 4)
            similar_to = scores.get("similar_to", "")
            result["similar_to"] = data["drug_names"].get(similar_to, similar_to)

        # Add MR (genetic/causal) evidence if available
        if "mr_score" in scores:
            result["mr_score"] = round(scores["mr_score"], 4)
            result["mr_genetic_targets"] = scores.get("mr_genetic_targets", 0)

        # Add graph evidence if requested
        if use_evidence:
            evidence = _get_graph_evidence(compound, scores.get("disease_entity", ""), data)
            result["evidence"] = evidence

        results.append(result)

    return results


def _combine_scores_v2(
    transe_scores: dict,
    pykeen_scores: dict,
    mol_sim_scores: dict,
    mol_emb_scores: dict,
    all_compounds: list[str],
    gene_sig_scores: dict | None = None,
    proximity_scores: dict | None = None,
    txgnn_scores: dict | None = None,
    mr_scores: dict | None = None,
) -> dict:
    """
    Combine scores from multiple pillars into a unified ranking.

    Strategy:
    - Normalize each pillar's scores to [0, 1] via percentile rank
    - Assign base weights per pillar type
    - Dynamically redistribute weights from inactive pillars to active ones
    - Weights ALWAYS sum to 1.0 for active pillars
    - Additive convergence bonus: +0.05 per extra pillar beyond 1
    """
    if gene_sig_scores is None:
        gene_sig_scores = {}
    if proximity_scores is None:
        proximity_scores = {}
    if txgnn_scores is None:
        txgnn_scores = {}
    if mr_scores is None:
        mr_scores = {}

    # Helper: compute percentile ranks for a score dict
    def percentile_rank(score_dict):
        if not score_dict:
            return {}, {}
        sorted_comps = sorted(score_dict.keys(), key=lambda c: -score_dict[c][0])
        total = len(sorted_comps)
        pcts = {}
        ranks = {}
        for i, comp in enumerate(sorted_comps):
            pcts[comp] = 1.0 - (i / total)
            ranks[comp] = i + 1
        return pcts, ranks

    transe_pct, transe_ranks = percentile_rank(transe_scores)
    pykeen_pct, pykeen_ranks = percentile_rank(pykeen_scores)

    # Base weights for each pillar (ideal when all active)
    base_weights = {
        "transe": 0.10,
        "pykeen": 0.10,
        "txgnn": 0.20,      # State-of-the-art GNN
        "mol_fp": 0.05,
        "mol_emb": 0.08,
        "gene_sig": 0.12,
        "proximity": 0.15,  # Validated approach (Barabási lab)
        "mr": 0.15,         # Mendelian randomization (CAUSAL evidence)
        "docking": 0.05,    # Structure-based docking (future)
    }

    # Determine which pillars are active
    active = {}
    if transe_scores:
        active["transe"] = base_weights["transe"]
    if pykeen_scores:
        active["pykeen"] = base_weights["pykeen"]
    if mol_sim_scores:
        active["mol_fp"] = base_weights["mol_fp"]
    if mol_emb_scores:
        active["mol_emb"] = base_weights["mol_emb"]
    if txgnn_scores:
        active["txgnn"] = base_weights["txgnn"]
    if gene_sig_scores:
        active["gene_sig"] = base_weights["gene_sig"]
    if proximity_scores:
        active["proximity"] = base_weights["proximity"]
    # MR is handled as a bonus, not a weighted pillar

    # Redistribute: normalize active weights to sum to 1.0
    total_weight = sum(active.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in active.items()}
    else:
        weights = {}

    combined = {}
    all_scored = (
        set(transe_scores.keys()) | set(pykeen_scores.keys()) |
        set(txgnn_scores.keys()) | set(mol_sim_scores.keys()) |
        set(mol_emb_scores.keys()) | set(gene_sig_scores.keys()) |
        set(proximity_scores.keys())
    )

    for compound in all_scored:
        scores = {}
        pillars_hit = 0

        # Collect per-drug pillar contributions: (base_weight, normalized_score)
        drug_pillars = []

        # Pillar 3a: TransE
        if compound in transe_scores:
            raw_score, relation, disease_entity = transe_scores[compound]
            pct = transe_pct.get(compound, 0)
            scores["transe_score"] = raw_score
            scores["transe_percentile"] = pct
            scores["transe_rank"] = transe_ranks.get(compound, 0)
            scores["transe_relation"] = relation
            scores["disease_entity"] = disease_entity
            drug_pillars.append((base_weights["transe"], pct))
            pillars_hit += 1

        # Pillar 3b: PyKEEN RotatE
        if compound in pykeen_scores:
            raw_score, relation, disease_entity = pykeen_scores[compound]
            pct = pykeen_pct.get(compound, 0)
            scores["pykeen_score"] = raw_score
            scores["pykeen_percentile"] = pct
            scores["pykeen_rank"] = pykeen_ranks.get(compound, 0)
            if "disease_entity" not in scores:
                scores["disease_entity"] = disease_entity
                scores["transe_relation"] = relation
            drug_pillars.append((base_weights["pykeen"], pct))
            pillars_hit += 1

        # Pillar 6: TxGNN
        if compound in txgnn_scores:
            txgnn_score, txgnn_rank = txgnn_scores[compound]
            scores["txgnn_score"] = txgnn_score
            scores["txgnn_rank"] = txgnn_rank
            # Normalize: rank 1 = 1.0, rank 100 = 0.0
            txgnn_pct = max(0, 1.0 - (txgnn_rank / 100.0)) if txgnn_rank > 0 else 0
            drug_pillars.append((base_weights["txgnn"], txgnn_pct))
            pillars_hit += 1

        # Pillar 1a: Fingerprint similarity
        if compound in mol_sim_scores:
            sim, similar_to = mol_sim_scores[compound]
            scores["mol_similarity"] = sim
            scores["similar_to"] = similar_to
            drug_pillars.append((base_weights["mol_fp"], sim))
            pillars_hit += 1

        # Pillar 1b: ChemBERTa learned similarity
        if compound in mol_emb_scores:
            sim, similar_to = mol_emb_scores[compound]
            scores["mol_emb_similarity"] = sim
            scores["mol_emb_similar_to"] = similar_to
            drug_pillars.append((base_weights["mol_emb"], sim))
            pillars_hit += 1

        # Pillar 4: Gene signature reversal
        if compound in gene_sig_scores:
            sig_score, sig_rank = gene_sig_scores[compound]
            scores["gene_sig_score"] = sig_score
            scores["gene_sig_rank"] = sig_rank
            # Normalize: rank 1 = 1.0, rank 50 = 0.0
            sig_pct = max(0, 1.0 - (sig_rank / 50.0)) if sig_rank > 0 else 0
            drug_pillars.append((base_weights["gene_sig"], sig_pct))
            pillars_hit += 1

        # Pillar 5: Network proximity (STRING PPI)
        if compound in proximity_scores:
            prox_score, prox_dist = proximity_scores[compound]
            scores["proximity_score"] = prox_score
            scores["proximity_distance"] = prox_dist
            drug_pillars.append((base_weights["proximity"], prox_score))
            pillars_hit += 1

        # Pillar 7: Mendelian Randomization (causal evidence)
        # MR acts as a BONUS rather than weighted pillar — it boosts drugs
        # with genetic evidence rather than introducing new candidates.
        mr_bonus = 0.0
        if compound in mr_scores:
            mr_score, mr_hits = mr_scores[compound]
            scores["mr_score"] = mr_score
            scores["mr_genetic_targets"] = mr_hits
            # Additive bonus: up to +0.15 for strong genetic evidence
            mr_bonus = 0.15 * mr_score
            pillars_hit += 1

        # Per-drug dynamic weighting: normalize weights for pillars that
        # actually scored THIS drug (not the global active set).
        # This prevents penalizing drugs missing from a pillar's coverage.
        drug_weight_total = sum(w for w, _ in drug_pillars)
        if drug_weight_total > 0:
            weighted_sum = sum((w / drug_weight_total) * s for w, s in drug_pillars)
        else:
            weighted_sum = 0.0

        # Convergence bonus: additive, +0.05 per extra pillar beyond 1
        convergence_bonus = 0.05 * max(0, pillars_hit - 1)
        final_score = weighted_sum + convergence_bonus + mr_bonus

        scores["combined_score"] = final_score
        scores["pillars_hit"] = pillars_hit
        combined[compound] = scores

    return combined


def _get_graph_evidence(compound: str, disease_entity: str, data: dict) -> list[str]:
    """
    Get evidence for a drug-disease relationship from the DRKG graph.

    Looks for:
    1. Direct compound→disease relationships
    2. Shared gene/protein targets (compound→gene←disease paths)
    """
    triplets = data["triplets"]
    drug_names = data["drug_names"]
    evidence = []

    # Direct relationships
    direct = triplets[
        (triplets["head"] == compound) & (triplets["tail"] == disease_entity)
    ]
    for _, row in direct.iterrows():
        rel = _parse_relation(row["relation"])
        evidence.append(f"Direct: {rel} (source: {row['relation'].split('::')[0]})")

    # Shared targets: compound→gene and disease→gene (or gene→disease)
    compound_genes = triplets[
        (triplets["head"] == compound) & (triplets["tail"].str.startswith("Gene::"))
    ]["tail"].unique()

    if len(compound_genes) > 0:
        disease_genes = set(
            triplets[
                (triplets["head"] == disease_entity)
                & (triplets["tail"].str.startswith("Gene::"))
            ]["tail"].unique()
        ) | set(
            triplets[
                (triplets["tail"] == disease_entity)
                & (triplets["head"].str.startswith("Gene::"))
            ]["head"].unique()
        )

        shared = set(compound_genes) & disease_genes
        if shared:
            gene_names = [g.split("::")[1] for g in list(shared)[:5]]
            evidence.append(f"Shared targets: {', '.join(gene_names)}" +
                          (f" (+{len(shared)-5} more)" if len(shared) > 5 else ""))

    return evidence


def search_simple(disease_name: str, top_k: int = 20) -> list[dict]:
    """
    Simple TransE-only search (faster, no molecular similarity).
    Used for validation and quick queries.
    """
    return search(disease_name, top_k=top_k, use_molecular_similarity=False, use_evidence=False)


def _parse_relation(relation: str) -> str:
    """Convert DRKG relation string to human-readable type."""
    rel_map = {
        "GNBR::T::Compound:Disease": "treats",
        "Hetionet::CtD::Compound:Disease": "treats",
        "GNBR::C::Compound:Disease": "inhibits/treats",
        "GNBR::Pa::Compound:Disease": "palliates",
        "GNBR::J::Compound:Disease": "role in pathogenesis",
        "GNBR::Mp::Compound:Disease": "mechanism related",
    }
    return rel_map.get(relation, relation)


# Curated DrugBank ID → drug name mapping for common drugs.
# This is a bootstrap for the PoC - will be replaced by API lookups.
DRUGBANK_NAMES = {
    # Common repurposing candidates and well-known drugs
    "DB00945": "Aspirin",
    "DB00331": "Metformin",
    "DB00203": "Sildenafil",
    "DB01041": "Thalidomide",
    "DB11817": "Baricitinib",
    "DB00608": "Chloroquine",
    "DB01611": "Hydroxychloroquine",
    "DB00563": "Methotrexate",
    "DB00619": "Imatinib",
    "DB01050": "Ibuprofen",
    "DB00316": "Acetaminophen",
    "DB01174": "Phenobarbital",
    "DB00564": "Carbamazepine",
    "DB01238": "Aripiprazole",
    "DB00543": "Amoxicillin",
    "DB00898": "Ethanol",
    "DB00741": "Hydrocortisone",
    "DB00635": "Prednisone",
    "DB00959": "Methylprednisolone",
    "DB01234": "Dexamethasone",
    "DB00091": "Cyclosporine",
    "DB00993": "Azathioprine",
    "DB01033": "Mercaptopurine",
    "DB00244": "Mesalazine",
    "DB00795": "Sulfasalazine",
    "DB00482": "Celecoxib",
    "DB00586": "Diclofenac",
    "DB00788": "Naproxen",
    "DB01009": "Ketoprofen",
    "DB00328": "Indomethacin",
    "DB00997": "Doxorubicin",
    "DB01229": "Paclitaxel",
    "DB00515": "Cisplatin",
    "DB00361": "Vinorelbine",
    "DB00694": "Daunorubicin",
    "DB01248": "Docetaxel",
    "DB00570": "Vinblastine",
    "DB00309": "Vindesine",
    "DB01169": "Arsenic trioxide",
    "DB00642": "Pemetrexed",
    "DB00544": "Fluorouracil",
    "DB00432": "Trifluridine",
    "DB01101": "Capecitabine",
    "DB00441": "Gemcitabine",
    "DB01005": "Hydroxyurea",
    "DB00398": "Sorafenib",
    "DB01259": "Lapatinib",
    "DB00530": "Erlotinib",
    "DB00072": "Trastuzumab",
    "DB00002": "Cetuximab",
    "DB06290": "Simeprevir",
    "DB09102": "Daclatasvir",
    "DB08934": "Sofosbuvir",
    "DB00705": "Delavirdine",
    "DB00625": "Efavirenz",
    "DB00879": "Emtricitabine",
    "DB00709": "Lamivudine",
    "DB00300": "Tenofovir",
    "DB01072": "Atazanavir",
    "DB01264": "Darunavir",
    "DB00701": "Amprenavir",
    "DB01601": "Lopinavir",
    "DB00503": "Ritonavir",
    "DB00932": "Tipranavir",
    "DB00224": "Indinavir",
    "DB01232": "Saquinavir",
    "DB00220": "Nelfinavir",
    "DB06817": "Raltegravir",
    "DB09101": "Dolutegravir",
    "DB00696": "Ergotamine",
    "DB00252": "Phenytoin",
    "DB00313": "Valproic acid",
    "DB01174": "Phenobarbital",
    "DB00909": "Zonisamide",
    "DB00996": "Gabapentin",
    "DB00555": "Lamotrigine",
    "DB01068": "Clonazepam",
    "DB00334": "Olanzapine",
    "DB01224": "Quetiapine",
    "DB00734": "Risperidone",
    "DB01267": "Paliperidone",
    "DB00477": "Chlorpromazine",
    "DB01239": "Chlorprothixene",
    "DB00502": "Haloperidol",
    "DB00363": "Clozapine",
    "DB01104": "Sertraline",
    "DB00472": "Fluoxetine",
    "DB01175": "Escitalopram",
    "DB00715": "Paroxetine",
    "DB00285": "Venlafaxine",
    "DB00543": "Amoxicillin",
    "DB01060": "Amoxicillin",
    "DB00567": "Cephalexin",
    "DB01212": "Ceftriaxone",
    "DB00438": "Ceftazidime",
    "DB01331": "Cefdinir",
    "DB00415": "Ampicillin",
    "DB01327": "Cefazolin",
    "DB01413": "Cefepime",
    "DB00798": "Gentamicin",
    "DB00684": "Tobramycin",
    "DB00479": "Amikacin",
    "DB01082": "Streptomycin",
    "DB00759": "Tetracycline",
    "DB00254": "Doxycycline",
    "DB01017": "Minocycline",
    "DB01321": "Azithromycin",
    "DB01211": "Clarithromycin",
    "DB00199": "Erythromycin",
    "DB00218": "Moxifloxacin",
    "DB00537": "Ciprofloxacin",
    "DB01155": "Gemifloxacin",
    "DB00365": "Grepafloxacin",
    "DB00817": "Rosuvastatin",
    "DB00227": "Lovastatin",
    "DB01076": "Atorvastatin",
    "DB00641": "Simvastatin",
    "DB01098": "Rosuvastatin",
    "DB00678": "Losartan",
    "DB00177": "Valsartan",
    "DB01340": "Cilnidipine",
    "DB00966": "Telmisartan",
    "DB00876": "Eprosartan",
    "DB01029": "Irbesartan",
    "DB00335": "Atenolol",
    "DB01136": "Carvedilol",
    "DB00612": "Bisoprolol",
    "DB00264": "Metoprolol",
    "DB01182": "Propafenone",
    "DB00571": "Propranolol",
    "DB00661": "Verapamil",
    "DB00381": "Amlodipine",
    "DB00622": "Nicardipine",
    "DB01023": "Felodipine",
    "DB00528": "Lercanidipine",
    "DB01115": "Nifedipine",
    "DB00390": "Digoxin",
    "DB00575": "Clonidine",
    "DB01024": "Mycophenolic acid",
    "DB00877": "Sirolimus",
    "DB00864": "Tacrolimus",
    "DB04835": "Maraviroc",
    "DB00674": "Galantamine",
    "DB00843": "Donepezil",
    "DB01043": "Memantine",
    "DB00989": "Rivastigmine",
    "DB01004": "Ganciclovir",
    "DB00577": "Valacyclovir",
    "DB00787": "Acyclovir",
    "DB00529": "Foscarnet",
    "DB00369": "Cidofovir",
    "DB00900": "Didanosine",
    "DB00495": "Zidovudine",
    "DB00649": "Stavudine",
    "DB00238": "Nevirapine",
    "DB00688": "Mycophenolate mofetil",
    "DB00091": "Cyclosporine",
    "DB04845": "Ixazomib",
    "DB06616": "Bosutinib",
    "DB08901": "Ponatinib",
    "DB00171": "ATP",
    "DB00158": "Folic acid",
    "DB00162": "Vitamin A",
    "DB00145": "Glycine",
    "DB00120": "Phenylalanine",
    "DB00160": "Alanine",
    "DB01593": "Zinc",
    "DB09130": "Copper",
    "DB14487": "Zinc acetate",
}
