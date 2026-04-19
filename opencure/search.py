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

        # v4 Phase 5: unified-KG TransE (DRKG+PrimeKG+OpenTargets)
        unified_model, unified_tf = _load_unified_kg()

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
                "unified_model": unified_model,
                "unified_tf": unified_tf,
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


def _load_unified_kg():
    """Load unified-KG TransE model (v4 Phase 5) if trained."""
    try:
        from opencure.scoring.unified_kg_scorer import load_unified_model
        return load_unified_model()
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

    # Step 2c: Pillar 10 - PrimeKG knowledge graph scoring
    primekg_scores = {}
    try:
        from opencure.scoring.primekg_scorer import score_drugs_for_disease_primekg
        print("[Pillar 10] PrimeKG knowledge graph scoring...")
        primekg_scores = score_drugs_for_disease_primekg(
            disease_name, data["compounds"]
        )
        if primekg_scores:
            print(f"  Scored {len(primekg_scores)} compounds with PrimeKG")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] PrimeKG scoring failed: {e}")

    # Step 2d: v4 Phase 5 — unified-KG scorer (DRKG+PrimeKG+OpenTargets TransE)
    unified_scores: dict = {}
    if data.get("unified_model") is not None:
        try:
            from opencure.scoring.unified_kg_scorer import score_drugs_for_disease_unified
            print("[Pillar 2b] Unified-KG TransE (DRKG+PrimeKG+OT)...")
            for de in disease_entities:
                out = score_drugs_for_disease_unified(
                    disease_entity=de,
                    model=data["unified_model"],
                    triples_factory=data["unified_tf"],
                    compound_entities=data["compounds"],
                    top_k=500,
                )
                for compound, score_tuple in out.items():
                    if compound not in unified_scores or score_tuple[0] > unified_scores[compound][0]:
                        unified_scores[compound] = score_tuple
            if unified_scores:
                print(f"  Scored {len(unified_scores)} compounds with unified-KG TransE")
        except Exception as e:
            print(f"  [WARN] unified-KG scoring failed: {e}")

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
    if primekg_scores:
        active_pillars.append("PrimeKG")
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

    # v5 supplementary pillar 4b: mechanistic signature reversal
    # Supplements L1000CDS2 (which typically matches <10 DrugBank drugs per
    # disease). Uses OT disease-gene associations + ChEMBL bioactivities to
    # score every drug that pharmacologically modulates disease-associated
    # genes. Coverage: ~thousands of drugs/disease vs ~5-10 from L1000CDS2.
    try:
        from opencure.scoring.mechanistic_reversal import score_mechanistic_reversal
        disease_entity_list = [de for de, _ in disease_matches]
        mech_scores = score_mechanistic_reversal(
            disease_entity_list, data["compounds"], top_k=500,
        )
        if mech_scores:
            # Merge into gene_sig_scores if the drug isn't already there from
            # L1000CDS2 (L1000 takes precedence because it's direct expression
            # evidence; mechanistic_reversal is a pharmacological proxy).
            added = 0
            for compound, (score, n_hits, best_gene) in mech_scores.items():
                if compound in gene_sig_scores:
                    continue
                # Convert our [0,1] score to a pseudo-rank (1 = best)
                pseudo_rank = max(1, int(1 / (score + 0.01)))
                gene_sig_scores[compound] = (score, pseudo_rank)
                added += 1
            if added:
                if "Gene Signatures" not in active_pillars:
                    active_pillars.append("Gene Signatures")
                print(f"  + {added} mechanistic reversers (OT genes × ChEMBL activities)")
    except Exception as e:
        print(f"  [INFO] mechanistic reversal skipped: {e}")

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

    # DTI prediction (DeepPurpose) — only top TransE candidates for speed
    dti_scores = {}
    try:
        from opencure.scoring.dti_predictor import score_drugs_for_disease_dti
        print("[Pillar 11] DeepPurpose DTI prediction (top 200 only)...")
        smiles_map = data.get("smiles_map", {})
        if smiles_map and transe_scores:
            # Score only top 200 TransE candidates to keep runtime manageable
            top_compounds = sorted(transe_scores.keys(), key=lambda c: -transe_scores[c][0])[:200]
            dti_scores = score_drugs_for_disease_dti(
                disease_name, top_compounds, smiles_map, data["triplets"]
            )
            if dti_scores:
                active_pillars.append("DTI")
                print(f"  Scored {len(dti_scores)} compounds by DTI")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] DTI prediction failed: {e}")

    # ADMET/Toxicity filtering and drug-likeness scoring
    admet_scores = {}
    admet_toxic = set()
    try:
        from opencure.scoring.admet_filter import score_drugs_for_disease_admet
        print("[Pillar 9] ADMET/Toxicity filtering...")
        smiles_map = data.get("smiles_map", {})
        if smiles_map:
            admet_scores, admet_toxic = score_drugs_for_disease_admet(
                data["compounds"], smiles_map
            )
            if admet_toxic:
                print(f"  Filtered {len(admet_toxic)} toxic compounds")
            if admet_scores:
                active_pillars.append("ADMET")
                print(f"  Scored {len(admet_scores)} compounds by drug-likeness")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] ADMET filtering failed: {e}")

    print(f"\nCombining {len(active_pillars)} active pillars: {', '.join(active_pillars)}")

    # v3 pipeline: grouped scoring with hard filters
    # v5 canonical scoring: grouped combiner only. No legacy fallback —
    # pillar_groups + grouped_combiner are core modules, their absence is
    # a broken install, not a runtime branch.
    from opencure.scoring.pillar_groups import (
        group_kg_scores, group_structural_scores, group_network_scores, build_feature_matrix
    )
    from opencure.scoring.grouped_combiner import combine_grouped_scores
    from opencure.filters.drug_filter import filter_compounds
    from opencure.scoring.admet_filter import load_cached_predictions

    # Hard filter BEFORE combining — reject non-therapeutic compounds
    admet_cache = load_cached_predictions()
    all_compounds = (
        set(transe_scores) | set(pykeen_scores) | set(primekg_scores)
        | set(unified_scores)
        | set(mol_sim_scores) | set(mol_emb_scores) | set(dti_scores)
        | set(proximity_scores) | set(gene_sig_scores) | set(mr_scores)
        | set(admet_scores) | set(txgnn_scores)
    )
    all_compounds -= admet_toxic  # legacy ADMET filter

    # v5 hard filter: SMILES rules + metabolite/IUPAC heuristics +
    # ChEMBL phase + critical ADMET (FDA-approved bypass per stage).
    kept, rejections = filter_compounds(
        list(all_compounds),
        data["smiles_map"],
        admet_cache=admet_cache,
        check_chembl=True,
        name_map=data.get("drug_names"),
    )
    if rejections:
        print(f"  Filtered {sum(rejections.values())} non-therapeutic compounds: {rejections}")
    kept_set = set(kept)

    # Group correlated pillars
    kg_group = group_kg_scores(transe_scores, pykeen_scores, primekg_scores, unified_scores)
    structural_group = group_structural_scores(mol_sim_scores, mol_emb_scores, dti_scores)
    network_group = group_network_scores(proximity_scores, gene_sig_scores)

    # Build grouped feature matrix (only for kept compounds)
    features = build_feature_matrix(
        kg_group, structural_group, network_group,
        txgnn_scores, mr_scores, admet_scores,
        kept_set,
    )

    # Combine with grouped combiner
    combined = combine_grouped_scores(features)

    # Attach per-pillar scores using ONE canonical field name per concept.
    # See opencure/scoring/common.py::PILLAR_FIELDS for the full list.
    # grouped_combiner already emits group-level: kg_score, structural_score,
    # network_score, txgnn_score, mr_score, admet_score.  Here we surface the
    # individual pillar-level scores too, using canonical names.
    for compound, scores in combined.items():
        if compound in transe_scores:
            raw, rel, disease_entity = transe_scores[compound]
            scores["transe_score"] = raw
            scores["transe_rank"] = 0  # filled below
            scores["transe_relation"] = rel
            scores["disease_entity"] = disease_entity
        if compound in pykeen_scores:
            raw, rel, de = pykeen_scores[compound]
            scores["pykeen_score"] = raw
        if compound in primekg_scores:
            raw, rel, de = primekg_scores[compound]
            scores["primekg_score"] = raw
        if compound in unified_scores:
            norm, rk, rel = unified_scores[compound]
            scores["unified_score"] = norm
            scores["unified_rank"] = rk
        if compound in txgnn_scores:
            raw, rank = txgnn_scores[compound]
            # grouped_combiner already set txgnn_score as rank-normalized [0,1].
            # Preserve the rank but DO NOT overwrite the combiner's normalized
            # version.  Raw probability is available in txgnn_scores dict for
            # anyone who needs it (not persisted to JSON).
            scores["txgnn_rank"] = rank
        if compound in mol_sim_scores:
            sim, similar_to = mol_sim_scores[compound]
            scores["mol_similarity"] = sim
            scores["similar_to"] = similar_to
        if compound in mol_emb_scores:
            sim, similar_to = mol_emb_scores[compound]
            scores["mol_emb_similarity"] = sim
            scores["mol_emb_similar_to"] = similar_to
        if compound in proximity_scores:
            ps, pd = proximity_scores[compound]
            scores["proximity_score"] = ps
            scores["proximity_distance"] = pd
        if compound in gene_sig_scores:
            sig_score, sig_rank = gene_sig_scores[compound]
            scores["gene_sig_score"] = sig_score
            scores["gene_sig_rank"] = sig_rank
        if compound in mr_scores:
            mr_s, mr_t = mr_scores[compound]
            # grouped_combiner sets mr_score already; we keep mr_genetic_targets
            scores["mr_genetic_targets"] = mr_t
        if compound in dti_scores:
            dti_s, dti_t, _ = dti_scores[compound]
            scores["dti_score"] = dti_s
            scores["dti_best_target"] = dti_t
        if compound in admet_scores:
            admet_s, admet_f, _ = admet_scores[compound]
            # grouped_combiner sets admet_score; we keep flags
            scores["admet_flags"] = admet_f

    # Fill transe_rank from percentile rank
    sorted_transe = sorted(transe_scores.keys(), key=lambda c: -transe_scores[c][0])
    for i, c in enumerate(sorted_transe):
        if c in combined:
            combined[c]["transe_rank"] = i + 1

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
            # v3 group-level scores
            "kg_group_score": round(scores.get("kg_score", 0), 4),
            "structural_group_score": round(scores.get("structural_score", 0), 4),
            "network_group_score": round(scores.get("network_score", 0), 4),
            "txgnn_group_score": round(scores.get("txgnn_score", 0), 4),
            "mr_group_score": round(scores.get("mr_score", 0), 4),
            "admet_multiplier": round(scores.get("admet_multiplier", 0.7), 4),
            "efficacy_score": round(scores.get("efficacy_score", 0), 4),
            "groups_hit": scores.get("groups_hit", 0),
        }

        # Pillar 1a: Molecular fingerprint similarity
        if "mol_similarity" in scores:
            result["mol_similarity"] = round(scores["mol_similarity"], 4)
            similar_to = scores.get("similar_to", "")
            result["similar_to"] = data["drug_names"].get(similar_to, similar_to)

        # Pillar 1b: ChemBERTa learned similarity
        if "mol_emb_similarity" in scores:
            result["mol_emb_similarity"] = round(scores["mol_emb_similarity"], 4)
            emb_sim_to = scores.get("mol_emb_similar_to", "")
            result["mol_emb_similar_to"] = data["drug_names"].get(emb_sim_to, emb_sim_to)

        # Pillar 3b: PyKEEN RotatE
        if "pykeen_score" in scores:
            result["pykeen_score"] = round(scores["pykeen_score"], 4)
            result["pykeen_rank"] = scores.get("pykeen_rank", 0)

        # Pillar 6: TxGNN
        if "txgnn_score" in scores:
            result["txgnn_score"] = round(scores["txgnn_score"], 4)
            result["txgnn_rank"] = scores.get("txgnn_rank", 0)

        # Pillar 4: Gene signature reversal
        if "gene_sig_score" in scores:
            result["gene_sig_score"] = round(scores["gene_sig_score"], 4)
            result["gene_sig_rank"] = scores.get("gene_sig_rank", 0)

        # Pillar 5: Network proximity
        if "proximity_score" in scores:
            result["proximity_score"] = round(scores["proximity_score"], 4)
            result["proximity_distance"] = scores.get("proximity_distance", 0)

        # Pillar 7: Mendelian Randomization (causal genetic evidence)
        if "mr_score" in scores:
            result["mr_score"] = round(scores["mr_score"], 4)
            result["mr_genetic_targets"] = scores.get("mr_genetic_targets", 0)

        # Pillar 9: ADMET drug-likeness
        if "admet_score" in scores:
            result["admet_score"] = round(scores["admet_score"], 4)
            result["admet_flags"] = scores.get("admet_flags", "")

        # Pillar 10: PrimeKG knowledge graph
        if "primekg_score" in scores:
            result["primekg_score"] = round(scores["primekg_score"], 4)

        # Pillar 11: DeepPurpose DTI
        if "dti_score" in scores:
            result["dti_score"] = round(scores["dti_score"], 4)
            result["dti_best_target"] = scores.get("dti_best_target", "")

        # Add graph evidence if requested
        if use_evidence:
            evidence = _get_graph_evidence(compound, scores.get("disease_entity", ""), data)
            result["evidence"] = evidence

        results.append(result)

    return results



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
