"""
LLM-powered mechanistic explanation for drug repurposing predictions.

Uses Graph-based Retrieval Augmented Generation (GraphRAG):
1. Extract multi-hop paths between drug and disease in the DRKG knowledge graph
2. Format paths as human-readable mechanistic chains
3. Feed paths + evidence to Claude API for mechanistic hypothesis generation

This replaces the keyword-based BioGPT approach with real mechanistic reasoning.
Researchers get explanations like: "Drug X inhibits kinase Y, which phosphorylates Z,
which drives disease pathway W" instead of generic text.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

# Check for API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def extract_kg_paths(
    drug_entity: str,
    disease_entity: str,
    triplets,
    max_hops: int = 3,
    max_paths: int = 15,
) -> list[list[tuple]]:
    """
    Extract multi-hop paths between drug and disease in DRKG.

    Uses BFS to find shortest paths through the knowledge graph.
    Each path is a list of (head, relation, tail) triplets.

    Args:
        drug_entity: DRKG compound entity (e.g., 'Compound::DB00945')
        disease_entity: DRKG disease entity (e.g., 'Disease::MESH:D000544')
        triplets: DRKG triplets DataFrame
        max_hops: Maximum path length
        max_paths: Maximum number of paths to return

    Returns:
        List of paths, where each path is a list of (head, relation, tail) tuples
    """
    # Build adjacency list from triplets (both directions)
    adj = {}
    for _, row in triplets.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]

        if h not in adj:
            adj[h] = []
        adj[h].append((r, t))

        # Also add reverse edges for bidirectional search
        if t not in adj:
            adj[t] = []
        adj[t].append((f"REV::{r}", h))

    # BFS from drug entity
    paths = []
    queue = deque()
    queue.append((drug_entity, [(drug_entity, "", "")]))
    visited = {drug_entity}

    while queue and len(paths) < max_paths:
        current, path = queue.popleft()

        if len(path) > max_hops + 1:
            break

        if current == disease_entity and len(path) > 1:
            # Found a path - convert to (head, relation, tail) format
            formatted_path = []
            for i in range(len(path) - 1):
                h = path[i][0]
                r = path[i + 1][1]
                t = path[i + 1][0]
                formatted_path.append((h, r, t))
            paths.append(formatted_path)
            continue

        if current in adj:
            for relation, neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, relation, current)]))

    return paths


def extract_kg_paths_fast(
    drug_entity: str,
    disease_entity: str,
    triplets,
    max_paths: int = 10,
) -> list[dict]:
    """
    Fast path extraction: find shared intermediaries between drug and disease.

    Instead of full BFS (slow on 5.9M triplets), finds:
    1. Direct drug->disease connections
    2. Drug->Gene<->Disease paths (shared gene targets)
    3. Drug->Gene->Gene->Disease paths (2-hop via PPI)

    Returns list of path dicts with human-readable descriptions.
    """
    paths = []

    # 1. Direct connections
    direct = triplets[
        (triplets["head"] == drug_entity) &
        (triplets["tail"] == disease_entity)
    ]
    for _, row in direct.iterrows():
        paths.append({
            "type": "direct",
            "hops": 1,
            "chain": [drug_entity, row["relation"], disease_entity],
            "relation": row["relation"],
        })

    # 2. Shared gene targets (drug->gene AND disease->gene)
    drug_genes = set(
        triplets[
            (triplets["head"] == drug_entity) &
            (triplets["tail"].str.startswith("Gene::"))
        ]["tail"].unique()
    )
    # Also reverse
    drug_genes |= set(
        triplets[
            (triplets["tail"] == drug_entity) &
            (triplets["head"].str.startswith("Gene::"))
        ]["head"].unique()
    )

    disease_genes = set(
        triplets[
            (triplets["head"] == disease_entity) &
            (triplets["tail"].str.startswith("Gene::"))
        ]["tail"].unique()
    ) | set(
        triplets[
            (triplets["tail"] == disease_entity) &
            (triplets["head"].str.startswith("Gene::"))
        ]["head"].unique()
    )

    shared_genes = drug_genes & disease_genes
    for gene in list(shared_genes)[:10]:
        # Get the specific relations
        drug_rel = triplets[
            ((triplets["head"] == drug_entity) & (triplets["tail"] == gene)) |
            ((triplets["tail"] == drug_entity) & (triplets["head"] == gene))
        ]["relation"].values
        disease_rel = triplets[
            ((triplets["head"] == disease_entity) & (triplets["tail"] == gene)) |
            ((triplets["tail"] == disease_entity) & (triplets["head"] == gene))
        ]["relation"].values

        paths.append({
            "type": "shared_target",
            "hops": 2,
            "chain": [drug_entity, drug_rel[0] if len(drug_rel) > 0 else "targets",
                      gene,
                      disease_rel[0] if len(disease_rel) > 0 else "associated_with",
                      disease_entity],
            "gene": gene,
        })

    # 3. Drug->Gene->Gene->Disease (2-hop via gene-gene interactions)
    if len(paths) < 5 and drug_genes:
        for drug_gene in list(drug_genes)[:5]:
            # Find genes connected to this drug's target
            neighbor_genes = set(
                triplets[
                    (triplets["head"] == drug_gene) &
                    (triplets["tail"].str.startswith("Gene::"))
                ]["tail"].unique()
            )
            # Check if any connect to disease
            bridge_genes = neighbor_genes & disease_genes
            for bg in list(bridge_genes)[:3]:
                paths.append({
                    "type": "gene_bridge",
                    "hops": 3,
                    "chain": [drug_entity, "targets", drug_gene,
                              "interacts_with", bg,
                              "associated_with", disease_entity],
                    "drug_gene": drug_gene,
                    "bridge_gene": bg,
                })

    return paths[:max_paths]


def format_paths_for_llm(paths: list[dict], drug_names: dict) -> str:
    """
    Convert KG paths to human-readable text for LLM consumption.

    Args:
        paths: List of path dicts from extract_kg_paths_fast
        drug_names: Entity -> name mapping

    Returns:
        Formatted text describing the mechanistic connections
    """
    # Load gene symbol mapping for readable names
    try:
        from opencure.scoring.mendelian_randomization import _load_entrez_to_symbol
        entrez_map = _load_entrez_to_symbol()
    except Exception:
        entrez_map = {}

    def entity_name(entity: str) -> str:
        """Convert DRKG entity to readable name."""
        if entity in drug_names:
            return drug_names[entity]
        if "::" in entity:
            parts = entity.split("::")
            entity_type = parts[0]
            entity_id = parts[1] if len(parts) > 1 else ""
            if entity_type == "Gene" and entity_id in entrez_map:
                return entrez_map[entity_id]
            return f"{entity_type} {entity_id}"
        return entity

    def relation_name(relation: str) -> str:
        """Convert DRKG relation to readable name."""
        rel_map = {
            "GNBR::T::Compound:Disease": "treats",
            "Hetionet::CtD::Compound:Disease": "treats",
            "GNBR::C::Compound:Disease": "inhibits/treats",
            "GNBR::Pa::Compound:Disease": "palliates",
            "GNBR::E+::Compound:Gene": "increases expression of",
            "GNBR::E-::Compound:Gene": "decreases expression of",
            "GNBR::A::Compound:Gene": "activates",
            "GNBR::N::Compound:Gene": "inhibits",
            "GNBR::B::Compound:Gene": "binds to",
            "Hetionet::CbG::Compound:Gene": "binds to",
            "Hetionet::CdG::Compound:Gene": "downregulates",
            "Hetionet::CuG::Compound:Gene": "upregulates",
            "Hetionet::DaG::Disease:Gene": "associated with",
            "Hetionet::DdG::Disease:Gene": "downregulates",
            "Hetionet::DuG::Disease:Gene": "upregulates",
            "Hetionet::GiG::Gene:Gene": "interacts with",
            "Hetionet::GcG::Gene:Gene": "co-expressed with",
        }
        for key, val in rel_map.items():
            if key in relation:
                return val
        # Extract readable part from relation string
        if "::" in relation:
            parts = relation.split("::")
            return parts[1] if len(parts) > 1 else relation
        return relation

    lines = []
    for i, path in enumerate(paths, 1):
        if path["type"] == "direct":
            drug = entity_name(path["chain"][0])
            rel = relation_name(path["chain"][1])
            disease = entity_name(path["chain"][2])
            lines.append(f"{i}. DIRECT: {drug} --[{rel}]--> {disease}")

        elif path["type"] == "shared_target":
            drug = entity_name(path["chain"][0])
            rel1 = relation_name(path["chain"][1])
            gene = entity_name(path["chain"][2])
            rel2 = relation_name(path["chain"][3])
            disease = entity_name(path["chain"][4])
            lines.append(
                f"{i}. SHARED TARGET: {drug} --[{rel1}]--> {gene} "
                f"--[{rel2}]--> {disease}"
            )

        elif path["type"] == "gene_bridge":
            drug = entity_name(path["chain"][0])
            gene1 = entity_name(path["chain"][2])
            gene2 = entity_name(path["chain"][4])
            disease = entity_name(path["chain"][6])
            lines.append(
                f"{i}. PATHWAY: {drug} --[targets]--> {gene1} "
                f"--[interacts with]--> {gene2} --[associated with]--> {disease}"
            )

    return "\n".join(lines) if lines else "No direct knowledge graph paths found."


def generate_mechanistic_explanation(
    drug_name: str,
    disease_name: str,
    kg_paths_text: str,
    evidence_summary: str = "",
    mr_evidence: str = "",
) -> dict:
    """
    Generate a mechanistic explanation using Claude API (GraphRAG).

    Args:
        drug_name: Human-readable drug name
        disease_name: Human-readable disease name
        kg_paths_text: Formatted KG paths from format_paths_for_llm
        evidence_summary: Additional evidence context (PubMed, trials, etc.)
        mr_evidence: Mendelian randomization evidence string

    Returns:
        dict with: hypothesis, mechanism, confidence, validation_experiments
    """
    if not ANTHROPIC_API_KEY:
        # Fallback: generate a basic hypothesis from paths
        return _generate_fallback_hypothesis(drug_name, disease_name, kg_paths_text)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""You are a pharmacology expert analyzing a computational drug repurposing prediction.

Drug: {drug_name}
Disease: {disease_name}

Knowledge Graph Paths (from the Drug Repurposing Knowledge Graph):
{kg_paths_text}

{f'Additional Evidence: {evidence_summary}' if evidence_summary else ''}
{f'Genetic/MR Evidence: {mr_evidence}' if mr_evidence else ''}

Based on these knowledge graph connections and evidence, provide:

1. MECHANISM: A specific mechanistic hypothesis (2-3 sentences) for how {drug_name} could treat {disease_name}. Be precise about molecular pathways.

2. CONFIDENCE: Rate your confidence in this hypothesis as HIGH/MEDIUM/LOW with one sentence explaining why.

3. VALIDATION: List 2-3 specific experiments that would validate or refute this hypothesis.

Be concise and scientifically rigorous. Focus on the molecular mechanism."""

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Parse structured response
        result = {
            "hypothesis": text,
            "mechanism": "",
            "confidence": "MEDIUM",
            "validation_experiments": [],
            "source": "claude-3-haiku",
        }

        # Try to extract sections
        if "MECHANISM:" in text:
            parts = text.split("MECHANISM:")
            if len(parts) > 1:
                mech = parts[1].split("CONFIDENCE:")[0] if "CONFIDENCE:" in parts[1] else parts[1]
                result["mechanism"] = mech.strip()[:500]

        if "CONFIDENCE:" in text:
            parts = text.split("CONFIDENCE:")
            if len(parts) > 1:
                conf_text = parts[1].split("VALIDATION:")[0] if "VALIDATION:" in parts[1] else parts[1]
                conf_text = conf_text.strip()
                if "HIGH" in conf_text.upper()[:20]:
                    result["confidence"] = "HIGH"
                elif "LOW" in conf_text.upper()[:20]:
                    result["confidence"] = "LOW"

        if "VALIDATION:" in text:
            parts = text.split("VALIDATION:")
            if len(parts) > 1:
                val_text = parts[1].strip()
                # Extract numbered items
                experiments = []
                for line in val_text.split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        experiments.append(line.lstrip("0123456789.-) "))
                result["validation_experiments"] = experiments[:3]

        return result

    except Exception as e:
        return _generate_fallback_hypothesis(drug_name, disease_name, kg_paths_text)


def _generate_fallback_hypothesis(
    drug_name: str, disease_name: str, kg_paths_text: str
) -> dict:
    """Generate a basic hypothesis from KG paths without LLM."""
    if "SHARED TARGET" in kg_paths_text:
        # Extract gene names from paths
        lines = kg_paths_text.split("\n")
        genes = []
        for line in lines:
            if "SHARED TARGET" in line:
                # Parse "Drug --[binds to]--> Gene 1234 --[associated with]--> Disease"
                parts = line.split("-->")
                if len(parts) >= 2:
                    gene_part = parts[1].strip().rstrip(" -")
                    genes.append(gene_part)

        gene_str = ", ".join(genes[:3])
        hypothesis = (
            f"{drug_name} may treat {disease_name} through shared molecular targets "
            f"({gene_str}). The drug modulates these targets, which are also "
            f"genetically or functionally linked to {disease_name} pathology."
        )
    elif "DIRECT" in kg_paths_text:
        hypothesis = (
            f"A direct treatment relationship between {drug_name} and {disease_name} "
            f"exists in the biomedical knowledge graph, supported by published evidence."
        )
    elif "PATHWAY" in kg_paths_text:
        hypothesis = (
            f"{drug_name} may indirectly affect {disease_name} through a multi-step "
            f"pathway involving protein-protein interactions between drug targets "
            f"and disease-associated genes."
        )
    else:
        hypothesis = (
            f"Computational analysis suggests {drug_name} as a potential candidate "
            f"for {disease_name} based on multi-pillar AI scoring, but the specific "
            f"mechanism requires further investigation."
        )

    return {
        "hypothesis": hypothesis,
        "mechanism": hypothesis,
        "confidence": "LOW",
        "validation_experiments": [
            f"In vitro assay testing {drug_name} effect on {disease_name}-related cell models",
            f"Gene expression profiling after {drug_name} treatment in relevant cell types",
            f"Literature review of {drug_name}'s known molecular targets and their role in {disease_name}",
        ],
        "source": "knowledge_graph_paths",
    }


def explain_prediction(
    drug_entity: str,
    disease_entity: str,
    disease_name: str,
    triplets,
    drug_names: dict,
    evidence_summary: str = "",
    mr_score: float = 0,
    mr_targets: int = 0,
) -> dict:
    """
    Main entry point: generate a full mechanistic explanation for a prediction.

    Combines KG path extraction with LLM reasoning.
    """
    drug_name = drug_names.get(drug_entity, drug_entity)

    # Extract KG paths
    paths = extract_kg_paths_fast(drug_entity, disease_entity, triplets)

    # Format for LLM
    paths_text = format_paths_for_llm(paths, drug_names)

    # MR evidence string
    mr_evidence = ""
    if mr_score > 0:
        mr_evidence = (
            f"Mendelian randomization analysis found {mr_targets} drug target gene(s) "
            f"with causal genetic evidence linking them to {disease_name} "
            f"(MR score: {mr_score:.2f}). This provides causal support beyond correlation."
        )

    # Generate explanation
    explanation = generate_mechanistic_explanation(
        drug_name, disease_name, paths_text, evidence_summary, mr_evidence
    )

    explanation["kg_paths"] = paths
    explanation["kg_paths_text"] = paths_text
    explanation["drug_name"] = drug_name
    explanation["disease_name"] = disease_name

    return explanation
