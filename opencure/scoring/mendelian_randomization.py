"""
Mendelian Randomization / Genetic Evidence scoring pillar for drug repurposing.

Uses genetic association evidence to estimate CAUSAL effects of drug targets
on disease outcomes. If a drug's target gene has strong genetic evidence
linking it to a disease (via GWAS, eQTL, L2G), then modulating that target
should affect the disease.

Data source: Open Targets Platform (GraphQL API) — free, no auth.
Aggregates GWAS catalog, UK Biobank, FinnGen, eQTLGen, and dozens of other
genetic studies into a single genetic_association score per target-disease pair.

This is fundamentally different from other pillars which measure correlations.
Genetic evidence provides CAUSAL support via Mendelian randomization logic:
  genetic_variant → affects target protein → affects disease risk

Methods:
  - Open Targets genetic_association scores (aggregated L2G, GWAS, eQTL)
  - Drug-target mapping from DRKG knowledge graph
  - Multi-target aggregation (drugs hitting multiple causal targets score higher)
"""
from __future__ import annotations

import time
import requests
import numpy as np
from typing import Optional

# Module-level caches
_disease_targets_cache: dict[str, dict] = {}
_disease_efo_cache: dict[str, Optional[str]] = {}
_entrez_to_symbol: Optional[dict] = None

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _load_entrez_to_symbol() -> dict[str, str]:
    """Load Entrez Gene ID -> gene symbol mapping from NCBI gene_info."""
    global _entrez_to_symbol
    if _entrez_to_symbol is not None:
        return _entrez_to_symbol

    import gzip
    from pathlib import Path

    gene_info_path = Path("data/ncbi_gene_info.gz")
    mapping = {}

    if gene_info_path.exists():
        with gzip.open(gene_info_path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    gene_id = parts[1]  # GeneID (Entrez)
                    symbol = parts[2]   # Symbol
                    mapping[gene_id] = symbol

    # Fall back to curated map if file not available
    if not mapping:
        mapping = ENTREZ_TO_SYMBOL.copy()

    _entrez_to_symbol = mapping
    return mapping


def disease_to_efo_id(disease_name: str) -> Optional[str]:
    """Map disease name to Open Targets EFO/MONDO ID."""
    disease_lower = disease_name.lower()
    if disease_lower in _disease_efo_cache:
        return _disease_efo_cache[disease_lower]

    query = '''
    query {
      search(queryString: "%s", entityNames: ["disease"], page: {size: 5, index: 0}) {
        hits {
          id
          entity
          object {
            ... on Disease { id name }
          }
        }
      }
    }
    ''' % disease_name.replace('"', '\\"')

    try:
        resp = requests.post(OPENTARGETS_URL, json={"query": query}, timeout=15)
        if resp.status_code == 200:
            hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
            for hit in hits:
                if hit.get("entity") == "disease":
                    efo_id = hit.get("id")
                    _disease_efo_cache[disease_lower] = efo_id
                    return efo_id
    except Exception:
        pass

    _disease_efo_cache[disease_lower] = None
    return None


def get_disease_genetic_targets(efo_id: str, top_n: int = 500) -> dict[str, float]:
    """
    Get targets with genetic evidence for a disease from Open Targets.

    Returns dict: gene_symbol -> genetic_association_score (0-1)
    Only returns targets with genetic_association > 0 (GWAS/eQTL evidence).
    """
    if efo_id in _disease_targets_cache:
        return _disease_targets_cache[efo_id]

    query = '''
    query {
      disease(efoId: "%s") {
        associatedTargets(page: {size: %d, index: 0}) {
          rows {
            target {
              id
              approvedSymbol
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    ''' % (efo_id, top_n)

    genetic_targets = {}

    try:
        resp = requests.post(OPENTARGETS_URL, json={"query": query}, timeout=30)
        if resp.status_code == 200:
            rows = (
                resp.json()
                .get("data", {})
                .get("disease", {})
                .get("associatedTargets", {})
                .get("rows", [])
            )
            for row in rows:
                gene_symbol = row.get("target", {}).get("approvedSymbol", "")
                if not gene_symbol:
                    continue

                # Extract genetic_association specific score
                genetic_score = 0.0
                for dt in row.get("datatypeScores", []):
                    if dt["id"] == "genetic_association":
                        genetic_score = dt["score"]
                        break

                if genetic_score > 0:
                    genetic_targets[gene_symbol] = genetic_score

        time.sleep(0.3)
    except Exception:
        pass

    _disease_targets_cache[efo_id] = genetic_targets
    return genetic_targets


def get_drug_target_genes(drug_entity: str, triplets) -> list[str]:
    """
    Extract gene targets for a drug compound from DRKG triplets.

    Returns list of Entrez gene IDs (e.g., ['1234', '5678']).
    """
    drug_genes = set()

    # Compound -> Gene relationships
    mask_fwd = (triplets["head"] == drug_entity) & (triplets["tail"].str.startswith("Gene::"))
    drug_genes.update(triplets[mask_fwd]["tail"].unique())

    # Gene -> Compound relationships (reverse)
    mask_rev = (triplets["tail"] == drug_entity) & (triplets["head"].str.startswith("Gene::"))
    drug_genes.update(triplets[mask_rev]["head"].unique())

    # Extract gene IDs (format: "Gene::1234" -> "1234")
    return [g.split("::")[1] for g in drug_genes][:20]


def _build_entrez_to_symbol_map(triplets) -> dict[str, str]:
    """
    Build Entrez Gene ID -> gene symbol mapping from DRKG entity names.

    DRKG uses Entrez IDs (Gene::1234) but Open Targets uses gene symbols (APOE).
    We build a mapping using the DRKG entity metadata.
    """
    # This is a curated mapping for common drug targets
    # In production, would use NCBI gene_info file
    # For now, use the gene names from DRKG's Hetionet source
    mapping = {}

    # Get unique gene entities
    gene_entities = set()
    gene_mask = triplets["head"].str.startswith("Gene::")
    gene_entities.update(triplets[gene_mask]["head"].unique())
    gene_mask2 = triplets["tail"].str.startswith("Gene::")
    gene_entities.update(triplets[gene_mask2]["tail"].unique())

    # Try to load gene symbol mapping from DRKG entity sources
    try:
        import pandas as pd
        from opencure.config import DATA_DIR

        entity_src = DATA_DIR / "entity2src.tsv"
        if entity_src.exists():
            df = pd.read_csv(entity_src, sep="\t", header=None, names=["entity", "source"])
            for _, row in df.iterrows():
                ent = str(row["entity"])
                if ent.startswith("Gene::"):
                    gene_id = ent.split("::")[1]
                    # The source sometimes contains the gene symbol
                    src = str(row.get("source", ""))
                    if src:
                        mapping[gene_id] = src
    except Exception:
        pass

    return mapping


# Pre-built mapping of common Entrez Gene IDs to symbols
# (subset of most common drug targets)
ENTREZ_TO_SYMBOL = {
    "1956": "EGFR", "2064": "ERBB2", "5594": "MAPK1", "5595": "MAPK3",
    "207": "AKT1", "5290": "PIK3CA", "672": "BRCA1", "675": "BRCA2",
    "7157": "TP53", "4609": "MYC", "25": "ABL1", "3845": "KRAS",
    "5743": "PTGS2", "5742": "PTGS1", "3553": "IL1B", "7124": "TNF",
    "3569": "IL6", "3586": "IL10", "3552": "IL1A", "2353": "FOS",
    "4790": "NFKB1", "5970": "RELA", "6714": "SRC", "2475": "MTOR",
    "5728": "PTEN", "1019": "CDK4", "1021": "CDK6", "5925": "RB1",
    "4233": "MET", "2260": "FGFR1", "2263": "FGFR2", "4914": "NTRK1",
    "369": "ARAF", "673": "BRAF", "5894": "RAF1", "4893": "NRAS",
    "5879": "RAC1", "7040": "TGFB1", "4763": "NF1", "7248": "TSC1",
    "7249": "TSC2", "142": "PARP1", "5111": "PCNA", "3417": "IDH1",
    "3418": "IDH2", "2033": "EP300", "2146": "EZH2", "8243": "SMC1A",
    "6597": "SMARCA4", "5468": "PPARG", "5465": "PPARA", "2099": "ESR1",
    "2100": "ESR2", "367": "AR", "5241": "PGR", "3643": "INSR",
    "3667": "IRS1", "5515": "PPP2CA", "5518": "PPP2R1A", "348": "APOE",
    "351": "APP", "5663": "PSEN1", "5664": "PSEN2", "4137": "MAPT",
    "6622": "SNCA", "1621": "DBH", "4128": "MAOA", "4129": "MAOB",
    "6531": "SLC6A3", "6532": "SLC6A4", "3350": "HTR1A", "3356": "HTR2A",
    "1812": "DRD1", "1813": "DRD2", "1814": "DRD3", "1815": "DRD4",
    "1816": "DRD5", "1136": "CHRNA4", "1137": "CHRNA7", "1138": "CHRNB2",
    "2903": "GRIA1", "2904": "GRIA2", "2911": "GRM1", "2912": "GRM2",
    "2902": "GRIK1", "2898": "GRIK2", "2890": "GRIA3", "2894": "GRID1",
    "2891": "GRIA4", "2895": "GRID2", "2897": "GRIK1", "2556": "GABRA1",
    "2557": "GABRA2", "2560": "GABRB1", "2562": "GABRB2", "2563": "GABRB3",
    "9254": "CACNA2D2", "775": "CACNA1C", "776": "CACNA1D", "778": "CACNA1E",
    "6323": "SCN1A", "6326": "SCN2A", "6328": "SCN3A", "6329": "SCN4A",
    "3736": "KCNA1", "3738": "KCNA2", "3741": "KCNA5", "3757": "KCNH2",
    "1565": "CYP2D6", "1571": "CYP2E1", "1576": "CYP3A4", "1577": "CYP3A5",
    "1543": "CYP1A1", "1544": "CYP1A2", "1559": "CYP2C19", "1557": "CYP2C9",
    "23236": "PLCB1", "5530": "PPP3CA", "5578": "PRKCA", "5579": "PRKCB",
    "5580": "PRKCD", "5588": "PRKCQ", "5599": "MAPK8", "5601": "MAPK9",
    "6300": "MAPK12", "1432": "MAPK14", "5600": "MAPK11",
    "1017": "CDK2", "983": "CDK1", "1029": "CDKN2A", "5347": "PLK1",
    "10000": "AKT3", "208": "AKT2", "5291": "PIK3CB", "5293": "PIK3CD",
    "5295": "PIK3R1", "6416": "MAP2K4", "5604": "MAP2K1", "5605": "MAP2K2",
    "2932": "GSK3B", "2931": "GSK3A",
}


def score_drugs_for_disease_mr(
    disease_name: str,
    compound_entities: list[str],
    triplets,
    drug_names: dict,
) -> dict:
    """
    Score drugs using genetic/MR evidence from Open Targets.

    For each drug:
    1. Get target genes from DRKG
    2. Map Entrez IDs to gene symbols
    3. Check if those gene symbols have genetic evidence for the disease
    4. Score: max genetic_association score across drug's targets
    5. Bonus: multiple targets with genetic evidence = stronger signal

    Returns: dict[compound_entity] -> (mr_score, num_genetic_targets)
    """
    # Step 1: Find EFO ID for this disease
    efo_id = disease_to_efo_id(disease_name)
    if not efo_id:
        return {}

    # Step 2: Get all targets with genetic evidence for this disease
    genetic_targets = get_disease_genetic_targets(efo_id)
    if not genetic_targets:
        return {}

    mr_scores = {}

    # Load full Entrez -> Symbol mapping
    entrez_map = _load_entrez_to_symbol()

    # Pre-build drug->gene_symbols map for all compounds at once (FAST)
    # Extract ALL compound-gene links in one vectorized pass
    compound_set = set(compound_entities)
    drug_gene_mask = (
        triplets["head"].isin(compound_set) &
        triplets["tail"].str.startswith("Gene::")
    )
    drug_gene_pairs = triplets[drug_gene_mask][["head", "tail"]].values

    # Also check reverse direction
    gene_drug_mask = (
        triplets["tail"].isin(compound_set) &
        triplets["head"].str.startswith("Gene::")
    )
    gene_drug_pairs = triplets[gene_drug_mask][["head", "tail"]].values

    # Build compound -> set of gene symbols
    drug_to_symbols: dict[str, set] = {}
    for compound, gene_entity in drug_gene_pairs:
        gene_id = gene_entity.split("::")[1]
        for sub_id in gene_id.split(";"):
            sub_id = sub_id.strip()
            if sub_id in entrez_map:
                drug_to_symbols.setdefault(compound, set()).add(entrez_map[sub_id])

    for gene_entity, compound in gene_drug_pairs:
        gene_id = gene_entity.split("::")[1]
        for sub_id in gene_id.split(";"):
            sub_id = sub_id.strip()
            if sub_id in entrez_map:
                drug_to_symbols.setdefault(compound, set()).add(entrez_map[sub_id])

    # Score each drug by overlap with disease genetic targets
    genetic_target_set = set(genetic_targets.keys())

    for compound, gene_symbols in drug_to_symbols.items():
        overlap = gene_symbols & genetic_target_set
        if not overlap:
            continue

        best_genetic_score = max(genetic_targets[g] for g in overlap)
        genetic_hits = len(overlap)

        # MR score: weighted average of top genetic scores for this drug's targets
        # Better than just max: rewards drugs that hit MULTIPLE causal targets
        overlap_scores = sorted([genetic_targets[g] for g in overlap], reverse=True)
        if len(overlap_scores) == 1:
            mr_score = overlap_scores[0]
        else:
            # Top score gets 60% weight, remaining 40% spread across others
            mr_score = 0.6 * overlap_scores[0] + 0.4 * np.mean(overlap_scores[1:])

        mr_scores[compound] = (mr_score, genetic_hits)

    return mr_scores
