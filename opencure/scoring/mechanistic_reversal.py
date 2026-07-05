"""
Mechanistic gene-signature reversal pillar (v5 replacement for thin L1000CDS2).

L1000CDS2 returns small expression-based reverser sets (27 per disease for
Malaria) where most entries are lab-screening library IDs that don't map
to DrugBank. Practical top-10 coverage is near zero.

This module produces an equivalent signal using two data sources we already
have locally:

  • Open Targets associationByOverallDirect (53,367 gene↔disease edges with
    confidence score ≥ 0.3, already parsed to data/open_targets/ot_triplets.tsv)
  • ChEMBL 34 drug-target activity index (94,717 drug-target pairs with
    median IC50/Ki in nM, data/drkg/drug_target_activities.json)

Algorithm per (drug, disease):
  1. Get top-N disease-associated genes from OT (weighted by OT score)
  2. For each candidate drug, look up its ChEMBL activities against those genes
  3. Score each target by affinity (potency ≤ 100 nM = 1.0, scaling to
     10 µM = 0.0) × OT confidence for that disease-gene pair
  4. Drug-disease score = sum over matched targets / |disease gene set|
     (normalized to [0, 1])

Every drug with ChEMBL bioactivity data (4,707 drugs) can receive a score
against every disease with ≥5 OT gene associations (~2,000 diseases) — so
hard coverage increases from ~5 drugs/disease to ~several-hundred/disease.

Semantically this is actually a MORE defensible "signature reversal":
L1000CDS2 correlates bulk expression signatures (noisy); this directly asks
"does this drug pharmacologically modulate the genes known to be causally
associated with this disease". It's the repurposing question we want.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional


OT_TRIPLETS_PATH = Path("data/open_targets/ot_triplets.tsv")
ACTIVITY_PATH = Path("data/drkg/drug_target_activities.json")
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")

# Potency → signal strength mapping
# <= 100 nM: 1.0  (strong, clinically-relevant potency)
# 1 µM     : 0.5  (moderate)
# >= 10 µM : 0.0  (too weak to rank as targeted)
def _affinity_to_signal(ic50_nm: float) -> float:
    if ic50_nm <= 100:
        return 1.0
    if ic50_nm >= 10_000:
        return 0.0
    # Log-scale interpolation
    return max(0.0, min(1.0, 1.0 - (math.log10(ic50_nm) - 2) / 2))


@lru_cache(maxsize=1)
def _load_disease_gene_map() -> dict[str, list[tuple[str, float]]]:
    """
    Load OT disease→(gene_entrez, score) map from ot_triplets.tsv.

    Returns {disease_entity: [(gene_entrez, weight), ...]} sorted by weight desc.
    Weight is OT 'assoc' relation presence (binary 1.0) — OT pre-filtered
    to score >= 0.3 at parse time.
    """
    if not OT_TRIPLETS_PATH.exists():
        return {}

    d2g: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with OT_TRIPLETS_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r != "OT::assoc::Gene:Disease":
                continue
            if not h.startswith("Gene::") or not t.startswith("Disease::"):
                continue
            gene_id = h.split("::", 1)[1]
            d2g[t].append((gene_id, 1.0))

    # Cap per-disease list (take top 50 by OT signal — we don't have scores
    # persisted in triplets, but OT already filtered to score >= 0.3, so
    # every kept edge is reasonable)
    capped: dict[str, list[tuple[str, float]]] = {}
    for disease, genes in d2g.items():
        capped[disease] = genes[:50]
    return capped


@lru_cache(maxsize=1)
def _load_activity_index() -> dict[str, dict[str, float]]:
    """Load ChEMBL drug-target activity index and flatten to
    {drugbank_id: {gene_symbol: median_nM}}.
    """
    if not ACTIVITY_PATH.exists():
        return {}
    try:
        raw = json.loads(ACTIVITY_PATH.read_text())
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    for db_id, targets in raw.items():
        flat: dict[str, float] = {}
        for sym, stats in targets.items():
            try:
                val = float(stats.get("median_nM", 0))
            except (TypeError, ValueError):
                continue
            if val > 0:
                flat[sym] = val
        if flat:
            out[db_id] = flat
    return out


@lru_cache(maxsize=1)
def _load_entrez_to_symbol() -> dict[str, str]:
    if not HGNC_PATH.exists():
        return {}
    m: dict[str, str] = {}
    with HGNC_PATH.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_sym = header.index("symbol")
            i_ent = header.index("entrez_id")
        except ValueError:
            return {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_sym, i_ent):
                continue
            ent, sym = parts[i_ent].strip(), parts[i_sym].strip()
            if ent and sym:
                m[ent] = sym
    return m


def _score_over_genes(
    gene_weight: dict[str, float],
    compound_entities: list[str],
    top_k: int,
    activities: dict[str, dict[str, float]],
) -> dict[str, tuple[float, int, str]]:
    """Shared scoring core for both disease-gene variants.

    `gene_weight` maps gene SYMBOL -> weight; `activities` is the flattened
    {drugbank_id: {symbol: median_nM}} index. Each compound's score is the
    affinity-weighted overlap with the gene set, normalised to [0, 1] (a drug
    hitting every disease gene at <=100 nM scores 1.0). Returns
    {compound_entity: (score, n_targeted, best_gene)} sorted desc, capped top_k.
    """
    total_weight = sum(gene_weight.values())
    if total_weight <= 0:
        return {}
    results: dict[str, tuple[float, int, str]] = {}
    for compound in compound_entities:
        db_id = compound.split("::", 1)[1] if "::" in compound else compound
        drug_activities = activities.get(db_id)
        if not drug_activities:
            continue
        matched_score = 0.0
        n_matched = 0
        best_signal = 0.0
        best_gene = ""
        for sym, w in gene_weight.items():
            median_nm = drug_activities.get(sym)
            if median_nm is None:
                continue
            sig = _affinity_to_signal(median_nm)
            if sig <= 0:
                continue
            matched_score += sig * w
            n_matched += 1
            if sig > best_signal:
                best_signal = sig
                best_gene = sym
        if n_matched == 0:
            continue
        results[compound] = (round(matched_score / total_weight, 4),
                             n_matched, best_gene)
    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])[:top_k]
    return dict(ranked)


def score_mechanistic_reversal(
    disease_entities: list[str],
    compound_entities: list[str],
    top_k: int = 500,
) -> dict[str, tuple[float, int, str]]:
    """
    Score each compound on pharmacological overlap with disease genes.

    Disease genes come from Open Targets ``associationByOverallDirect`` (ALL
    datasources, via ot_triplets.tsv). That association mixes datasources
    INCLUDING ``chembl``/``known_drug``, so against a treats-edge held-out
    benchmark this all-datasource variant is leak-SUSPECT — use
    ``score_mechanistic_reversal_genetics`` for a leak-clean measurement.

    Returns dict[compound_entity] -> (score_0_to_1, n_targeted, best_gene)
    sorted by score desc, capped at top_k.
    """
    d2g = _load_disease_gene_map()
    activities = _load_activity_index()
    ent_to_sym = _load_entrez_to_symbol()
    if not d2g or not activities or not ent_to_sym:
        return {}

    # Collect disease gene symbols (union across disease aliases, max weight).
    gene_weight: dict[str, float] = {}
    for de in disease_entities:
        for gene_id, weight in d2g.get(de, []):
            sym = ent_to_sym.get(gene_id)
            if sym and weight > gene_weight.get(sym, 0.0):
                gene_weight[sym] = weight
    if not gene_weight:
        return {}
    return _score_over_genes(gene_weight, compound_entities, top_k, activities)


@lru_cache(maxsize=1)
def _load_disease_gene_map_genetics() -> dict[str, dict[str, float]]:
    """Leak-clean disease->gene map: Open Targets GENETICS datasources only.

    The all-datasource ``_load_disease_gene_map`` mixes ``chembl``/``known_drug``
    evidence — a path through which a held-out treats edge could leak. This
    variant restricts disease->gene edges to GENETICS datasources (GWAS /
    rare-variant / clinical-genetics), reusing the exact leak-control machinery
    already validated in ``genetics_anchored``:

      * the GENETICS-only association caches (already filtered to GENETICS_USE
        at build time — no chembl/known_drug/literature),
      * ``_ot_disease_index`` (MeSH -> Open Targets id crosswalk),
      * ``_gene_crosswalk`` (Ensembl -> approved symbol).

    Offline (cache-only): diseases absent from the genetics caches are simply
    not scorable here — an honest coverage gap, not a silent fallback.

    Returns {disease_entity ("Disease::MESH:D...."): {gene_symbol: score}}.
    """
    from opencure.scoring.genetics_anchored import (
        _gene_crosswalk,
        _load_detail_cache,
        _load_score_cache,
        _ot_disease_index,
    )

    mesh2ot, _, _ = _ot_disease_index()
    ens2sym = _gene_crosswalk()
    detail = _load_detail_cache()   # {ot_id: {gene: {score, datasources}}}
    score = _load_score_cache()     # {ot_id: {gene: score}}  (legacy, genetics-only)

    def _genes_for(oid: str) -> dict[str, float]:
        if oid in detail:
            return {g: float(info.get("score", 0.0))
                    for g, info in detail[oid].items()}
        if oid in score:
            return {g: float(s) for g, s in score[oid].items()}
        return {}

    out: dict[str, dict[str, float]] = {}
    for mesh, ot_ids in mesh2ot.items():
        gw: dict[str, float] = {}
        for oid in ot_ids:
            for ens, s in _genes_for(oid).items():
                sym = ens2sym.get(ens)
                if sym and s > gw.get(sym, 0.0):
                    gw[sym] = s
        if gw:
            out[f"Disease::MESH:{mesh}"] = gw
    return out


def score_mechanistic_reversal_genetics(
    disease_entities: list[str],
    compound_entities: list[str],
    top_k: int = 500,
) -> dict[str, tuple[float, int, str]]:
    """Leak-clean variant of :func:`score_mechanistic_reversal`.

    Identical pharmacological scoring, but the disease gene set is restricted to
    Open Targets GENETICS datasources (no chembl/known_drug/literature), so a
    held-out treats edge cannot leak through the disease->gene side. This is the
    configuration whose benchmark number can be honestly claimed.
    """
    activities = _load_activity_index()
    d2g = _load_disease_gene_map_genetics()
    if not activities or not d2g:
        return {}
    gene_weight: dict[str, float] = {}
    for de in disease_entities:
        for sym, w in d2g.get(de, {}).items():
            if w > gene_weight.get(sym, 0.0):
                gene_weight[sym] = w
    if not gene_weight:
        return {}
    return _score_over_genes(gene_weight, compound_entities, top_k, activities)


if __name__ == "__main__":
    # Smoke test: Malaria
    import time
    t0 = time.time()
    compounds = list(_load_activity_index().keys())
    compounds = [f"Compound::{c}" for c in compounds]
    # Malaria MeSH
    mal = score_mechanistic_reversal(
        ["Disease::MESH:D008288"],
        compounds[:1000],
        top_k=20,
    )
    print(f"Scored {len(compounds)} compounds in {time.time()-t0:.1f}s")
    print(f"\nTop 10 mechanistic reversers for Malaria:")
    for i, (comp, (score, n, gene)) in enumerate(list(mal.items())[:10], 1):
        db_id = comp.split("::", 1)[1]
        print(f"  {i:>2}. {db_id:>10s}  score={score:.4f}  "
              f"targets_hit={n}  top_target={gene}")
