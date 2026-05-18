"""
Mechanism explanation via graph paths.

For a (drug, disease) candidate, find short paths through DRKG and render
them as natural-language mechanistic hypotheses. This is the explainability
layer that turns "combined_score = 0.52" into

    Clarithromycin → INHIBITS → 50S ribosomal protein L14
                   → PRESENT_IN → P. falciparum apicoplast
                   → ESSENTIAL_FOR → Malaria

Design:
  - One-time precompute (indexed) of the full DRKG into a directed graph.
  - For each candidate, do a bounded bidirectional BFS (cutoff 3 hops).
  - Rank paths by (a) path length (shorter = better), (b) relation
    specificity (INHIBITS beats ASSOCIATION), (c) intermediate-node
    non-hubness (avoid paths through generic housekeeping genes).
  - Render via a relation → English verb map.

Storage: paths are computed on demand because pre-computing all-pairs is
expensive. Individual lookups are ~10-50 ms on a pre-built NetworkX graph.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


DRKG_PATH = Path("data/drkg/drkg.tsv")
ADJ_CACHE_PATH = Path("data/drkg/path_adjacency.pkl")
_ADJ: Optional[dict] = None  # entity -> list[(neighbor, rel)]

# Only load edges of these relation families (drug-target, target-disease, target-
# target, drug-disease) — the minimum needed to form drug→...→disease paths.
_KEEP_RELATION_PREFIXES = (
    "DRUGBANK::target",
    "DRUGBANK::treats",
    "DRUGBANK::enzyme",
    "DGIDB::",
    "Hetionet::CbG",
    "Hetionet::CuG",
    "Hetionet::CdG",
    "Hetionet::CtD",
    "Hetionet::CpD",
    "Hetionet::DaG",
    "Hetionet::DuG",
    "Hetionet::DdG",
    "Hetionet::GiG",
    "Hetionet::Gr>G",
    "STRING::BINDING",
    "STRING::INHIBITION",
    "STRING::ACTIVATION",
    "STRING::CATALYSIS",
    "bioarx::DrugHumGen",
    "bioarx::HumGenHumGen",
    "GNBR::T::",
    "GNBR::L::",
    "GNBR::J::",
    "GNBR::E::",
    "GNBR::N::",
    "GNBR::Mp::",
    "GNBR::A+::",
    "GNBR::A-::",
)


# Relation → English phrasing. Missing relations fall back to "is associated with".
_REL_PHRASES: dict[str, str] = {
    # DrugBank
    "DRUGBANK::treats::Compound:Disease": "treats",
    "DRUGBANK::target::Compound:Gene": "targets",
    "DRUGBANK::enzyme::Compound:Gene": "is metabolized by",
    "DRUGBANK::carrier::Compound:Gene": "is carried by",
    "DRUGBANK::transporter::Compound:Gene": "is transported by",
    "DRUGBANK::ddi-interactor-in::Compound:Compound": "interacts with",
    "DRUGBANK::x-atc::Compound:Atc": "is classified as",
    # DGIDB drug-target
    "DGIDB::INHIBITOR::Gene:Compound": "inhibits",
    "DGIDB::ACTIVATOR::Gene:Compound": "activates",
    "DGIDB::AGONIST::Gene:Compound": "is an agonist of",
    "DGIDB::ANTAGONIST::Gene:Compound": "antagonizes",
    "DGIDB::BLOCKER::Gene:Compound": "blocks",
    "DGIDB::BINDER::Gene:Compound": "binds",
    "DGIDB::CHANNEL BLOCKER::Gene:Compound": "blocks channel",
    "DGIDB::MODULATOR::Gene:Compound": "modulates",
    "DGIDB::ALLOSTERIC MODULATOR::Gene:Compound": "allosterically modulates",
    # Hetionet disease/gene
    "Hetionet::DaG::Disease:Gene": "is linked to",
    "Hetionet::DdG::Disease:Gene": "downregulates",
    "Hetionet::DuG::Disease:Gene": "upregulates",
    "Hetionet::CtD::Compound:Disease": "treats",
    "Hetionet::CpD::Compound:Disease": "palliates",
    "Hetionet::CrC::Compound:Compound": "resembles",
    "Hetionet::CbG::Compound:Gene": "binds",
    "Hetionet::CuG::Compound:Gene": "upregulates",
    "Hetionet::CdG::Compound:Gene": "downregulates",
    # STRING / interactions
    "STRING::BINDING::Gene:Gene": "binds",
    "STRING::INHIBITION::Gene:Gene": "inhibits",
    "STRING::ACTIVATION::Gene:Gene": "activates",
    "STRING::CATALYSIS::Gene:Gene": "catalyzes",
    "STRING::REACTION::Gene:Gene": "reacts with",
    "STRING::OTHER::Gene:Gene": "interacts with",
    # GNBR literature-mined
    "GNBR::T::Compound:Disease": "is a therapeutic for",
    "GNBR::L::Gene:Disease": "is linked to",
    "GNBR::J::Gene:Disease": "is associated with",
    "GNBR::E::Compound:Gene": "affects",
    "GNBR::Mp::Compound:Gene": "is a modulator of",
    "GNBR::N::Compound:Gene": "inhibits",
    "GNBR::A+::Compound:Gene": "agonizes",
    "GNBR::A-::Compound:Gene": "antagonizes",
}


# Generic/housekeeping entities that make paths uninformative
_HUB_GENES = {
    "Gene::7157",   # TP53
    "Gene::1956",   # EGFR
    "Gene::3845",   # KRAS
    "Gene::4893",   # NRAS
    "Gene::7124",   # TNF
    "Gene::3569",   # IL6
    "Gene::3458",   # IFNG
    "Gene::4790",   # NFKB1
    "Gene::5599",   # MAPK8
    "Gene::7422",   # VEGFA
}


# Relation specificity score (0=vague, 1=specific)
_REL_SPECIFICITY: dict[str, float] = {
    "DRUGBANK::treats::Compound:Disease": 1.0,
    "DRUGBANK::target::Compound:Gene": 0.9,
    "DGIDB::INHIBITOR::Gene:Compound": 0.9,
    "DGIDB::AGONIST::Gene:Compound": 0.9,
    "DGIDB::ANTAGONIST::Gene:Compound": 0.9,
    "DGIDB::BLOCKER::Gene:Compound": 0.85,
    "DGIDB::ACTIVATOR::Gene:Compound": 0.85,
    "Hetionet::CtD::Compound:Disease": 1.0,
    "Hetionet::CbG::Compound:Gene": 0.8,
    "Hetionet::DaG::Disease:Gene": 0.7,
    "Hetionet::DdG::Disease:Gene": 0.75,
    "Hetionet::DuG::Disease:Gene": 0.75,
    "STRING::BINDING::Gene:Gene": 0.6,
    "STRING::INHIBITION::Gene:Gene": 0.7,
    "STRING::ACTIVATION::Gene:Gene": 0.7,
    "STRING::OTHER::Gene:Gene": 0.3,
    "GNBR::T::Compound:Disease": 0.9,
    "GNBR::L::Gene:Disease": 0.6,
    "GNBR::J::Gene:Disease": 0.5,
}


def _load_adjacency() -> dict:
    """Build/load compact filtered adjacency list from DRKG.

    Returns dict[entity] -> list[(neighbor, relation)]. Undirected (both
    forward and reverse edges stored, reverse prefixed with '~').

    Cached to disk as pickle for fast subsequent loads (~3s vs 60s).
    """
    global _ADJ
    if _ADJ is not None:
        return _ADJ

    import pickle

    if ADJ_CACHE_PATH.exists():
        try:
            with ADJ_CACHE_PATH.open("rb") as f:
                _ADJ = pickle.load(f)
            return _ADJ
        except Exception:
            pass

    if not DRKG_PATH.exists():
        raise FileNotFoundError(f"{DRKG_PATH} not found")

    adj: dict[str, list[tuple[str, str]]] = {}
    n_edges = 0
    with DRKG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if not r.startswith(_KEEP_RELATION_PREFIXES):
                continue
            adj.setdefault(h, []).append((t, r))
            adj.setdefault(t, []).append((h, f"~{r}"))
            n_edges += 2

    # Prune nodes with > 5000 neighbors (mega-hubs add no signal and explode BFS)
    _MAX_NEIGHBORS = 5000
    for k in list(adj):
        if len(adj[k]) > _MAX_NEIGHBORS:
            adj[k] = adj[k][:_MAX_NEIGHBORS]

    ADJ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ADJ_CACHE_PATH.open("wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)

    _ADJ = adj
    return adj


def _bfs_paths(adj: dict, src: str, dst: str, cutoff: int = 3, max_paths: int = 200) -> list[list[str]]:
    """Bounded BFS, returns up to max_paths simple paths from src to dst with
    length <= cutoff."""
    if src not in adj or dst not in adj:
        return []
    results: list[list[str]] = []
    stack: list[tuple[list[str], set[str]]] = [([src], {src})]
    while stack and len(results) < max_paths:
        path, visited = stack.pop()
        if len(path) - 1 >= cutoff:
            continue
        node = path[-1]
        for nb, _ in adj.get(node, []):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb == dst:
                results.append(new_path)
                if len(results) >= max_paths:
                    break
            else:
                stack.append((new_path, visited | {nb}))
    return results


def _edge_relations(adj: dict, a: str, b: str) -> list[str]:
    """Return all relations on the a→b edge."""
    return [r for nb, r in adj.get(a, []) if nb == b]


def _short_name(entity: str) -> str:
    """Produce a compact display label for an entity."""
    if "::" not in entity:
        return entity
    kind, ident = entity.split("::", 1)
    # Prefix gene IDs so they read as "Gene 23406" rather than bare "23406"
    if kind == "Gene":
        return f"gene-{ident}"
    if kind == "Biological Process":
        return f"BP:{ident}"
    if kind == "Pathway":
        return f"PW:{ident}"
    if kind == "Molecular Function":
        return f"MF:{ident}"
    return ident


_REVERSE_PHRASES: dict[str, str] = {
    "treats": "is treated by",
    "targets": "is a target of",
    "inhibits": "is inhibited by",
    "activates": "is activated by",
    "blocks": "is blocked by",
    "binds": "is bound by",
    "upregulates": "is upregulated by",
    "downregulates": "is downregulated by",
    "antagonizes": "is antagonized by",
    "catalyzes": "is catalyzed by",
    "palliates": "is palliated by",
    "resembles": "resembles",
    "is linked to": "is linked to",
    "is associated with": "is associated with",
    "is a therapeutic for": "is treated by",
    "is classified as": "has member",
    "is metabolized by": "metabolizes",
    "is carried by": "carries",
    "is transported by": "transports",
    "interacts with": "interacts with",
    "modulates": "is modulated by",
    "allosterically modulates": "is allosterically modulated by",
    "is an agonist of": "is agonized by",
    "affects": "is affected by",
    "is a modulator of": "is modulated by",
    "agonizes": "is agonized by",
    "reacts with": "reacts with",
    "blocks channel": "channel blocked by",
}


def _phrase_for_rel(rel: str) -> str:
    if rel.startswith("~"):
        base = rel[1:]
        forward = _REL_PHRASES.get(base, "is related to")
        return _REVERSE_PHRASES.get(forward, f"is {forward} by")
    return _REL_PHRASES.get(rel, "is associated with")


def _path_score(path: list[str], adj: dict) -> float:
    """Higher = better path (short, specific relations, non-hub intermediates)."""
    if len(path) < 2:
        return 0.0
    length_penalty = 1.0 / len(path)
    spec_scores: list[float] = []
    for a, b in zip(path[:-1], path[1:]):
        rels = _edge_relations(adj, a, b)
        best = 0.0
        for r in rels:
            base = r[1:] if r.startswith("~") else r
            best = max(best, _REL_SPECIFICITY.get(base, 0.4))
        spec_scores.append(best)
    spec = sum(spec_scores) / len(spec_scores) if spec_scores else 0.4
    hub_penalty = 1.0
    for node in path[1:-1]:
        if node in _HUB_GENES:
            hub_penalty *= 0.7
    return length_penalty * spec * hub_penalty


def explain_path(
    drug_entity: str,
    disease_entity: str,
    max_paths: int = 3,
    cutoff: int = 3,
    entity_name_map: Optional[dict[str, str]] = None,
) -> list[dict]:
    """
    Find mechanistic paths drug → disease through DRKG.

    Args:
        drug_entity: e.g. "Compound::DB01211"
        disease_entity: e.g. "Disease::MESH:D008288"
        max_paths: max number of paths to return
        cutoff: max path length (number of edges)
        entity_name_map: entity → human name for readable output

    Returns: list of dicts with "path" (list of entities), "narration" (str),
             "score" (float), sorted by score descending. Empty if none found.
    """
    adj = _load_adjacency()
    if drug_entity not in adj or disease_entity not in adj:
        return []

    paths = _bfs_paths(adj, drug_entity, disease_entity, cutoff=cutoff, max_paths=300)
    if not paths:
        return []

    scored = [(p, _path_score(p, adj)) for p in paths]
    scored.sort(key=lambda t: t[1], reverse=True)

    out = []
    for path, score in scored[:max_paths]:
        narration = _render_narration(path, adj, entity_name_map or {})
        out.append({
            "path": path,
            "narration": narration,
            "score": round(score, 3),
            "length": len(path) - 1,
        })
    return out


def _render_narration(path: list[str], adj: dict, name_map: dict[str, str]) -> str:
    """Render a path as '<drug> → <phrase> → <intermediate> → … → <disease>'."""
    def label(entity: str) -> str:
        return name_map.get(entity) or _short_name(entity)

    segments = [label(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        rels = _edge_relations(adj, a, b)
        best_rel = ""
        best_spec = -1.0
        for rel in rels:
            base = rel[1:] if rel.startswith("~") else rel
            spec = _REL_SPECIFICITY.get(base, 0.4)
            if spec > best_spec:
                best_spec = spec
                best_rel = rel
        phrase = _phrase_for_rel(best_rel)
        segments.append(f"—[{phrase}]→ {label(b)}")
    return " ".join(segments)


@lru_cache(maxsize=1)
def graph_ready() -> bool:
    try:
        _load_adjacency()
        return True
    except Exception:
        return False
