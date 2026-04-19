"""
Backfill ``shared_targets`` and ``shared_target_count`` on every candidate
using the unified disease-gene index (OT + DRKG/GNBR) plus DRKG's own
drug-target edges.

The inline compute in ``opencure/search.py::_get_graph_evidence`` only
consulted DRKG's *own* disease-gene edges, which are empty for most
neglected-tropical diseases (Schisto has 0 DRKG disease-gene edges).
With the unified index built by ``scripts/build_disease_gene_index.py``,
we can surface genuine drug-disease shared targets on the post-processing
path.

Output schema per candidate:
  shared_targets:       list[str]   # gene symbols, up to 20
  shared_target_count:  int

Usage
-----
    python3 scripts/backfill_shared_targets.py
    python3 scripts/backfill_shared_targets.py Malaria Tuberculosis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
DISEASE_GENE_INDEX = Path("data/disease_gene_index.json")
ACTIVITIES_PATH = Path("data/drkg/drug_target_activities.json")
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")
DRKG_PATH = Path("data/drkg/drkg.tsv")


_drug_targets_cache: dict[str, set[str]] | None = None
_entrez_to_symbol: dict[str, str] | None = None


def load_entrez_to_symbol() -> dict[str, str]:
    global _entrez_to_symbol
    if _entrez_to_symbol is None:
        _entrez_to_symbol = {}
        if HGNC_PATH.exists():
            import csv
            with HGNC_PATH.open() as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    ent = (row.get("entrez_id") or "").strip()
                    sym = (row.get("symbol") or "").strip()
                    if ent and sym:
                        _entrez_to_symbol[ent] = sym
    return _entrez_to_symbol


def build_drug_targets_from_drkg() -> dict[str, set[str]]:
    """Return {Compound::DBxxxxx: {gene_symbol, ...}} from DRKG + ChEMBL
    bioactivities. Union of both sources maximizes coverage."""
    import pandas as pd
    ent_to_sym = load_entrez_to_symbol()
    drug_targets: dict[str, set[str]] = {}

    # Source 1: DRKG compound→gene edges (any relation)
    if DRKG_PATH.exists():
        for chunk in pd.read_csv(
            DRKG_PATH, sep="\t", header=None,
            names=["h", "r", "t"], chunksize=500_000,
        ):
            m = (chunk["h"].str.startswith("Compound::")
                 & chunk["t"].str.startswith("Gene::"))
            for _, row in chunk.loc[m].iterrows():
                entrez = row["t"].split("::", 1)[1]
                sym = ent_to_sym.get(entrez)
                if sym:
                    drug_targets.setdefault(row["h"], set()).add(sym)

    # Source 2: ChEMBL drug_target_activities (measured potency)
    if ACTIVITIES_PATH.exists():
        acts = json.loads(ACTIVITIES_PATH.read_text())
        for drug_id, tgt_map in acts.items():
            drug_targets.setdefault(f"Compound::{drug_id}", set()).update(tgt_map.keys())

    return drug_targets


def load_drug_targets() -> dict[str, set[str]]:
    global _drug_targets_cache
    if _drug_targets_cache is None:
        print("  Building drug-target index (DRKG edges + ChEMBL bioactivities)…")
        _drug_targets_cache = build_drug_targets_from_drkg()
        total = sum(len(v) for v in _drug_targets_cache.values())
        print(f"    {len(_drug_targets_cache):,} drugs with "
              f"{total:,} drug-target pairs")
    return _drug_targets_cache


def _resolve_disease_entity(disease_name: str) -> str | None:
    try:
        from opencure.data.drkg import load_embeddings, find_disease_entities
        _, _, ent2id, _, _ = load_embeddings()
        matches = find_disease_entities(ent2id, disease_name)
        if matches:
            m = matches[0]
            return m[0] if isinstance(m, tuple) else m
    except Exception:
        pass
    return None


def backfill(path: Path, index: dict[str, list[str]],
             drug_targets: dict[str, set[str]]) -> int:
    data = json.loads(path.read_text())
    candidates = data.get("candidates", [])
    if not candidates:
        return 0
    disease_name = data.get("disease") or path.stem.replace("_", " ")
    disease_entity = data.get("disease_entity") or _resolve_disease_entity(disease_name)
    if not disease_entity:
        return 0
    disease_genes = set(index.get(disease_entity, []))
    if not disease_genes:
        return 0

    n_nonzero = 0
    for cand in candidates:
        drug_id = cand.get("drug_id", "")
        key = drug_id if drug_id.startswith("Compound::") else f"Compound::{drug_id}"
        dt = drug_targets.get(key, set())
        shared = sorted(dt & disease_genes)
        cand["shared_targets"] = shared[:20]
        cand["shared_target_count"] = len(shared)
        if shared:
            n_nonzero += 1
    path.write_text(json.dumps(data, indent=2))
    return n_nonzero


def main() -> None:
    if not DISEASE_GENE_INDEX.exists():
        sys.exit(
            f"Missing {DISEASE_GENE_INDEX}.\n"
            "  Run: python3 scripts/build_disease_gene_index.py"
        )
    index = json.loads(DISEASE_GENE_INDEX.read_text())
    drug_targets = load_drug_targets()

    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in {"screening_summary", "novel_candidates",
                                          "opencure_database"})

    total_nonzero = 0
    total_cands = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n = backfill(f, index, drug_targets)
        ncand = len(json.loads(f.read_text()).get("candidates", []))
        total_nonzero += n
        total_cands += ncand
        print(f"  {f.name}: {n}/{ncand} candidates now carry shared_targets")
    print(f"\nDone. {total_nonzero}/{total_cands} "
          f"({100*total_nonzero/max(total_cands,1):.0f}%) candidates with ≥1 shared target.")


if __name__ == "__main__":
    main()
