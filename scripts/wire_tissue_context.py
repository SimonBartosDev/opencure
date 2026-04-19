"""
Post-processor: populate ``tissue_context`` on every candidate in every v5
result JSON. Pre-fix results carried ``tissue_context: {}`` because the
pillar was only invoked when drug-disease shared targets were non-empty —
which happens on a minority of candidates. This backfill runs the same
GTEx/DISEASE_TISSUE_MAP scoring the live pipeline now runs, using
disease-associated genes when shared targets are absent.

Usage
-----
    python3 scripts/wire_tissue_context.py                # all result JSONs
    python3 scripts/wire_tissue_context.py Malaria
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.scoring.tissue_context import score_tissue_context


RESULTS_DIR = Path("experiments/results")
DISEASE_GENE_INDEX_PATH = Path("data/disease_gene_index.json")
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")


_DISEASE_GENE_INDEX: dict[str, list[str]] | None = None
_SYMBOL_TO_ENTREZ: dict[str, str] | None = None


def _load_index() -> dict[str, list[str]]:
    global _DISEASE_GENE_INDEX
    if _DISEASE_GENE_INDEX is None:
        if DISEASE_GENE_INDEX_PATH.exists():
            _DISEASE_GENE_INDEX = json.loads(DISEASE_GENE_INDEX_PATH.read_text())
        else:
            _DISEASE_GENE_INDEX = {}
    return _DISEASE_GENE_INDEX


def _load_symbol_to_entrez() -> dict[str, str]:
    global _SYMBOL_TO_ENTREZ
    if _SYMBOL_TO_ENTREZ is None:
        _SYMBOL_TO_ENTREZ = {}
        if HGNC_PATH.exists():
            import csv
            with HGNC_PATH.open() as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    ent = (row.get("entrez_id") or "").strip()
                    sym = (row.get("symbol") or "").strip()
                    if ent and sym:
                        _SYMBOL_TO_ENTREZ[sym] = ent
    return _SYMBOL_TO_ENTREZ


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


def _disease_genes(disease_name: str) -> set[str]:
    """Return ``{"Gene::<entrez>", ...}`` for a disease.

    Reads the unified index (OT + DRKG/GNBR) built by
    ``scripts/build_disease_gene_index.py`` and converts HGNC symbols to
    Entrez IDs for compatibility with ``score_tissue_context``.
    """
    entity = _resolve_disease_entity(disease_name)
    if not entity:
        return set()
    sym_to_ent = _load_symbol_to_entrez()
    return {
        f"Gene::{sym_to_ent[sym]}"
        for sym in _load_index().get(entity, [])
        if sym in sym_to_ent
    }


def backfill(path: Path) -> int:
    data = json.load(path.open())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    if not candidates:
        return 0
    disease_name = data.get("disease") or path.stem.replace("_", " ")

    # Disease-level gene set (cached once per file)
    disease_gene_set = _disease_genes(disease_name)

    # score_tissue_context expects ``Gene::<entrez>`` entries. shared_targets
    # on candidates are HGNC symbols — convert via the HGNC map so the
    # matrix lookup actually hits. Fall back to the disease-wide gene set
    # (already in Entrez form) when a candidate has no shared targets.
    sym_to_ent = _load_symbol_to_entrez()

    n_populated = 0
    for cand in candidates:
        shared = cand.get("shared_targets") or []
        if shared:
            gene_set = {f"Gene::{sym_to_ent[s]}" for s in shared if s in sym_to_ent}
        else:
            gene_set = disease_gene_set
        ctx = score_tissue_context(disease_name, gene_set)
        cand["tissue_context"] = ctx
        if ctx.get("tissues"):
            n_populated += 1

    json.dump(data, path.open("w"), indent=2)
    return n_populated


def main() -> None:
    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in {"screening_summary", "novel_candidates",
                                          "opencure_database"})
    total = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n = backfill(f)
        print(f"  {f.name}: {n} tissue_context populated")
        total += n
    print(f"\nDone. {total} candidates now carry tissue_context.")


if __name__ == "__main__":
    main()
