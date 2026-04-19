"""
Post-processor: annotate every candidate with a ``structural_alerts`` field
sourced from RDKit PAINS + Brenk + NIH catalogs.

Output on each candidate:
  structural_alerts: {
    has_pains:      bool,
    pains_families: [str, ...],   # sorted, e.g. ['BRENK', 'NIH']
    n_alerts:       int,
    alert_names:    [str, ...],   # up to 5 example pattern names
  }

Usage
-----
    python3 scripts/annotate_structural_alerts.py
    python3 scripts/annotate_structural_alerts.py Malaria Tuberculosis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opencure.filters.pains import annotate_structural_alerts


RESULTS_DIR = Path("experiments/results")
SMILES_PATH = Path("data/drkg/compound_smiles.tsv")


def load_smiles_map() -> dict[str, str]:
    """Return {DrugBank_id: SMILES} from the DRKG compound-SMILES cache.

    The TSV header is: drugbank_id\\tentity\\tsmiles (with entity always
    ``Compound::<drugbank_id>``). We index by bare DrugBank ID since
    candidates in result JSONs use that form.
    """
    m: dict[str, str] = {}
    if not SMILES_PATH.exists():
        return m
    with SMILES_PATH.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            drugbank_id, _entity, smiles = parts[0], parts[1], parts[2]
            if drugbank_id and smiles:
                m[drugbank_id] = smiles
    return m


def backfill(path: Path, smiles_map: dict[str, str]) -> tuple[int, int, int]:
    data = json.loads(path.read_text())
    candidates = data.get("candidates", [])
    if not candidates:
        return 0, 0, 0
    n_alerts_any = n_pains = 0
    for cand in candidates:
        drug_id = cand.get("drug_id", "")
        smiles = smiles_map.get(drug_id, "")
        ann = annotate_structural_alerts(smiles)
        cand["structural_alerts"] = ann
        if ann.get("n_alerts", 0):
            n_alerts_any += 1
        if ann.get("has_pains"):
            n_pains += 1
    path.write_text(json.dumps(data, indent=2))
    return len(candidates), n_alerts_any, n_pains


def main() -> None:
    smiles_map = load_smiles_map()
    if not smiles_map:
        sys.exit(f"Missing SMILES cache at {SMILES_PATH}")
    print(f"Loaded {len(smiles_map):,} SMILES records")

    if len(sys.argv) > 1:
        files = [RESULTS_DIR / f"{d}.json" for d in sys.argv[1:]]
    else:
        files = sorted(p for p in RESULTS_DIR.glob("*.json")
                       if p.stem not in {"screening_summary", "novel_candidates",
                                          "opencure_database"})
    total = any_alerts = pains_ct = 0
    for f in files:
        if not f.exists():
            print(f"  [skip] {f.name}")
            continue
        n, a, p = backfill(f, smiles_map)
        total += n
        any_alerts += a
        pains_ct += p
        print(f"  {f.name}: {a}/{n} flagged, {p} PAINS")
    print(f"\nDone. Any alert: {any_alerts}/{total} "
          f"({100*any_alerts/max(total,1):.0f}%); "
          f"PAINS-specific: {pains_ct}/{total} "
          f"({100*pains_ct/max(total,1):.0f}%)")


if __name__ == "__main__":
    main()
