"""
Build a PRE-2020 disease->gene genetics snapshot from Open Targets release 20.02
(February 2020), to harden the temporal holdout against posterior-leak.

WHY
---
The temporal conditional-lift test (scripts/popularity_residualized_lift.py on
data/eval/time_sliced_test.jsonl) grades 2020-2023 drug approvals using genetics
read from a ~2024-2026 Open Targets snapshot. A 2024 genetic association can
already encode the target that drove a 2020-2023 approval (genetics evidence
accrues partly from drug programs / their trials), so a temporal "win" is not
clean proof of foresight. Using a Feb-2020 snapshot — which provably predates
every test approval — removes that confound.

WHAT
----
OT 20.02 ships `20.02_association_data.json.gz`: one JSON record per
target-disease association with `association_score.datatypes.genetic_association`
— the aggregate human-genetics score (gwas_catalog, eva, gene2phenotype,
genomics_england, uniprot, phewas_catalog, postgap, ot_genetics_portal). It
EXCLUDES known_drug, literature, somatic-mutation and animal-model datatypes, so
it is leak-free genetics by construction, and dated Feb 2020.

This streams the gz, keeps only the EFO/MONDO disease ids needed by the holdout
sets, and writes {disease_id: {ensembl_gene: genetic_association_score}}.

OUTPUT: data/open_targets/genetics_pre2020_efo.json
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetics_anchored_benchmark_v3 as gab

GZ = Path("data/open_targets/archive_20.02/20.02_association_data.json.gz")
OUT = Path("data/open_targets/genetics_pre2020_efo.json")
HOLDOUTS = ["data/eval/holdout_test.jsonl", "data/eval/time_sliced_test.jsonl"]


def main() -> None:
    mesh2efo, _ = gab.build_mesh_to_efo()
    needed: set[str] = set()
    for hp in HOLDOUTS:
        for line in open(hp):
            mesh = json.loads(line)["disease"].split(":")[-1]
            for efo in mesh2efo.get(mesh, []):
                needed.add(efo)
    print(f"Needed EFO/MONDO disease ids across holdouts: {len(needed)}")

    gen: dict[str, dict[str, float]] = defaultdict(dict)
    n = 0
    with gzip.open(GZ, "rt") as fh:
        for line in fh:
            n += 1
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            did = (r.get("disease") or {}).get("id")
            if did not in needed:
                continue
            sc = ((r.get("association_score") or {})
                  .get("datatypes", {}).get("genetic_association", 0.0))
            if not sc or sc <= 0:
                continue
            tid = (r.get("target") or {}).get("id")
            if not tid:
                continue
            if sc > gen[did].get(tid, 0.0):
                gen[did][tid] = sc
            if n % 2_000_000 == 0:
                print(f"  scanned {n:,} records, {len(gen)} needed diseases hit")
    OUT.write_text(json.dumps(gen))
    print(f"Scanned {n:,} records. Diseases with pre-2020 genetics: {len(gen)} "
          f"/ {len(needed)} needed. Saved {OUT}")


if __name__ == "__main__":
    main()
