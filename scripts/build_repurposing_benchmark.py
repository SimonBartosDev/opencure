"""
Build an honest *retrospective repurposing* benchmark (WS1 decisive eval).

The existing time-sliced test set (data/eval/time_sliced_test.jsonl) is
"post-2020 first approvals" — mostly brand-new molecular entities. Repurposing
is the opposite: an OLD drug gaining a NEW indication. A method scored on
new-drug approvals is being graded on the wrong task.

This script keeps only the genuine repurposing events: a post-2020 pair
(drug X, disease B) where X already has a CURATED treats edge to some other
disease A != B in the 2020 DRKG. That guarantees X is an established drug, so
the post-2020 X->B edge is a real repurposing — exactly what OpenCure claims
to predict, and naturally leak-free (DRKG is 2020-vintage; the X->B edge is
not in it).

Output: data/eval/repurposing_benchmark.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

DRKG = Path("data/drkg/drkg.tsv")
TIME_SLICED = Path("data/eval/time_sliced_test.jsonl")
OUT = Path("data/eval/repurposing_benchmark.jsonl")

# Curated treats edges only — GNBR::T is text-mined co-occurrence (54k noisy
# edges) and would mark almost every drug as "treats something".
CURATED_TREATS = {
    "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease",
}


def main() -> None:
    # drug entity -> set of diseases it treats in the 2020 DRKG
    treats: dict[str, set[str]] = {}
    for line in DRKG.open():
        p = line.rstrip("\n").split("\t")
        if len(p) == 3 and p[1] in CURATED_TREATS:
            treats.setdefault(p[0], set()).add(p[2])
    print(f"Established drugs in 2020 DRKG: {len(treats)}")

    kept, dropped_new, dropped_same = [], 0, 0
    with TIME_SLICED.open() as f:
        rows = [json.loads(ln) for ln in f]
    print(f"Time-sliced post-2020 pairs: {len(rows)}")

    for d in rows:
        drug, disease = d["compound"], d["disease"]
        prior = treats.get(drug, set())
        if not prior:
            dropped_new += 1            # brand-new drug, not a repurposing
            continue
        other = prior - {disease}
        if not other:
            dropped_same += 1           # only treats this disease in DRKG
            continue
        d["prior_indications"] = len(other)
        kept.append(d)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for d in kept:
            f.write(json.dumps(d) + "\n")

    diseases = {d["disease"] for d in kept}
    print(f"\nDropped {dropped_new} brand-new drugs (no prior indication)")
    print(f"Dropped {dropped_same} with no OTHER indication")
    print(f"Kept {len(kept)} genuine repurposing events across "
          f"{len(diseases)} diseases")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
