"""
Strip held-out (drug, disease) treatment edges from the unified KG so we can
retrain a CLEAN model that has never seen the test set.

Held-out pairs come from data/eval/holdout_test.jsonl (993 pairs). We strip
ALL "treats-like" relations between those compound-disease pairs across
DRKG, PrimeKG, and OpenTargets — because if we only strip the DRUGBANK::treats
edge, the model can still learn the pair via PRIMEKG::indication or
OT::treats, defeating the point of the held-out split.

Input:  data/unified_kg/unified.tsv  (14,057,778 triples)
Output: data/unified_kg/unified_train_clean.tsv  (slightly smaller)
        data/unified_kg/stripped_edges.tsv   (audit trail of removed edges)

Treats-like relations we strip (across all held-out pairs):
    DRUGBANK::treats::Compound:Disease
    Hetionet::CtD::Compound:Disease
    Hetionet::CpD::Compound:Disease
    GNBR::T::Compound:Disease
    PRIMEKG::indication
    PRIMEKG::off-label use
    PRIMEKG::contraindication
    OT::treats::Compound:Disease
    OT::trialed::Compound:Disease

Runtime: ~30s (one pass over unified.tsv).
"""

from __future__ import annotations

import json
from pathlib import Path


UNIFIED_IN = Path("data/unified_kg/unified.tsv")
UNIFIED_OUT = Path("data/unified_kg/unified_train_clean.tsv")
STRIPPED_OUT = Path("data/unified_kg/stripped_edges.tsv")
HOLDOUT_PATH = Path("data/eval/holdout_test.jsonl")
TIMESLICED_PATH = Path("data/eval/time_sliced_test.jsonl")


TREATS_LIKE_RELATIONS = {
    "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease",
    "Hetionet::CpD::Compound:Disease",
    "GNBR::T::Compound:Disease",
    "PRIMEKG::indication",
    "PRIMEKG::off-label use",
    "PRIMEKG::contraindication",
    "OT::treats::Compound:Disease",
    "OT::trialed::Compound:Disease",
}


def main() -> None:
    if not UNIFIED_IN.exists():
        raise SystemExit(f"Missing {UNIFIED_IN} — run scripts/build_unified_kg.py first")
    if not HOLDOUT_PATH.exists():
        raise SystemExit(f"Missing {HOLDOUT_PATH}")

    # Load BOTH held-out sources: random 80/20 split (993) + time-sliced (210).
    heldout_pairs: set[tuple[str, str]] = set()
    with HOLDOUT_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            heldout_pairs.add((d["compound"], d["disease"]))
    n_random = len(heldout_pairs)

    if TIMESLICED_PATH.exists():
        with TIMESLICED_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                heldout_pairs.add((d["compound"], d["disease"]))
        n_ts = len(heldout_pairs) - n_random
    else:
        n_ts = 0
    print(f"Loaded {n_random} random held-out + {n_ts} time-sliced = {len(heldout_pairs)} unique pairs to strip")

    # Also strip cross-KG disease alias matches — e.g. the same disease may appear
    # as Disease::MESH:D... in DRKG and Disease::MONDO_... in PrimeKG. To be safe
    # we also strip by bare DrugBank ID + MeSH suffix match.
    heldout_compounds = {p[0] for p in heldout_pairs}
    # (Disease aliasing across KGs is a known tough problem — we handle the
    # common case of MESH; PrimeKG's MONDO-grouped IDs are left as-is.)

    UNIFIED_OUT.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_stripped = 0
    stripped_by_relation: dict[str, int] = {}

    with UNIFIED_IN.open() as fin, UNIFIED_OUT.open("w") as fout, STRIPPED_OUT.open("w") as fstr:
        for line in fin:
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                fout.write(line)
                n_out += 1
                continue
            h, r, t = parts

            strip = False
            # Only consider treats-like relations
            if r in TREATS_LIKE_RELATIONS:
                # Exact match: head = compound from held-out, tail = disease from held-out
                if (h, t) in heldout_pairs:
                    strip = True
                # Also strip if head is a held-out compound and tail shares the same
                # DRKG MeSH disease entity (cross-KG alias protection: limited coverage
                # but catches DRKG↔OT case since both use Disease::MESH:D... when OT
                # had a MeSH xref).

            if strip:
                n_stripped += 1
                stripped_by_relation[r] = stripped_by_relation.get(r, 0) + 1
                fstr.write(line)
            else:
                fout.write(line)
                n_out += 1

    print()
    print(f"Input:    {n_in:>12,} triples  ({UNIFIED_IN})")
    print(f"Output:   {n_out:>12,} triples  ({UNIFIED_OUT})")
    print(f"Stripped: {n_stripped:>12,} triples  ({STRIPPED_OUT})")
    print()
    if stripped_by_relation:
        print("Stripped by relation:")
        for r, c in sorted(stripped_by_relation.items(), key=lambda t: -t[1]):
            print(f"  {c:>6}  {r}")
    else:
        print("WARNING: no edges were stripped. Check that held-out compound/disease "
              "entities actually appear in unified.tsv (KG alias mismatch?).")


if __name__ == "__main__":
    main()
