#!/usr/bin/env python3
"""
Build a CLEAN, LEAK-FREE gene direction-of-effect table for OpenCure.

For a causal gene G we want the *disease-causing direction* of dosage/function:
  - LoF mechanism  (loss/reduced activity is harmful)  -> therapy must INCREASE G
                                                          -> want AGONIST/ACTIVATOR
                                                             (ChEMBL POSITIVE MODULATOR)
  - GoF mechanism  (excess activity is harmful)         -> therapy must REDUCE G
                                                          -> want INHIBITOR/ANTAGONIST
                                                             (ChEMBL NEGATIVE MODULATOR)

SOURCES (genetics/constraint, NOT drug-derived -> leak-free):
  1. ClinGen Dosage Sensitivity (gene curation list TSV)
       Haploinsufficiency score 3 (sufficient evidence)  => LoF mechanism
       Triplosensitivity     score 3 (sufficient evidence) => GoF/dosage-excess mechanism
     (scores 0,1,2,30,40 are not sufficient evidence -> ignored)
  2. gnomAD v2.1.1 gene constraint
       pLI > 0.9 OR LOEUF < 0.35  => LoF-intolerant => losing function is harmful
                                   => LoF mechanism likely (lower-confidence prior)

CONFLICT RESOLUTION (conservative):
  ClinGen (when score==3) is highest confidence and wins.
  If ClinGen gives both HI==3 and TS==3 -> genuinely dosage-sensitive both ways -> unknown.
  gnomAD only ever votes LoF; it is a prior used when ClinGen is silent.
  If ClinGen says GoF and gnomAD prior says LoF -> ClinGen wins (it is direct dosage evidence).

OUTPUT: data/genetics/gene_direction.tsv
  gene_symbol  mechanism(LoF|GoF|unknown)  basis  confidence(high|medium|low)  detail
"""
from __future__ import annotations
import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GDIR = ROOT / "data" / "genetics"
CLINGEN = GDIR / "clingen_dosage.tsv"
GNOMAD = GDIR / "gnomad_constraint.txt.bgz"
OUT = GDIR / "gene_direction.tsv"

PLI_T = 0.9
LOEUF_T = 0.35


def parse_clingen():
    """gene -> {'hi': int|None, 'ts': int|None}. Only scores 0..3 are real evidence levels."""
    out = {}
    with open(CLINGEN) as fh:
        # skip comment header lines beginning with '#', the last '#' line is the header
        header = None
        rows = []
        for line in fh:
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            rows.append(line.rstrip("\n").split("\t"))
    gi = header.index("Gene Symbol")
    hi = header.index("Haploinsufficiency Score")
    ts = header.index("Triplosensitivity Score")
    for r in rows:
        if len(r) <= max(gi, hi, ts):
            continue
        sym = r[gi].strip()
        if not sym:
            continue

        def _score(v):
            v = v.strip()
            try:
                iv = int(v)
            except ValueError:
                return None
            return iv if iv in (0, 1, 2, 3) else None  # 30/40 = autosomal recessive / unlikely; not dosage

        out[sym] = {"hi": _score(r[hi]), "ts": _score(r[ts])}
    return out


def parse_gnomad():
    """gene -> {'pli': float|None, 'loeuf': float|None}."""
    out = {}
    with gzip.open(GNOMAD, "rt") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        gi = header.index("gene")
        pi = header.index("pLI")
        li = header.index("oe_lof_upper")  # LOEUF

        def _f(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        for r in rd:
            if len(r) <= max(gi, pi, li):
                continue
            sym = r[gi].strip()
            if not sym:
                continue
            pli = _f(r[pi])
            loeuf = _f(r[li])
            # keep the most LoF-intolerant transcript per gene symbol
            prev = out.get(sym)
            cur = {"pli": pli, "loeuf": loeuf}
            if prev is None:
                out[sym] = cur
            else:
                # prefer higher pli / lower loeuf
                if (pli or 0) > (prev["pli"] or 0):
                    out[sym] = cur
    return out


def resolve(sym, cg, gn):
    """Return (mechanism, basis, confidence, detail)."""
    hi = cg.get("hi") if cg else None
    ts = cg.get("ts") if cg else None
    cg_lof = hi == 3
    cg_gof = ts == 3
    if cg_lof and cg_gof:
        return "unknown", "clingen", "low", "HI=3 and TS=3 (dosage-sensitive both ways)"
    if cg_lof:
        return "LoF", "clingen", "high", "ClinGen haploinsufficiency=3 (sufficient)"
    if cg_gof:
        return "GoF", "clingen", "high", "ClinGen triplosensitivity=3 (sufficient)"

    # gnomAD prior: only ever LoF-mechanism evidence
    pli = gn.get("pli") if gn else None
    loeuf = gn.get("loeuf") if gn else None
    lof_intol = (pli is not None and pli > PLI_T) or (loeuf is not None and loeuf < LOEUF_T)
    if lof_intol:
        return "LoF", "gnomad", "medium", f"LoF-intolerant (pLI={pli}, LOEUF={loeuf})"
    return "unknown", "none", "none", ""


def main():
    cg = parse_clingen()
    gn = parse_gnomad()
    print(f"ClinGen genes parsed : {len(cg)}")
    print(f"gnomAD genes parsed  : {len(gn)}")

    all_syms = set(cg) | set(gn)
    rows = []
    counts = {"LoF": 0, "GoF": 0, "unknown": 0}
    for sym in sorted(all_syms):
        mech, basis, conf, detail = resolve(sym, cg.get(sym), gn.get(sym))
        if mech == "unknown" and basis == "none":
            continue  # don't store no-information rows
        counts[mech] = counts.get(mech, 0) + 1
        rows.append((sym, mech, basis, conf, detail))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["gene_symbol", "mechanism", "basis", "confidence", "detail"])
        w.writerows(rows)

    print(f"\nwritten -> {OUT}  ({len(rows)} informative genes)")
    print(f"  LoF (want activator)  : {counts['LoF']}")
    print(f"  GoF (want inhibitor)  : {counts['GoF']}")
    print(f"  unknown (conflict)    : {counts['unknown']}")


if __name__ == "__main__":
    main()
