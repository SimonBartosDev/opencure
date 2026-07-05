#!/usr/bin/env python3
"""
Four-filter cross-indication query WITH a CLEAN, LEAK-FREE gene direction layer.

Difference vs cross_indication_query.py:
  Direction-of-effect comes from data/genetics/gene_direction.tsv, built from
  ClinGen dosage sensitivity + gnomAD constraint (pure genetics/constraint, NOT
  drug-derived). No Open Targets clinical_precedence is used -> no leak.

DIRECTION SEMANTICS
  gene mechanism LoF  -> losing G activity causes disease -> therapy must INCREASE G
                      -> required ChEMBL parent_type = POSITIVE MODULATOR (agonist/activator)
  gene mechanism GoF  -> excess G activity causes disease -> therapy must REDUCE G
                      -> required ChEMBL parent_type = NEGATIVE MODULATOR (inhibitor/antagonist)

A candidate is direction-CONCORDANT iff the gene mechanism is KNOWN and the drug's
ChEMBL action direction OPPOSES the disease-causing direction. Everything else is
labelled `direction_unknown` (the gene has no clean mechanism) and is NOT credible.
Drugs whose direction DISCORDS a known mechanism are dropped.

Reuses chembl helpers & filter logic from cross_indication_query.py.
"""
from __future__ import annotations
import csv
import json
import sqlite3
from pathlib import Path

import cross_indication_query as base

ROOT = Path(__file__).resolve().parents[1]
GENE_DIR_TSV = ROOT / "data" / "genetics" / "gene_direction.tsv"
OUT = ROOT / "experiments" / "eval" / "cross_indication_leads_directioned.json"

POS = base.POS  # "POSITIVE MODULATOR" (want this if gene LoF)
NEG = base.NEG  # "NEGATIVE MODULATOR" (want this if gene GoF)


def load_gene_direction():
    """gene_symbol -> {mechanism, basis, confidence, required_parent}."""
    out = {}
    for r in csv.DictReader(open(GENE_DIR_TSV), delimiter="\t"):
        mech = r["mechanism"]
        if mech == "LoF":
            req = POS
        elif mech == "GoF":
            req = NEG
        else:
            req = None
        out[r["gene_symbol"]] = {
            "mechanism": mech,
            "basis": r["basis"],
            "confidence": r["confidence"],
            "required_parent": req,
        }
    return out


def main():
    diseases = base.load_covered_diseases()
    gdir = load_gene_direction()
    con = sqlite3.connect(str(base.CHEMBL_DB))

    results = []
    total_candidates = 0
    n_concordant = 0
    n_unknown = 0
    diseases_with_concordant = 0

    for D in diseases:
        bname = D["disease"]
        b_ids = D["ot_disease_ids"]
        leads = []
        for gene, ginfo in D["genes"].items():
            gd = gdir.get(gene)
            req_parent = gd["required_parent"] if gd else None
            mech = gd["mechanism"] if gd else "unknown"
            basis = gd["basis"] if gd else "none"
            conf = gd["confidence"] if gd else "none"

            drugs = base.chembl_gene_drugs(con, gene)
            for drug in drugs:
                parents = drug["parents"] - {"OTHER"}
                if not parents:
                    continue  # no usable direction sense from the drug side
                if req_parent is None:
                    dir_status = "direction_unknown"
                else:
                    if req_parent in parents:
                        dir_status = "concordant"
                    else:
                        # drug opposes the wrong way relative to a KNOWN mechanism -> drop
                        continue
                inds = base.drug_indications(con, drug["molregno"])
                appr_ok, other_inds = base.approved_elsewhere(inds, b_ids, bname)
                if not appr_ok:
                    continue  # filter 3
                tried, _ = base.tried_for_B(inds, b_ids, bname)
                if tried:
                    continue  # filter 4
                leads.append({
                    "drug_name": drug["name"] or drug["chembl_id"],
                    "drug_chembl_id": drug["chembl_id"],
                    "causal_gene": gene,
                    "gene_ensembl": ginfo["ensembl"],
                    "genetics_score": round(ginfo["score"], 4),
                    "gene_mechanism": mech,
                    "mechanism_basis": basis,
                    "mechanism_confidence": conf,
                    "drug_action_types": sorted(drug["action_types"]),
                    "drug_direction_class": sorted(parents),
                    "required_direction": req_parent,
                    "direction_status": dir_status,
                    "approved_for": other_inds[:12],
                    "untried_for_B": True,
                })
        rank = {"concordant": 0, "direction_unknown": 1}
        leads.sort(key=lambda x: (rank[x["direction_status"]],
                                  0 if x["mechanism_confidence"] == "high" else 1,
                                  -x["genetics_score"]))
        conc = [l for l in leads if l["direction_status"] == "concordant"]
        if conc:
            diseases_with_concordant += 1
        n_concordant += len(conc)
        n_unknown += sum(1 for l in leads if l["direction_status"] == "direction_unknown")
        total_candidates += len(leads)
        results.append({
            "disease": bname,
            "ot_disease_ids": b_ids,
            "n_leads": len(leads),
            "n_concordant": len(conc),
            "leads": leads,
        })

    summary = {
        "n_covered_diseases": len(diseases),
        "direction_layer": "ClinGen dosage + gnomAD constraint (leak-free)",
        "total_candidates": total_candidates,
        "n_concordant_clean": n_concordant,
        "n_direction_unknown": n_unknown,
        "n_diseases_with_concordant": diseases_with_concordant,
    }
    out = {"summary": summary, "diseases": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print("=== CROSS-INDICATION FOUR-FILTER (CLEAN DIRECTION LAYER) ===")
    print(f"covered diseases scanned     : {summary['n_covered_diseases']}")
    print(f"total candidate leads        : {summary['total_candidates']}")
    print(f"  direction-CONCORDANT clean : {summary['n_concordant_clean']}")
    print(f"  direction-unknown          : {summary['n_direction_unknown']}")
    print(f"diseases w/ >=1 concordant   : {summary['n_diseases_with_concordant']}")
    print(f"written -> {OUT}")
    print("\n=== CLEAN DIRECTION-CONCORDANT LEADS ===")
    any_c = False
    for r in results:
        for l in r["leads"]:
            if l["direction_status"] != "concordant":
                continue
            any_c = True
            print(f"  [{l['mechanism_confidence']:6} {l['gene_mechanism']}] {r['disease']:26} <- "
                  f"{l['drug_name']} ({'/'.join(l['drug_action_types'])} of {l['causal_gene']}, "
                  f"basis={l['mechanism_basis']}, gscore={l['genetics_score']}) "
                  f"approved: {', '.join(l['approved_for'][:2])}")
    if not any_c:
        print("  (none -- all leads remain direction_unknown)")


if __name__ == "__main__":
    main()
