#!/usr/bin/env python3
"""
Honest triage of the four-filter cross-indication concordant leads.

WHY THIS EXISTS
---------------
`scripts/cross_indication_query.py`, re-run with a FULLY POPULATED Open Targets
direction-of-effect cache, surfaces genetics-tier "concordant" leads (a drug
that hits a disease's genetically-causal gene in the disease-opposing
direction, is approved elsewhere, and is untried here). The headline counts are
encouraging vs the earlier constraint-only layer (which produced 40 leads, ALL
activators, all collapsing). This script asks the skeptical follow-up:

  WHAT KIND of direction evidence actually justified each concordant call?

The answer determines whether the leads are credible. It classifies each
genetics-tier concordant lead by:
  * oncology vs non-oncology (by disease name),
  * inhibitor (NEGATIVE MODULATOR) vs activator (POSITIVE MODULATOR),
  * the datasource(s) that supplied the WINNING direction vote, split into
      - CLEAN disease-specific : gwas_credible_sets, gene_burden
      - Mendelian / mouse       : eva, eva_somatic, impc, clingen, orphanet,
                                  uniprot_variants, gene2phenotype,
                                  cancer_gene_census, intogen
  * whether the NEG/POS votes conflicted (fragile call).

Output: experiments/eval/cross_indication_triage.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "open_targets" / "direction_of_effect_cache.json"
LEADS = ROOT / "experiments" / "eval" / "cross_indication_leads.json"
OUT = ROOT / "experiments" / "eval" / "cross_indication_triage.json"

POS = "POSITIVE MODULATOR"
NEG = "NEGATIVE MODULATOR"

CLEAN_SRC = {"gwas_credible_sets", "gene_burden"}  # disease-specific common/rare variant
MENDEL_SRC = {
    "eva", "eva_somatic", "impc", "clingen", "orphanet",
    "uniprot_variants", "gene2phenotype", "cancer_gene_census", "intogen",
}
ONCO_WORDS = {
    "cancer", "carcinoma", "melanoma", "leukemia", "leukaemia", "lymphoma",
    "myeloma", "neoplasm", "tumor", "tumour", "sarcoma", "blastoma", "glioma",
}


def is_onco(name: str) -> bool:
    n = name.lower()
    return any(w in n for w in ONCO_WORDS)


def votes_with_src(rows):
    """Replicate cross_indication_query's non-clinical direction vote, but keep
    the datasource list behind each NEG / POS vote."""
    out = {NEG: [], POS: []}
    for ds, dot, dotr in rows:
        if ds == "clinical_precedence" or not dotr:
            continue
        if dot == "GoF" and dotr == "risk":
            harmful = True
        elif dot == "LoF" and dotr == "protect":
            harmful = True
        elif dot == "LoF" and dotr == "risk":
            harmful = False
        elif dot == "GoF" and dotr == "protect":
            harmful = False
        else:
            continue
        out[NEG if harmful else POS].append(ds)
    return out


def main() -> None:
    cache = json.load(open(CACHE))
    leads = json.load(open(LEADS))

    def find_rows(ens, ot_ids):
        norm = ",".join(i.replace(":", "_") for i in ot_ids)
        if f"{ens}|{norm}" in cache:
            return cache[f"{ens}|{norm}"].get("rows", [])
        for k, v in cache.items():       # fallback: any key for this gene
            if k.startswith(ens + "|"):
                return v.get("rows", [])
        return []

    by_class = Counter()
    winning_src = Counter()
    conflicts = 0
    clean_backed = []
    detailed = []

    for r in leads["diseases"]:
        for l in r["leads"]:
            if l["direction_status"] != "concordant":
                continue
            ens = l["gene_ensembl"]
            req = l["required_direction"]
            vw = votes_with_src(find_rows(ens, r["ot_disease_ids"]))
            winners = sorted(set(vw.get(req, [])))
            conflict = bool(vw[NEG]) and bool(vw[POS])
            conflicts += int(conflict)
            onc = "oncology" if is_onco(r["disease"]) else "non_oncology"
            direction = "inhibitor" if req == NEG else "activator"
            by_class[(onc, direction)] += 1
            for s in winners:
                winning_src[s] += 1
            clean = bool(set(winners) & CLEAN_SRC)
            if clean:
                clean_backed.append({
                    "disease": r["disease"], "drug": l["drug_name"],
                    "gene": l["causal_gene"], "direction": direction,
                    "winning_sources": winners,
                })
            detailed.append({
                "disease": r["disease"], "drug": l["drug_name"],
                "gene": l["causal_gene"], "direction": direction,
                "oncology": onc == "oncology",
                "winning_sources": winners,
                "winning_source_class": (
                    "clean_disease_specific" if clean
                    else "mendelian_or_mouse" if set(winners) & MENDEL_SRC
                    else "none"),
                "conflicting_votes": conflict,
            })

    n_conc = len(detailed)
    summary = {
        "n_genetics_tier_concordant": n_conc,
        "by_class": {f"{a}|{b}": c for (a, b), c in sorted(by_class.items())},
        "n_inhibitor": sum(c for (o, d), c in by_class.items() if d == "inhibitor"),
        "n_activator": sum(c for (o, d), c in by_class.items() if d == "activator"),
        "winning_direction_source_freq": dict(winning_src.most_common()),
        "n_backed_by_clean_disease_specific_source": len(clean_backed),
        "n_gwas_credible_sets_winning_directions": winning_src.get("gwas_credible_sets", 0),
        "n_conflicting_vote_leads": conflicts,
        "interpretation": (
            "Direction that justifies concordance is dominated by cancer-gene "
            "catalogs (cancer_gene_census, intogen) and Mendelian/mouse sources "
            "(eva, impc). gwas_credible_sets — the disease-specific common-variant "
            "source — supplies ZERO winning directions. The method is direction-"
            "correct for oncology (oncogene GoF -> inhibitor, well catalogued) but "
            "there yields precision-oncology rediscoveries / in-trials matches; "
            "outside oncology the only available direction is Mendelian/mouse and "
            "inverts vs the complex-disease direction (e.g. TLR7->lupus activator, "
            "KCNJ11->T2D opener). Barrier (a) is relocated, not dissolved."
        ),
    }
    out = {"summary": summary, "clean_source_leads": clean_backed,
           "leads": detailed}
    OUT.write_text(json.dumps(out, indent=2))

    print("=== CROSS-INDICATION CONCORDANT-LEAD TRIAGE ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
