"""
Disease subtype stratification (v5 D4).

For clinically-heterogeneous diseases, molecularly-defined subtypes respond
differently to drugs. Running one prediction per "Breast cancer" collapses
HER2+, HR+, and triple-negative into one noisy average. Predicting per
subtype is how results become actionable.

This module exposes a curated mapping: disease → list of subtypes, where
each subtype has:
  - display_name
  - defining molecular feature (gene/biomarker)
  - relevant MeSH / EFO / MONDO entity when available

Used by experiments/systematic_screening.py to optionally expand a disease
entry into per-subtype predictions.

v1 covers the most clinically-consequential ~15 diseases. Expansion list
in extend_subtypes() at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Subtype:
    display_name: str
    marker: str
    mesh_id: Optional[str] = None
    rationale: str = ""


DISEASE_SUBTYPES: dict[str, list[Subtype]] = {
    "Breast cancer": [
        Subtype("HER2-positive", "ERBB2 amplification", "D018719",
                "Responds to trastuzumab/pertuzumab; distinct drug class"),
        Subtype("Hormone-receptor-positive (HR+)", "ESR1+ / PGR+",
                rationale="Tamoxifen, aromatase inhibitors, CDK4/6 inhibitors"),
        Subtype("Triple-negative", "ER−/PR−/HER2−",
                rationale="No targeted therapy; standard-of-care chemo only — high unmet need"),
        Subtype("BRCA-mutant", "BRCA1/2 loss-of-function",
                rationale="PARP-inhibitor-sensitive; distinct DNA-repair biology"),
    ],
    "Lung cancer": [
        Subtype("EGFR-mutant NSCLC", "EGFR L858R / del19",
                rationale="TKI-responsive (erlotinib/osimertinib)"),
        Subtype("ALK-rearranged NSCLC", "ALK fusion",
                rationale="Crizotinib/alectinib-responsive"),
        Subtype("KRAS G12C NSCLC", "KRAS G12C",
                rationale="Sotorasib/adagrasib-responsive"),
        Subtype("Small-cell lung cancer (SCLC)", "Neuroendocrine",
                rationale="Different biology; platinum+etoposide; checkpoint inhibitors"),
    ],
    "Alzheimers disease": [
        Subtype("APOE4-carrier Alzheimer's", "APOE ε4",
                rationale="Higher lecanemab ARIA risk; different progression rate"),
        Subtype("Familial / early-onset AD", "APP / PSEN1 / PSEN2 mutations",
                rationale="Mendelian; dominant-negative biology differs from sporadic"),
        Subtype("Sporadic late-onset AD", "polygenic risk",
                rationale="Most common form; target of most current drug development"),
    ],
    "Parkinsons disease": [
        Subtype("LRRK2-associated PD", "LRRK2 G2019S",
                rationale="Potentially LRRK2-inhibitor responsive"),
        Subtype("GBA-associated PD", "GBA heterozygote",
                rationale="Faster progression; different glucocerebrosidase biology"),
        Subtype("Sporadic PD", "polygenic",
                rationale="Majority; target of most small-molecule DMTs"),
    ],
    "Colorectal cancer": [
        Subtype("MSI-high CRC", "mismatch-repair deficiency",
                rationale="Checkpoint-inhibitor responsive (pembrolizumab)"),
        Subtype("RAS-wildtype CRC", "KRAS/NRAS wildtype",
                rationale="Cetuximab/panitumumab responsive"),
        Subtype("BRAF V600E CRC", "BRAF V600E",
                rationale="Encorafenib + cetuximab combo responsive"),
    ],
    "Melanoma": [
        Subtype("BRAF V600-mutant melanoma", "BRAF V600E/K",
                rationale="BRAF/MEK inhibitor responsive"),
        Subtype("NRAS-mutant melanoma", "NRAS Q61",
                rationale="MEK inhibitor + immunotherapy"),
        Subtype("Wild-type / triple-wild melanoma", "BRAF/NRAS/NF1 wildtype",
                rationale="Checkpoint inhibitors; otherwise limited targeted options"),
    ],
    "Glioblastoma": [
        Subtype("IDH-mutant glioma", "IDH1/2 mutation",
                rationale="Better prognosis; ivosidenib/vorasidenib responsive"),
        Subtype("IDH-wildtype GBM", "IDH wildtype",
                rationale="Classical aggressive GBM; temozolomide + radiation; needs new options"),
        Subtype("MGMT-methylated GBM", "MGMT promoter methylation",
                rationale="Better temozolomide response"),
    ],
    "Acute myeloid leukemia": [
        Subtype("FLT3-mutant AML", "FLT3-ITD or FLT3-TKD",
                rationale="Midostaurin/gilteritinib responsive"),
        Subtype("IDH-mutant AML", "IDH1/2",
                rationale="Ivosidenib/enasidenib responsive"),
        Subtype("TP53-mutant AML", "TP53",
                rationale="Poor prognosis; limited targeted options — high unmet need"),
    ],
    "Asthma": [
        Subtype("Type-2-high (eosinophilic) asthma", "IL-4/IL-13/IL-5 high, eosinophils",
                rationale="Anti-IL-5 (mepolizumab), anti-IL-4Rα (dupilumab) responsive"),
        Subtype("Type-2-low asthma", "neutrophilic / paucigranulocytic",
                rationale="Poor biologic response; different treatment algorithm"),
    ],
    "Inflammatory bowel disease": [
        Subtype("TNF-responsive IBD", "serum TNF",
                rationale="Infliximab/adalimumab responsive"),
        Subtype("JAK-responsive IBD", "JAK/STAT pathway activation",
                rationale="Tofacitinib/upadacitinib responsive"),
        Subtype("IL-23-responsive IBD", "IL23/Th17 axis",
                rationale="Ustekinumab/risankizumab responsive"),
    ],
    "Rheumatoid arthritis": [
        Subtype("Seropositive RA", "RF+ or anti-CCP+",
                rationale="Earlier biologic; worse prognosis"),
        Subtype("Seronegative RA", "RF− / anti-CCP−",
                rationale="Different progression; biologic response less predictable"),
    ],
    "Sickle cell disease": [
        Subtype("HbSS (classical)", "homozygous βS",
                rationale="Most severe; voxelotor, crizanlizumab, hydroxyurea standard"),
        Subtype("HbSβ0-thalassemia", "compound heterozygote",
                rationale="Similar severity to HbSS"),
        Subtype("HbSC / HbSβ+", "milder compound heterozygotes",
                rationale="Milder phenotype; different management"),
    ],
    "Cystic fibrosis": [
        Subtype("F508del-homozygous CF", "CFTR F508del/F508del",
                rationale="Elexacaftor-tezacaftor-ivacaftor (Trikafta) responsive"),
        Subtype("Gating-mutation CF", "e.g. G551D",
                rationale="Ivacaftor monotherapy responsive"),
        Subtype("Minimal-function CF", "severe CFTR mutations",
                rationale="Poor modulator response; highest unmet need"),
    ],
}


def get_subtypes(disease_name: str) -> list[dict]:
    """Return list of subtype dicts for a disease; empty list if no subtypes curated."""
    subs = DISEASE_SUBTYPES.get(disease_name)
    if not subs:
        return []
    return [
        {
            "display_name": s.display_name,
            "marker": s.marker,
            "mesh_id": s.mesh_id,
            "rationale": s.rationale,
        }
        for s in subs
    ]


def has_subtypes(disease_name: str) -> bool:
    return disease_name in DISEASE_SUBTYPES


def list_subtyped_diseases() -> list[str]:
    return sorted(DISEASE_SUBTYPES.keys())


if __name__ == "__main__":
    import json
    for d in list_subtyped_diseases():
        subs = get_subtypes(d)
        print(f"\n{d}  ({len(subs)} subtypes):")
        for s in subs:
            print(f"  • {s['display_name']:40s}  [{s['marker']}]")
