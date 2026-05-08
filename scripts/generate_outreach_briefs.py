"""Generate lab-outreach briefs for the 22 NTDs + 19 rare-disease set.

Produces one-page briefs designed for cold + warm outreach to
academic labs, mission-aligned non-profits (DNDi, MMV, FIND), and
disease foundations. Different format than the wet-lab briefs (which
are technical hand-offs); these focus on *partnership pitch*:

  - Why this disease, why now
  - Top 5 OpenCure predictions with one-line rationales
  - Suggested assay + concrete readout
  - Three named target labs / orgs with one-line context
  - "Ask" (cost / time / compound) and "Offer" (authorship / data)

Output:
    docs/outreach/<disease_key>.md         per-disease brief
    docs/lab_outreach_briefs.md            consolidated index

Designed to read straight from ``experiments/results/<disease>.json``
+ ``opencure/eval/disease_classes.yaml`` so re-running after each
v7 screen automatically refreshes every brief.

Usage:
    python3 scripts/generate_outreach_briefs.py
    python3 scripts/generate_outreach_briefs.py --diseases Schistosomiasis Chagas_disease
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencure.scoring.common import AGGREGATE_RESULT_FILES
from opencure.scoring.per_class_ensemble import route_disease

OUTREACH_DIR = Path("docs/outreach")
INDEX_PATH = Path("docs/lab_outreach_briefs.md")
RESULTS_DIR = Path("experiments/results")


# ---- Disease-specific outreach metadata --------------------------------

# Hand-curated outreach metadata. Most diseases get a generic
# "academic lab + foundation + national institute" triple; the four
# lead diseases get expanded named-PI entries.
TARGET_LABS: dict[str, dict] = {
    "Schistosomiasis": {
        "biology": (
            "Schistosoma mansoni / haematobium / japonicum. Adult worms "
            "in mesenteric and urogenital venous plexus. ~200 M infected "
            "globally; praziquantel is the only WHO-recommended drug, "
            "resistance is emerging."
        ),
        "labs": [
            "Conor Caffrey lab (UC San Diego) — schisto screening specialist",
            "Jenner Institute Helminth Vaccine Centre (Oxford)",
            "DNDi Helminth cluster",
            "Imperial-Wellcome SCI Foundation",
            "KEMRI Centre for Geographic Medicine Research",
        ],
        "assay": "Adult-worm viability assay (S. mansoni NMRI strain), 48-72 h",
        "readout": "WST-1 reduction or direct microscopic motility",
        "ask": "20-30 µL of purchased compound at 1, 10, 100 µM",
        "lead": True,
    },
    "Chagas_disease": {
        "biology": (
            "Trypanosoma cruzi infection; cardiomyopathy in 30 % of chronic "
            "cases. Benznidazole and nifurtimox are toxic and only 30-60 % "
            "curative. ~7 M infected, largely Latin America."
        ),
        "labs": [
            "DNDi Chagas cluster (lead non-profit)",
            "Fundação Oswaldo Cruz (Fiocruz) — established screening cascade",
            "Texas A&M Chagas Center",
            "Mundo Sano Foundation",
        ],
        "assay": "In-vitro amastigote screen on infected Vero cells, 72 h",
        "readout": "Parasite count via beta-galactosidase reporter",
        "ask": "10 µL at 1 / 10 µM",
        "lead": True,
    },
    "Sickle_cell_disease": {
        "biology": (
            "Hemoglobin S polymerisation under deoxygenation, RBC sickling, "
            "vaso-occlusion. Hydroxyurea, voxelotor, crizanlizumab approved; "
            "still high unmet need (acute pain crises, cumulative organ "
            "damage). ~100 K affected in the US, ~5 M worldwide."
        ),
        "labs": [
            "CureSCi consortium (NIH) — central US sickle-cell network",
            "Doris Duke Charitable Foundation — sickle-cell granting program",
            "St. Jude Children's Research Hospital — pediatric SCD",
            "Imperial College London / Hammersmith Sickle Cell Centre",
        ],
        "assay": "Anti-sickling assay on SS RBCs ex vivo, deoxygenation challenge",
        "readout": "Sickling rate by morphometry; HbF induction (qPCR)",
        "ask": "30 µL at 0.1 / 1 / 10 µM",
        "lead": True,
    },
    "Niemann-Pick_disease": {
        "biology": (
            "Lysosomal storage disorder caused by NPC1/NPC2 mutations "
            "(Niemann-Pick C) or SMPD1 (Niemann-Pick A/B). Cholesterol "
            "trafficking defect, neurodegeneration, hepatosplenomegaly. "
            "Miglustat slows disease but is not curative. Pediatric onset, "
            "fatal."
        ),
        "labs": [
            "Ara Parseghian Medical Research Foundation (NPC granting org)",
            "NPUK (UK patient foundation)",
            "Forbes Porter lab (NIH/NICHD) — NPC1 mechanism",
            "Daniel Ory lab (Washington University)",
        ],
        "assay": "Filipin staining for unesterified cholesterol in patient fibroblasts",
        "readout": "Filipin intensity reduction; LAMP1 co-localisation",
        "ask": "10 µL at 0.1 / 1 / 10 µM in patient-derived fibroblasts",
        "lead": True,
    },
    # Generic NTD / rare-disease entries; lab list is a starting point
    # rather than a vetted match.
    "Leishmaniasis": {
        "biology": "Leishmania spp. (visceral, cutaneous). Pentamidine, sodium stibogluconate, miltefosine — toxic, resistance widespread.",
        "labs": ["DNDi Leishmaniasis cluster", "MSF Access Campaign",
                 "Walter Reed Army Institute of Research"],
        "assay": "Intracellular amastigote screen (THP-1 infected with L. donovani)",
        "readout": "Parasite count at 72 h; selectivity vs host viability",
    },
    "Tuberculosis": {
        "biology": "M. tuberculosis. MDR/XDR-TB rising. Bedaquiline / pretomanid the most recent additions; long treatment duration limits adherence.",
        "labs": ["TB Alliance", "FIND Diagnostics", "Stewart Cole lab (EPFL)"],
        "assay": "MIC against M. tuberculosis H37Rv per CLSI broth microdilution",
        "readout": "MIC after 7 d incubation",
    },
    "Buruli_ulcer": {
        "biology": "M. ulcerans skin infection; necrotic ulcers, mycolactone toxin. Rifampicin + clarithromycin curative but not always available.",
        "labs": ["WHO Buruli ulcer task force", "Pasteur Institute — Mycolactone group",
                 "FIND Diagnostics"],
        "assay": "MIC vs M. ulcerans Agy99 (BSL-2)",
        "readout": "Growth inhibition over 14 d",
    },
    "Trachoma": {
        "biology": "Chlamydia trachomatis ocular infection; leading infectious cause of blindness. Azithromycin MDA effective but reinfection common.",
        "labs": ["International Trachoma Initiative", "ITI partner network",
                 "Carter Center Trachoma Control Program"],
        "assay": "Inhibition of C. trachomatis replication in HeLa cells",
        "readout": "Inclusion-forming units at 48 h",
    },
    "Lymphatic_filariasis": {
        "biology": "Wuchereria bancrofti / Brugia malayi. Elephantiasis, hydrocele. Ivermectin + albendazole MDA.",
        "labs": ["Filarial Research Group (Liverpool School of Tropical Medicine)",
                 "Mectizan Donation Program", "Smith College — Filariasis lab"],
        "assay": "Adult-worm viability (B. malayi)",
        "readout": "Motility / WST-1 reduction at 72 h",
    },
    "Onchocerciasis": {
        "biology": "Onchocerca volvulus, river blindness. Ivermectin MDA + Wolbachia targeting (doxycycline).",
        "labs": ["MDP (Mectizan Donation Program)", "Dezouré-Salama Group (Cameroon)",
                 "Sightsavers"],
        "assay": "Adult-worm viability (O. ochengi as proxy)",
        "readout": "Motility / WST-1 over 72 h",
    },
    "Echinococcosis": {
        "biology": "Echinococcus granulosus / multilocularis cystic disease. Albendazole + surgery; recurrence common.",
        "labs": ["Klaus Brehm lab (Würzburg) — Echinococcus cell culture",
                 "WHO Echinococcosis network"],
        "assay": "Protoscolex viability (E. multilocularis)",
        "readout": "Eosin uptake at 72 h",
    },
    "Cysticercosis": {
        "biology": "Taenia solium larval cysts. Neurocysticercosis a major epilepsy cause. Albendazole + corticosteroids.",
        "labs": ["TPSU/Cysticercosis Working Group of Peru",
                 "Hector Garcia lab (Universidad Peruana Cayetano Heredia)"],
        "assay": "Cyst viability (T. crassiceps proxy)",
        "readout": "Eosin / motility at 72 h",
    },
    "Rabies": {
        "biology": "Rabies lyssavirus. Universally fatal post-symptom; PEP only intervention. ~59 K deaths/year.",
        "labs": ["Pasteur Institute — Rabies network",
                 "World Rabies Day partners"],
        "assay": "Viral replication assay (CVS-11 strain in BHK cells)",
        "readout": "Plaque count at 48 h",
    },
    "Scabies": {
        "biology": "Sarcoptes scabiei var. hominis. Permethrin + ivermectin standard.",
        "labs": ["WHO Neglected Tropical Diseases (NTD) team",
                 "James Cook University — Scabies Research"],
        "assay": "Mite viability (S. scabiei) ex vivo",
        "readout": "Survival at 24-48 h",
    },
    "Ascariasis": {
        "biology": "Ascaris lumbricoides. Mebendazole / albendazole standard; resistance emerging.",
        "labs": ["Liverpool School of Tropical Medicine — Helminth lab",
                 "DOLF Project (Death of Lymphatic Filariasis)"],
        "assay": "Adult-worm viability (A. suum proxy)",
        "readout": "Motility / WST-1 at 72 h",
    },
    "Hookworm_infection": {
        "biology": "Necator americanus, Ancylostoma duodenale. Anaemia in pregnancy. Mebendazole standard.",
        "labs": ["Hookworm Vaccine Initiative (Sabin)",
                 "Peter Hotez lab (Texas Children's)"],
        "assay": "L3 viability ex vivo",
        "readout": "Motility at 72 h",
    },
    "Visceral_leishmaniasis": {
        "biology": "Leishmania donovani / infantum. Fatal if untreated; HIV co-infection accelerates.",
        "labs": ["DNDi Visceral Leishmaniasis cluster",
                 "WHO South-East Asia Regional Office",
                 "Walter Reed Army Institute"],
        "assay": "Intracellular amastigote screen",
        "readout": "Parasite count at 72 h",
    },
    "Cutaneous_leishmaniasis": {
        "biology": "Leishmania major / tropica skin lesions. Disfiguring; toxic conventional therapy.",
        "labs": ["DNDi", "MSF Access Campaign", "WHO Pan American Health Org"],
        "assay": "Promastigote / intracellular amastigote screen",
        "readout": "Parasite count at 72 h",
    },
    "Dengue": {
        "biology": "Dengue virus 1-4 (Flaviviridae). Antibody-dependent enhancement complicates vaccines. Supportive care only.",
        "labs": ["Viralink", "Singapore-MIT Alliance for Research and Technology",
                 "Imperial College London — Dengue Research"],
        "assay": "Viral RNA reduction (DENV-2 NGC strain in Huh-7.5)",
        "readout": "qRT-PCR at 48 h post-infection",
    },
    "HIV": {
        "biology": "HIV-1 / HIV-2; lentivirus integrating into host genome. ART now durable but lifelong; no cure.",
        "labs": ["amfAR (Foundation for AIDS Research)", "Bill & Melinda Gates Foundation HIV team",
                 "International AIDS Society"],
        "assay": "HIV-1 replication assay (NL4-3 in CEM-T4 cells)",
        "readout": "p24 ELISA at 5 d",
    },
    "Hepatitis_C": {
        "biology": "HCV genotype 1-6. DAAs curative for most patients; remaining unmet need is access and resistance-associated substitutions.",
        "labs": ["Hepatitis C Research Network", "WHO Hepatitis Programme",
                 "Sonia Patel lab (Imperial College)"],
        "assay": "Replicon assay (Huh-7.5 cells with subgenomic replicon)",
        "readout": "Replicon RNA reduction at 72 h",
    },
    "African_trypanosomiasis": {
        "biology": "Trypanosoma brucei gambiense / rhodesiense. Sleeping sickness. Fexinidazole approved 2018; pre-fexinidazole drugs were toxic.",
        "labs": ["DNDi HAT cluster", "WHO HAT Elimination Programme",
                 "Pierre Escudero lab (CIRAD Montpellier)"],
        "assay": "Bloodstream-form viability (T. b. brucei)",
        "readout": "WST-1 reduction at 72 h",
    },
    "Leprosy": {
        "biology": "M. leprae. WHO MDT (rifampicin + clofazimine + dapsone) curative; nerve damage often irreversible by diagnosis.",
        "labs": ["LEPRA / Leprosy Mission International",
                 "Schieffelin Institute of Health-Research (India)"],
        "assay": "MIC against M. leprae (mouse footpad model is gold-standard but slow; cell culture surrogate via M. ulcerans / M. smegmatis)",
        "readout": "Growth inhibition at 14 d",
    },
    # ---- Rare diseases (19) ------------------------------------------
    "Fragile_X_syndrome": {
        "biology": "FMR1 CGG-repeat expansion silencing FMRP. mGluR5 / GABA-B / endocannabinoid pathway dysregulation.",
        "labs": ["FRAXA Research Foundation", "Cure Fragile X (Holzer Foundation)",
                 "Stephen Warren lab (Emory)"],
        "assay": "FMRP-deficient neuron phenotype rescue (iPSC-derived)",
        "readout": "Synaptic density / mGluR5 signalling at 5-7 days",
    },
    "Duchenne_muscular_dystrophy": {
        "biology": "Dystrophin loss-of-function. Glucocorticoids + exon-skipping ASOs; gene therapy emerging.",
        "labs": ["CureDuchenne", "Parent Project Muscular Dystrophy",
                 "Lou Kunkel lab (Boston Children's)"],
        "assay": "Muscle-cell phenotype in DMD iPSC-derived myocytes",
        "readout": "Membrane integrity / dystrophin-glycoprotein complex stability",
    },
    "Neurofibromatosis": {
        "biology": "NF1 / NF2 tumor-suppressor loss; benign nerve-sheath tumors, malignant transformation risk.",
        "labs": ["Children's Tumor Foundation", "NF Therapeutic Acceleration Network"],
        "assay": "Schwann-cell viability / proliferation (Nf1-/- MPNST lines)",
        "readout": "CellTiter-Glo at 72 h",
    },
    "Marfan_syndrome": {
        "biology": "FBN1 mutations; aortic aneurysm. ARBs delay progression but don't reverse.",
        "labs": ["Marfan Foundation", "Hal Dietz lab (Johns Hopkins)"],
        "assay": "TGF-β-driven extracellular-matrix remodelling in iPSC-derived smooth muscle",
        "readout": "TGF-β signalling and fibrillin-1 deposition",
    },
    "Ehlers-Danlos_syndrome": {
        "biology": "Heterogeneous collagen / connective-tissue disorders. No specific therapy; symptomatic management.",
        "labs": ["Ehlers-Danlos Society", "Norris-Stokes lab (Imperial College)"],
        "assay": "Patient-fibroblast collagen secretion + crosslinking",
        "readout": "Collagen-I deposition density",
    },
    "Gaucher_disease": {
        "biology": "GBA1 deficiency; glucocerebroside accumulation. Enzyme replacement (imiglucerase) standard. Type 2/3 neurological forms unmet.",
        "labs": ["National Gaucher Foundation", "Anthony Futerman lab (Weizmann)",
                 "Mark Sands lab (Washington University)"],
        "assay": "GCase activity rescue in patient fibroblasts",
        "readout": "Enzyme activity by 4-MU-glucoside fluorometric assay",
    },
    "Fabry_disease": {
        "biology": "GLA deficiency; α-galactosidase A. Enzyme replacement + chaperone (migalastat) standard. Cardiac variant unmet.",
        "labs": ["Fabry International Network", "Tony Futerman lab (Weizmann)"],
        "assay": "α-Gal A activity / Gb3 accumulation in patient fibroblasts",
        "readout": "Gb3 LC-MS at 72 h",
    },
    "Hunter_syndrome": {
        "biology": "MPS-II, IDS deficiency. Glycosaminoglycan accumulation. Idursulfase standard but doesn't cross BBB.",
        "labs": ["MPS Society", "National MPS Society", "Ed Wraith lab (Manchester)"],
        "assay": "IDS activity rescue / GAG accumulation in patient fibroblasts",
        "readout": "DMB-GAG assay at 72 h",
    },
    "Pompe_disease": {
        "biology": "GAA deficiency. Glycogen accumulation in lysosomes. Alglucosidase alfa standard; muscle penetration limited.",
        "labs": ["AMDA / IPA (International Pompe Association)",
                 "Andrea Ballabio lab (TIGEM Naples)"],
        "assay": "GAA activity / glycogen clearance in patient fibroblasts or iPSC-cardiomyocytes",
        "readout": "Glycogen quantification by PAS staining",
    },
    "Spinal_muscular_atrophy": {
        "biology": "SMN1 loss; SMN2 backup. Nusinersen, risdiplam, onasemnogene abeparvovec all approved. Refractory phenotypes remain.",
        "labs": ["Cure SMA Foundation", "SMA Trust",
                 "Adrian Krainer lab (CSHL) — SMN splicing"],
        "assay": "SMN protein level rescue in patient fibroblasts / iPSC-motor-neurons",
        "readout": "SMN immunostaining / Western blot",
    },
    "Friedreich's_ataxia": {
        "biology": "FXN GAA-repeat expansion silencing frataxin. Mitochondrial iron overload, cardiomyopathy, neurodegeneration. Omaveloxolone approved 2023.",
        "labs": ["Friedreich's Ataxia Research Alliance (FARA)",
                 "Hélène Puccio lab (IGBMC Strasbourg)"],
        "assay": "Frataxin level / mitochondrial function in patient fibroblasts",
        "readout": "Aconitase activity / FXN immunostaining",
    },
    "Rett_syndrome": {
        "biology": "MECP2 X-linked dominant; female-predominant. Trofinetide approved 2023; no curative therapy.",
        "labs": ["Rett Syndrome Research Trust", "International Rett Syndrome Foundation",
                 "Adrian Bird lab (Edinburgh)"],
        "assay": "MECP2-deficient neuron phenotype rescue (iPSC-derived)",
        "readout": "Synaptic density / electrophysiology at 5-7 weeks",
    },
    "Tay-Sachs_disease": {
        "biology": "HEXA deficiency. GM2 ganglioside accumulation. No approved therapy; substrate-reduction therapy in trials.",
        "labs": ["NTSAD (National Tay-Sachs & Allied Diseases Association)",
                 "Don Mahuran lab (Hospital for Sick Children, Toronto)"],
        "assay": "HexA activity / GM2 accumulation in patient fibroblasts",
        "readout": "MUGS fluorometric assay / GM2 LC-MS",
    },
    "Wilson's_disease": {
        "biology": "ATP7B copper-transport defect. Penicillamine, trientine, zinc standard. Hepatic / neurological presentations.",
        "labs": ["Wilson Disease Association", "Peter Ferenci lab (Vienna)"],
        "assay": "Copper-handling / ATP7B-deficient hepatocyte phenotype",
        "readout": "Cu accumulation by ICP-MS / NRF2 signalling",
    },
    "Phenylketonuria": {
        "biology": "PAH deficiency. Sapropterin, pegvaliase. Lifelong dietary restriction.",
        "labs": ["National PKU Alliance", "Stuart Maby lab (UCSD)"],
        "assay": "PAH activity rescue / Phe clearance in patient hepatocytes (iPSC-derived)",
        "readout": "Phe → Tyr conversion at 72 h",
    },
    "Spinocerebellar_ataxia": {
        "biology": "Heterogeneous (SCA1-50). Polyglutamine and noncoding-repeat expansions. No disease-modifying therapy.",
        "labs": ["National Ataxia Foundation", "Ataxia UK",
                 "Stefan Pulst lab (University of Utah)"],
        "assay": "Patient-iPSC neuron viability under stress",
        "readout": "Aggregate formation / electrophysiology at 5 weeks",
    },
    "Acromegaly": {
        "biology": "Growth-hormone hypersecretion (pituitary adenoma). Somatostatin analogues, pegvisomant. Treatment-resistant cases unmet.",
        "labs": ["Pituitary Society", "Shlomo Melmed lab (Cedars-Sinai)"],
        "assay": "GH secretion suppression in pituitary tumor cells",
        "readout": "GH ELISA at 24 h",
    },
    "Cystic_fibrosis": {
        "biology": "CFTR mutations. Trikafta covers ~90 % of patients; minor-allele patients still unmet.",
        "labs": ["Cystic Fibrosis Foundation", "CF Trust UK"],
        "assay": "CFTR-mediated chloride flux in patient organoid forskolin assay",
        "readout": "Forskolin-induced swelling at 1 h",
    },
}


# Rare-disease keys we cover where TARGET_LABS doesn't carry a curated
# entry — they get the generic template.
GENERIC_FALLBACK = {
    "biology": (
        "Mechanism details available in the dashboard's per-disease "
        "evidence panel. Add disease-specific context here."
    ),
    "labs": [
        "[Disease-area academic lab] — add specific PI",
        "[Disease foundation] — add granting organisation",
        "[National research institute] — add country-specific partner",
    ],
    "assay": "Disease-relevant cellular phenotypic assay",
    "readout": "Phenotypic readout matched to disease pathology",
}


# ---- Brief assembly -----------------------------------------------------

def _top_predictions(result_path: Path, k: int = 5) -> list[dict]:
    if not result_path.exists():
        return []
    data = json.loads(result_path.read_text())
    candidates = data.get("candidates") or data.get("top_candidates") or []
    return candidates[:k]


def _format_top_predictions(cands: list[dict]) -> str:
    if not cands:
        return "_(populated after the next v7 screen)_"
    lines = []
    for i, c in enumerate(cands, start=1):
        name = c.get("drug_name") or c.get("drug_id") or "?"
        prob = c.get("ensemble_prob")
        prob_str = f"  (p = {prob:.2f})" if isinstance(prob, (int, float)) else ""
        rationale = c.get("relation_type") or c.get("similar_to") or ""
        rationale_str = f" — {rationale}" if rationale else ""
        lines.append(f"{i}. **{name}**{prob_str}{rationale_str}")
    return "\n".join(lines)


def _render_brief(disease_key: str, ctx: dict, predictions: list[dict]) -> str:
    disease_name = disease_key.replace("_", " ")
    is_lead = ctx.get("lead", False)
    klass = route_disease(disease_name) or "unmapped"
    lab_lines = "\n".join(f"- {lab}" for lab in ctx["labs"])

    body = f"""# {disease_name} — outreach brief

> **Disease class:** `{klass}` &nbsp; | &nbsp; **Lead disease:** {'**yes**' if is_lead else 'no'}

## Why this disease

{ctx["biology"]}

## Top OpenCure predictions (v7)

{_format_top_predictions(predictions)}

Per-candidate mechanism, conformal interval, red-team critique, and
suggested-assay details are in
`experiments/results/briefs/{disease_key}_top5.md`.

## Suggested assay

- **Assay:** {ctx["assay"]}
- **Readout:** {ctx["readout"]}
"""
    if "ask" in ctx:
        body += f"- **Ask (compound supply):** {ctx['ask']}\n"

    body += f"""
## Target labs / partners

{lab_lines}

## What OpenCure offers

- Co-authorship on any published validation work.
- Continuous re-screening as the platform updates (currently quarterly).
- Open-data deposit of all results; nothing is held back.
- Wet-lab brief regenerated from this disease's `briefs/` file each release.

## Citation

Cite the v7 methods preprint (DOI on the homepage) plus the prediction
snapshot DOI shown on the per-disease dashboard page.
"""
    return body


def _render_index(briefs: list[tuple[str, dict]]) -> str:
    """Consolidated index for docs/lab_outreach_briefs.md."""
    lines = [
        "# OpenCure Lab Outreach Briefs (NTDs + rare diseases)",
        "",
        "One-page partnership briefs for each of the 22 NTDs and 19 "
        "rare diseases the v7 platform screens. Pair each brief with the "
        "live dashboard URL and the content-fingerprinted Zenodo "
        "snapshot. Customize per-PI when reaching out.",
        "",
        "Lead diseases (deepest curation, highest-priority outreach):",
        "**Schistosomiasis · Chagas disease · Sickle cell disease · "
        "Niemann-Pick disease**.",
        "",
        "Per-disease briefs live under `docs/outreach/<disease_key>.md`.",
        "",
        "## Index",
        "",
        "| Disease | Class | Brief |",
        "|---------|-------|-------|",
    ]
    for disease_key, ctx in briefs:
        klass = route_disease(disease_key.replace("_", " ")) or "_(unmapped)_"
        is_lead = "**lead**" if ctx.get("lead") else ""
        lines.append(
            f"| {disease_key.replace('_', ' ')} | `{klass}` | "
            f"[`outreach/{disease_key}.md`](outreach/{disease_key}.md) "
            f"{is_lead} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diseases", nargs="*",
                        help="Specific disease keys (no .json); empty = all NTDs+rare")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTREACH_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Default disease set = the 41 keys in TARGET_LABS plus any other
    # NTD / rare disease present in TARGET_DISEASES. Drop the
    # "_disease" → "" rename-mismatches deterministically.
    disease_keys = args.diseases or sorted(TARGET_LABS.keys())

    written: list[tuple[str, dict]] = []
    for disease_key in disease_keys:
        ctx = TARGET_LABS.get(disease_key, GENERIC_FALLBACK)
        # Allow JSON files named with spaces or underscores.
        json_path = args.results_dir / f"{disease_key}.json"
        if not json_path.exists():
            json_path = args.results_dir / f"{disease_key.replace('_', ' ')}.json"
        predictions = _top_predictions(json_path, k=5) if json_path.exists() else []
        brief = _render_brief(disease_key, ctx, predictions)
        out_path = args.out_dir / f"{disease_key}.md"
        out_path.write_text(brief)
        written.append((disease_key, ctx))

    args.index.write_text(_render_index(written))
    print(f"Wrote {len(written)} per-disease briefs to {args.out_dir}/")
    print(f"Updated index at {args.index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
