"""
Tissue-context pillar (v5 C2).

A drug's effect depends on where its targets are expressed. A CNS drug
whose target isn't expressed in neurons is a non-starter. Conversely, a
predicted drug-disease pair where the disease-gene is highly expressed
in the disease's tissue AND the drug's target is in the same tissue gets
a real boost.

Data: GTEx v8 bulk tissue expression (54 tissues × ~56k Ensembl genes,
median TPM). Downloaded to data/sources_2024/gtex/gtex_median_tpm.gct.

Mapping from disease → relevant tissue(s):
  Curated in DISEASE_TISSUE_MAP below. For uncurated diseases we fall
  back to 'Whole Blood' (poor but universal).

Output per (disease, gene list):
  - mean expression of those genes in disease-relevant tissue(s)
  - tissue-specificity score (1 = tissue-specific, 0 = ubiquitous)
  - disease_tissue_match_score in [0, 1] used as a pillar modifier

This DOES NOT replace any pillar but augments combined_score by ±15%
via `tissue_context_modifier` when the evidence is strong enough.
"""

from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


GTEX_PATH = Path("data/sources_2024/gtex/gtex_median_tpm.gct")
GTEX_GZ_PATH = Path("data/sources_2024/gtex/gtex_median_tpm.gct.gz")
HGNC_PATH = Path("data/mappings/hgnc_complete_set.txt")


# Curated disease → tissue(s) mapping. Tissues use exact GTEx labels.
DISEASE_TISSUE_MAP: dict[str, list[str]] = {
    # Neurology
    "Alzheimers disease":   ["Brain - Cortex", "Brain - Hippocampus", "Brain - Frontal Cortex (BA9)"],
    "Parkinsons disease":   ["Brain - Substantia nigra", "Brain - Putamen (basal ganglia)", "Brain - Caudate (basal ganglia)"],
    "Huntingtons disease":  ["Brain - Caudate (basal ganglia)", "Brain - Putamen (basal ganglia)"],
    "Amyotrophic lateral sclerosis": ["Brain - Spinal cord (cervical c-1)", "Brain - Cortex", "Muscle - Skeletal"],
    "Multiple sclerosis":   ["Brain - Cortex", "Brain - Cerebellum"],
    "Epilepsy":             ["Brain - Hippocampus", "Brain - Cortex"],
    "Depression":           ["Brain - Frontal Cortex (BA9)", "Brain - Anterior cingulate cortex (BA24)"],
    "Anxiety":              ["Brain - Amygdala", "Brain - Hippocampus"],
    "Schizophrenia":        ["Brain - Frontal Cortex (BA9)", "Brain - Cortex"],
    "Bipolar disorder":     ["Brain - Frontal Cortex (BA9)", "Brain - Cortex"],
    # Cardio
    "Atrial fibrillation":  ["Heart - Atrial Appendage"],
    "Heart failure":        ["Heart - Left Ventricle", "Heart - Atrial Appendage"],
    "Hypertension":         ["Artery - Aorta", "Artery - Tibial", "Kidney - Cortex"],
    "Coronary artery disease": ["Artery - Coronary", "Artery - Aorta"],
    "Atherosclerosis":      ["Artery - Aorta", "Artery - Coronary"],
    "Pulmonary hypertension": ["Lung", "Artery - Coronary"],
    # Cancer — tissue of origin
    "Breast cancer":        ["Breast - Mammary Tissue"],
    "Lung cancer":          ["Lung"],
    "Colorectal cancer":    ["Colon - Sigmoid", "Colon - Transverse"],
    "Pancreatic cancer":    ["Pancreas"],
    "Prostate cancer":      ["Prostate"],
    "Ovarian cancer":       ["Ovary"],
    "Leukemia":             ["Whole Blood"],
    "Lymphoma":             ["Whole Blood", "Spleen"],
    "Melanoma":             ["Skin - Sun Exposed (Lower leg)", "Skin - Not Sun Exposed (Suprapubic)"],
    "Glioblastoma":         ["Brain - Cortex", "Brain - Frontal Cortex (BA9)"],
    "Multiple myeloma":     ["Whole Blood", "Spleen"],
    # Metabolic
    "Type 2 diabetes":      ["Pancreas", "Liver", "Muscle - Skeletal", "Adipose - Subcutaneous"],
    "Obesity":              ["Adipose - Subcutaneous", "Adipose - Visceral (Omentum)", "Liver"],
    "Osteoporosis":         ["Muscle - Skeletal"],  # bone not in GTEx
    # Autoimmune / inflammatory
    "Rheumatoid arthritis": ["Whole Blood", "Muscle - Skeletal"],
    "Lupus":                ["Whole Blood", "Skin - Not Sun Exposed (Suprapubic)"],
    "Psoriasis":            ["Skin - Not Sun Exposed (Suprapubic)", "Skin - Sun Exposed (Lower leg)"],
    "Inflammatory bowel disease": ["Colon - Transverse", "Small Intestine - Terminal Ileum"],
    "Crohns disease":       ["Small Intestine - Terminal Ileum", "Colon - Transverse"],
    "Ulcerative colitis":   ["Colon - Sigmoid", "Colon - Transverse"],
    "Asthma":               ["Lung"],
    "COPD":                 ["Lung"],
    "Cystic fibrosis":      ["Lung", "Pancreas"],
    "Idiopathic pulmonary fibrosis": ["Lung"],
    # Infectious — host tissues
    "Malaria":              ["Whole Blood", "Liver", "Spleen"],
    "Tuberculosis":         ["Lung", "Whole Blood"],
    "HIV":                  ["Whole Blood", "Spleen"],
    "Hepatitis C":          ["Liver"],
    "Dengue":               ["Whole Blood", "Liver"],
    "COVID-19":             ["Lung", "Whole Blood"],
    "Schistosomiasis":      ["Liver", "Whole Blood"],
    "Leishmaniasis":        ["Whole Blood", "Spleen", "Liver"],
    "Chagas disease":       ["Heart - Left Ventricle", "Whole Blood"],
    # Renal / hepatic
    "Chronic kidney disease": ["Kidney - Cortex", "Kidney - Medulla"],
    "Liver cirrhosis":      ["Liver"],
    # Rare
    "Sickle cell disease":  ["Whole Blood", "Spleen"],
    "Gaucher disease":      ["Spleen", "Liver", "Whole Blood"],
    "Fabry disease":        ["Kidney - Cortex", "Heart - Left Ventricle"],
    "Duchenne muscular dystrophy": ["Muscle - Skeletal", "Heart - Left Ventricle"],
    "Fragile X syndrome":   ["Brain - Cortex"],
    "Marfan syndrome":      ["Artery - Aorta", "Heart - Left Ventricle"],
    "Neurofibromatosis":    ["Brain - Cortex", "Nerve - Tibial"],
    "Ehlers-Danlos syndrome": ["Skin - Not Sun Exposed (Suprapubic)"],
    # General / other
    "Sepsis":               ["Whole Blood"],
    "Endometriosis":        ["Uterus"],
}


@lru_cache(maxsize=1)
def _load_gtex() -> Optional[dict]:
    """Load GTEx median TPM.

    Returns dict with:
      'genes': list of (ensembl_id, symbol)
      'tissues': list of tissue names
      'matrix': dict[gene_symbol] -> list[tpm]  (aligned to tissues)
    """
    path = GTEX_PATH if GTEX_PATH.exists() else None
    if path is None and GTEX_GZ_PATH.exists():
        path = GTEX_GZ_PATH

    if path is None:
        return None

    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        _ = f.readline()  # "#1.2"
        _ = f.readline()  # "<nrows> <ncols>"
        header = f.readline().rstrip("\n").split("\t")
        tissues = header[2:]
        matrix: dict[str, list[float]] = {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            symbol = parts[1].strip()
            try:
                vals = [float(x) for x in parts[2:]]
            except ValueError:
                continue
            # Store by gene symbol; take max if duplicate (paralogs)
            prev = matrix.get(symbol)
            if prev is None or sum(vals) > sum(prev):
                matrix[symbol] = vals
    return {"tissues": tissues, "matrix": matrix}


@lru_cache(maxsize=1)
def _load_entrez_to_symbol() -> dict[str, str]:
    if not HGNC_PATH.exists():
        return {}
    m: dict[str, str] = {}
    with HGNC_PATH.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        i_sym = header.index("symbol")
        i_ent = header.index("entrez_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_sym, i_ent):
                continue
            ent = parts[i_ent].strip()
            sym = parts[i_sym].strip()
            if ent and sym:
                m[ent] = sym
    return m


def _tissue_indices(tissue_names: list[str], all_tissues: list[str]) -> list[int]:
    return [i for i, t in enumerate(all_tissues) if t in tissue_names]


def score_tissue_context(
    disease_name: str,
    gene_entities: Iterable[str],  # e.g. {'Gene::5599', 'Gene::7157', ...}
) -> dict:
    """
    Score how well a set of genes is expressed in the disease-relevant tissue(s).

    Returns:
        {
          "tissues": [names],
          "n_genes": int,
          "n_expressed": int,                 # genes with TPM >= 1 in at least one tissue
          "mean_tpm_in_tissue": float,
          "max_tpm_in_tissue": float,
          "tissue_specificity": float,        # higher = more specific
          "context_modifier": float in [0.85, 1.15]   # multiplier for pillar scores
        }
    """
    gtex = _load_gtex()
    if gtex is None:
        return {"context_modifier": 1.0, "tissues": [], "n_genes": 0}

    tissues = DISEASE_TISSUE_MAP.get(disease_name, ["Whole Blood"])
    tissue_idx = _tissue_indices(tissues, gtex["tissues"])
    if not tissue_idx:
        return {"context_modifier": 1.0, "tissues": tissues, "n_genes": 0}

    ent_to_sym = _load_entrez_to_symbol()

    tpms_per_gene: list[float] = []
    n_expressed = 0
    for ge in gene_entities:
        # Gene::<entrez>
        if "::" not in ge:
            continue
        entrez = ge.split("::", 1)[1]
        sym = ent_to_sym.get(entrez)
        if not sym:
            continue
        row = gtex["matrix"].get(sym)
        if row is None:
            continue
        tpm_vals = [row[i] for i in tissue_idx]
        max_tpm = max(tpm_vals)
        tpms_per_gene.append(max_tpm)
        if max_tpm >= 1.0:
            n_expressed += 1

    if not tpms_per_gene:
        return {"context_modifier": 1.0, "tissues": tissues, "n_genes": 0}

    mean_tpm = sum(tpms_per_gene) / len(tpms_per_gene)
    max_tpm = max(tpms_per_gene)

    # Tissue specificity: compare in-tissue mean to all-tissue mean
    # (cheap, not perfect). For every gene compute ratio.
    total_tissues = len(gtex["tissues"])
    all_means = 0.0
    in_means = 0.0
    for ge in gene_entities:
        if "::" not in ge:
            continue
        sym = ent_to_sym.get(ge.split("::", 1)[1])
        if not sym:
            continue
        row = gtex["matrix"].get(sym)
        if row is None:
            continue
        all_means += sum(row) / total_tissues
        in_means += sum(row[i] for i in tissue_idx) / max(len(tissue_idx), 1)
    specificity = 0.0
    if all_means > 0:
        specificity = min(1.0, max(0.0, (in_means - all_means) / (in_means + all_means + 1e-9) + 0.5))

    # Context modifier: boost if expressed + tissue-specific; damp if poorly expressed
    log_tpm = min(1.0, max(0.0, _log_scale(mean_tpm)))
    # Modifier in [0.85, 1.15] centered at 1.0 for neutral signal
    modifier = 0.85 + 0.30 * (0.6 * log_tpm + 0.4 * specificity)

    return {
        "tissues": tissues,
        "n_genes": len(tpms_per_gene),
        "n_expressed": n_expressed,
        "mean_tpm_in_tissue": round(mean_tpm, 2),
        "max_tpm_in_tissue": round(max_tpm, 2),
        "tissue_specificity": round(specificity, 3),
        "context_modifier": round(modifier, 3),
    }


def _log_scale(tpm: float) -> float:
    """Map TPM to [0,1] via log scaling (TPM ~100 ≈ 1.0)."""
    import math
    return math.log1p(tpm) / math.log1p(100)


if __name__ == "__main__":
    # Smoke test
    test_genes = {"Gene::5621", "Gene::5071", "Gene::6622"}  # PRNP, PARK2, SNCA — PD-relevant
    for disease in ["Parkinsons disease", "Alzheimers disease", "Heart failure", "Breast cancer"]:
        r = score_tissue_context(disease, test_genes)
        print(f"{disease:25s}  modifier={r.get('context_modifier')} tissues={r.get('tissues')}  genes={r.get('n_genes')}/{r.get('n_expressed')} expressed")
