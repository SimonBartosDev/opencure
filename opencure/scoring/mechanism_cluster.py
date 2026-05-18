"""
Cross-disease mechanism clustering.

After the full 61-disease screen runs, this module identifies drugs that
surface as candidates across multiple *unrelated* diseases via a shared
molecular mechanism. These are OpenCure's polypharmacology picks —
candidates for "one drug, many indications" repositioning, particularly
valuable for pandemic preparedness (a drug that hits infection + sepsis +
ARDS targets a shared biology no single-disease screen would identify).

Input:  per-disease result JSONs under experiments/results/
Output: experiments/results/mechanism_clusters.json + Markdown summary.

Algorithm (per drug):
  1. Collect every disease where this drug scored >= MIN_SCORE
  2. Group diseases by ICD-like category (if known) to penalize clusters
     of trivially-related diseases (e.g. several cancers from the same
     organ system).
  3. Score cluster as:
         n_diseases × mean_score × category_diversity × pathway_coherence
     - n_diseases: raw repurposing breadth
     - mean_score: average combined_score across hits
     - category_diversity: 1 - max(fraction any single category)
     - pathway_coherence: fraction of hit diseases whose disease_genes
       overlap with the drug's targets (proxy for shared mechanism)
  4. Rank clusters; emit top N.

The cluster page on the dashboard lets a viewer see (e.g.)
    Sirolimus (mTOR inhibitor)
      → Malaria, Tuberculosis, HIV, Leishmaniasis
      shared mechanism: autophagy induction via mTOR

which is a narrative no single-disease page can show.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Minimum per-disease score for the drug to count as a "hit" in that disease
MIN_SCORE_THRESHOLD: float = 0.35

# Minimum number of diseases a drug must hit to form a cluster
MIN_CLUSTER_SIZE: int = 3


# Coarse disease category map. Used to compute category_diversity so that a
# drug that hits 5 cancers doesn't out-rank a drug that hits 3 completely
# unrelated conditions. Only categories we care about for the narrative.
DISEASE_CATEGORIES: dict[str, str] = {
    # Infectious
    "Malaria": "infectious",
    "Tuberculosis": "infectious",
    "HIV": "infectious",
    "Hepatitis C": "infectious",
    "Dengue": "infectious",
    "COVID-19": "infectious",
    "Schistosomiasis": "infectious",
    "Leishmaniasis": "infectious",
    "Chagas disease": "infectious",
    # Neurodegen
    "Alzheimers disease": "neurodegen",
    "Parkinsons disease": "neurodegen",
    "Huntingtons disease": "neurodegen",
    "Amyotrophic lateral sclerosis": "neurodegen",
    "Multiple sclerosis": "neurodegen",
    # Psychiatric
    "Depression": "psychiatric",
    "Anxiety": "psychiatric",
    "Bipolar disorder": "psychiatric",
    "Schizophrenia": "psychiatric",
    "Epilepsy": "neurological",
    # Cardiovascular
    "Atrial fibrillation": "cardio",
    "Heart failure": "cardio",
    "Hypertension": "cardio",
    "Coronary artery disease": "cardio",
    "Atherosclerosis": "cardio",
    "Pulmonary hypertension": "cardio",
    # Cancer
    "Breast cancer": "oncology",
    "Colorectal cancer": "oncology",
    "Lung cancer": "oncology",
    "Pancreatic cancer": "oncology",
    "Prostate cancer": "oncology",
    "Ovarian cancer": "oncology",
    "Leukemia": "oncology",
    "Lymphoma": "oncology",
    "Melanoma": "oncology",
    "Glioblastoma": "oncology",
    "Multiple myeloma": "oncology",
    # Metabolic
    "Type 2 diabetes": "metabolic",
    "Obesity": "metabolic",
    "Osteoporosis": "metabolic",
    # Autoimmune / inflammatory
    "Rheumatoid arthritis": "autoimmune",
    "Lupus": "autoimmune",
    "Psoriasis": "autoimmune",
    "Inflammatory bowel disease": "autoimmune",
    "Crohns disease": "autoimmune",
    "Ulcerative colitis": "autoimmune",
    "Asthma": "autoimmune",
    "COPD": "respiratory",
    "Cystic fibrosis": "respiratory",
    "Idiopathic pulmonary fibrosis": "respiratory",
    # Rare/genetic
    "Sickle cell disease": "rare_genetic",
    "Gaucher disease": "rare_genetic",
    "Fabry disease": "rare_genetic",
    "Duchenne muscular dystrophy": "rare_genetic",
    "Ehlers-Danlos syndrome": "rare_genetic",
    "Fragile X syndrome": "rare_genetic",
    "Marfan syndrome": "rare_genetic",
    "Neurofibromatosis": "rare_genetic",
    # Other
    "Chronic kidney disease": "renal",
    "Liver cirrhosis": "hepatic",
    "Sepsis": "infectious",
    "Endometriosis": "gynecological",
}


def _category_for(disease_name: str) -> str:
    return DISEASE_CATEGORIES.get(disease_name, "other")


def _category_diversity(categories: list[str]) -> float:
    if not categories:
        return 0.0
    from collections import Counter
    c = Counter(categories)
    max_frac = max(c.values()) / len(categories)
    return round(1.0 - max_frac, 3)


def compute_clusters(
    results_dir: Path,
    min_score: float = MIN_SCORE_THRESHOLD,
    min_cluster: int = MIN_CLUSTER_SIZE,
) -> list[dict]:
    """
    Build mechanism clusters across disease result JSONs.

    Returns list of cluster dicts sorted by cluster_strength desc.
    """
    drug_hits: dict[str, list[dict]] = defaultdict(list)

    for jf in sorted(results_dir.glob("*.json")):
        if any(x in jf.name for x in ("opencure", "screening", "novel", "mechanism")):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        cands = data.get("candidates", []) if isinstance(data, dict) else data
        if not cands:
            continue
        disease_name = jf.stem.replace("_", " ")
        for c in cands:
            score = c.get("combined_score", 0)
            if score < min_score:
                continue
            drug_hits[c.get("drug_name", "?")].append({
                "disease": disease_name,
                "score": score,
                "category": _category_for(disease_name),
                "rank_in_disease": (c.get("rank") or 0),
                "drug_id": c.get("drug_id", ""),
                "pillars_hit": c.get("pillars_hit", 0),
                "shared_targets": c.get("shared_target_count", 0),
                "mr_genetic_targets": c.get("mr_genetic_targets", 0),
                "top_path": c.get("mechanistic_hypothesis", ""),
            })

    clusters: list[dict] = []
    for drug_name, hits in drug_hits.items():
        if len(hits) < min_cluster:
            continue
        diseases = [h["disease"] for h in hits]
        categories = [h["category"] for h in hits]
        mean_score = sum(h["score"] for h in hits) / len(hits)
        diversity = _category_diversity(categories)
        # Pathway coherence: fraction of hits with any shared_targets or MR
        coherent = sum(1 for h in hits if (h["shared_targets"] or h["mr_genetic_targets"]))
        pathway_coherence = coherent / len(hits)

        strength = len(hits) * mean_score * (0.5 + 0.5 * diversity) * (0.5 + 0.5 * pathway_coherence)

        clusters.append({
            "drug_name": drug_name,
            "drug_id": hits[0]["drug_id"],
            "n_diseases": len(hits),
            "diseases": diseases,
            "categories": categories,
            "category_diversity": diversity,
            "pathway_coherence": round(pathway_coherence, 3),
            "mean_score": round(mean_score, 3),
            "cluster_strength": round(strength, 3),
            "hits": hits,
        })

    clusters.sort(key=lambda c: c["cluster_strength"], reverse=True)
    return clusters


def render_markdown(clusters: list[dict], top: int = 20) -> str:
    lines = [
        "# OpenCure cross-disease mechanism clusters",
        "",
        "Drugs that appear as high-score candidates across multiple unrelated ",
        "diseases. Ranked by cluster_strength = N × mean_score × category_diversity × pathway_coherence.",
        "",
        "| Drug | N diseases | Diversity | Mean score | Strength | Categories |",
        "|---|---|---|---|---|---|",
    ]
    for c in clusters[:top]:
        cats = ", ".join(sorted(set(c["categories"])))
        lines.append(
            f"| {c['drug_name']} | {c['n_diseases']} | {c['category_diversity']} "
            f"| {c['mean_score']} | {c['cluster_strength']} | {cats} |"
        )
    lines.append("")
    lines.append("## Details")
    for c in clusters[:top]:
        lines.append(f"\n### {c['drug_name']}  (strength {c['cluster_strength']})")
        lines.append(f"Hits ({c['n_diseases']}): " + ", ".join(h["disease"] for h in c["hits"]))
        paths = [h["top_path"] for h in c["hits"] if h.get("top_path")]
        if paths:
            lines.append("")
            lines.append("Representative mechanism path:")
            lines.append(f"> {paths[0]}")
    return "\n".join(lines)


def main() -> None:
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results")
    clusters = compute_clusters(d)
    out_json = d / "mechanism_clusters.json"
    out_md = d / "mechanism_clusters.md"
    out_json.write_text(json.dumps(clusters, indent=2))
    out_md.write_text(render_markdown(clusters))
    print(f"Found {len(clusters)} clusters (>= {MIN_CLUSTER_SIZE} diseases, >= {MIN_SCORE_THRESHOLD} score)")
    for c in clusters[:10]:
        print(f"  {c['cluster_strength']:>6.2f}  {c['drug_name']:25s}  n={c['n_diseases']}  div={c['category_diversity']}  mean={c['mean_score']}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
