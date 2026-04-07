"""
Novelty assessment for drug repurposing candidates.

The core question: Is this prediction NOVEL (a potential new discovery)
or is it just re-discovering a known treatment?

A drug that already has 10,000 PubMed articles and 10 clinical trials
for a disease is NOT a repurposing candidate - it's the existing standard
of care. The interesting candidates are drugs that have:
- Strong computational support (high KG score, molecular similarity)
- But LIMITED or NO existing literature/trials for this specific disease
- They may be approved for OTHER diseases (that's what makes them repurposable)

This module computes a novelty score that measures how SURPRISING
a prediction is, given what's already known.
"""


def compute_novelty_score(report_dict: dict) -> dict:
    """
    Compute a novelty score for a drug-disease candidate.

    High novelty = strong computational prediction + weak existing evidence
    Low novelty = prediction that's already well-known

    Returns dict with:
        novelty_score: float 0-1 (1 = maximally novel)
        novelty_level: BREAKTHROUGH / NOVEL / EMERGING / KNOWN / ESTABLISHED
        interpretation: str
    """
    # Evidence of EXISTING knowledge (reduces novelty)
    pubmed = report_dict.get("pubmed_total", report_dict.get("pubmed_articles", 0))
    trials = report_dict.get("clinical_trials_total", 0)
    repurposing_articles = report_dict.get("pubmed_repurposing_total", report_dict.get("pubmed_repurposing", 0))
    max_citations = report_dict.get("max_citations", 0)
    faers = report_dict.get("faers_cooccurrences", 0)

    # Safety: ensure numeric types
    pubmed = int(pubmed) if isinstance(pubmed, (int, float)) else 0
    trials = int(trials) if isinstance(trials, (int, float)) else 0
    repurposing_articles = int(repurposing_articles) if isinstance(repurposing_articles, (int, float)) else 0
    max_citations = int(max_citations) if isinstance(max_citations, (int, float)) else 0
    faers = int(faers) if isinstance(faers, (int, float)) else 0

    # Evidence of COMPUTATIONAL support (increases novelty value)
    combined_score = report_dict.get("combined_score", 0)
    pillars = report_dict.get("pillars_hit", 0)
    shared_targets = report_dict.get("shared_target_count", report_dict.get("shared_targets", 0))
    direct_relations = len(report_dict.get("direct_relations", [])) if isinstance(report_dict.get("direct_relations"), list) else report_dict.get("direct_relations", 0)

    # Compute "existing knowledge" score (0 = unknown, 1 = extremely well known)
    knowledge_score = 0.0

    if pubmed > 1000:
        knowledge_score += 0.4  # Extensively studied
    elif pubmed > 100:
        knowledge_score += 0.3
    elif pubmed > 10:
        knowledge_score += 0.15
    elif pubmed > 0:
        knowledge_score += 0.05

    if trials > 5:
        knowledge_score += 0.3  # Many trials = established treatment
    elif trials > 0:
        knowledge_score += 0.15

    if max_citations > 500:
        knowledge_score += 0.15
    elif max_citations > 100:
        knowledge_score += 0.1

    if faers > 100:
        knowledge_score += 0.1  # Widely used in this context
    elif faers > 10:
        knowledge_score += 0.05

    knowledge_score = min(1.0, knowledge_score)

    # Compute "computational support" score
    comp_score = min(1.0, combined_score)  # Already 0-1 ish

    # Novelty = strong computation + weak existing knowledge
    # High novelty means: the AI says it should work, but nobody has published about it
    if knowledge_score > 0:
        novelty_score = comp_score * (1.0 - knowledge_score)
    else:
        novelty_score = comp_score  # No existing knowledge = maximum novelty potential

    # Classify
    if knowledge_score >= 0.6:
        level = "ESTABLISHED"
        interpretation = (
            f"Well-established drug-disease relationship ({pubmed} articles, "
            f"{trials} trials). Not a repurposing candidate."
        )
    elif knowledge_score >= 0.3:
        level = "KNOWN"
        interpretation = (
            f"Known association with moderate evidence ({pubmed} articles, "
            f"{trials} trials). Already being studied."
        )
    elif knowledge_score >= 0.1 and repurposing_articles > 0:
        level = "EMERGING"
        interpretation = (
            f"Emerging repurposing candidate. Some evidence exists ({pubmed} articles, "
            f"{repurposing_articles} mention repurposing) but not yet validated in trials."
        )
    elif knowledge_score > 0:
        level = "NOVEL"
        interpretation = (
            f"Novel prediction with minimal existing evidence ({pubmed} articles). "
            f"Computational support suggests this warrants investigation."
        )
    else:
        level = "BREAKTHROUGH"
        interpretation = (
            "No published evidence found for this drug-disease pair. "
            "Strong computational prediction suggests a potential new discovery. "
            "Requires experimental validation."
        )

    # MR bonus: causal genetic evidence makes novel predictions more credible
    mr_score = report_dict.get("mr_score", 0)
    if isinstance(mr_score, (int, float)) and mr_score > 0.5:
        novelty_score = min(1.0, novelty_score * 1.2)  # 20% bonus for strong MR

    return {
        "novelty_score": round(novelty_score, 3),
        "novelty_level": level,
        "knowledge_score": round(knowledge_score, 3),
        "computational_score": round(comp_score, 3),
        "interpretation": interpretation,
    }


def compute_literature_gap_score(disease_name: str) -> float:
    """
    Compute a literature gap score for a disease.

    Diseases with fewer total publications represent bigger knowledge gaps
    where computational predictions are most valuable.

    Returns: 0-1 score (1 = maximal gap, 0 = well-studied disease)
    """
    import requests

    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f'"{disease_name}"[Title/Abstract]',
            "rettype": "count",
            "retmode": "json",
        }
        resp = requests.get(base, params=params, timeout=10)
        if resp.status_code == 200:
            count = int(resp.json().get("esearchresult", {}).get("count", 0))

            # Scoring: fewer papers = bigger gap
            if count < 100:
                return 1.0    # Very understudied
            elif count < 1000:
                return 0.7
            elif count < 10000:
                return 0.4
            elif count < 100000:
                return 0.2
            else:
                return 0.0    # Extremely well-studied
    except Exception:
        pass

    return 0.3  # Default: moderate gap


def compute_mechanism_novelty(
    drug_name: str,
    disease_name: str,
    drug_targets: list,
) -> float:
    """
    Assess whether the drug's mechanism of action is novel for this disease.

    A drug with a known mechanism class (e.g., "kinase inhibitor") being applied
    to a disease where that class has never been tried = high mechanism novelty.

    Returns: 0-1 score (1 = completely novel mechanism for this disease)
    """
    import requests

    if not drug_targets:
        return 0.5  # Unknown

    # Check if ANY of the drug's targets have been studied for this disease
    targets_studied = 0
    targets_checked = 0

    for target in drug_targets[:5]:
        target_name = target if isinstance(target, str) else str(target)
        try:
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": f'"{target_name}" AND "{disease_name}"',
                "rettype": "count",
                "retmode": "json",
            }
            resp = requests.get(base, params=params, timeout=8)
            if resp.status_code == 200:
                count = int(resp.json().get("esearchresult", {}).get("count", 0))
                targets_checked += 1
                if count > 5:
                    targets_studied += 1
        except Exception:
            pass

        import time
        time.sleep(0.35)  # PubMed rate limit

    if targets_checked == 0:
        return 0.5

    # If no targets have been studied for this disease = novel mechanism
    ratio_novel = 1.0 - (targets_studied / targets_checked)
    return ratio_novel


def is_known_treatment(report_dict: dict) -> bool:
    """
    Check if a drug is likely an already-approved treatment for this disease.

    Returns True if the evidence strongly suggests this is a known treatment,
    not a repurposing candidate.
    """
    pubmed = report_dict.get("pubmed_total", report_dict.get("pubmed_articles", 0))
    trials = report_dict.get("clinical_trials_total", 0)

    # Safety: ensure numeric types
    pubmed = int(pubmed) if isinstance(pubmed, (int, float)) else 0
    trials = int(trials) if isinstance(trials, (int, float)) else 0

    # If there are many trials AND many papers, it's almost certainly
    # an existing treatment, not a repurposing candidate
    if trials >= 5 and pubmed >= 500:
        return True

    return False
