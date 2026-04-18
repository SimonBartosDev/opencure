"""
Generate comprehensive evidence reports for drug repurposing candidates.

Combines:
- DRKG knowledge graph evidence (shared targets, direct relations)
- PubMed literature evidence (publication counts, key papers)
- Semantic Scholar (citation-weighted papers, 200M+ corpus)
- ClinicalTrials.gov data (existing trials, phases)
- BioGPT AI hypothesis generation (mechanistic explanations)
- PubMedBERT abstract analysis (relation classification)

Produces a confidence assessment for each candidate.
"""

import time
from dataclasses import dataclass, field

from opencure.evidence.pubmed import search_drug_disease_evidence
from opencure.evidence.clinical_trials import search_trials
from opencure.evidence.semantic_scholar import search_drug_disease_papers
from opencure.evidence.literature_ai import (
    generate_hypothesis,
    analyze_abstract_for_relation,
)
from opencure.evidence.faers import compute_repurposing_signal
from opencure.evidence.gene_signatures import check_signature_reversal
from opencure.evidence.novelty import compute_novelty_score, is_known_treatment
from opencure.data.failed_trials import check_failed_trials
from opencure.data.opentargets import search_disease, get_disease_targets


@dataclass
class EvidenceReport:
    drug_name: str
    drug_id: str
    disease_name: str
    disease_entity: str

    # Scores from pipeline (all 11 pillars)
    combined_score: float = 0.0
    pillars_hit: int = 0
    # Pillar 3a: TransE
    transe_rank: int = 0
    transe_score: float = 0.0
    # Pillar 3b: RotatE/PyKEEN
    pykeen_score: float = 0.0
    pykeen_rank: int = 0
    # Pillar 6: TxGNN
    txgnn_score: float = 0.0
    txgnn_rank: int = 0
    # Pillar 1a: Fingerprint similarity
    mol_similarity: float = 0.0
    similar_to: str = ""
    # Pillar 1b: ChemBERTa learned similarity
    mol_emb_similarity: float = 0.0
    mol_emb_similar_to: str = ""
    # Pillar 4: Gene signature
    gene_sig_score: float = 0.0
    gene_sig_rank: int = 0
    # Pillar 5: Network proximity
    proximity_score: float = 0.0
    proximity_distance: float = 0.0
    # Pillar 9: ADMET
    admet_score: float = 0.0
    admet_flags: str = ""
    # Pillar 10: PrimeKG
    primekg_score: float = 0.0
    # Pillar 11: DeepPurpose DTI
    dti_score: float = 0.0
    dti_best_target: str = ""

    # v3: Group-level scores (from grouped_combiner.py)
    kg_group_score: float = 0.0
    structural_group_score: float = 0.0
    network_group_score: float = 0.0
    txgnn_group_score: float = 0.0
    mr_group_score: float = 0.0
    admet_multiplier: float = 0.7
    efficacy_score: float = 0.0
    groups_hit: int = 0

    # Graph evidence
    direct_relations: list = field(default_factory=list)
    shared_targets: list = field(default_factory=list)
    shared_target_count: int = 0

    # Literature evidence
    pubmed_total: int = 0
    pubmed_treatment_total: int = 0
    pubmed_repurposing_total: int = 0
    key_papers: list = field(default_factory=list)
    repurposing_papers: list = field(default_factory=list)

    # Clinical trial evidence
    clinical_trials_total: int = 0
    clinical_trials: list = field(default_factory=list)
    trial_phases: dict = field(default_factory=dict)

    # Semantic Scholar evidence
    semantic_scholar_papers: list = field(default_factory=list)
    most_cited_paper: dict = field(default_factory=dict)
    max_citations: int = 0

    # FAERS real-world evidence
    faers_signal: str = ""  # strong/moderate/weak/none
    faers_total_reports: int = 0
    faers_cooccurrences: int = 0
    faers_interpretation: str = ""

    # Gene signature reversal
    signature_reversal_found: bool = False
    signature_reversal_rank: int = 0
    signature_reversal_score: float = 0.0
    signature_top_reversers: list = field(default_factory=list)
    signature_interpretation: str = ""

    # AI-generated evidence
    ai_hypothesis: str = ""
    abstract_analyses: list = field(default_factory=list)
    ai_positive_signals: int = 0

    # Failed trial data
    has_failed_trial: bool = False
    failed_trial_phase: str = ""
    failed_trial_count: int = 0
    failed_trial_penalty: float = 1.0
    failed_trial_details: list = field(default_factory=list)

    # Mendelian Randomization (causal evidence)
    mr_score: float = 0.0
    mr_genetic_targets: int = 0

    # LLM Explanation
    mechanistic_hypothesis: str = ""
    kg_paths_text: str = ""
    validation_experiments: list = field(default_factory=list)

    # Novelty assessment
    novelty_score: float = 0.0
    novelty_level: str = ""  # BREAKTHROUGH, NOVEL, EMERGING, KNOWN, ESTABLISHED
    is_known_treatment: bool = False

    # Confidence assessment
    confidence: str = ""  # HIGH, MEDIUM, LOW, NOVEL
    confidence_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "drug_id": self.drug_id,
            "disease_name": self.disease_name,
            "combined_score": self.combined_score,
            "pillars_hit": self.pillars_hit,
            # All 11 pillar scores
            "transe_rank": self.transe_rank,
            "transe_score": self.transe_score,
            "pykeen_score": self.pykeen_score,
            "pykeen_rank": self.pykeen_rank,
            "txgnn_score": self.txgnn_score,
            "txgnn_rank": self.txgnn_rank,
            "mol_similarity": self.mol_similarity,
            "similar_to": self.similar_to,
            "mol_emb_similarity": self.mol_emb_similarity,
            "mol_emb_similar_to": self.mol_emb_similar_to,
            "gene_sig_score": self.gene_sig_score,
            "gene_sig_rank": self.gene_sig_rank,
            "proximity_score": self.proximity_score,
            "proximity_distance": self.proximity_distance,
            "admet_score": self.admet_score,
            "admet_flags": self.admet_flags,
            "primekg_score": self.primekg_score,
            "dti_score": self.dti_score,
            "dti_best_target": self.dti_best_target,
            # v3 group scores
            "kg_group_score": self.kg_group_score,
            "structural_group_score": self.structural_group_score,
            "network_group_score": self.network_group_score,
            "txgnn_group_score": self.txgnn_group_score,
            "mr_group_score": self.mr_group_score,
            "admet_multiplier": self.admet_multiplier,
            "efficacy_score": self.efficacy_score,
            "groups_hit": self.groups_hit,
            "direct_relations": self.direct_relations,
            "shared_target_count": self.shared_target_count,
            "shared_targets": self.shared_targets[:10],
            "pubmed_total": self.pubmed_total,
            "pubmed_treatment_total": self.pubmed_treatment_total,
            "pubmed_repurposing_total": self.pubmed_repurposing_total,
            "key_papers": [p for p in self.key_papers[:5]],
            "repurposing_papers": [p for p in self.repurposing_papers[:3]],
            "semantic_scholar_papers": self.semantic_scholar_papers[:5],
            "most_cited_paper": self.most_cited_paper,
            "max_citations": self.max_citations,
            "faers_signal": self.faers_signal,
            "faers_total_reports": self.faers_total_reports,
            "faers_cooccurrences": self.faers_cooccurrences,
            "faers_interpretation": self.faers_interpretation,
            "signature_reversal_found": self.signature_reversal_found,
            "signature_reversal_rank": self.signature_reversal_rank,
            "signature_top_reversers": self.signature_top_reversers[:5],
            "signature_interpretation": self.signature_interpretation,
            "ai_hypothesis": self.ai_hypothesis,
            "abstract_analyses": self.abstract_analyses[:3],
            "has_failed_trial": self.has_failed_trial,
            "failed_trial_phase": self.failed_trial_phase,
            "failed_trial_count": self.failed_trial_count,
            "failed_trial_penalty": self.failed_trial_penalty,
            "mr_score": self.mr_score,
            "mr_genetic_targets": self.mr_genetic_targets,
            "mechanistic_hypothesis": self.mechanistic_hypothesis,
            "kg_paths_text": self.kg_paths_text,
            "validation_experiments": self.validation_experiments,
            "novelty_score": self.novelty_score,
            "novelty_level": self.novelty_level,
            "is_known_treatment": self.is_known_treatment,
            "clinical_trials_total": self.clinical_trials_total,
            "clinical_trials": self.clinical_trials[:5],
            "trial_phases": self.trial_phases,
            "confidence": self.confidence,
            "confidence_reasons": self.confidence_reasons,
        }


def generate_evidence_report(
    result: dict,
    disease_name: str,
) -> EvidenceReport:
    """
    Generate a full evidence report for a single drug-disease prediction.

    Args:
        result: Search result dict from opencure.search
        disease_name: Human-readable disease name (for API queries)

    Returns:
        EvidenceReport with all evidence gathered
    """
    report = EvidenceReport(
        drug_name=result["drug_name"],
        drug_id=result["drug_id"],
        disease_name=disease_name,
        disease_entity=result.get("disease_entity", ""),
        combined_score=result.get("combined_score", 0),
        pillars_hit=result.get("pillars_hit", 0),
        # Copy ALL pillar scores from search result
        transe_rank=result.get("transe_rank", 0),
        transe_score=result.get("transe_score", 0),
        pykeen_score=result.get("pykeen_score", 0),
        pykeen_rank=result.get("pykeen_rank", 0),
        txgnn_score=result.get("txgnn_score", 0),
        txgnn_rank=result.get("txgnn_rank", 0),
        mol_similarity=result.get("mol_similarity", 0),
        similar_to=result.get("similar_to", ""),
        mol_emb_similarity=result.get("mol_emb_similarity", 0),
        mol_emb_similar_to=result.get("mol_emb_similar_to", ""),
        gene_sig_score=result.get("gene_sig_score", 0),
        gene_sig_rank=result.get("gene_sig_rank", 0),
        proximity_score=result.get("proximity_score", 0),
        proximity_distance=result.get("proximity_distance", 0),
        admet_score=result.get("admet_score", 0),
        admet_flags=result.get("admet_flags", ""),
        primekg_score=result.get("primekg_score", 0),
        dti_score=result.get("dti_score", 0),
        dti_best_target=result.get("dti_best_target", ""),
        # v3 group-level scores
        kg_group_score=result.get("kg_group_score", 0),
        structural_group_score=result.get("structural_group_score", 0),
        network_group_score=result.get("network_group_score", 0),
        txgnn_group_score=result.get("txgnn_group_score", 0),
        mr_group_score=result.get("mr_group_score", 0),
        admet_multiplier=result.get("admet_multiplier", 0.7),
        efficacy_score=result.get("efficacy_score", 0),
        groups_hit=result.get("groups_hit", 0),
    )

    # Graph evidence (already in result)
    graph_evidence = result.get("evidence", [])
    for e in graph_evidence:
        if e.startswith("Direct:"):
            report.direct_relations.append(e)
        elif e.startswith("Shared targets:"):
            report.shared_targets = e.replace("Shared targets: ", "").split(", ")
            report.shared_target_count = len(report.shared_targets)
            # Handle "+N more" suffix
            for t in report.shared_targets:
                if "more" in t:
                    try:
                        extra = int(t.replace("(+", "").replace(" more)", ""))
                        report.shared_target_count = len(report.shared_targets) - 1 + extra
                    except ValueError:
                        pass

    # PubMed evidence
    drug_query_name = result["drug_name"]
    if drug_query_name == result["drug_id"]:
        # Name wasn't resolved, skip PubMed search
        pass
    else:
        pubmed = search_drug_disease_evidence(drug_query_name, disease_name)
        report.pubmed_total = pubmed.get("total_articles", 0)
        report.pubmed_treatment_total = pubmed.get("total_treatment_articles", 0)
        report.pubmed_repurposing_total = pubmed.get("total_repurposing_articles", 0)
        report.key_papers = pubmed.get("articles", [])
        report.repurposing_papers = pubmed.get("repurposing_articles", [])
        time.sleep(0.5)

    # ClinicalTrials.gov evidence
    if drug_query_name != result["drug_id"]:
        trials = search_trials(drug_query_name, disease_name)
        total_trials = trials.get("total_trials", 0)
        report.clinical_trials_total = int(total_trials) if not isinstance(total_trials, (list, dict)) else 0
        report.clinical_trials = trials.get("trials", [])
        report.trial_phases = trials.get("phase_counts", {})
        time.sleep(0.3)

    # Semantic Scholar evidence (citation-weighted)
    if drug_query_name != result["drug_id"]:
        try:
            s2_data = search_drug_disease_papers(drug_query_name, disease_name, limit=5)
            s2_papers = s2_data.get("papers", []) + s2_data.get("repurposing_papers", [])

            # Find most cited paper
            for p in s2_papers:
                citations = p.get("citationCount", 0) or 0
                paper_info = {
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "citations": citations,
                    "url": p.get("url", ""),
                    "authors": ", ".join(
                        a.get("name", "") for a in (p.get("authors") or [])[:3]
                    ),
                }
                report.semantic_scholar_papers.append(paper_info)
                if citations > report.max_citations:
                    report.max_citations = citations
                    report.most_cited_paper = paper_info

            time.sleep(0.5)
        except Exception:
            pass

    # Failed trial check
    if drug_query_name != result["drug_id"]:
        try:
            trial_check = check_failed_trials(drug_query_name, disease_name)
            report.has_failed_trial = trial_check.get("has_failed", False)
            report.failed_trial_phase = trial_check.get("failed_phase", "")
            report.failed_trial_count = trial_check.get("failed_count", 0)
            report.failed_trial_penalty = trial_check.get("penalty", 1.0)
            report.failed_trial_details = trial_check.get("details", [])
            time.sleep(0.3)
        except Exception:
            pass

    # FAERS real-world evidence
    if drug_query_name != result["drug_id"]:
        try:
            faers = compute_repurposing_signal(drug_query_name, disease_name)
            report.faers_signal = faers.get("signal_strength", "none")
            report.faers_total_reports = faers.get("total_drug_reports", 0)
            report.faers_cooccurrences = faers.get("disease_cooccurrences", 0)
            report.faers_interpretation = faers.get("interpretation", "")
            time.sleep(0.3)
        except Exception:
            pass

    # Gene signature reversal (L1000CDS2)
    if drug_query_name != result["drug_id"]:
        try:
            sig = check_signature_reversal(drug_query_name, disease_name)
            report.signature_reversal_found = sig.get("found", False)
            report.signature_reversal_rank = sig.get("rank", 0)
            report.signature_reversal_score = sig.get("score", 0)
            report.signature_top_reversers = sig.get("top_reversers", [])
            report.signature_interpretation = sig.get("interpretation", "")
            time.sleep(0.3)
        except Exception:
            pass

    # AI abstract analysis (PubMedBERT-based relation classification)
    abstracts_to_analyze = []
    for paper in report.key_papers[:3]:
        abstract = paper.get("abstract_snippet", "")
        if abstract and len(abstract) > 50:
            abstracts_to_analyze.append(abstract)

    for abstract in abstracts_to_analyze:
        try:
            analysis = analyze_abstract_for_relation(abstract, drug_query_name, disease_name)
            report.abstract_analyses.append(analysis)
            if analysis.get("relation_type") == "treats":
                report.ai_positive_signals += 1
        except Exception:
            pass

    # BioGPT hypothesis generation
    if drug_query_name != result["drug_id"]:
        try:
            # Build context from what we've found
            context_parts = []
            if report.shared_target_count > 0:
                context_parts.append(f"they share {report.shared_target_count} gene targets")
            if report.direct_relations:
                context_parts.append("a direct treatment relationship exists in the knowledge graph")
            if report.pubmed_total > 0:
                context_parts.append(f"{report.pubmed_total} research articles mention both")
            context = ", ".join(context_parts) if context_parts else ""

            report.ai_hypothesis = generate_hypothesis(drug_query_name, disease_name, context)
        except Exception:
            pass

    # MR causal evidence (from search result)
    report.mr_score = result.get("mr_score", 0) or 0
    report.mr_genetic_targets = result.get("mr_genetic_targets", 0) or 0

    # Confidence assessment
    report.confidence, report.confidence_reasons = _assess_confidence(report)

    # Novelty assessment
    novelty = compute_novelty_score(report.to_dict())
    report.novelty_score = novelty["novelty_score"]
    report.novelty_level = novelty["novelty_level"]
    report.is_known_treatment = is_known_treatment(report.to_dict())

    if report.is_known_treatment:
        report.confidence_reasons.append(
            "NOTE: This appears to be an already-approved treatment, not a repurposing candidate"
        )

    return report


def _assess_confidence(report: EvidenceReport) -> tuple[str, list[str]]:
    """
    Assess confidence level for a drug-disease repurposing candidate.

    Returns (confidence_level, list_of_reasons)
    """
    reasons = []
    score = 0

    # Clinical trials are the strongest evidence
    if report.clinical_trials_total > 0:
        score += 3
        reasons.append(f"{report.clinical_trials_total} clinical trial(s) found on ClinicalTrials.gov")
        if "PHASE3" in report.trial_phases or "PHASE 3" in str(report.trial_phases):
            score += 2
            reasons.append("Phase 3 trial exists")

    # PubMed evidence
    if report.pubmed_total > 50:
        score += 2
        reasons.append(f"{report.pubmed_total} PubMed articles on this drug-disease pair")
    elif report.pubmed_total > 10:
        score += 1
        reasons.append(f"{report.pubmed_total} PubMed articles found")
    elif report.pubmed_total > 0:
        reasons.append(f"{report.pubmed_total} PubMed article(s) - limited literature")

    if report.pubmed_repurposing_total > 0:
        score += 1
        reasons.append(f"{report.pubmed_repurposing_total} article(s) specifically mention repurposing/repositioning")

    # Direct DRKG relations
    if report.direct_relations:
        score += 1
        reasons.append(f"Direct treatment relation in knowledge graph ({len(report.direct_relations)} source(s))")

    # Shared targets
    if report.shared_target_count > 20:
        score += 1
        reasons.append(f"{report.shared_target_count} shared gene targets between drug and disease")
    elif report.shared_target_count > 5:
        reasons.append(f"{report.shared_target_count} shared gene targets")

    # Multi-pillar support
    if report.pillars_hit > 1:
        score += 1
        reasons.append(f"Supported by {report.pillars_hit} independent AI pillars")

    # Molecular similarity
    if report.mol_similarity > 0.5:
        reasons.append(f"High molecular similarity ({report.mol_similarity:.2f}) to {report.similar_to}")

    # Failed trial penalty (CRITICAL - reduces false positives)
    if report.has_failed_trial:
        if report.failed_trial_penalty <= 0.3:
            score -= 3
            reasons.append(f"WARNING: Drug FAILED {report.failed_trial_phase} trial(s) for this disease ({report.failed_trial_count} terminated/withdrawn)")
        elif report.failed_trial_penalty <= 0.6:
            score -= 2
            reasons.append(f"CAUTION: Drug had terminated/withdrawn trials ({report.failed_trial_phase}) for this disease")
        else:
            score -= 1
            reasons.append(f"Note: Drug had early-phase trial failures for this disease")

    # FAERS real-world evidence (independent of computational predictions)
    if report.faers_signal == "strong":
        score += 2
        reasons.append(f"Strong FAERS signal: {report.faers_cooccurrences} adverse event reports link drug to disease ({report.faers_total_reports} total reports)")
    elif report.faers_signal == "moderate":
        score += 1
        reasons.append(f"Moderate FAERS signal: {report.faers_cooccurrences} adverse event reports mention drug-disease co-occurrence")

    # Gene signature reversal (biological mechanism evidence)
    if report.signature_reversal_found and report.signature_reversal_rank <= 10:
        score += 2
        reasons.append(f"Drug ranks #{report.signature_reversal_rank} for reversing disease gene signature (L1000CDS2 biological evidence)")
    elif report.signature_reversal_found and report.signature_reversal_rank <= 50:
        score += 1
        reasons.append(f"Drug ranks #{report.signature_reversal_rank} for reversing disease gene signature")

    # Semantic Scholar citation evidence
    if report.max_citations > 100:
        score += 1
        reasons.append(f"Highly cited paper ({report.max_citations} citations): \"{report.most_cited_paper.get('title', '')[:60]}...\"")
    elif report.max_citations > 20:
        reasons.append(f"Supporting paper with {report.max_citations} citations")

    # AI abstract analysis
    if report.ai_positive_signals > 0:
        score += 1
        reasons.append(f"AI analysis of {report.ai_positive_signals} abstract(s) found positive treatment signals")

    # BioGPT hypothesis
    if report.ai_hypothesis:
        reasons.append(f"AI-generated hypothesis available")

    # Mendelian Randomization (causal genetic evidence)
    if report.mr_score > 0.7:
        score += 2
        reasons.append(f"Strong causal genetic evidence (MR score: {report.mr_score:.2f}, {report.mr_genetic_targets} genetic targets)")
    elif report.mr_score > 0.3:
        score += 1
        reasons.append(f"Moderate genetic evidence supporting causality (MR score: {report.mr_score:.2f})")

    # Classify
    if score >= 5:
        return "HIGH", reasons
    elif score >= 3:
        return "MEDIUM", reasons
    elif score >= 1:
        return "LOW", reasons
    else:
        if report.combined_score > 0.9:
            reasons.append("Strong computational prediction but no published evidence found - potentially novel discovery")
            return "NOVEL", reasons
        return "LOW", reasons


def generate_batch_reports(
    results: list[dict],
    disease_name: str,
    max_candidates: int = 10,
) -> list[EvidenceReport]:
    """Generate evidence reports for top candidates."""
    reports = []
    for result in results[:max_candidates]:
        print(f"  Gathering evidence for {result['drug_name']} ({result['drug_id']})...")
        report = generate_evidence_report(result, disease_name)
        reports.append(report)
    return reports
