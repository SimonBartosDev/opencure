"""
Search ClinicalTrials.gov for trial data supporting drug-disease predictions.

Uses the ClinicalTrials.gov API v2 (free, no auth needed).
"""

import requests

from opencure.config import CLINICALTRIALS_URL


def search_trials(
    drug_name: str,
    disease_name: str,
    max_results: int = 10,
) -> dict:
    """
    Search ClinicalTrials.gov for trials involving a drug-disease pair.

    Returns dict with: total_trials, trials (list), phases breakdown
    """
    params = {
        "query.intr": drug_name,
        "query.cond": disease_name,
        "pageSize": max_results,
        "format": "json",
    }

    try:
        resp = requests.get(CLINICALTRIALS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"drug": drug_name, "disease": disease_name, "total_trials": 0, "trials": [], "error": str(e)}

    studies = data.get("studies", [])
    total_raw = data.get("totalCount", len(studies))
    total = int(total_raw) if isinstance(total_raw, (int, float)) else len(studies)

    trials = []
    phase_counts = {}

    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        status = status_mod.get("overallStatus", "")
        phases = design.get("phases", [])
        phase_str = ", ".join(phases) if phases else "N/A"

        # Count phases
        for p in phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1

        # Enrollment
        enrollment = ""
        enroll_info = design.get("enrollmentInfo", {})
        if enroll_info:
            enrollment = str(enroll_info.get("count", ""))

        trials.append({
            "nct_id": nct_id,
            "title": title,
            "status": status,
            "phase": phase_str,
            "enrollment": enrollment,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        })

    return {
        "drug": drug_name,
        "disease": disease_name,
        "total_trials": total,
        "trials": trials,
        "phase_counts": phase_counts,
    }
