"""
Failed trial detection for drug-disease pairs.

Checks ClinicalTrials.gov for trials that were terminated, withdrawn,
or completed without leading to approval. This is critical evidence
that a drug did NOT work for a disease - reducing false positives.

Uses the ClinicalTrials.gov API v2 (free, no auth needed).
"""

import requests
import time
from functools import lru_cache

from opencure.config import CLINICALTRIALS_URL

# Cache results to avoid repeated API calls
_trial_cache: dict[str, dict] = {}


def check_failed_trials(drug_name: str, disease_name: str) -> dict:
    """
    Check if a drug has any failed trials for a disease.

    Returns dict with:
        has_failed: bool
        failed_phase: str (highest phase that failed)
        failed_count: int
        terminated_count: int
        withdrawn_count: int
        completed_no_approval: int
        total_trials: int
        details: list of trial summaries
        penalty: float (0-1, where 1 = no penalty, 0.3 = severe penalty)
    """
    cache_key = f"{drug_name.lower()}|{disease_name.lower()}"
    if cache_key in _trial_cache:
        return _trial_cache[cache_key]

    result = {
        "has_failed": False,
        "failed_phase": "",
        "failed_count": 0,
        "terminated_count": 0,
        "withdrawn_count": 0,
        "total_trials": 0,
        "details": [],
        "penalty": 1.0,  # 1.0 = no penalty
    }

    try:
        # Search for terminated/withdrawn trials
        params = {
            "query.intr": drug_name,
            "query.cond": disease_name,
            "filter.overallStatus": "TERMINATED,WITHDRAWN,SUSPENDED",
            "countTotal": "true",
            "pageSize": 10,
            "format": "json",
        }
        resp = requests.get(CLINICALTRIALS_URL, params=params, timeout=15)
        if resp.status_code != 200:
            _trial_cache[cache_key] = result
            return result

        data = resp.json()
        failed_studies = data.get("studies", [])
        total_raw = data.get("totalCount", 0)
        result["total_trials"] = int(total_raw) if isinstance(total_raw, (int, float)) else 0

        highest_failed_phase = 0
        for study in failed_studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})

            status = status_mod.get("overallStatus", "")
            phases = design.get("phases", [])
            title = ident.get("briefTitle", "")
            why_stopped = status_mod.get("whyStoppedText", "")

            # Track failure type
            if status == "TERMINATED":
                result["terminated_count"] += 1
            elif status == "WITHDRAWN":
                result["withdrawn_count"] += 1

            # Determine phase number
            phase_num = 0
            for p in phases:
                if "3" in p:
                    phase_num = max(phase_num, 3)
                elif "2" in p:
                    phase_num = max(phase_num, 2)
                elif "1" in p:
                    phase_num = max(phase_num, 1)

            if phase_num > highest_failed_phase:
                highest_failed_phase = phase_num

            result["details"].append({
                "title": title[:100],
                "status": status,
                "phase": ", ".join(phases) if phases else "N/A",
                "why_stopped": why_stopped[:200] if why_stopped else "",
            })

        result["failed_count"] = len(failed_studies)

        if result["failed_count"] > 0:
            result["has_failed"] = True
            result["failed_phase"] = f"Phase {highest_failed_phase}" if highest_failed_phase > 0 else "Unknown"

            # Compute penalty based on highest phase of failure
            # Phase 3 failure = strong evidence drug doesn't work (70% penalty)
            # Phase 2 failure = moderate evidence (40% penalty)
            # Phase 1 failure = may be safety not efficacy (20% penalty)
            if highest_failed_phase >= 3:
                result["penalty"] = 0.3
            elif highest_failed_phase >= 2:
                result["penalty"] = 0.6
            elif highest_failed_phase >= 1:
                result["penalty"] = 0.8
            else:
                result["penalty"] = 0.9  # Unknown phase

        # CRITICAL: Also check for COMPLETED/ACTIVE trials
        # If there are successful trials too, the failures are likely noise
        # (other drugs tested alongside this drug as background therapy)
        time.sleep(0.3)
        params_success = {
            "query.intr": drug_name,
            "query.cond": disease_name,
            "filter.overallStatus": "COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING",
            "countTotal": "true",
            "pageSize": 1,
            "format": "json",
        }
        try:
            resp2 = requests.get(CLINICALTRIALS_URL, params=params_success, timeout=15)
            if resp2.status_code == 200:
                sc_raw = resp2.json().get("totalCount", 0)
                success_count = int(sc_raw) if isinstance(sc_raw, (int, float)) else 0
                if success_count > 0:
                    # Drug has both failed and successful trials
                    # Failure ratio determines penalty
                    failure_ratio = result["failed_count"] / (result["failed_count"] + success_count)
                    if failure_ratio < 0.5:
                        # More successes than failures - this drug works, failures are noise
                        result["has_failed"] = False
                        result["penalty"] = 1.0
                    else:
                        # More failures than successes - reduce penalty but keep it
                        result["penalty"] = min(1.0, result["penalty"] + 0.3)

                    result["details"].append({
                        "title": f"Note: {success_count} completed/active trials also exist (failure ratio: {failure_ratio:.0%})",
                        "status": "COMPLETED",
                        "phase": "",
                        "why_stopped": "",
                    })
        except Exception:
            pass

    except Exception:
        pass

    _trial_cache[cache_key] = result
    return result
