"""Query Open Targets Platform API for disease-target associations."""

import requests

from opencure.config import OPENTARGETS_URL


def search_disease(disease_name: str) -> list[dict]:
    """
    Search Open Targets for a disease by name.

    Returns list of dicts with: id (EFO ID), name, description
    """
    query = """
    query SearchDisease($queryString: String!) {
      search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 5}) {
        hits {
          id
          name
          description
          entity
        }
      }
    }
    """
    try:
        resp = requests.post(
            OPENTARGETS_URL,
            json={"query": query, "variables": {"queryString": disease_name}},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json()["data"]["search"]["hits"]
        return [{"id": h["id"], "name": h["name"], "description": h.get("description", "")} for h in hits]
    except Exception as e:
        print(f"  [WARN] Open Targets search failed: {e}")
        return []


def get_disease_targets(efo_id: str, limit: int = 25) -> list[dict]:
    """
    Get top targets associated with a disease from Open Targets.

    Returns list of dicts with: target_id, gene_symbol, score
    """
    query = """
    query DiseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
          count
          rows {
            target {
              id
              approvedSymbol
              approvedName
            }
            score
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            OPENTARGETS_URL,
            json={"query": query, "variables": {"efoId": efo_id, "size": limit}},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["data"]["disease"]
        if not data:
            return []

        targets = []
        for row in data["associatedTargets"]["rows"]:
            targets.append({
                "target_id": row["target"]["id"],
                "gene_symbol": row["target"]["approvedSymbol"],
                "gene_name": row["target"]["approvedName"],
                "score": row["score"],
            })
        return targets
    except Exception as e:
        print(f"  [WARN] Open Targets target query failed: {e}")
        return []


def get_known_drugs_for_disease(efo_id: str, limit: int = 50) -> list[dict]:
    """
    Get known drug-disease associations from Open Targets.

    Returns list of dicts with: drug_id, drug_name, molecule_type, phase, status
    """
    query = """
    query DiseaseKnownDrugs($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        knownDrugs(size: $size) {
          count
          rows {
            drug {
              id
              name
              drugType
              maximumClinicalTrialPhase
            }
            phase
            status
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            OPENTARGETS_URL,
            json={"query": query, "variables": {"efoId": efo_id, "size": limit}},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["data"]["disease"]
        if not data or not data.get("knownDrugs"):
            return []

        drugs = []
        for row in data["knownDrugs"]["rows"]:
            drugs.append({
                "drug_id": row["drug"]["id"],
                "drug_name": row["drug"]["name"],
                "drug_type": row["drug"]["drugType"],
                "max_phase": row["drug"]["maximumClinicalTrialPhase"],
                "trial_phase": row["phase"],
                "trial_status": row.get("status", ""),
            })
        return drugs
    except Exception as e:
        print(f"  [WARN] Open Targets known drugs query failed: {e}")
        return []
