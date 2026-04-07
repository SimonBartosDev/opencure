"""
Semantic Scholar API for fetching academic papers.

Free API with 200M+ papers, includes abstracts and citation data.
More reliable than PubMed E-utilities for batch queries.
"""

import time
import requests

S2_API = "https://api.semanticscholar.org/graph/v1"


def search_papers(
    query: str,
    limit: int = 10,
    fields: str = "title,abstract,year,authors,citationCount,url,externalIds",
) -> list[dict]:
    """
    Search Semantic Scholar for papers.

    Returns list of paper dicts with: paperId, title, abstract, year, authors, citationCount, url
    """
    try:
        resp = requests.get(
            f"{S2_API}/paper/search",
            params={"query": query, "limit": limit, "fields": fields},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        elif resp.status_code == 429:
            time.sleep(2)
            return search_papers(query, limit, fields)
    except Exception:
        pass
    return []


def search_drug_disease_papers(
    drug_name: str,
    disease_name: str,
    limit: int = 10,
) -> dict:
    """
    Search for papers about a drug-disease relationship.

    Returns dict with total count and papers.
    """
    # Direct search
    query = f"{drug_name} {disease_name}"
    papers = search_papers(query, limit=limit)

    # Repurposing-specific search
    time.sleep(0.5)
    repurposing_query = f"{drug_name} {disease_name} repurposing OR repositioning"
    repurposing_papers = search_papers(repurposing_query, limit=5)

    # Deduplicate
    seen = set()
    unique_papers = []
    for p in papers:
        pid = p.get("paperId", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique_papers.append(p)

    unique_repurposing = []
    for p in repurposing_papers:
        pid = p.get("paperId", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique_repurposing.append(p)

    return {
        "drug": drug_name,
        "disease": disease_name,
        "total_papers": len(unique_papers),
        "papers": unique_papers,
        "repurposing_papers": unique_repurposing,
    }
