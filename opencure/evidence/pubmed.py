"""
Search PubMed for evidence supporting drug-disease repurposing predictions.

Uses NCBI E-utilities API (free, no key needed for low volume).
"""

from __future__ import annotations

import time
import requests
import xml.etree.ElementTree as ET

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """
    Search PubMed for articles matching a query.

    Returns list of dicts with: pmid, title, authors, journal, year, abstract_snippet
    """
    # Step 1: Search for PMIDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        total = int(data.get("esearchresult", {}).get("count", 0))
    except Exception as e:
        return []

    if not pmids:
        return []

    time.sleep(0.4)  # Rate limit

    # Step 2: Fetch article details
    articles = _fetch_article_details(pmids)

    # Add total count to first result
    if articles:
        articles[0]["total_results"] = total

    return articles


def _fetch_article_details(pmids: list[str]) -> list[dict]:
    """Fetch article details from PubMed for a list of PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    articles = []
    try:
        root = ET.fromstring(resp.text)
        for article_elem in root.findall(".//PubmedArticle"):
            article = _parse_article(article_elem)
            if article:
                articles.append(article)
    except ET.ParseError:
        return []

    return articles


def _parse_article(elem) -> dict | None:
    """Parse a PubmedArticle XML element into a dict."""
    try:
        medline = elem.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid = medline.findtext("PMID", "")
        article = medline.find("Article")
        if article is None:
            return None

        title = article.findtext("ArticleTitle", "")

        # Authors
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author")[:3]:
                last = author.findtext("LastName", "")
                init = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {init}".strip())
            if len(author_list.findall("Author")) > 3:
                authors.append("et al.")

        # Journal and year
        journal_elem = article.find("Journal")
        journal = ""
        year = ""
        if journal_elem is not None:
            journal = journal_elem.findtext("Title", "")
            ji = journal_elem.find("JournalIssue")
            if ji is not None:
                pd = ji.find("PubDate")
                if pd is not None:
                    year = pd.findtext("Year", "")
                    if not year:
                        medline_date = pd.findtext("MedlineDate", "")
                        if medline_date:
                            year = medline_date[:4]

        # Abstract
        abstract_elem = article.find("Abstract")
        abstract = ""
        if abstract_elem is not None:
            parts = []
            for text in abstract_elem.findall("AbstractText"):
                label = text.get("Label", "")
                content = text.text or ""
                if label:
                    parts.append(f"{label}: {content}")
                else:
                    parts.append(content)
            abstract = " ".join(parts)

        # Truncate abstract for display
        abstract_snippet = abstract[:300] + "..." if len(abstract) > 300 else abstract

        return {
            "pmid": pmid,
            "title": title,
            "authors": ", ".join(authors),
            "journal": journal,
            "year": year,
            "abstract_snippet": abstract_snippet,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except Exception:
        return None


def search_drug_disease_evidence(
    drug_name: str,
    disease_name: str,
    max_results: int = 5,
) -> dict:
    """
    Search PubMed for evidence of a drug-disease relationship.

    Performs multiple targeted searches:
    1. Direct drug + disease
    2. Drug + disease + "repurpos*"
    3. Drug + disease + "treatment" or "therapy"

    Returns dict with: total_articles, articles, repurposing_articles, query_used
    """
    # Clean names for search
    drug_clean = drug_name.replace("'", "").strip()
    disease_clean = disease_name.replace("'", "").strip()

    # Search 1: Direct association
    query1 = f'"{drug_clean}" AND "{disease_clean}"'
    articles1 = search_pubmed(query1, max_results=max_results)
    total1 = articles1[0].get("total_results", 0) if articles1 else 0

    time.sleep(0.4)

    # Search 2: Repurposing context
    query2 = f'"{drug_clean}" AND "{disease_clean}" AND (repurpos* OR repositioning)'
    articles2 = search_pubmed(query2, max_results=3)
    total2 = articles2[0].get("total_results", 0) if articles2 else 0

    time.sleep(0.4)

    # Search 3: Treatment/therapy context
    query3 = f'"{drug_clean}" AND "{disease_clean}" AND (treatment OR therapy OR therapeutic)'
    articles3 = search_pubmed(query3, max_results=3)
    total3 = articles3[0].get("total_results", 0) if articles3 else 0

    # Deduplicate by PMID
    seen_pmids = set()
    all_articles = []
    for a in articles1 + articles3:
        if a["pmid"] not in seen_pmids:
            seen_pmids.add(a["pmid"])
            all_articles.append(a)

    repurposing_articles = []
    for a in articles2:
        if a["pmid"] not in seen_pmids:
            seen_pmids.add(a["pmid"])
        repurposing_articles.append(a)

    return {
        "drug": drug_name,
        "disease": disease_name,
        "total_articles": total1,
        "total_treatment_articles": total3,
        "total_repurposing_articles": total2,
        "articles": all_articles[:max_results],
        "repurposing_articles": repurposing_articles,
        "query": query1,
    }
