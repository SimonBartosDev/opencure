#!/usr/bin/env python3
"""
OpenCure Researcher Outreach Tool

Finds researchers who publish on diseases we've screened and generates
personalized outreach emails highlighting our novel drug predictions.

Usage:
    python scripts/find_researchers.py                    # All 25 diseases
    python scripts/find_researchers.py --disease "Malaria" # Single disease
    python scripts/find_researchers.py --dry-run           # Preview queries only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import os
import time
import xml.etree.ElementTree as ET

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}\b")

NOVEL_CANDIDATES_PATH = "experiments/results/novel_candidates.json"
OUTPUT_CSV = "researchers_outreach.csv"


def search_pmids(query: str, max_results: int = 20) -> list[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "date",
        "retmode": "json",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  [WARN] PubMed search failed: {e}")
        return []


def fetch_authors_detailed(pmids: list[str]) -> list[dict]:
    """Fetch full author details including affiliations and emails from PubMed."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] PubMed fetch failed: {e}")
        return []

    researchers = []
    try:
        root = ET.fromstring(resp.text)
        for article_elem in root.findall(".//PubmedArticle"):
            paper = _parse_paper_with_authors(article_elem)
            if paper:
                researchers.extend(paper)
    except ET.ParseError:
        pass

    return researchers


def _parse_paper_with_authors(elem) -> list[dict]:
    """Parse a PubmedArticle and extract all authors with affiliations."""
    results = []
    try:
        medline = elem.find(".//MedlineCitation")
        if medline is None:
            return []

        pmid = medline.findtext("PMID", "")
        article = medline.find("Article")
        if article is None:
            return []

        title = article.findtext("ArticleTitle", "")

        journal_elem = article.find("Journal")
        year = ""
        if journal_elem is not None:
            ji = journal_elem.find("JournalIssue")
            if ji is not None:
                pd = ji.find("PubDate")
                if pd is not None:
                    year = pd.findtext("Year", "")

        author_list = article.find("AuthorList")
        if author_list is None:
            return []

        for author in author_list.findall("Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if not last:
                continue

            name = f"{fore} {last}".strip()

            # Extract all affiliations
            affiliations = []
            emails = []
            for aff_elem in author.findall("AffiliationInfo"):
                aff_text = aff_elem.findtext("Affiliation", "")
                if aff_text:
                    affiliations.append(aff_text)
                    # Extract email from affiliation text
                    found_emails = EMAIL_RE.findall(aff_text)
                    emails.extend(found_emails)

            results.append({
                "name": name,
                "affiliation": "; ".join(affiliations),
                "emails": list(set(emails)),
                "paper_title": title,
                "paper_year": year,
                "pmid": pmid,
            })

    except Exception:
        pass

    return results


def load_novel_candidates() -> dict[str, list[dict]]:
    """Load novel candidates grouped by disease."""
    with open(NOVEL_CANDIDATES_PATH) as f:
        data = json.load(f)

    candidates = data.get("candidates", data)
    if isinstance(candidates, dict):
        candidates = candidates.get("candidates", [])

    by_disease = {}
    for c in candidates:
        disease = c.get("disease", "Unknown")
        by_disease.setdefault(disease, []).append(c)

    return by_disease


def generate_email_draft(
    researcher_name: str,
    disease: str,
    paper_title: str,
    top_predictions: list[dict],
) -> str:
    """Generate a personalized outreach email."""
    drug_lines = []
    for p in top_predictions[:3]:
        drug = p.get("drug_name", "Unknown")
        score = p.get("combined_score", 0)
        novelty = p.get("novelty_level", "NOVEL")
        drug_lines.append(f"  - {drug} (novelty: {novelty}, AI score: {score:.3f})")

    drugs_text = "\n".join(drug_lines)

    return f"""Subject: OpenCure AI predictions for {disease} - potential drug repurposing candidates

Dear {researcher_name.split()[0] if researcher_name else "Researcher"},

I came across your work "{paper_title}" and wanted to share some computational drug repurposing predictions that may be relevant to your research on {disease}.

OpenCure is an open-source AI platform that uses 8 independent scoring methods (knowledge graph embeddings, GNN predictions, molecular similarity, genetic causality, gene expression reversal, network proximity, and more) to identify approved drugs with potential for repurposing.

Our analysis identified the following novel candidates for {disease}:
{drugs_text}

These predictions have strong multi-pillar computational support but limited existing literature, suggesting they could represent genuinely new therapeutic hypotheses worth investigating.

The full results, methodology, and code are freely available:
- Web app: [OpenCure URL]
- GitHub: https://github.com/SimonBartosDev/opencure

We would welcome your expert assessment of these predictions. All findings are open-source under Apache 2.0 — our mission is to accelerate drug repurposing for underserved diseases.

Best regards,
The OpenCure Team"""


def find_researchers_for_disease(
    disease: str,
    candidates: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """Find researchers publishing on a disease and match with our predictions."""

    # Build PubMed query for recent drug repurposing researchers
    query = (
        f'"{disease}" AND '
        f'("drug repurposing" OR "drug repositioning" OR "therapeutic" OR "treatment") '
        f'AND ("2023"[Date - Publication] : "2026"[Date - Publication])'
    )

    if dry_run:
        print(f"  Query: {query}")
        return []

    # Search PubMed
    pmids = search_pmids(query, max_results=20)
    if not pmids:
        print(f"  No PubMed results for {disease}")
        return []

    time.sleep(0.4)

    # Fetch detailed author info
    authors = fetch_authors_detailed(pmids)
    if not authors:
        print(f"  No authors extracted for {disease}")
        return []

    # Deduplicate by name, keep the one with email if available
    seen = {}
    for a in authors:
        name = a["name"]
        if name not in seen or (a["emails"] and not seen[name]["emails"]):
            seen[name] = a

    # Sort: researchers with emails first, then by year
    researchers = sorted(
        seen.values(),
        key=lambda x: (0 if x["emails"] else 1, -(int(x["paper_year"]) if x["paper_year"].isdigit() else 0)),
    )

    # Top predictions for this disease (sorted by novelty score)
    top_preds = sorted(candidates, key=lambda c: -c.get("novelty_score", 0))[:5]

    # Generate outreach entries
    results = []
    for r in researchers[:15]:  # Max 15 per disease
        email = r["emails"][0] if r["emails"] else ""
        draft = generate_email_draft(r["name"], disease, r["paper_title"], top_preds)

        results.append({
            "disease": disease,
            "researcher_name": r["name"],
            "affiliation": r["affiliation"][:200],
            "email": email,
            "paper_title": r["paper_title"][:150],
            "paper_year": r["paper_year"],
            "pmid": r["pmid"],
            "email_draft": draft,
        })

    with_email = sum(1 for r in results if r["email"])
    print(f"  Found {len(results)} researchers ({with_email} with email)")

    return results


def main():
    parser = argparse.ArgumentParser(description="OpenCure Researcher Outreach Tool")
    parser.add_argument("--disease", type=str, help="Search for a single disease only")
    parser.add_argument("--dry-run", action="store_true", help="Preview queries without hitting APIs")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCure Researcher Outreach Tool")
    print("=" * 60)

    # Load novel candidates
    by_disease = load_novel_candidates()
    print(f"Loaded {sum(len(v) for v in by_disease.values())} novel candidates across {len(by_disease)} diseases\n")

    # Filter to single disease if specified
    if args.disease:
        if args.disease not in by_disease:
            # Try fuzzy match
            matches = [d for d in by_disease if args.disease.lower() in d.lower()]
            if matches:
                args.disease = matches[0]
            else:
                print(f"Disease '{args.disease}' not found. Available: {list(by_disease.keys())}")
                return
        diseases = {args.disease: by_disease[args.disease]}
    else:
        diseases = by_disease

    # Find researchers for each disease
    all_results = []
    for disease, candidates in diseases.items():
        print(f"\n[{disease}] Searching for researchers...")
        results = find_researchers_for_disease(disease, candidates, dry_run=args.dry_run)
        all_results.extend(results)
        time.sleep(1)  # Rate limit between diseases

    if args.dry_run:
        print("\nDry run complete. No API calls made.")
        return

    if not all_results:
        print("\nNo researchers found.")
        return

    # Write CSV
    fieldnames = [
        "disease", "researcher_name", "affiliation", "email",
        "paper_title", "paper_year", "pmid", "email_draft",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    total = len(all_results)
    with_email = sum(1 for r in all_results if r["email"])
    diseases_covered = len(set(r["disease"] for r in all_results))

    print(f"\n{'=' * 60}")
    print(f"  OUTREACH SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Researchers found: {total}")
    print(f"  With email: {with_email}")
    print(f"  Diseases covered: {diseases_covered}")
    print(f"  Saved to: {args.output}")
    print()

    if with_email > 0:
        print(f"  Top researchers with email:")
        for r in all_results:
            if r["email"]:
                print(f"    {r['researcher_name']} ({r['disease']}) — {r['email']}")


if __name__ == "__main__":
    main()
