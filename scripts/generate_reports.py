#!/usr/bin/env python3
"""
Generate professional PDF research reports for each screened disease.

Creates one PDF per disease in reports/ with:
- Executive summary table
- Per-candidate deep dives with evidence
- Methodology overview
- References

Usage:
    python scripts/generate_reports.py
    python scripts/generate_reports.py --disease "Malaria"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

RESULTS_DIR = Path("experiments/results")
REPORTS_DIR = Path("reports")

# Colors
DARK_BLUE = (20, 40, 80)
MEDIUM_BLUE = (41, 98, 168)
LIGHT_BLUE = (220, 235, 250)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (100, 100, 100)
LIGHT_GRAY = (240, 240, 240)
GREEN = (34, 139, 34)
ORANGE = (210, 120, 20)
RED = (180, 40, 40)


def _safe(text: str) -> str:
    """Replace non-latin1 characters to avoid fpdf2 Helvetica errors."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class OpenCureReport(FPDF):
    """Custom PDF with OpenCure branding."""

    def __init__(self, disease_name: str):
        super().__init__()
        self.disease_name = disease_name
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() == 1:
            return  # Title page has custom header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"OpenCure Report: {self.disease_name}", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 16, 200, 16)
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 10, "OpenCure | Open-source AI drug repurposing | github.com/SimonBartosDev/opencure | Apache 2.0", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*DARK_BLUE)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*MEDIUM_BLUE)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(4)

    def subsection_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*MEDIUM_BLUE)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*BLACK)
        self.set_x(10)
        self.multi_cell(190, 5, _safe(text))
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*BLACK)
        self.set_x(10)
        self.multi_cell(190, 5, f"  - {_safe(text)}")

    def confidence_badge(self, confidence: str):
        colors = {
            "HIGH": GREEN,
            "MEDIUM": ORANGE,
            "LOW": GRAY,
            "NOVEL": MEDIUM_BLUE,
            "UNRESOLVED": RED,
        }
        color = colors.get(confidence, GRAY)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.cell(25, 6, f" {confidence} ", fill=True)
        self.set_text_color(*BLACK)

    def novelty_badge(self, level: str):
        colors = {
            "BREAKTHROUGH": (139, 0, 139),
            "NOVEL": MEDIUM_BLUE,
            "EMERGING": ORANGE,
            "KNOWN": GRAY,
            "ESTABLISHED": (100, 100, 100),
        }
        color = colors.get(level, GRAY)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.cell(30, 5, f" {level} ", fill=True)
        self.set_text_color(*BLACK)


def generate_title_page(pdf: OpenCureReport, disease: str, data: dict):
    """Generate the title page."""
    pdf.add_page()

    # Background header block
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 0, 210, 100, "F")

    # Title
    pdf.set_y(20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 15, "OpenCure", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "AI Drug Repurposing Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, f"Drug Repurposing Report:", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 14, disease, align="C", new_x="LMARGIN", new_y="NEXT")

    # Summary stats below the header
    pdf.set_y(110)
    pdf.set_text_color(*BLACK)

    candidates = data.get("candidates", [])
    valid = [c for c in candidates if c.get("confidence") not in ("UNRESOLVED", "ERROR")]
    cc = {}
    for c in valid:
        conf = c.get("confidence", "UNKNOWN")
        cc[conf] = cc.get(conf, 0) + 1

    novelty_counts = {}
    for c in valid:
        nl = c.get("novelty_level", "UNKNOWN")
        novelty_counts[nl] = novelty_counts.get(nl, 0) + 1

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 10, "Report Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Stats table
    stats = [
        ("Total candidates analyzed", str(len(valid))),
        ("HIGH confidence", str(cc.get("HIGH", 0))),
        ("MEDIUM confidence", str(cc.get("MEDIUM", 0))),
        ("LOW confidence", str(cc.get("LOW", 0))),
        ("NOVEL (no literature)", str(cc.get("NOVEL", 0))),
        ("BREAKTHROUGH novelty", str(novelty_counts.get("BREAKTHROUGH", 0))),
        ("NOVEL novelty", str(novelty_counts.get("NOVEL", 0))),
        ("Scoring pillars used", "8 (TransE, RotatE, TxGNN, Fingerprints, ChemBERTa, MR, Gene Sig, PPI)"),
        ("Evidence sources", "PubMed, ClinicalTrials.gov, Semantic Scholar, FDA FAERS, Open Targets"),
    ]

    pdf.set_font("Helvetica", "", 10)
    for label, value in stats:
        pdf.set_text_color(*GRAY)
        pdf.cell(90, 7, f"  {label}:", align="R")
        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"  {value}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "github.com/SimonBartosDev/opencure | Apache 2.0 License", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "All predictions require experimental validation. Not a clinical recommendation.", align="C")


def generate_summary_table(pdf: OpenCureReport, candidates: list[dict]):
    """Generate the executive summary table."""
    pdf.add_page()
    pdf.section_title("Executive Summary")
    pdf.body_text("Top drug repurposing candidates ranked by multi-pillar AI scoring. "
                  "Confidence reflects strength of published evidence; novelty measures how surprising the prediction is.")
    pdf.ln(2)

    # Table header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*DARK_BLUE)
    pdf.set_text_color(*WHITE)

    col_widths = [8, 35, 16, 20, 24, 13, 16, 13, 13, 32]
    headers = ["#", "Drug", "Score", "Conf.", "Novelty", "Pillars", "PubMed", "Trials", "MR", "Similar To"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 7.5)
    for i, c in enumerate(candidates):
        if c.get("confidence") in ("UNRESOLVED", "ERROR"):
            continue

        bg = LIGHT_BLUE if i % 2 == 0 else WHITE
        pdf.set_fill_color(*bg)
        pdf.set_text_color(*BLACK)

        row = [
            str(i + 1),
            c.get("drug_name", "?")[:20],
            f"{c.get('combined_score', 0):.3f}",
            c.get("confidence", "?"),
            c.get("novelty_level", "?"),
            str(c.get("pillars_hit", 0)),
            str(c.get("pubmed_total", 0)),
            str(c.get("clinical_trials_total", 0)),
            f"{c.get('mr_score', 0):.2f}",
            c.get("similar_to", "-")[:12],
        ]
        for w, val in zip(col_widths, row):
            pdf.cell(w, 6, val, border=1, fill=True, align="C")
        pdf.ln()


def generate_candidate_page(pdf: OpenCureReport, candidate: dict, rank: int):
    """Generate a detailed page for one candidate drug."""
    if candidate.get("confidence") in ("UNRESOLVED", "ERROR"):
        return

    drug = candidate.get("drug_name", "Unknown")
    drug_id = candidate.get("drug_id", "")

    # Check if we need a new page
    if pdf.get_y() > 60:
        pdf.add_page()

    # Drug header
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 10, f"#{rank}: {drug}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(50, 5, f"DrugBank: {drug_id}")
    pdf.cell(40, 5, f"Score: {candidate.get('combined_score', 0):.4f}")
    pdf.cell(30, 5, f"Pillars: {candidate.get('pillars_hit', 0)}")
    pdf.cell(0, 5, f"Rank: #{candidate.get('transe_rank', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Badges
    pdf.confidence_badge(candidate.get("confidence", "UNKNOWN"))
    pdf.cell(3, 6, "")
    pdf.novelty_badge(candidate.get("novelty_level", "UNKNOWN"))
    pdf.ln(8)

    # Why this drug?
    reasons = candidate.get("confidence_reasons", [])
    if reasons:
        pdf.subsection_title("Why This Drug?")
        for reason in reasons:
            pdf.bullet(reason)
        pdf.ln(2)

    # Novelty interpretation
    novelty_interp = candidate.get("novelty_interpretation", "")
    if not novelty_interp:
        # Build from novelty_level
        nl = candidate.get("novelty_level", "")
        ns = candidate.get("novelty_score", 0)
        if nl == "BREAKTHROUGH":
            novelty_interp = f"No published evidence found for this drug-disease pair. Strong computational prediction (novelty score: {ns:.2f}) suggests a potential new discovery."
        elif nl == "NOVEL":
            novelty_interp = f"Minimal existing literature. Computational support suggests this warrants investigation (novelty score: {ns:.2f})."
        elif nl == "EMERGING":
            novelty_interp = f"Some evidence exists but not yet validated in trials (novelty score: {ns:.2f})."

    if novelty_interp:
        pdf.subsection_title("Novelty Assessment")
        pdf.body_text(novelty_interp)

    # Evidence grid
    pdf.subsection_title("Evidence Summary")

    evidence_items = []

    # Literature
    pubmed = candidate.get("pubmed_total", 0)
    repurp = candidate.get("pubmed_repurposing_total", 0)
    if pubmed > 0:
        evidence_items.append(f"PubMed: {pubmed} articles ({repurp} mention repurposing)")
    else:
        evidence_items.append("PubMed: No articles found (novel prediction)")

    # Clinical trials
    trials = candidate.get("clinical_trials_total", 0)
    if trials > 0:
        phases = candidate.get("trial_phases", {})
        phase_str = ", ".join(f"{k}: {v}" for k, v in phases.items()) if phases else ""
        evidence_items.append(f"Clinical trials: {trials} found{' (' + phase_str + ')' if phase_str else ''}")

    # Molecular similarity
    sim = candidate.get("mol_similarity", 0)
    sim_to = candidate.get("similar_to", "")
    if sim > 0 and sim_to:
        evidence_items.append(f"Molecular similarity: {sim:.3f} to {sim_to}")

    # MR / Genetic
    mr = candidate.get("mr_score", 0)
    mr_targets = candidate.get("mr_genetic_targets", 0)
    if mr > 0:
        evidence_items.append(f"Genetic evidence (MR): score {mr:.3f} ({mr_targets} causal target{'s' if mr_targets != 1 else ''})")

    # FAERS
    faers = candidate.get("faers_signal", "")
    faers_co = candidate.get("faers_cooccurrences", 0)
    if faers and faers != "none":
        evidence_items.append(f"Real-world evidence (FAERS): {faers} signal ({faers_co} co-occurrences)")

    # Gene signature
    if candidate.get("signature_reversal_found"):
        sig_rank = candidate.get("signature_reversal_rank", 0)
        evidence_items.append(f"Gene signature reversal: rank #{sig_rank} among disease-reversing drugs")

    # Shared targets
    shared = candidate.get("shared_target_count", 0)
    if shared > 0:
        evidence_items.append(f"Shared gene targets with disease: {shared}")

    # Failed trials (safety flag)
    if candidate.get("has_failed_trial"):
        fc = candidate.get("failed_trial_count", 0)
        fp = candidate.get("failed_trial_phase", "")
        evidence_items.append(f"WARNING: {fc} failed/terminated trial(s) ({fp})")

    for item in evidence_items:
        pdf.bullet(item)
    pdf.ln(2)

    # Key publications
    papers = candidate.get("key_papers", [])
    if papers:
        pdf.subsection_title("Key Publications")
        for p in papers[:3]:
            title = p.get("title", "")[:100]
            authors = p.get("authors", "")
            year = p.get("year", "")
            journal = p.get("journal", "")
            pdf.set_x(15)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(185, 4, _safe(title))
            pdf.set_x(15)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(185, 4, _safe(f"{authors} ({year}) {journal}"))
            pdf.ln(2)

    # Most cited paper
    mcp = candidate.get("most_cited_paper", {})
    if mcp and mcp.get("title"):
        citations = mcp.get("citations", 0)
        if citations > 10:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GRAY)
            pdf.cell(0, 5, _safe(f"Most cited: \"{mcp['title'][:80]}...\" ({citations} citations)"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # Separator
    pdf.set_draw_color(*LIGHT_GRAY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


def generate_methodology_page(pdf: OpenCureReport):
    """Generate the methodology overview."""
    pdf.add_page()
    pdf.section_title("Methodology")

    pdf.body_text(
        "OpenCure uses 8 independent AI scoring pillars to evaluate each drug-disease pair. "
        "These capture different biological signals and are combined using per-drug dynamic weighting "
        "with convergence bonuses for multi-pillar agreement."
    )

    pillars = [
        ("TransE (Knowledge Graph)", "Learns statistical patterns from 5.87 million biomedical relationships in the Drug Repurposing Knowledge Graph (DRKG). Drugs that occupy similar vector space positions to known treatments score highly."),
        ("RotatE (Complex Embeddings)", "Rotation-based embeddings in complex vector space that capture symmetric and asymmetric relation patterns. Trained via PyKEEN on DRKG."),
        ("TxGNN (Graph Neural Network)", "Harvard's state-of-the-art GNN for zero-shot drug repurposing (Nature Medicine 2024). Pre-computed predictions for 60 diseases, 49% better than prior baselines."),
        ("Molecular Fingerprints", "Morgan/ECFP fingerprints (radius=2, 2048 bits) via RDKit. Measures Tanimoto structural similarity to drugs known to treat the disease."),
        ("ChemBERTa (Learned Embeddings)", "Transformer-based molecular embeddings from SMILES strings. Captures functional similarity that traditional fingerprints miss."),
        ("Mendelian Randomization", "Uses Open Targets genetic association scores as causal evidence. If a drug's target gene has GWAS evidence linking it to the disease, the prediction has causal support."),
        ("Gene Signature Reversal", "Queries L1000CDS2 for drugs that reverse the disease's gene expression signature. A drug that opposes the disease transcriptome may correct it."),
        ("Network Proximity", "Shortest-path distance between drug target proteins and disease gene proteins in the STRING protein-protein interaction network (Barabasi lab approach)."),
    ]

    for name, desc in pillars:
        pdf.subsection_title(name)
        pdf.body_text(desc)

    pdf.ln(3)
    pdf.section_title("Evidence Sources")
    pdf.body_text(
        "After AI scoring, each candidate undergoes evidence gathering from 5+ sources: "
        "PubMed (literature), ClinicalTrials.gov (clinical trials), Semantic Scholar (citation-weighted papers), "
        "FDA FAERS (adverse event reports indicating real-world drug-disease co-occurrence), "
        "and Open Targets (genetic associations). Candidates are then assessed for confidence "
        "(strength of evidence) and novelty (how surprising the prediction is)."
    )

    pdf.ln(3)
    pdf.section_title("Disclaimer")
    pdf.body_text(
        "All predictions are computational hypotheses that require experimental validation. "
        "This report is not a clinical recommendation. OpenCure is a research tool designed to "
        "accelerate the identification of drug repurposing candidates for further investigation. "
        "The platform is open-source (Apache 2.0) and nonprofit."
    )


def generate_report(disease_name: str, data: dict, output_path: Path):
    """Generate a complete PDF report for one disease."""
    pdf = OpenCureReport(disease_name)

    candidates = data.get("candidates", [])
    valid = [c for c in candidates if c.get("confidence") not in ("UNRESOLVED", "ERROR")]

    # Sort by combined_score descending
    valid.sort(key=lambda c: -c.get("combined_score", 0))

    # 1. Title page
    generate_title_page(pdf, disease_name, data)

    # 2. Executive summary table
    generate_summary_table(pdf, valid)

    # 3. Per-candidate deep dives
    pdf.add_page()
    pdf.section_title("Candidate Deep Dives")
    pdf.body_text(f"Detailed evidence for each of the {len(valid)} drug candidates, ranked by combined AI score.")
    pdf.ln(3)

    for i, c in enumerate(valid):
        generate_candidate_page(pdf, c, i + 1)

    # 4. Methodology
    generate_methodology_page(pdf)

    # Save
    pdf.output(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Generate OpenCure PDF reports")
    parser.add_argument("--disease", type=str, help="Generate for one disease only")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)

    # Find all disease result files
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    skip = {"screening_summary.json", "opencure_database.json", "novel_candidates.json", "chagas_report.json"}

    print("=" * 60)
    print("  OpenCure PDF Report Generator")
    print("=" * 60)

    generated = 0
    for f in result_files:
        if f.name in skip:
            continue

        with open(f) as fh:
            data = json.load(fh)

        disease = data.get("disease", f.stem.replace("_", " "))

        if args.disease and args.disease.lower() not in disease.lower():
            continue

        safe_name = f.stem
        output_path = REPORTS_DIR / f"{safe_name}_OpenCure_Report.pdf"

        print(f"  Generating: {disease}...")
        try:
            generate_report(disease, data, output_path)
            generated += 1
            print(f"    Saved: {output_path}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nGenerated {generated} PDF reports in {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
