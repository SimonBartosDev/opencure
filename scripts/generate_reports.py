#!/usr/bin/env python3
"""
Generate professional PDF research reports for each screened disease.

Creates comprehensive, researcher-grade PDFs with full pillar breakdowns,
evidence tables, clinical trial details, and publication references.

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

# Color palette
NAVY = (15, 30, 60)
DARK_BLUE = (20, 40, 80)
BLUE = (41, 98, 168)
LIGHT_BLUE = (220, 235, 250)
PALE_BLUE = (240, 247, 255)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
DARK_GRAY = (60, 60, 60)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
VERY_LIGHT_GRAY = (245, 245, 245)
GREEN = (34, 120, 50)
DARK_GREEN = (20, 80, 35)
ORANGE = (200, 110, 20)
RED = (170, 40, 40)
PURPLE = (100, 20, 140)
TEAL = (0, 120, 120)


def _s(text) -> str:
    """Make text safe for Helvetica (latin-1 only)."""
    if text is None:
        return ""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


class Report(FPDF):
    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease
        self.set_auto_page_break(auto=True, margin=22)
        self._in_title = False

    def header(self):
        if self._in_title:
            return
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY)
        self.set_draw_color(*LIGHT_GRAY)
        self.cell(95, 6, _s(f"OpenCure Drug Repurposing Report  |  {self.disease}"), align="L")
        self.cell(95, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*GRAY)
        self.cell(0, 8, _s("OpenCure  |  Open-source AI Drug Repurposing  |  github.com/SimonBartosDev/opencure  |  Apache 2.0  |  Not a clinical recommendation"), align="C")

    # ---- layout helpers ----

    def _section(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.cell(0, 9, _s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def _subsection(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLUE)
        self.set_x(10)
        self.cell(190, 7, _s(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def _text(self, text: str, size: float = 9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*DARK_GRAY)
        self.set_x(10)
        self.multi_cell(190, 4.5, _s(text))
        self.ln(1.5)

    def _bold_text(self, text: str, size: float = 9):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*BLACK)
        self.set_x(10)
        self.multi_cell(190, 4.5, _s(text))
        self.ln(1)

    def _bullet(self, text: str, indent: int = 14):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*DARK_GRAY)
        self.set_x(indent)
        self.multi_cell(190 - indent + 10, 4.5, _s(f"-  {text}"))
        self.ln(0.5)

    def _kv_row(self, label: str, value: str, label_w: int = 55):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*GRAY)
        self.set_x(14)
        self.cell(label_w, 5, _s(label))
        self.set_text_color(*BLACK)
        self.set_font("Helvetica", "B", 8.5)
        self.multi_cell(190 - label_w - 14, 5, _s(value))

    def _badge(self, text: str, color: tuple):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        w = max(self.get_string_width(f" {text} ") + 4, 22)
        self.cell(w, 5.5, _s(f" {text} "), fill=True)
        self.set_text_color(*BLACK)

    def _separator(self):
        self.set_draw_color(*LIGHT_GRAY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def _table_header(self, widths: list, headers: list):
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for w, h in zip(widths, headers):
            self.cell(w, 6, _s(h), border=1, fill=True, align="C")
        self.ln()

    def _table_row(self, widths: list, values: list, even: bool):
        bg = PALE_BLUE if even else WHITE
        self.set_fill_color(*bg)
        self.set_text_color(*BLACK)
        self.set_font("Helvetica", "", 7)
        for w, v in zip(widths, values):
            self.cell(w, 5.5, _s(v), border=1, fill=True, align="C")
        self.ln()


# ============================================================
#  TITLE PAGE
# ============================================================
def _title_page(pdf: Report, disease: str, data: dict):
    pdf._in_title = True
    pdf.add_page()

    # Header block
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 95, "F")

    # Accent line
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 95, 210, 3, "F")

    pdf.set_y(18)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 14, "OpenCure", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 200, 230)
    pdf.cell(0, 7, "Open-Source AI Drug Repurposing Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 8, "Drug Repurposing Candidates Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, _s(disease), align="C", new_x="LMARGIN", new_y="NEXT")

    # Stats section
    candidates = data.get("candidates", [])
    valid = [c for c in candidates if c.get("confidence") not in ("UNRESOLVED", "ERROR")]
    cc = {}
    for c in valid:
        cc[c.get("confidence", "?")] = cc.get(c.get("confidence", "?"), 0) + 1
    nc = {}
    for c in valid:
        nc[c.get("novelty_level", "?")] = nc.get(c.get("novelty_level", "?"), 0) + 1

    pdf.set_y(108)
    pdf._section("Report Overview")

    rows = [
        ("Candidates analyzed", str(len(valid))),
        ("HIGH confidence", str(cc.get("HIGH", 0))),
        ("MEDIUM confidence", str(cc.get("MEDIUM", 0))),
        ("LOW confidence", str(cc.get("LOW", 0))),
        ("NOVEL confidence", str(cc.get("NOVEL", 0))),
        ("", ""),
        ("BREAKTHROUGH novelty", str(nc.get("BREAKTHROUGH", 0))),
        ("NOVEL novelty", str(nc.get("NOVEL", 0))),
        ("EMERGING novelty", str(nc.get("EMERGING", 0))),
        ("KNOWN/ESTABLISHED", str(nc.get("KNOWN", 0) + nc.get("ESTABLISHED", 0))),
        ("", ""),
        ("AI Scoring Pillars", "8 independent methods"),
        ("Evidence Sources", "6 biomedical databases"),
        ("Ensemble AUC-ROC", "0.998 on known treatments"),
    ]
    for label, value in rows:
        if not label:
            pdf.ln(2)
            continue
        pdf._kv_row(label, value)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.set_x(10)
    pdf.cell(190, 5, _s(f"Report generated: {datetime.now().strftime('%B %d, %Y')}"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 5, "github.com/SimonBartosDev/opencure  |  Apache 2.0 License", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 5, "All predictions are computational hypotheses requiring experimental validation.", align="C")

    pdf._in_title = False


# ============================================================
#  EXECUTIVE SUMMARY
# ============================================================
def _summary_page(pdf: Report, valid: list):
    pdf.add_page()
    pdf._section("Executive Summary")
    pdf._text(
        "Candidates ranked by combined multi-pillar AI score. The Confidence column reflects "
        "strength of existing published evidence (HIGH = clinical trials + literature; LOW = limited evidence). "
        "The Novelty column indicates how surprising the prediction is (BREAKTHROUGH = no prior literature; "
        "ESTABLISHED = well-known treatment). The most interesting candidates for new discoveries have "
        "HIGH or MEDIUM confidence with NOVEL or BREAKTHROUGH novelty."
    )
    pdf.ln(2)

    w = [8, 32, 15, 18, 24, 11, 15, 12, 12, 15, 28]
    h = ["#", "Drug", "Score", "Conf.", "Novelty", "Plrs", "PubMed", "Trials", "MR", "FAERS", "Similar To"]
    pdf._table_header(w, h)

    for i, c in enumerate(valid):
        row = [
            str(i + 1),
            c.get("drug_name", "?")[:18],
            f"{c.get('combined_score', 0):.3f}",
            c.get("confidence", "?"),
            c.get("novelty_level", "?")[:10],
            str(c.get("pillars_hit", 0)),
            str(c.get("pubmed_total", 0)),
            str(c.get("clinical_trials_total", 0)),
            f"{c.get('mr_score', 0):.2f}",
            c.get("faers_signal", "-")[:6],
            c.get("similar_to", "-")[:16],
        ]
        pdf._table_row(w, row, i % 2 == 0)


# ============================================================
#  CANDIDATE DEEP DIVE
# ============================================================
CONF_COLORS = {"HIGH": GREEN, "MEDIUM": ORANGE, "LOW": GRAY, "NOVEL": BLUE, "UNRESOLVED": RED}
NOV_COLORS = {"BREAKTHROUGH": PURPLE, "NOVEL": BLUE, "EMERGING": ORANGE, "KNOWN": GRAY, "ESTABLISHED": DARK_GRAY}


def _candidate_page(pdf: Report, c: dict, rank: int):
    if c.get("confidence") in ("UNRESOLVED", "ERROR"):
        return

    drug = c.get("drug_name", "Unknown")
    drug_id = c.get("drug_id", "")

    # Force new page for each candidate
    pdf.add_page()

    # ---- Header bar ----
    pdf.set_fill_color(*NAVY)
    pdf.rect(10, pdf.get_y() - 2, 190, 12, "F")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*WHITE)
    pdf.set_x(14)
    pdf.cell(0, 10, _s(f"Candidate #{rank}:  {drug}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ---- Key metrics row ----
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY)
    metrics = [
        f"DrugBank: {drug_id}",
        f"AI Score: {c.get('combined_score', 0):.4f}",
        f"Pillars: {c.get('pillars_hit', 0)}/8",
        f"KG Rank: #{c.get('transe_rank', 'N/A')}",
        f"Novelty: {c.get('novelty_score', 0):.2f}",
    ]
    pdf.set_x(12)
    pdf.cell(190, 5, _s("   |   ".join(metrics)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- Badges ----
    pdf.set_x(12)
    pdf._badge(c.get("confidence", "?"), CONF_COLORS.get(c.get("confidence"), GRAY))
    pdf.cell(3, 5.5, "")
    pdf._badge(c.get("novelty_level", "?"), NOV_COLORS.get(c.get("novelty_level"), GRAY))

    mol_sim = c.get("mol_similarity", 0)
    sim_to = c.get("similar_to", "")
    if mol_sim > 0 and sim_to:
        pdf.cell(3, 5.5, "")
        pdf._badge(f"Sim: {mol_sim:.2f} to {sim_to}", TEAL)
    pdf.ln(7)

    # ---- Confidence Reasoning ----
    reasons = c.get("confidence_reasons", [])
    if reasons:
        pdf._subsection("Confidence Assessment")
        for r in reasons:
            pdf._bullet(r)
        pdf.ln(2)

    # ---- Novelty Assessment ----
    nl = c.get("novelty_level", "")
    ns = c.get("novelty_score", 0)
    pdf._subsection("Novelty Assessment")
    if nl == "BREAKTHROUGH":
        pdf._text(f"BREAKTHROUGH (score: {ns:.2f}): No published evidence found for this drug-disease pair. "
                  "This is a completely novel computational prediction — if validated experimentally, "
                  "it would represent a new therapeutic hypothesis not yet explored in the literature.")
    elif nl == "NOVEL":
        pubmed = c.get("pubmed_total", 0)
        pdf._text(f"NOVEL (score: {ns:.2f}): Only {pubmed} article(s) found in PubMed for this drug-disease pair. "
                  "The computational evidence substantially exceeds the published evidence, "
                  "suggesting this is an under-explored therapeutic avenue.")
    elif nl == "EMERGING":
        pdf._text(f"EMERGING (score: {ns:.2f}): Some published evidence exists but no validated clinical trials. "
                  "This is an active research area where computational support adds value.")
    elif nl == "KNOWN":
        pdf._text(f"KNOWN (score: {ns:.2f}): Moderate published evidence exists. This drug-disease "
                  "association has been studied but may not yet be in clinical use for this indication.")
    elif nl == "ESTABLISHED":
        pdf._text(f"ESTABLISHED (score: {ns:.2f}): Well-documented drug-disease relationship. "
                  "This is likely an existing treatment, not a repurposing candidate.")
    if c.get("is_known_treatment"):
        pdf._bold_text("Note: This appears to be an already-approved treatment for this disease.")

    # ---- Pillar-by-Pillar Evidence ----
    pdf._subsection("Scoring Pillar Breakdown")

    pillar_data = [
        ("Knowledge Graph (TransE)", f"Rank #{c.get('transe_rank', 'N/A')} out of 10,551 drugs in DRKG embedding space. "
         f"Lower rank = drug vector is closer to disease vector in 400-dim translation space."),
        ("Molecular Similarity", f"Tanimoto similarity: {mol_sim:.3f}" + (f" to {sim_to}" if sim_to else "") +
         ". Computed from Morgan/ECFP fingerprints (radius=2, 2048 bits). "
         f"Measures 2D structural overlap with known treatments for this disease." if mol_sim > 0 else
         "No molecular similarity data (drug may lack SMILES structure)."),
        ("Mendelian Randomization", f"MR score: {c.get('mr_score', 0):.3f} | Causal targets: {c.get('mr_genetic_targets', 0)}. "
         "Based on Open Targets GWAS genetic association scores. "
         "A positive MR score means the drug's gene targets have causal genetic evidence linking them to this disease "
         "(not just correlation)." if c.get("mr_score", 0) > 0 else
         "No causal genetic evidence found linking this drug's targets to the disease via GWAS."),
    ]

    for name, desc in pillar_data:
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*NAVY)
        pdf.cell(190, 5, _s(name), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DARK_GRAY)
        pdf.set_x(16)
        pdf.multi_cell(182, 4, _s(desc))
        pdf.ln(2)

    # Gene signature
    pdf.set_x(12)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(190, 5, "Gene Signature Reversal (L1000CDS2)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_x(16)
    sig_interp = c.get("signature_interpretation", "")
    if c.get("signature_reversal_found"):
        pdf.multi_cell(182, 4, _s(f"MATCH: Drug ranks #{c.get('signature_reversal_rank', '?')} among drugs "
                                   f"that reverse the disease gene expression signature. {sig_interp}"))
    elif sig_interp:
        pdf.multi_cell(182, 4, _s(sig_interp))
    else:
        pdf.multi_cell(182, 4, "No gene signature reversal data available for this drug-disease pair.")
    pdf.ln(2)

    # ---- Literature Evidence ----
    pubmed = c.get("pubmed_total", 0)
    treatment = c.get("pubmed_treatment_total", 0)
    repurp = c.get("pubmed_repurposing_total", 0)

    pdf._subsection("Literature Evidence")

    # Stats box
    pdf.set_fill_color(*VERY_LIGHT_GRAY)
    pdf.set_draw_color(*LIGHT_GRAY)
    y0 = pdf.get_y()
    pdf.rect(12, y0, 186, 14, "DF")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*BLACK)
    pdf.set_xy(14, y0 + 2)
    pdf.cell(45, 5, _s(f"PubMed articles: {pubmed}"))
    pdf.cell(50, 5, _s(f"Treatment-related: {treatment}"))
    pdf.cell(50, 5, _s(f"Repurposing mentions: {repurp}"))
    pdf.cell(0, 5, _s(f"Max citations: {c.get('max_citations', 0)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(14, y0 + 8)
    faers_interp = c.get("faers_interpretation", "")
    pdf.cell(0, 5, _s(f"FAERS: {c.get('faers_signal', 'none')} ({c.get('faers_cooccurrences', 0)} co-occ / {c.get('faers_total_reports', 0)} total reports)"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Key papers
    papers = c.get("key_papers", [])
    if papers:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*NAVY)
        pdf.set_x(12)
        pdf.cell(190, 5, _s(f"Key Publications ({len(papers)} shown of {pubmed} total)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        for p in papers[:5]:
            pdf.set_x(16)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(180, 4, _s(p.get("title", "")))
            pdf.set_x(16)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*GRAY)
            authors = p.get("authors", "")
            journal = p.get("journal", "")
            year = p.get("year", "")
            pmid = p.get("pmid", "")
            pdf.multi_cell(180, 3.5, _s(f"{authors} ({year}) {journal}  [PMID: {pmid}]"))
            pdf.ln(1.5)

    # Repurposing-specific papers
    rppapers = c.get("repurposing_papers", [])
    if rppapers:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*PURPLE)
        pdf.set_x(12)
        pdf.cell(190, 5, _s(f"Repurposing-Specific Publications ({len(rppapers)})"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for p in rppapers[:3]:
            pdf.set_x(16)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(180, 4, _s(p.get("title", "")))
            pdf.set_x(16)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(180, 3.5, _s(f"{p.get('authors', '')} ({p.get('year', '')}) [PMID: {p.get('pmid', '')}]"))
            pdf.ln(1.5)

    # Semantic Scholar highly-cited papers
    s2papers = c.get("semantic_scholar_papers", [])
    if s2papers:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*TEAL)
        pdf.set_x(12)
        pdf.cell(190, 5, _s(f"Highly-Cited Papers (Semantic Scholar, {len(s2papers)} shown)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for p in s2papers[:3]:
            cit = p.get("citations", 0)
            pdf.set_x(16)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(180, 4, _s(f"[{cit} citations] {p.get('title', '')}"))
            pdf.set_x(16)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(180, 3.5, _s(f"{p.get('authors', '')} ({p.get('year', '')})"))
            pdf.ln(1.5)

    # ---- Clinical Trials ----
    trials = c.get("clinical_trials", [])
    trials_total = c.get("clinical_trials_total", 0)
    if trials_total > 0 or trials:
        pdf._subsection(f"Clinical Trials ({trials_total} found on ClinicalTrials.gov)")
        phases = c.get("trial_phases", {})
        if phases:
            phase_str = ", ".join(f"{k}: {v}" for k, v in phases.items())
            pdf._text(f"Phase distribution: {phase_str}")

        for t in trials[:5]:
            pdf.set_x(14)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*BLACK)
            status = t.get("status", "")
            phase = t.get("phase", "")
            nct = t.get("nct_id", "")
            status_color = GREEN if "COMPLETED" in status.upper() else (ORANGE if "RECRUITING" in status.upper() else GRAY)
            pdf.set_text_color(*status_color)
            pdf.cell(22, 4, _s(f"[{status[:12]}]"))
            pdf.set_text_color(*BLACK)
            pdf.cell(20, 4, _s(f"Phase: {phase}"))
            pdf.set_text_color(*GRAY)
            pdf.cell(0, 4, _s(f"  {nct}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(14)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*DARK_GRAY)
            pdf.multi_cell(182, 4, _s(t.get("title", "")))
            pdf.ln(1.5)

    # ---- Safety Flags ----
    if c.get("has_failed_trial"):
        pdf._subsection("Safety Warning: Failed Trials")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*RED)
        pdf.set_x(12)
        fc = c.get("failed_trial_count", 0)
        fp = c.get("failed_trial_phase", "")
        penalty = c.get("failed_trial_penalty", 1.0)
        pdf.multi_cell(186, 5, _s(f"WARNING: {fc} trial(s) were terminated or withdrawn ({fp}). "
                                   f"Confidence penalty factor: {penalty:.2f}. "
                                   "This drug has previously failed for this indication. "
                                   "Proceed with caution if considering re-investigation."))
        pdf.ln(2)

    # ---- FAERS Interpretation ----
    faers_interp = c.get("faers_interpretation", "")
    if faers_interp:
        pdf._subsection("Real-World Evidence (FDA FAERS)")
        pdf._text(faers_interp)

    # ---- Shared Targets ----
    shared = c.get("shared_targets", [])
    if shared and len(shared) > 0:
        pdf._subsection(f"Shared Gene Targets ({c.get('shared_target_count', len(shared))} genes)")
        targets_str = ", ".join(shared[:20])
        if len(shared) > 20:
            targets_str += f" ... and {len(shared) - 20} more"
        pdf._text(f"Drug and disease share these gene targets in DRKG: {targets_str}")

    # ---- Direct KG Relations ----
    direct = c.get("direct_relations", [])
    if direct:
        pdf._subsection("Direct Knowledge Graph Relations")
        for rel in direct[:5]:
            pdf._bullet(rel)
        pdf.ln(1)

    # ---- Abstract Analysis ----
    analyses = c.get("abstract_analyses", [])
    positive = [a for a in analyses if a.get("relation_type") == "treats"]
    if analyses:
        pdf._subsection("AI Abstract Analysis (PubMedBERT)")
        pdf._text(f"Analyzed {len(analyses)} abstracts. "
                  f"{len(positive)} showed positive treatment signals. "
                  "Relation types: " + ", ".join(set(a.get("relation_type", "?") for a in analyses)) + ".")


# ============================================================
#  METHODOLOGY (detailed)
# ============================================================
def _methodology_pages(pdf: Report):
    pdf.add_page()
    pdf._section("Methodology: Multi-Pillar AI Scoring")

    pdf._text(
        "OpenCure evaluates each drug-disease pair using 8 independent computational methods "
        "(pillars), each capturing a different biological signal. Scores are combined using per-drug "
        "dynamic weighting: only pillars that produce a score for a given drug contribute to its "
        "final rank. A convergence bonus of +0.05 is added for each additional pillar beyond 1 "
        "that scores the drug, rewarding multi-source agreement."
    )
    pdf.ln(2)

    pillars = [
        ("Pillar 1: TransE Knowledge Graph Embeddings",
         "Method: TransE learns 400-dimensional vector representations of all 97,238 entities in the "
         "Drug Repurposing Knowledge Graph (DRKG, 5.87M triplets from 6 databases). For a query disease, "
         "we compute the translation vector d - r (where r is the treatment relation) and rank all drugs "
         "by cosine distance to this target vector.\n"
         "Interpretation: Low rank means the drug occupies a similar position in biomedical knowledge space "
         "as drugs known to treat this disease.\n"
         "Weight: 0.10"),

        ("Pillar 2: RotatE Complex Rotation Embeddings",
         "Method: RotatE represents relations as rotations in complex vector space (200 dimensions), "
         "trained via PyKEEN on DRKG. This captures asymmetric relations (A treats B but B doesn't treat A) "
         "better than TransE.\n"
         "Interpretation: Complementary to TransE; drugs scoring highly on both have stronger KG support.\n"
         "Weight: 0.10"),

        ("Pillar 3: TxGNN Graph Neural Network",
         "Method: Harvard Medical School's Therapeutic Graph Neural Network (Nature Medicine 2024). "
         "Uses message-passing on a heterogeneous biomedical graph to predict drug-disease associations. "
         "Pre-computed predictions for 60 diseases. Achieves 49% improvement over prior baselines.\n"
         "Interpretation: State-of-the-art deep learning prediction, independent of embedding methods.\n"
         "Weight: 0.20"),

        ("Pillar 4: Molecular Fingerprint Similarity",
         "Method: Morgan/ECFP fingerprints (radius=2, 2048 bits) computed via RDKit for each drug. "
         "Tanimoto similarity measured against drugs known to treat the disease.\n"
         "Interpretation: Structurally similar molecules often have similar biological activity. "
         "A drug with Tanimoto > 0.5 to a known treatment likely shares pharmacological properties.\n"
         "Weight: 0.05"),

        ("Pillar 5: ChemBERTa Transformer Embeddings",
         "Method: ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) encodes SMILES molecular structures into "
         "768-dim contextual embeddings. Cosine similarity measured against known treatment embeddings.\n"
         "Interpretation: Captures functional molecular similarity that 2D fingerprints miss, including "
         "3D conformational effects learned from millions of molecules.\n"
         "Weight: 0.08"),

        ("Pillar 6: Gene Signature Reversal (L1000CDS2)",
         "Method: Queries the L1000CDS2 API for drugs that reverse the disease's gene expression signature. "
         "If a disease upregulates genes A, B, C, a drug that downregulates A, B, C may therapeutically "
         "correct the disease transcriptome.\n"
         "Interpretation: Direct functional evidence that the drug opposes the disease at the gene expression level. "
         "Independent of structural or network-based methods.\n"
         "Weight: 0.12"),

        ("Pillar 7: Network Proximity (STRING PPI)",
         "Method: Measures shortest-path distance between drug target proteins and disease gene proteins "
         "in the STRING protein-protein interaction network (16,201 proteins, 236,930 interactions, "
         "confidence > 700). Based on the Barabasi lab approach (PNAS 2021).\n"
         "Interpretation: If a drug's targets are topologically close to disease genes in the interactome, "
         "the drug may modulate disease pathways even without directly targeting disease genes.\n"
         "Weight: 0.15"),

        ("Pillar 8: Mendelian Randomization (Causal Genetics)",
         "Method: Uses Open Targets genetic association scores from GWAS studies. For each drug target gene, "
         "checks if genome-wide association studies provide causal evidence linking that gene to the disease. "
         "Score formula: single target = genetic_score; multiple = 0.6 x top + 0.4 x mean(rest).\n"
         "Interpretation: Unlike correlation-based methods, MR provides causal evidence that modulating "
         "the drug's target gene affects the disease. This is the strongest form of genetic evidence.\n"
         "Additive bonus: +0.15 x mr_score"),
    ]

    for name, desc in pillars:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf._subsection(name)
        pdf._text(desc)
        pdf.ln(1)

    # Evidence sources
    pdf.add_page()
    pdf._section("Post-Scoring Evidence Gathering")
    pdf._text(
        "After AI scoring ranks candidates, the top 10 per disease undergo comprehensive evidence "
        "gathering from 6 external databases. This evidence does NOT affect the AI score but informs "
        "the confidence and novelty assessments."
    )

    sources = [
        ("PubMed (NCBI)", "Full-text search for drug+disease co-occurrence, treatment context, "
         "and repurposing mentions. Provides article counts, key papers, and abstracts."),
        ("ClinicalTrials.gov", "Active and completed clinical trials for this drug-disease pair. "
         "Phase distribution provides a measure of how far investigation has progressed."),
        ("Semantic Scholar", "Citation-weighted paper search (200M+ corpus). Identifies highly-cited "
         "papers that establish drug-disease connections."),
        ("FDA FAERS", "Adverse event reports where drug and disease co-occur. Real-world pharmacovigilance "
         "signal independent of published research."),
        ("Open Targets", "Genetic association evidence, target tractability, and pathway data. "
         "Provides the genetic foundation for Mendelian Randomization scoring."),
        ("L1000CDS2", "LINCS L1000 gene expression database. Identifies drugs whose transcriptomic "
         "signature opposes the disease signature."),
    ]

    for name, desc in sources:
        pdf._subsection(name)
        pdf._text(desc)

    # Confidence scoring
    pdf._section("Confidence & Novelty Scoring")
    pdf._text(
        "Confidence (HIGH/MEDIUM/LOW/NOVEL) is computed from a point system: "
        "+3 for clinical trials, +2 for 50+ PubMed articles, +1 for repurposing mentions, "
        "+1 for direct KG relations, +1 for multi-pillar support, +2 for strong FAERS signal, "
        "+2 for MR > 0.7, +1 for highly-cited paper, -3 for failed trials. "
        "HIGH >= 5 points, MEDIUM >= 3, LOW >= 1."
    )
    pdf._text(
        "Novelty measures how surprising a prediction is: "
        "novelty = computational_score x (1 - knowledge_score). "
        "BREAKTHROUGH = zero published evidence; NOVEL = minimal evidence; "
        "EMERGING = some evidence but no trials; KNOWN = moderate evidence; "
        "ESTABLISHED = well-documented treatment."
    )

    # Disclaimer
    pdf._section("Disclaimer")
    pdf._text(
        "All predictions in this report are computational hypotheses generated by machine learning models. "
        "They require experimental validation before any clinical consideration. This report is not "
        "a clinical recommendation and should not be used to guide patient treatment decisions. "
        "OpenCure is a research tool designed to accelerate hypothesis generation for drug repurposing. "
        "The platform is open-source (Apache 2.0) and operated as a nonprofit."
    )
    pdf._text(
        "Researchers interested in validating these predictions are encouraged to contact us at "
        "opencure.research@gmail.com or visit github.com/SimonBartosDev/opencure."
    )


# ============================================================
#  MAIN REPORT GENERATOR
# ============================================================
def generate_report(disease: str, data: dict, path: Path):
    pdf = Report(disease)

    candidates = data.get("candidates", [])
    valid = [c for c in candidates if c.get("confidence") not in ("UNRESOLVED", "ERROR")]
    valid.sort(key=lambda c: -c.get("combined_score", 0))

    # 1. Title
    _title_page(pdf, disease, data)

    # 2. Summary table
    _summary_page(pdf, valid)

    # 3. Deep dives
    for i, c in enumerate(valid):
        _candidate_page(pdf, c, i + 1)

    # 4. Methodology
    _methodology_pages(pdf)

    pdf.output(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", type=str)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)

    skip = {"screening_summary.json", "opencure_database.json", "novel_candidates.json", "chagas_report.json"}

    print("=" * 60)
    print("  OpenCure PDF Report Generator v2")
    print("=" * 60)

    generated = 0
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name in skip:
            continue
        with open(f) as fh:
            data = json.load(fh)
        disease = data.get("disease", f.stem.replace("_", " "))
        if args.disease and args.disease.lower() not in disease.lower():
            continue

        out = REPORTS_DIR / f"{f.stem}_OpenCure_Report.pdf"
        print(f"  {disease}...", end=" ", flush=True)
        try:
            generate_report(disease, data, out)
            generated += 1
            pages = Report(disease)
            print(f"OK -> {out}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nGenerated {generated} reports in {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
