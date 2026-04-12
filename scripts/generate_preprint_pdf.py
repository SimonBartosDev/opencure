#!/usr/bin/env python3
"""Generate a formatted PDF preprint from PREPRINT.md for bioRxiv submission."""

from __future__ import annotations
import re
from pathlib import Path
from fpdf import FPDF


def _s(text) -> str:
    if text is None:
        return ""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


class PreprintPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        self._line_num = 0

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Times", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "OpenCure: Multi-Pillar AI Drug Repurposing Platform", align="L")
        self.cell(0, 6, str(self.page_no()), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(10, 14, 200, 14)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "bioRxiv preprint  |  Not peer-reviewed  |  CC-BY 4.0", align="C")


def parse_md(md_text: str) -> list[dict]:
    """Parse markdown into structured sections."""
    sections = []
    current = None

    for line in md_text.split("\n"):
        if line.startswith("# "):
            current = {"type": "h1", "text": line[2:].strip(), "content": []}
            sections.append(current)
        elif line.startswith("## "):
            current = {"type": "h2", "text": line[3:].strip(), "content": []}
            sections.append(current)
        elif line.startswith("### "):
            current = {"type": "h3", "text": line[4:].strip(), "content": []}
            sections.append(current)
        elif line.startswith("```"):
            if current:
                current["content"].append({"type": "code_toggle"})
        elif line.startswith("|"):
            if current:
                current["content"].append({"type": "table_row", "text": line})
        elif line.startswith("- "):
            if current:
                current["content"].append({"type": "list", "text": line[2:]})
        elif line.startswith("1. ") or (len(line) > 3 and line[0].isdigit() and line[1] == "."):
            if current:
                current["content"].append({"type": "olist", "text": re.sub(r"^\d+\.\s*", "", line)})
        elif line.strip().startswith("**") and line.strip().endswith("**"):
            if current:
                current["content"].append({"type": "bold_line", "text": line.strip().strip("*")})
        elif line.strip():
            if current:
                current["content"].append({"type": "text", "text": line.strip()})
        else:
            if current:
                current["content"].append({"type": "blank"})

    return sections


def render_pdf(sections: list[dict], output: Path):
    pdf = PreprintPDF()

    # ---- TITLE PAGE ----
    pdf.add_page()
    pdf.ln(20)

    # "PREPRINT" label
    pdf.set_font("Times", "I", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "PREPRINT  |  Not peer-reviewed", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Title
    title = sections[0]["text"] if sections else "OpenCure"
    pdf.set_font("Times", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(20)
    pdf.multi_cell(170, 9, _s(title), align="C")
    pdf.ln(8)

    # Authors section
    author_section = [s for s in sections if s["type"] == "h2" and "Author" in s["text"]]
    if author_section:
        for item in author_section[0]["content"]:
            text = item.get("text", "")
            if not text:
                continue
            if text.startswith("*Corresponding"):
                pdf.set_font("Times", "I", 9)
                pdf.set_text_color(80, 80, 80)
            elif text.startswith("^"):
                pdf.set_font("Times", "", 9)
                pdf.set_text_color(80, 80, 80)
            else:
                pdf.set_font("Times", "B", 12)
                pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 6, _s(text), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # Horizontal rule
    pdf.set_draw_color(180, 180, 180)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(6)

    # Abstract
    abstract_section = [s for s in sections if s["type"] == "h2" and "Abstract" in s["text"]]
    if abstract_section:
        pdf.set_font("Times", "B", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 7, "Abstract", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        abstract_text = " ".join(
            item["text"] for item in abstract_section[0]["content"]
            if item["type"] == "text"
        )
        pdf.set_font("Times", "", 9.5)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(15)
        pdf.multi_cell(180, 4.5, _s(abstract_text))
        pdf.ln(3)

        # Keywords
        kw = [item for item in abstract_section[0]["content"] if item["type"] == "bold_line"]
        if kw:
            pdf.set_font("Times", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.set_x(15)
            pdf.multi_cell(180, 4.5, _s(kw[0]["text"].replace("Keywords:", "Keywords:  ")))
        pdf.ln(4)

    # ---- BODY SECTIONS ----
    skip_sections = {"Authors", "Abstract"}
    in_code_block = False
    in_references = False

    for section in sections:
        if section["type"] == "h1":
            continue  # Already handled title

        if section["text"] in skip_sections:
            continue

        if "Reference" in section["text"]:
            in_references = True

        if "Supplementary" in section["text"]:
            pdf.add_page()

        # Section heading
        if section["type"] == "h2":
            pdf.ln(4)
            pdf.set_font("Times", "B", 12)
            pdf.set_text_color(20, 40, 80)
            pdf.set_x(10)
            pdf.cell(190, 7, _s(section["text"]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif section["type"] == "h3":
            pdf.ln(2)
            pdf.set_font("Times", "BI", 10)
            pdf.set_text_color(40, 60, 100)
            pdf.set_x(10)
            pdf.cell(190, 6, _s(section["text"]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        # Content
        for item in section["content"]:
            if item["type"] == "code_toggle":
                in_code_block = not in_code_block
                continue

            if in_code_block:
                # Render code/diagram in monospace
                pdf.set_font("Courier", "", 7)
                pdf.set_text_color(60, 60, 60)
                pdf.set_x(15)
                pdf.multi_cell(180, 3.5, _s(item.get("text", "")))
                continue

            if item["type"] == "blank":
                pdf.ln(2)
                continue

            if item["type"] == "text":
                text = item["text"]
                # Handle inline bold with **text**
                clean = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

                if in_references:
                    pdf.set_font("Times", "", 8)
                    pdf.set_text_color(40, 40, 40)
                    pdf.set_x(10)
                    pdf.multi_cell(190, 4, _s(clean))
                    pdf.ln(1)
                else:
                    pdf.set_font("Times", "", 10)
                    pdf.set_text_color(30, 30, 30)
                    pdf.set_x(10)
                    pdf.multi_cell(190, 5, _s(clean))
                    pdf.ln(1.5)

            elif item["type"] == "bold_line":
                pdf.set_font("Times", "B", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(10)
                pdf.multi_cell(190, 5, _s(item["text"]))
                pdf.ln(1)

            elif item["type"] == "list":
                text = re.sub(r"\*\*(.*?)\*\*", r"\1", item["text"])
                pdf.set_font("Times", "", 9.5)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(15)
                pdf.multi_cell(183, 4.5, _s(f"-  {text}"))
                pdf.ln(0.5)

            elif item["type"] == "olist":
                text = re.sub(r"\*\*(.*?)\*\*", r"\1", item["text"])
                pdf.set_font("Times", "", 9.5)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(15)
                pdf.multi_cell(183, 4.5, _s(text))
                pdf.ln(0.5)

            elif item["type"] == "table_row":
                row = item["text"]
                cells = [c.strip() for c in row.split("|")[1:-1]]
                if all(c.replace("-", "").strip() == "" for c in cells):
                    continue  # separator row

                # Detect header (first table row)
                is_header = any(c in ("Category", "Disease", "#", "HIGH") for c in cells)

                if is_header:
                    pdf.set_font("Times", "B", 7)
                    pdf.set_fill_color(20, 40, 80)
                    pdf.set_text_color(255, 255, 255)
                else:
                    pdf.set_font("Times", "", 7)
                    pdf.set_fill_color(245, 248, 255)
                    pdf.set_text_color(30, 30, 30)

                n = len(cells)
                if n >= 7:
                    widths = [35, 45, 14, 17, 12, 16, 14][:n]
                    # Pad remaining
                    while len(widths) < n:
                        widths.append(14)
                else:
                    w = 190 // max(n, 1)
                    widths = [w] * n

                pdf.set_x(10)
                for w, c in zip(widths, cells):
                    pdf.cell(w, 5, _s(c[:20]), border=1, fill=True, align="C")
                pdf.ln()

    pdf.output(str(output))


def main():
    md_path = Path("PREPRINT.md")
    out_path = Path("OpenCure_Preprint.pdf")

    print("Parsing PREPRINT.md...")
    md_text = md_path.read_text()
    sections = parse_md(md_text)
    print(f"  {len(sections)} sections found")

    print("Rendering PDF...")
    render_pdf(sections, out_path)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
