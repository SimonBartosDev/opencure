#!/usr/bin/env python3
"""
Build the genetics-anchored OpenCure dashboard.

Reads the per-disease JSON written by `scripts/screen_genetics_anchored_all.py`
and renders a single honest HTML page (`docs/genetics_dashboard.html`).

HONESTY CONTRACT
----------------
This dashboard intentionally does NOT carry forward the old multi-pillar
schema (combined_score, pillar sub-scores, novelty). Leak-free evaluation
showed those rankings do not beat a popularity baseline — they are noise and
are not resurrected here.

The page presents exactly what the validated scorer produces:
  * COVERED diseases  — mechanism-grounded ranked leads: target gene, the
    Open Targets genetics datasources backing the gene link, the genetics
    association score, and the curated ChEMBL mechanism action_type.
  * NOT_ASSESSED diseases — listed explicitly with the reason. Honest
    silence (no genetic signal => no ranking) is a feature, not a gap.

No benchmark accuracy figure is published. The only performance statement is
the leak-free-validated relative one: ~5x a popularity baseline on the
genetics-covered subset.

Usage
-----
  python3 scripts/build_genetics_dashboard.py
"""
from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("experiments/results/genetics_anchored")
OUT_HTML = Path("docs/genetics_dashboard.html")

REASON_LABEL = {
    "no_genetics_associations":
        "Maps to Open Targets, but has no GWAS / rare-variant / "
        "clinical-genetics gene association.",
    "no_mechanism_drug":
        "Has human-genetic evidence, but no drug has a curated ChEMBL "
        "mechanism on any genetically-implicated gene.",
    "no_efo_mapping":
        "No MeSH/EFO mapping in the Open Targets disease ontology snapshot.",
    "unresolved_offline":
        "Could not be resolved from offline caches (network fetch disabled).",
    "other": "Not assessed.",
    "unknown": "Not assessed.",
}

CSS = """
:root{--bg:#f8f9fb;--surface:#fff;--border:#e2e8f0;--text:#1a202c;
--text2:#4a5568;--text3:#718096;--accent:#2563eb;--accent-l:#dbeafe;
--green:#059669;--green-l:#d1fae5;--amber:#b45309;--amber-l:#fef3c7;
--r:10px;--sh:0 1px 3px rgba(0,0,0,.08)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,system-ui,
sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
.hero{background:linear-gradient(135deg,#1e3a5f,#0f172a);color:#fff;
padding:2.4rem 2rem 1.8rem}
.hero h1{font-size:1.7rem;font-weight:700;letter-spacing:-.01em}
.hero .tag{color:#93c5fd;font-size:.95rem;margin-top:.35rem;font-weight:500}
.hero p{color:#cbd5e1;font-size:.9rem;margin-top:.8rem;max-width:60rem}
.hero .claim{margin-top:.7rem;font-size:.85rem;color:#fcd34d}
.stats{display:flex;gap:.8rem;margin-top:1.2rem;flex-wrap:wrap}
.stat{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);
border-radius:var(--r);padding:.7rem 1.1rem;min-width:120px;text-align:center}
.stat .n{font-size:1.5rem;font-weight:700}
.stat .l{font-size:.72rem;color:#94a3b8;text-transform:uppercase;
letter-spacing:.05em}
.stat.cov .n{color:#6ee7b7}
.stat.na .n{color:#fcd34d}
main{max-width:80rem;margin:0 auto;padding:1.6rem 2rem 4rem}
.bar{background:var(--surface);border:1px solid var(--border);
border-radius:var(--r);padding:.8rem 1rem;margin-bottom:1.4rem;
box-shadow:var(--sh);display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
.bar input,.bar select{padding:.45rem .7rem;border:1px solid var(--border);
border-radius:8px;font-size:.85rem;background:var(--surface);color:var(--text)}
.bar input{flex:1;min-width:180px}
.section-h{font-size:1.15rem;font-weight:700;margin:1.6rem 0 .6rem;
display:flex;align-items:center;gap:.5rem}
.section-h .badge{font-size:.7rem;font-weight:700;padding:.18rem .5rem;
border-radius:5px;letter-spacing:.04em}
.badge.cov{background:var(--green-l);color:var(--green)}
.badge.na{background:var(--amber-l);color:var(--amber)}
.note{font-size:.83rem;color:var(--text2);margin-bottom:.8rem}
.disease{background:var(--surface);border:1px solid var(--border);
border-radius:var(--r);margin-bottom:.7rem;box-shadow:var(--sh);
overflow:hidden}
.disease.na{border-left:3px solid var(--amber)}
.disease.cov{border-left:3px solid var(--green)}
.dhead{padding:.85rem 1.1rem;cursor:pointer;display:flex;align-items:center;
gap:.7rem}
.dhead:hover{background:#f8fafc}
.dhead .name{font-weight:600;font-size:.98rem}
.dhead .cat{font-size:.72rem;color:var(--text3);background:#f1f5f9;
padding:.12rem .5rem;border-radius:5px}
.dhead .meta{margin-left:auto;font-size:.82rem;color:var(--text3)}
.dhead .caret{color:var(--text3);font-size:.8rem;transition:transform .15s}
.disease.open .caret{transform:rotate(90deg)}
.dbody{display:none;padding:0 1.1rem 1rem}
.disease.open .dbody{display:block}
.na-reason{font-size:.86rem;color:var(--amber);background:var(--amber-l);
border-radius:8px;padding:.6rem .8rem}
.na-reason b{color:#92400e}
table{width:100%;border-collapse:collapse;font-size:.84rem}
thead{background:#f1f5f9}
th,td{padding:.5rem .6rem;text-align:left;border-bottom:1px solid var(--border)}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;
color:var(--text3)}
td.num{font-variant-numeric:tabular-nums}
.gene{font-weight:600;color:var(--accent)}
.ds{font-size:.74rem;color:var(--text2)}
.ds .chip{display:inline-block;background:#eef2ff;color:#4338ca;
padding:.05rem .4rem;border-radius:4px;margin:.05rem .12rem .05rem 0}
.ds .none{color:var(--text3);font-style:italic}
.act{font-size:.78rem;color:var(--text2)}
.scorebar{display:inline-block;width:44px;height:5px;background:#e2e8f0;
border-radius:3px;overflow:hidden;margin-right:.4rem;vertical-align:middle}
.scorefill{height:100%;background:var(--accent)}
.more{font-size:.8rem;color:var(--accent);cursor:pointer;margin-top:.5rem;
display:inline-block}
footer{max-width:80rem;margin:0 auto;padding:1.5rem 2rem 3rem;
font-size:.78rem;color:var(--text3);border-top:1px solid var(--border)}
.hidden{display:none}
"""

JS = """
function toggle(el){el.parentElement.classList.toggle('open');}
function showAll(btn){
  var tb=btn.previousElementSibling;
  tb.querySelectorAll('tr.extra').forEach(function(r){r.classList.remove('hidden');});
  btn.style.display='none';
}
function filt(){
  var q=document.getElementById('q').value.toLowerCase();
  var cat=document.getElementById('cat').value;
  document.querySelectorAll('.disease').forEach(function(d){
    var okq=d.dataset.name.indexOf(q)>-1;
    var okc=(cat==='')||(d.dataset.cat===cat);
    d.style.display=(okq&&okc)?'':'none';
  });
}
"""


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def load_results() -> tuple[list[dict], dict]:
    summary = json.loads(
        (RESULTS_DIR / "genetics_anchored_summary.json").read_text()
    )
    diseases = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "genetics_anchored_summary.json":
            continue
        diseases.append(json.loads(f.read_text()))
    return diseases, summary


def render_candidate_row(c: dict, rank: int, max_score: float,
                          extra: bool) -> str:
    name = esc(c.get("drug_name") or c.get("drug_id"))
    dbid = c.get("drugbank_id")
    if dbid:
        name += f' <span class="ds">({esc(dbid)})</span>'
    gene = esc(c.get("target_gene_symbol") or c.get("target_gene_id"))
    score = c.get("score", 0.0)
    pct = (score / max_score * 100) if max_score else 0
    ds = c.get("evidence_datasources") or []
    if ds:
        ds_html = "".join(f'<span class="chip">{esc(d)}</span>' for d in ds)
    else:
        ds_html = '<span class="none">score-cache; datasources n/a</span>'
    action = esc(c.get("action_type") or "-")
    cls = ' class="extra hidden"' if extra else ""
    return (
        f"<tr{cls}>"
        f'<td class="num">{rank}</td>'
        f"<td>{name}</td>"
        f'<td class="num">'
        f'<span class="scorebar"><span class="scorefill" '
        f'style="width:{pct:.0f}%"></span></span>{score:.4f}</td>'
        f'<td class="gene">{gene}</td>'
        f'<td class="act">{action}</td>'
        f'<td class="ds">{ds_html}</td>'
        f"</tr>"
    )


def render_covered(d: dict) -> str:
    cands = d.get("candidates") or []
    max_score = max((c.get("score", 0.0) for c in cands), default=1.0) or 1.0
    rows = []
    SHOWN = 10
    for i, c in enumerate(cands, 1):
        rows.append(render_candidate_row(c, i, max_score, extra=i > SHOWN))
    more = ""
    if len(cands) > SHOWN:
        more = (f'<span class="more" onclick="showAll(this)">'
                f'show all {len(cands)} leads &darr;</span>')
    ot = ", ".join(d.get("ot_disease_ids") or [])
    table = (
        '<table><thead><tr><th>#</th><th>Drug candidate</th>'
        '<th>Genetics score</th><th>Causal gene</th>'
        '<th>Mechanism (ChEMBL)</th><th>Genetics datasources</th>'
        '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
    )
    return (
        f'<div class="note">Open Targets entity: <code>{esc(ot)}</code> '
        f'&middot; {len(cands)} mechanism-grounded lead(s), ranked by the '
        f'genetics-datasource association score of the drug\'s causal '
        f'target gene.</div>'
        + table + more
    )


def render_not_assessed(d: dict) -> str:
    code = d.get("reason_code") or "unknown"
    label = REASON_LABEL.get(code, REASON_LABEL["other"])
    raw = d.get("reason") or ""
    ot = ", ".join(d.get("ot_disease_ids") or []) or "(none)"
    return (
        f'<div class="na-reason"><b>No human-genetic signal &mdash; '
        f'not assessed.</b><br>{esc(label)}'
        f'<br><span style="color:#92400e;font-size:.78rem">'
        f'reason code: <code>{esc(code)}</code> &middot; '
        f'OT entity: <code>{esc(ot)}</code></span>'
        f'<br><span style="color:#a16207;font-size:.78rem">'
        f'{esc(raw)}</span></div>'
    )


def render_disease(d: dict) -> str:
    covered = d.get("covered")
    cls = "cov" if covered else "na"
    name = esc(d["disease"])
    cat = esc(d.get("category") or "")
    if covered:
        meta = f'{d.get("n_candidates", 0)} leads'
        body = render_covered(d)
    else:
        meta = "not assessed"
        body = render_not_assessed(d)
    return (
        f'<div class="disease {cls}" data-name="{name.lower()}" '
        f'data-cat="{cat}">'
        f'<div class="dhead" onclick="toggle(this)">'
        f'<span class="caret">&#9656;</span>'
        f'<span class="name">{name}</span>'
        f'<span class="cat">{cat}</span>'
        f'<span class="meta">{meta}</span></div>'
        f'<div class="dbody">{body}</div></div>'
    )


def build() -> None:
    diseases, summary = load_results()
    covered = [d for d in diseases if d.get("covered")]
    not_assessed = [d for d in diseases if not d.get("covered")]
    covered.sort(key=lambda d: (-d.get("n_candidates", 0), d["disease"]))
    not_assessed.sort(key=lambda d: (d.get("reason_code") or "",
                                     d["disease"]))
    total = len(diseases)
    cats = sorted({d.get("category") or "" for d in diseases})

    # not_assessed reason tally
    na_reasons = summary.get("not_assessed_reasons", {})
    na_tally = " &middot; ".join(
        f"{n} {REASON_LABEL.get(c, c).split('.')[0].lower()}"
        for c, n in sorted(na_reasons.items(), key=lambda kv: -kv[1])
    )

    cat_opts = "".join(f'<option value="{esc(c)}">{esc(c)}</option>'
                       for c in cats)
    cov_html = "".join(render_disease(d) for d in covered)
    na_html = "".join(render_disease(d) for d in not_assessed)
    gen = summary.get("generated", datetime.now().isoformat(timespec="seconds"))

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenCure — Genetics-Anchored Repurposing Triage</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero">
  <h1>OpenCure</h1>
  <div class="tag">Genetics-anchored drug-repurposing triage tool</div>
  <p>OpenCure ranks repurposing candidates <b>only where human genetics
  implicates a causal target</b>. Each lead is a drug with a curated ChEMBL
  mechanism acting on a gene that GWAS, rare-variant or clinical-genetics
  evidence ties to the disease. Where genetics gives no signal, OpenCure
  says so and ranks nothing &mdash; it does not manufacture a list.</p>
  <p>This tool publishes <b>no benchmark accuracy figure</b>. The earlier
  multi-pillar similarity scores (knowledge-graph, structure, morphology)
  were shown by leak-free evaluation not to beat a popularity baseline and
  are not used here.</p>
  <div class="claim">Leak-free-validated: ~5&times; a popularity baseline on
  the genetics-covered disease subset (Hit@10 ~21%, median rank 64 of
  ~8000 candidates). This is a triage tool, not a predictor of clinical
  efficacy.</div>
  <div class="stats">
    <div class="stat"><div class="n">{total}</div>
      <div class="l">Diseases screened</div></div>
    <div class="stat cov"><div class="n">{len(covered)}</div>
      <div class="l">Genetics-covered</div></div>
    <div class="stat na"><div class="n">{len(not_assessed)}</div>
      <div class="l">Not assessed</div></div>
  </div>
</header>
<main>
  <div class="bar">
    <input id="q" placeholder="Filter by disease name&hellip;"
      oninput="filt()">
    <select id="cat" onchange="filt()">
      <option value="">All categories</option>{cat_opts}
    </select>
  </div>

  <div class="section-h"><span class="badge cov">COVERED</span>
    Genetics-covered diseases &mdash; mechanism-grounded leads
    ({len(covered)})</div>
  <div class="note">Each disease below has at least one human-genetic
    association and at least one drug with a curated mechanism on a
    genetically-implicated gene. Click a disease to see its ranked leads,
    causal target gene, the genetics datasources backing the link, and the
    drug's mechanism action type.</div>
  {cov_html}

  <div class="section-h"><span class="badge na">NOT ASSESSED</span>
    Diseases with no human-genetic signal ({len(not_assessed)})</div>
  <div class="note">These are listed deliberately. A disease appears here
    when it has no genetics-datasource gene association, when no drug has a
    curated mechanism on a genetically-implicated gene, or when it does not
    map to the Open Targets disease ontology. OpenCure does not rank these
    &mdash; honest silence is the correct output. Breakdown:
    {na_tally}.</div>
  {na_html}
</main>
<footer>
  Generated {esc(gen)} from
  <code>scripts/screen_genetics_anchored_all.py</code> via
  <code>opencure.scoring.genetics_anchored.score_disease</code>.
  Disease genetics: Open Targets genetics datasources only (GWAS,
  rare-variant, clinical genetics) &mdash; chembl/known-drug, literature and
  expression datasources excluded by construction. Drug side: curated ChEMBL
  <code>drug_mechanism</code> targets only. No <code>treats</code> /
  <code>indication</code> edge is read. Leak-free by construction.
</footer>
<script>{JS}</script>
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page)
    print(f"  wrote {OUT_HTML}  ({len(page):,} bytes)")
    print(f"  covered={len(covered)}  not_assessed={len(not_assessed)}  "
          f"total={total}")


if __name__ == "__main__":
    build()
