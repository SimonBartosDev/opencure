#!/usr/bin/env python3
"""Generate the OpenCure Explorer v2 — world-class biomedical dashboard.

Reads disease JSON files + database JSON, merges rich evidence data, and
produces docs/index.html: a self-contained interactive dashboard with
tabbed evidence panels, clickable PubMed links, clinical trial details,
pillar strength bars, cross-disease network, and CSV export.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

DB_PATH = Path("experiments/results/opencure_database.json")
RESULTS_DIR = Path("experiments/results")
OUT_PATH = Path("docs/index.html")


def load_data():
    """Load database + merge rich evidence from per-disease JSON files."""
    raw = json.loads(DB_PATH.read_text())
    db_candidates = raw["candidates"]

    # Build lookup from disease JSONs (rich data)
    rich = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("opencure_") or f.name in ("screening_summary.json", "novel_candidates.json"):
            continue
        try:
            d = json.loads(f.read_text())
            for c in d.get("candidates", []):
                key = (c.get("disease_name", d.get("disease", "")), c["drug_name"])
                rich[key] = c
        except (json.JSONDecodeError, KeyError):
            continue

    # Merge rich fields into db candidates
    merged = []
    for c in db_candidates:
        key = (c["disease"], c["drug_name"])
        r = rich.get(key, {})
        m = dict(c)
        m["key_papers"] = r.get("key_papers", [])[:5]
        m["repurposing_papers"] = r.get("repurposing_papers", [])[:3]
        m["semantic_scholar_papers"] = r.get("semantic_scholar_papers", [])[:5]
        m["most_cited_paper"] = r.get("most_cited_paper")
        m["faers_interpretation"] = r.get("faers_interpretation", "")
        m["signature_interpretation"] = r.get("signature_interpretation", "")
        m["signature_top_reversers"] = r.get("signature_top_reversers", [])[:5]
        m["trial_phases"] = r.get("trial_phases", {})
        m["has_failed_trial"] = r.get("has_failed_trial", False)
        m["failed_trial_phase"] = r.get("failed_trial_phase", "")
        m["failed_trial_count"] = r.get("failed_trial_count", 0)
        m["shared_targets_list"] = r.get("shared_targets", [])
        if isinstance(m["shared_targets_list"], int):
            m["shared_targets_list"] = []
        merged.append(m)

    return merged


def compute_cross_disease(candidates):
    drug_map = defaultdict(list)
    for c in candidates:
        if c["novelty_level"] in ("BREAKTHROUGH", "NOVEL"):
            drug_map[c["drug_name"]].append({
                "disease": c["disease"],
                "score": c["combined_score"],
                "pillars": c["pillars_hit"],
                "novelty": c["novelty_level"],
                "confidence": c["confidence"],
            })
    return {d: v for d, v in sorted(drug_map.items(), key=lambda x: -len(x[1])) if len(v) >= 2}


def compute_stats(candidates):
    diseases = set()
    novel = bt = high = 0
    for c in candidates:
        diseases.add(c["disease"])
        if c["novelty_level"] == "BREAKTHROUGH":
            bt += 1
        if c["novelty_level"] in ("BREAKTHROUGH", "NOVEL"):
            novel += 1
        if c["confidence"] == "HIGH":
            high += 1
    return {"total": len(candidates), "diseases": len(diseases), "novel": novel, "breakthrough": bt, "high_confidence": high}


def prep_js_candidates(candidates):
    out = []
    for c in candidates:
        out.append({
            "disease": c["disease"],
            "drug_name": c["drug_name"],
            "drug_id": c.get("drug_id", ""),
            "confidence": c["confidence"],
            "novelty_level": c["novelty_level"],
            "combined_score": round(c["combined_score"], 4),
            "pillars_hit": c["pillars_hit"],
            "mr_score": round(c.get("mr_score", 0), 4),
            "mr_genetic_targets": c.get("mr_genetic_targets", 0),
            "pubmed_articles": c.get("pubmed_articles", 0),
            "pubmed_treatment": c.get("pubmed_treatment", 0),
            "pubmed_repurposing": c.get("pubmed_repurposing", 0),
            "clinical_trials": c.get("clinical_trials", 0),
            "mol_similarity": round(c.get("mol_similarity", 0), 4),
            "similar_to": c.get("similar_to", ""),
            "transe_rank": c.get("transe_rank", 0),
            "shared_targets": c.get("shared_targets", 0),
            "shared_targets_list": c.get("shared_targets_list", []),
            "faers_signal": c.get("faers_signal", "none"),
            "faers_cooccurrences": c.get("faers_cooccurrences", 0),
            "faers_interpretation": c.get("faers_interpretation", ""),
            "signature_reversal": c.get("signature_reversal", False),
            "signature_rank": c.get("signature_rank", 0),
            "signature_interpretation": c.get("signature_interpretation", ""),
            "signature_top_reversers": c.get("signature_top_reversers", []),
            "novelty_interpretation": c.get("novelty_interpretation", ""),
            "confidence_reasons": c.get("confidence_reasons", ""),
            "is_known_treatment": c.get("is_known_treatment", False),
            "key_papers": c.get("key_papers", []),
            "repurposing_papers": c.get("repurposing_papers", []),
            "semantic_scholar_papers": c.get("semantic_scholar_papers", []),
            "most_cited_paper": c.get("most_cited_paper"),
            "trial_phases": c.get("trial_phases", {}),
            "has_failed_trial": c.get("has_failed_trial", False),
            "failed_trial_phase": c.get("failed_trial_phase", ""),
            "failed_trial_count": c.get("failed_trial_count", 0),
        })
    return out


def build_html(candidates, cross_disease, stats):
    diseases_list = sorted(set(c["disease"] for c in candidates))
    js_candidates = prep_js_candidates(candidates)
    data_json = json.dumps(js_candidates, separators=(",", ":"))
    cross_json = json.dumps(cross_disease, separators=(",", ":"))
    diseases_json = json.dumps(diseases_list, separators=(",", ":"))
    generated = datetime.now().strftime("%Y-%m-%d")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenCure Explorer - AI Drug Repurposing Dashboard</title>
<meta name="description" content="Interactive dashboard for 245 AI-predicted drug repurposing candidates across 25 neglected and rare diseases.">
<style>{CSS}</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <div class="hero-text">
      <h1>Open<span class="accent">Cure</span> Explorer</h1>
      <p class="subtitle">AI Drug Repurposing Dashboard</p>
      <p class="tagline">Multi-pillar computational predictions for 25 underserved diseases &middot; Updated {generated}</p>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-num">{stats['total']}</div><div class="stat-label">Candidates</div></div>
      <div class="stat-card"><div class="stat-num">{stats['diseases']}</div><div class="stat-label">Diseases</div></div>
      <div class="stat-card hl"><div class="stat-num">{stats['breakthrough']}</div><div class="stat-label">Breakthroughs</div></div>
      <div class="stat-card"><div class="stat-num">{stats['high_confidence']}</div><div class="stat-label">High Confidence</div></div>
    </div>
  </div>
</header>

<nav class="nav-bar" id="navbar">
  <a href="#predictions">Predictions</a>
  <a href="#cross-disease">Cross-Disease</a>
  <a href="#breakthroughs">Breakthroughs</a>
  <a href="#methodology">Methods</a>
  <a href="https://github.com/SimonBartosDev/opencure" target="_blank" rel="noopener">GitHub</a>
</nav>

<div class="guide-toggle" id="guide-toggle">
  <button onclick="document.getElementById('guide-box').classList.toggle('open')">How to read this dashboard</button>
  <div class="guide-box" id="guide-box">
    <p><strong>Confidence</strong>: HIGH = multiple evidence types agree; MEDIUM = some support; LOW = computational-only.</p>
    <p><strong>Novelty</strong>: BREAKTHROUGH = zero published literature; NOVEL = minimal; EMERGING/KNOWN/ESTABLISHED = increasing prior evidence.</p>
    <p><strong>Pillars</strong>: How many of the 8 independent AI methods support this prediction (more = stronger signal).</p>
    <p><strong>MR Score</strong>: Mendelian Randomization — causal genetic evidence (0-1, higher = stronger causal support from human genetics).</p>
    <p><strong>Score</strong>: Combined weighted score across all pillars with convergence bonus. Higher = more AI methods agree.</p>
    <p>Click any row to see full evidence: papers, trials, molecular data, and genetic support.</p>
  </div>
</div>

<main>

<section id="predictions" class="section">
  <h2>All Predictions</h2>
  <div class="filters">
    <select id="f-disease"><option value="">All Diseases</option></select>
    <select id="f-conf">
      <option value="">All Confidence</option>
      <option value="HIGH">HIGH</option>
      <option value="MEDIUM">MEDIUM</option>
      <option value="LOW">LOW</option>
    </select>
    <select id="f-nov">
      <option value="">All Novelty</option>
      <option value="BREAKTHROUGH">BREAKTHROUGH</option>
      <option value="NOVEL">NOVEL</option>
      <option value="EMERGING">EMERGING</option>
      <option value="KNOWN">KNOWN</option>
      <option value="ESTABLISHED">ESTABLISHED</option>
    </select>
    <select id="f-evidence">
      <option value="">All Evidence</option>
      <option value="trials">Has Clinical Trials</option>
      <option value="pubmed">Has PubMed Evidence</option>
      <option value="computational">Pure Computational</option>
    </select>
    <input type="text" id="f-search" placeholder="Search drug name...">
  </div>
  <div class="filter-row2">
    <label class="pill-filter"><input type="checkbox" id="f-known"> Hide known treatments</label>
    <label class="range-label">Min pillars: <input type="range" id="f-pillars" min="1" max="8" value="1"><span id="f-pillars-val">1</span></label>
    <span id="result-count" class="result-count"></span>
    <button class="btn-export" onclick="exportCSV()">Download CSV</button>
  </div>
  <div id="disease-header" class="disease-header" style="display:none"></div>
  <div class="table-wrap">
    <table id="pred-table">
      <thead><tr>
        <th data-sort="combined_score" class="sort-desc">Rank</th>
        <th data-sort="disease">Disease <span class="sa"></span></th>
        <th data-sort="drug_name">Drug <span class="sa"></span></th>
        <th data-sort="combined_score">Score <span class="sa"></span></th>
        <th data-sort="confidence">Confidence <span class="sa"></span></th>
        <th data-sort="novelty_level">Novelty <span class="sa"></span></th>
        <th data-sort="pillars_hit">Pillars <span class="sa"></span></th>
        <th data-sort="mr_score">MR <span class="sa"></span></th>
        <th>Evidence</th>
      </tr></thead>
      <tbody id="pred-body"></tbody>
    </table>
  </div>
</section>

<section id="cross-disease" class="section">
  <h2>Cross-Disease Drug Network</h2>
  <p class="section-desc">Drugs predicted as novel candidates for <strong>multiple diseases</strong> — cross-disease convergence from independent AI pillars suggests shared biological mechanisms and increases prediction confidence.</p>
  <div id="cross-cards" class="cross-grid"></div>
</section>

<section id="breakthroughs" class="section">
  <h2>Top Breakthrough Predictions</h2>
  <p class="section-desc">Highest-scoring predictions with <strong>no prior published evidence</strong> — genuinely novel computational discoveries. Click any card for full evidence.</p>
  <div id="bt-cards" class="bt-grid"></div>
</section>

<section id="methodology" class="section">
  <h2>Methodology: 8-Pillar AI Scoring</h2>
  <div class="method-grid">
    <div class="method-card"><div class="mi">1</div><h3>TransE Embeddings</h3><p>Knowledge graph link prediction using DRKG (5.87M biological relationships)</p></div>
    <div class="method-card"><div class="mi">2</div><h3>RotatE / PyKEEN</h3><p>Complex relation pattern scoring via rotation-based graph embeddings</p></div>
    <div class="method-card"><div class="mi">3</div><h3>TxGNN</h3><p>Graph neural network for therapeutic use prediction (Harvard)</p></div>
    <div class="method-card"><div class="mi">4</div><h3>Molecular Fingerprints</h3><p>RDKit ECFP similarity to known treatments for each disease</p></div>
    <div class="method-card"><div class="mi">5</div><h3>ChemBERTa</h3><p>Transformer-learned molecular representations capturing deep structural patterns</p></div>
    <div class="method-card"><div class="mi">6</div><h3>Gene Signatures</h3><p>L1000CDS2 transcriptomic reversal — drugs that reverse disease gene expression</p></div>
    <div class="method-card"><div class="mi">7</div><h3>Network Proximity</h3><p>STRING protein-protein interaction shortest paths between drug targets and disease genes</p></div>
    <div class="method-card"><div class="mi">8</div><h3>Mendelian Randomization</h3><p>Causal genetic evidence from Open Targets — strongest form of drug target validation</p></div>
  </div>
  <div class="method-note">
    <p>Each drug receives a <strong>dynamic weighted score</strong> across all applicable pillars, with convergence bonuses when multiple independent methods agree. Candidates are then enriched with evidence from PubMed, ClinicalTrials.gov, FDA FAERS, Semantic Scholar, and L1000CDS2.</p>
  </div>
</section>

<footer>
  <p>OpenCure is open-source (Apache 2.0) and free for all researchers. <a href="https://github.com/SimonBartosDev/opencure">GitHub</a></p>
</footer>

</main>

<!-- DETAIL PANEL -->
<div class="panel-overlay" id="panel-overlay" onclick="closePanel()"></div>
<div class="detail-panel" id="detail-panel">
  <div class="panel-header" id="panel-header"></div>
  <div class="panel-tabs" id="panel-tabs"></div>
  <div class="panel-body" id="panel-body"></div>
</div>

<script>
const DATA={data_json};
const CROSS={cross_json};
const DISEASES={diseases_json};
{JS}
</script>
</body>
</html>"""


CSS = r"""
:root{--bg:#f8f9fb;--surface:#fff;--border:#e2e8f0;--text:#1a202c;--text2:#4a5568;--text3:#718096;--accent:#2563eb;--accent-l:#dbeafe;--green:#059669;--green-l:#d1fae5;--orange:#d97706;--orange-l:#fef3c7;--red:#dc2626;--red-l:#fee2e2;--purple:#7c3aed;--purple-l:#ede9fe;--r:10px;--sh:0 1px 3px rgba(0,0,0,.08);--sh2:0 4px 14px rgba(0,0,0,.1)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
.hero{background:linear-gradient(135deg,#1e3a5f,#0f172a);color:#fff;padding:2.5rem 2rem 1.8rem}
.hero-inner{max-width:1200px;margin:0 auto}
.hero h1{font-size:2.2rem;font-weight:800;letter-spacing:-.03em}
.accent{color:#60a5fa}
.subtitle{font-size:1.1rem;color:#93c5fd;margin-top:.2rem;font-weight:500}
.tagline{font-size:.88rem;color:#94a3b8;margin-top:.15rem}
.stats-row{display:flex;gap:.8rem;margin-top:1.2rem;flex-wrap:wrap}
.stat-card{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:var(--r);padding:.8rem 1.2rem;min-width:110px;text-align:center}
.stat-card.hl{background:rgba(96,165,250,.15);border-color:rgba(96,165,250,.3)}
.stat-num{font-size:1.8rem;font-weight:800}
.stat-label{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-top:.1rem}
.nav-bar{position:sticky;top:0;z-index:100;background:var(--surface);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;gap:0;overflow-x:auto;box-shadow:var(--sh)}
.nav-bar a{padding:.75rem 1.1rem;text-decoration:none;color:var(--text2);font-size:.88rem;font-weight:500;white-space:nowrap;border-bottom:2px solid transparent;transition:all .15s}
.nav-bar a:hover{color:var(--accent);border-bottom-color:var(--accent)}
.guide-toggle{max-width:1200px;margin:.8rem auto 0;padding:0 1.5rem}
.guide-toggle button{background:none;border:1px solid var(--border);border-radius:8px;padding:.4rem .8rem;font-size:.82rem;color:var(--text2);cursor:pointer}
.guide-toggle button:hover{border-color:var(--accent);color:var(--accent)}
.guide-box{display:none;margin-top:.5rem;padding:.8rem 1rem;background:var(--accent-l);border-radius:8px;font-size:.85rem;color:#1e40af}
.guide-box.open{display:block}
.guide-box p{margin-bottom:.3rem}
main{max-width:1200px;margin:0 auto;padding:0 1.5rem 3rem}
.section{margin-top:2rem}
.section h2{font-size:1.4rem;font-weight:700;margin-bottom:.4rem}
.section-desc{color:var(--text2);margin-bottom:1rem;font-size:.92rem}
.filters{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem}
.filters select,.filters input[type=text]{padding:.5rem .7rem;border:1px solid var(--border);border-radius:8px;font-size:.85rem;background:var(--surface);color:var(--text);outline:none}
.filters select:focus,.filters input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-l)}
.filter-row2{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin-bottom:.8rem}
.pill-filter{font-size:.82rem;color:var(--text2);display:flex;align-items:center;gap:.3rem;cursor:pointer}
.range-label{font-size:.82rem;color:var(--text2);display:flex;align-items:center;gap:.4rem}
.range-label input[type=range]{width:80px}
.result-count{font-size:.82rem;color:var(--text3);margin-left:auto}
.btn-export{padding:.4rem .8rem;border:1px solid var(--accent);border-radius:8px;background:none;color:var(--accent);font-size:.82rem;font-weight:500;cursor:pointer}
.btn-export:hover{background:var(--accent);color:#fff}
.disease-header{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1rem 1.2rem;margin-bottom:.8rem;box-shadow:var(--sh)}
.table-wrap{overflow-x:auto;background:var(--surface);border-radius:var(--r);box-shadow:var(--sh);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.88rem}
thead{background:#f1f5f9}
th{padding:.65rem .8rem;text-align:left;font-weight:600;color:var(--text2);cursor:pointer;user-select:none;white-space:nowrap;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
th:hover{color:var(--accent)}
.sa{font-size:.7rem;margin-left:.15rem}
td{padding:.55rem .8rem;border-top:1px solid var(--border);vertical-align:middle}
tbody tr{cursor:pointer;transition:background .1s}
tbody tr:hover{background:#f8fafc}
.b{display:inline-block;padding:.12rem .5rem;border-radius:6px;font-size:.72rem;font-weight:600;letter-spacing:.02em}
.b-high{background:var(--green-l);color:var(--green)}.b-medium{background:var(--orange-l);color:var(--orange)}.b-low{background:var(--red-l);color:var(--red)}
.b-breakthrough{background:var(--purple-l);color:var(--purple)}.b-novel{background:var(--accent-l);color:var(--accent)}.b-emerging{background:var(--orange-l);color:var(--orange)}.b-known,.b-established{background:#f1f5f9;color:var(--text3)}
.ev-dots{display:flex;gap:.25rem;align-items:center}
.ev-dot{width:7px;height:7px;border-radius:50%;background:#cbd5e1}
.ev-dot.on{background:var(--accent)}
.ev-dot.trial{background:var(--green)}
.ev-dot.warn{background:var(--red)}
.score-bar{display:inline-block;width:50px;height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;vertical-align:middle;margin-right:.3rem}
.score-fill{height:100%;border-radius:3px;background:var(--accent)}
.rank-badge{font-size:.7rem;color:var(--text3);margin-left:.3rem}
/* Cross-disease */
.cross-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.8rem}
.cross-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1rem;box-shadow:var(--sh)}
.cross-card h3{font-size:1rem;font-weight:700;margin-bottom:.3rem}
.cross-card .dc{font-size:.8rem;color:var(--accent);font-weight:600;margin-bottom:.4rem}
.cross-tag{display:inline-block;background:#f1f5f9;color:var(--text2);padding:.15rem .5rem;border-radius:6px;font-size:.75rem;margin:.12rem;cursor:pointer;transition:all .1s}
.cross-tag:hover{background:var(--accent-l);color:var(--accent)}
/* Breakthroughs */
.bt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:.8rem}
.bt-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1.1rem;box-shadow:var(--sh);cursor:pointer;transition:box-shadow .15s}
.bt-card:hover{box-shadow:var(--sh2)}
.bt-card h3{font-size:.95rem;font-weight:700;margin-bottom:.1rem}
.bt-disease{font-size:.85rem;color:var(--accent);font-weight:500}
.bt-meta{font-size:.8rem;color:var(--text3);margin-top:.3rem}
.bt-pillars{display:flex;gap:.2rem;margin-top:.5rem;flex-wrap:wrap}
.pillar-bar-mini{height:4px;border-radius:2px;flex:1;min-width:20px}
/* Methodology */
.method-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:.8rem}
.method-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1rem;box-shadow:var(--sh)}
.mi{width:28px;height:28px;background:var(--accent);color:#fff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;margin-bottom:.5rem}
.method-card h3{font-size:.9rem;font-weight:700;margin-bottom:.2rem}
.method-card p{font-size:.82rem;color:var(--text2)}
.method-note{margin-top:1rem;padding:.8rem 1rem;background:var(--accent-l);border-radius:var(--r);font-size:.88rem;color:#1e40af}
footer{text-align:center;padding:1.5rem;color:var(--text3);font-size:.82rem;border-top:1px solid var(--border);margin-top:2.5rem}
footer a{color:var(--accent);text-decoration:none}
/* DETAIL PANEL */
.panel-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.35);z-index:999}
.panel-overlay.open{display:block}
.detail-panel{position:fixed;top:0;right:-700px;width:680px;max-width:100vw;height:100vh;background:var(--surface);z-index:1000;overflow-y:auto;box-shadow:-4px 0 20px rgba(0,0,0,.15);transition:right .25s ease}
.detail-panel.open{right:0}
.panel-header{padding:1.2rem 1.5rem;border-bottom:1px solid var(--border);position:relative}
.panel-header h2{font-size:1.3rem;font-weight:700;padding-right:2rem}
.panel-header .ph-disease{font-size:1rem;color:var(--accent);font-weight:500;margin-top:.1rem}
.panel-header .ph-badges{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.5rem}
.panel-close{position:absolute;top:1rem;right:1rem;background:none;border:none;font-size:1.4rem;cursor:pointer;color:var(--text3);width:32px;height:32px;display:flex;align-items:center;justify-content:center;border-radius:8px}
.panel-close:hover{background:#f1f5f9}
.failed-warn{background:var(--red-l);color:var(--red);padding:.5rem .8rem;border-radius:8px;font-size:.82rem;font-weight:600;margin-top:.5rem}
.panel-tabs{display:flex;border-bottom:1px solid var(--border);padding:0 1.5rem;overflow-x:auto}
.panel-tab{padding:.6rem 1rem;font-size:.85rem;font-weight:500;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.panel-tab:hover{color:var(--text)}
.panel-tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.panel-body{padding:1.2rem 1.5rem}
/* Panel content */
.pillar-rows{display:flex;flex-direction:column;gap:.4rem}
.pillar-row{display:flex;align-items:center;gap:.6rem}
.pillar-name{width:130px;font-size:.82rem;font-weight:500;color:var(--text2);flex-shrink:0}
.pillar-track{flex:1;height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden}
.pillar-fill{height:100%;border-radius:4px;transition:width .3s}
.pillar-val{font-size:.78rem;color:var(--text3);width:60px;text-align:right;flex-shrink:0}
.reason-list{list-style:none;padding:0;margin:.8rem 0}
.reason-list li{padding:.3rem 0;font-size:.85rem;color:var(--text2);border-bottom:1px solid #f1f5f9;padding-left:.8rem;position:relative}
.reason-list li::before{content:"";position:absolute;left:0;top:.55rem;width:4px;height:4px;border-radius:50%;background:var(--accent)}
.paper-card{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:.7rem .9rem;margin-bottom:.5rem;transition:border-color .15s}
.paper-card:hover{border-color:var(--accent)}
.paper-card .pc-title{font-size:.88rem;font-weight:600;color:var(--text)}
.paper-card .pc-title a{color:var(--accent);text-decoration:none}
.paper-card .pc-title a:hover{text-decoration:underline}
.paper-card .pc-meta{font-size:.78rem;color:var(--text3);margin-top:.2rem}
.paper-card .pc-badge{display:inline-block;font-size:.68rem;padding:.1rem .4rem;border-radius:4px;font-weight:600;margin-left:.3rem}
.pc-repurposing{background:var(--purple-l);color:var(--purple)}
.pc-cited{background:var(--green-l);color:var(--green)}
.trial-phase{display:inline-flex;align-items:center;gap:.3rem;background:#f1f5f9;border-radius:6px;padding:.3rem .6rem;margin:.2rem;font-size:.82rem;font-weight:500}
.trial-phase .tp-num{font-weight:700;color:var(--accent)}
.interp-box{background:#f8fafc;border-radius:8px;padding:.7rem .9rem;font-size:.85rem;color:var(--text2);margin:.5rem 0;line-height:1.5}
.interp-box strong{color:var(--text)}
.empty-state{text-align:center;padding:2rem;color:var(--text3);font-size:.88rem}
@media(max-width:768px){
.hero{padding:1.5rem 1rem 1.2rem}
.hero h1{font-size:1.6rem}
.stats-row{gap:.4rem}
.stat-card{padding:.5rem .7rem;min-width:80px}
.stat-num{font-size:1.3rem}
.filters{flex-direction:column}
.filters select,.filters input{width:100%}
.detail-panel{width:100vw}
.cross-grid,.bt-grid{grid-template-columns:1fr}
.filter-row2{flex-direction:column;align-items:flex-start}
}
"""

JS = r"""
let sortCol='combined_score',sortAsc=false,filtered=[...DATA];
(function init(){
  const sel=document.getElementById('f-disease');
  DISEASES.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;sel.appendChild(o)});
  ['f-disease','f-conf','f-nov','f-evidence'].forEach(id=>document.getElementById(id).addEventListener('change',applyFilters));
  document.getElementById('f-search').addEventListener('input',applyFilters);
  document.getElementById('f-known').addEventListener('change',applyFilters);
  const pr=document.getElementById('f-pillars');
  pr.addEventListener('input',()=>{document.getElementById('f-pillars-val').textContent=pr.value;applyFilters()});
  document.querySelectorAll('#pred-table th[data-sort]').forEach(th=>{
    th.addEventListener('click',()=>{
      const col=th.dataset.sort;
      if(sortCol===col)sortAsc=!sortAsc;else{sortCol=col;sortAsc=col==='disease'||col==='drug_name'}
      renderTable()
    })
  });
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closePanel()});
  applyFilters();renderCross();renderBT()
})();

function applyFilters(){
  const disease=document.getElementById('f-disease').value;
  const conf=document.getElementById('f-conf').value;
  const nov=document.getElementById('f-nov').value;
  const ev=document.getElementById('f-evidence').value;
  const search=document.getElementById('f-search').value.toLowerCase();
  const hideKnown=document.getElementById('f-known').checked;
  const minP=parseInt(document.getElementById('f-pillars').value);
  filtered=DATA.filter(c=>{
    if(disease&&c.disease!==disease)return false;
    if(conf&&c.confidence!==conf)return false;
    if(nov&&c.novelty_level!==nov)return false;
    if(search&&!c.drug_name.toLowerCase().includes(search))return false;
    if(hideKnown&&c.is_known_treatment)return false;
    if(c.pillars_hit<minP)return false;
    if(ev==='trials'&&c.clinical_trials<1)return false;
    if(ev==='pubmed'&&c.pubmed_articles<1)return false;
    if(ev==='computational'&&(c.pubmed_articles>0||c.clinical_trials>0))return false;
    return true;
  });
  document.getElementById('result-count').textContent=filtered.length+' of '+DATA.length;
  showDiseaseHeader(disease);
  renderTable();
}

function showDiseaseHeader(disease){
  const dh=document.getElementById('disease-header');
  if(!disease){dh.style.display='none';return}
  const all=DATA.filter(c=>c.disease===disease);
  const hi=all.filter(c=>c.confidence==='HIGH').length;
  const bt=all.filter(c=>c.novelty_level==='BREAKTHROUGH').length;
  const top3=all.sort((a,b)=>b.combined_score-a.combined_score).slice(0,3);
  dh.style.display='block';
  dh.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem">
    <div><strong style="font-size:1.1rem">${esc(disease)}</strong><br><span style="font-size:.85rem;color:var(--text2)">${all.length} candidates &middot; ${hi} high confidence &middot; ${bt} breakthroughs</span></div>
    <div style="font-size:.82rem;color:var(--text3)">Top: ${top3.map(c=>'<strong>'+esc(c.drug_name)+'</strong>').join(', ')}</div>
  </div>`;
}

function renderTable(){
  filtered.sort((a,b)=>{
    let va=a[sortCol],vb=b[sortCol];
    if(typeof va==='string'){va=va.toLowerCase();vb=vb.toLowerCase()}
    if(va<vb)return sortAsc?-1:1;if(va>vb)return sortAsc?1:-1;return 0
  });
  document.querySelectorAll('#pred-table th[data-sort]').forEach(th=>{
    const a=th.querySelector('.sa');if(a)a.textContent=th.dataset.sort===sortCol?(sortAsc?'\u25B2':'\u25BC'):''
  });
  const tbody=document.getElementById('pred-body');
  // Find global rank for each candidate
  const allSorted=[...DATA].sort((a,b)=>b.combined_score-a.combined_score);
  const rankMap=new Map();allSorted.forEach((c,i)=>rankMap.set(c.drug_name+'|'+c.disease,i+1));

  tbody.innerHTML=filtered.map(c=>{
    const rank=rankMap.get(c.drug_name+'|'+c.disease)||'-';
    const idx=DATA.indexOf(c);
    const hasPub=c.pubmed_articles>0;
    const hasTrial=c.clinical_trials>0;
    const hasFail=c.has_failed_trial;
    return `<tr onclick="openPanel(${idx})">
      <td style="text-align:center;color:var(--text3);font-size:.8rem">#${rank}</td>
      <td>${esc(c.disease)}</td>
      <td><strong>${esc(c.drug_name)}</strong></td>
      <td><span class="score-bar"><span class="score-fill" style="width:${Math.min(c.combined_score/1.25*100,100)}%"></span></span>${c.combined_score.toFixed(2)}</td>
      <td><span class="b b-${c.confidence.toLowerCase()}">${c.confidence}</span></td>
      <td><span class="b b-${c.novelty_level.toLowerCase()}">${c.novelty_level}</span></td>
      <td style="text-align:center">${c.pillars_hit}</td>
      <td>${c.mr_score>0?c.mr_score.toFixed(2):'-'}</td>
      <td><div class="ev-dots">${hasPub?'<span class="ev-dot on" title="PubMed"></span>':'<span class="ev-dot" title="No PubMed"></span>'}${hasTrial?'<span class="ev-dot trial" title="Clinical trials"></span>':'<span class="ev-dot" title="No trials"></span>'}${hasFail?'<span class="ev-dot warn" title="Failed trial"></span>':''}</div></td>
    </tr>`
  }).join('');
}

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}

// ---- DETAIL PANEL ----
let currentTab='summary';
function openPanel(idx){
  const c=DATA[idx];
  currentTab='summary';
  document.getElementById('panel-overlay').classList.add('open');
  document.getElementById('detail-panel').classList.add('open');
  // Header
  const ph=document.getElementById('panel-header');
  const allSorted=[...DATA].sort((a,b)=>b.combined_score-a.combined_score);
  const rank=allSorted.findIndex(x=>x.drug_name===c.drug_name&&x.disease===c.disease)+1;
  ph.innerHTML=`<button class="panel-close" onclick="closePanel()">&times;</button>
    <h2>${esc(c.drug_name)}</h2>
    <div class="ph-disease">${esc(c.disease)}</div>
    <div class="ph-badges">
      <span class="b b-${c.confidence.toLowerCase()}">${c.confidence}</span>
      <span class="b b-${c.novelty_level.toLowerCase()}">${c.novelty_level}</span>
      <span class="b" style="background:#f1f5f9;color:var(--text)">Score: ${c.combined_score.toFixed(2)} (#${rank})</span>
      <span class="b" style="background:#f1f5f9;color:var(--text)">${c.pillars_hit} pillars</span>
      ${c.drug_id?'<span class="b" style="background:#f1f5f9;color:var(--text)">'+esc(c.drug_id)+'</span>':''}
    </div>
    ${c.has_failed_trial?'<div class="failed-warn">Warning: '+c.failed_trial_count+' failed trial(s) reported'+(c.failed_trial_phase?' ('+esc(c.failed_trial_phase)+')':'')+'</div>':''}`;
  // Tabs
  const tabs=[{id:'summary',label:'Summary'},{id:'literature',label:'Literature ('+((c.key_papers||[]).length+(c.repurposing_papers||[]).length+(c.semantic_scholar_papers||[]).length)+')'},{id:'clinical',label:'Clinical'},{id:'molecular',label:'Molecular'},{id:'genetic',label:'Genetic'}];
  document.getElementById('panel-tabs').innerHTML=tabs.map(t=>`<div class="panel-tab${t.id==='summary'?' active':''}" onclick="switchTab('${t.id}',${idx})">${t.label}</div>`).join('');
  renderTab('summary',c);
  document.body.style.overflow='hidden';
}

function closePanel(){
  document.getElementById('panel-overlay').classList.remove('open');
  document.getElementById('detail-panel').classList.remove('open');
  document.body.style.overflow='';
}

function switchTab(tab,idx){
  currentTab=tab;
  document.querySelectorAll('.panel-tab').forEach(t=>t.classList.toggle('active',t.textContent.startsWith(tab.charAt(0).toUpperCase()+tab.slice(1))||t.onclick.toString().includes("'"+tab+"'")));
  // Fix active state
  document.querySelectorAll('.panel-tab').forEach(t=>{
    const match=t.getAttribute('onclick').match(/'(\w+)'/);
    t.classList.toggle('active',match&&match[1]===tab)
  });
  renderTab(tab,DATA[idx]);
}

function renderTab(tab,c){
  const pb=document.getElementById('panel-body');
  if(tab==='summary')pb.innerHTML=renderSummary(c);
  else if(tab==='literature')pb.innerHTML=renderLiterature(c);
  else if(tab==='clinical')pb.innerHTML=renderClinical(c);
  else if(tab==='molecular')pb.innerHTML=renderMolecular(c);
  else if(tab==='genetic')pb.innerHTML=renderGenetic(c);
}

function pillarBar(label,value,max,color,detail){
  const pct=max>0?Math.min(value/max*100,100):0;
  const col=pct>60?'var(--green)':pct>25?'var(--orange)':'#cbd5e1';
  return `<div class="pillar-row"><span class="pillar-name">${label}</span><div class="pillar-track"><div class="pillar-fill" style="width:${pct}%;background:${color||col}"></div></div><span class="pillar-val">${detail}</span></div>`;
}

function renderSummary(c){
  const transe=c.transe_rank>0?Math.max(0,1-c.transe_rank/500):0;
  const bars=[
    pillarBar('TransE',transe,1,null,c.transe_rank>0?'#'+c.transe_rank:'-'),
    pillarBar('Mol. Similarity',c.mol_similarity,1,null,c.mol_similarity>0?(c.mol_similarity*100).toFixed(0)+'%':'-'),
    pillarBar('MR Score',c.mr_score,1,null,c.mr_score>0?c.mr_score.toFixed(2):'-'),
    pillarBar('Gene Signature',c.signature_reversal?0.8:0,1,null,c.signature_reversal?'Rank '+c.signature_rank:'None'),
    pillarBar('FAERS Signal',c.faers_signal==='strong'?0.9:c.faers_signal==='moderate'?0.5:c.faers_signal==='weak'?0.2:0,1,null,c.faers_signal!=='none'?c.faers_signal:'-'),
    pillarBar('Shared Targets',Math.min(c.shared_targets/5,1),1,null,c.shared_targets||'-'),
    pillarBar('PubMed',Math.min(c.pubmed_articles/20,1),1,null,c.pubmed_articles||'0'),
    pillarBar('Clinical Trials',Math.min(c.clinical_trials/5,1),1,null,c.clinical_trials||'0'),
  ];
  const reasons=c.confidence_reasons?c.confidence_reasons.split(';').filter(r=>r.trim()):[];
  return `<h3 style="font-size:.95rem;margin-bottom:.6rem">Pillar Strength</h3>
    <div class="pillar-rows">${bars.join('')}</div>
    ${reasons.length?'<h3 style="font-size:.95rem;margin:1rem 0 .3rem">Confidence Reasoning</h3><ul class="reason-list">'+reasons.map(r=>'<li>'+esc(r.trim())+'</li>').join('')+'</ul>':''}
    ${c.novelty_interpretation?'<div class="interp-box"><strong>Novelty:</strong> '+esc(c.novelty_interpretation)+'</div>':''}`;
}

function paperCard(p,badge){
  const link=p.pmid?'https://pubmed.ncbi.nlm.nih.gov/'+p.pmid:(p.url||'');
  return `<div class="paper-card">
    <div class="pc-title">${link?'<a href="'+link+'" target="_blank" rel="noopener">'+esc(p.title||'Untitled')+'</a>':esc(p.title||'Untitled')}${badge||''}</div>
    <div class="pc-meta">${esc(p.authors||'')}${p.journal?' &middot; '+esc(p.journal):''}${p.year?' &middot; '+p.year:''}${p.citations?' &middot; <strong>'+p.citations+' citations</strong>':''}${p.pmid?' &middot; PMID: '+p.pmid:''}</div>
  </div>`;
}

function renderLiterature(c){
  let html='';
  // Most cited paper
  if(c.most_cited_paper&&c.most_cited_paper.title){
    html+='<h3 style="font-size:.95rem;margin-bottom:.5rem">Most Cited Paper</h3>';
    html+=paperCard(c.most_cited_paper,'<span class="pc-badge pc-cited">'+c.most_cited_paper.citations+' citations</span>');
  }
  // Repurposing papers
  if(c.repurposing_papers&&c.repurposing_papers.length){
    html+='<h3 style="font-size:.95rem;margin:1rem 0 .5rem">Repurposing-Specific Papers</h3>';
    c.repurposing_papers.forEach(p=>{html+=paperCard(p,'<span class="pc-badge pc-repurposing">Repurposing</span>')});
  }
  // Key papers
  if(c.key_papers&&c.key_papers.length){
    html+='<h3 style="font-size:.95rem;margin:1rem 0 .5rem">Key PubMed Papers</h3>';
    c.key_papers.forEach(p=>{html+=paperCard(p)});
  }
  // Semantic Scholar
  if(c.semantic_scholar_papers&&c.semantic_scholar_papers.length){
    html+='<h3 style="font-size:.95rem;margin:1rem 0 .5rem">Semantic Scholar</h3>';
    c.semantic_scholar_papers.forEach(p=>{html+=paperCard(p)});
  }
  if(!html)html='<div class="empty-state">No published literature found for this drug-disease combination.<br>This is a purely computational prediction.</div>';
  // Summary
  const total=(c.key_papers||[]).length+(c.repurposing_papers||[]).length+(c.semantic_scholar_papers||[]).length;
  return `<div style="margin-bottom:.8rem;font-size:.85rem;color:var(--text2)">${c.pubmed_articles} PubMed articles (${c.pubmed_treatment} treatment, ${c.pubmed_repurposing} repurposing) &middot; ${total} papers shown below</div>`+html;
}

function renderClinical(c){
  let html='';
  const phases=c.trial_phases||{};
  const phaseKeys=Object.keys(phases);
  if(phaseKeys.length){
    html+='<h3 style="font-size:.95rem;margin-bottom:.5rem">Trial Phases</h3><div style="display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.8rem">';
    phaseKeys.forEach(p=>{html+=`<div class="trial-phase">${esc(p.replace('PHASE','Phase '))}: <span class="tp-num">${phases[p]}</span></div>`});
    html+='</div>';
  }
  html+=`<div class="interp-box"><strong>Total trials:</strong> ${c.clinical_trials||0}</div>`;
  if(c.has_failed_trial){
    html+=`<div class="failed-warn" style="margin-top:.5rem">Failed trial warning: ${c.failed_trial_count} trial(s) failed${c.failed_trial_phase?' at '+esc(c.failed_trial_phase):''}</div>`;
  }
  // Link to ClinicalTrials.gov search
  const searchQ=encodeURIComponent(c.drug_name+' '+c.disease);
  html+=`<div style="margin-top:.8rem"><a href="https://clinicaltrials.gov/search?term=${searchQ}" target="_blank" rel="noopener" style="color:var(--accent);font-size:.88rem">Search ClinicalTrials.gov &rarr;</a></div>`;
  if(!c.clinical_trials&&!c.has_failed_trial)html='<div class="empty-state">No clinical trials found for this drug-disease combination.<br><a href="https://clinicaltrials.gov/search?term='+searchQ+'" target="_blank" rel="noopener" style="color:var(--accent)">Search ClinicalTrials.gov &rarr;</a></div>';
  return html;
}

function renderMolecular(c){
  let html='';
  // Similarity
  if(c.mol_similarity>0){
    html+=`<div class="interp-box"><strong>Molecular similarity:</strong> ${(c.mol_similarity*100).toFixed(1)}% to <strong>${esc(c.similar_to)}</strong><br><span style="font-size:.82rem;color:var(--text3)">Based on Morgan/ECFP fingerprints and ChemBERTa transformer embeddings</span></div>`;
  }
  // FAERS
  if(c.faers_interpretation){
    html+=`<h3 style="font-size:.95rem;margin:1rem 0 .4rem">FDA FAERS Real-World Signal</h3>`;
    html+=`<div class="interp-box"><strong>Signal strength:</strong> ${c.faers_signal} (${c.faers_cooccurrences.toLocaleString()} co-occurrences)<br>${esc(c.faers_interpretation)}</div>`;
  } else if(c.faers_signal!=='none'){
    html+=`<div class="interp-box"><strong>FAERS signal:</strong> ${c.faers_signal} (${c.faers_cooccurrences.toLocaleString()} co-occurrences)</div>`;
  }
  // Gene Signature
  if(c.signature_interpretation){
    html+=`<h3 style="font-size:.95rem;margin:1rem 0 .4rem">Gene Signature Reversal</h3>`;
    html+=`<div class="interp-box">${esc(c.signature_interpretation)}</div>`;
  }
  // Top reversers
  if(c.signature_top_reversers&&c.signature_top_reversers.length){
    html+=`<div style="margin-top:.5rem;font-size:.82rem;color:var(--text3)">Top reversing compounds: ${c.signature_top_reversers.map(r=>typeof r==='string'?esc(r):esc(r.name||r.drug||JSON.stringify(r))).join(', ')}</div>`;
  }
  // Shared targets
  if(c.shared_targets_list&&c.shared_targets_list.length){
    html+=`<h3 style="font-size:.95rem;margin:1rem 0 .4rem">Shared Protein Targets</h3>`;
    html+=`<div style="display:flex;flex-wrap:wrap;gap:.3rem">${c.shared_targets_list.map(t=>'<span class="b" style="background:#f1f5f9;color:var(--text)">'+esc(typeof t==='string'?t:t.name||JSON.stringify(t))+'</span>').join('')}</div>`;
  }
  if(!html)html='<div class="empty-state">No molecular evidence data available for this prediction.</div>';
  return html;
}

function renderGenetic(c){
  let html='';
  if(c.mr_score>0){
    const strength=c.mr_score>0.7?'Strong':c.mr_score>0.4?'Moderate':'Weak';
    const color=c.mr_score>0.7?'var(--green)':c.mr_score>0.4?'var(--orange)':'var(--text3)';
    html+=`<div class="interp-box"><strong>Mendelian Randomization:</strong> <span style="color:${color};font-weight:700">${strength}</span> causal evidence (score: ${c.mr_score.toFixed(4)})<br>${c.mr_genetic_targets} genetic target(s) with causal support from human GWAS data<br><span style="font-size:.82rem;color:var(--text3)">MR uses genetic variants as natural experiments to establish causal relationships between drug targets and disease outcomes</span></div>`;
  }
  if(c.transe_rank>0){
    html+=`<h3 style="font-size:.95rem;margin:1rem 0 .4rem">Knowledge Graph Evidence</h3>`;
    html+=`<div class="interp-box"><strong>TransE rank:</strong> #${c.transe_rank} out of ~10,500 drugs<br><span style="font-size:.82rem;color:var(--text3)">${c.transe_rank<50?'Very strong':''}${c.transe_rank>=50&&c.transe_rank<200?'Strong':''}${c.transe_rank>=200&&c.transe_rank<500?'Moderate':''}${c.transe_rank>=500?'Weak':''} signal from DRKG knowledge graph embedding (5.87M biological relationships)</span></div>`;
  }
  if(!html)html='<div class="empty-state">No genetic evidence data available for this prediction.</div>';
  return html;
}

// ---- CROSS-DISEASE ----
function renderCross(){
  const container=document.getElementById('cross-cards');
  const drugs=Object.entries(CROSS).sort((a,b)=>b[1].length-a[1].length);
  container.innerHTML=drugs.map(([drug,entries])=>`
    <div class="cross-card">
      <h3>${esc(drug)}</h3>
      <div class="dc">${entries.length} diseases predicted</div>
      <div>${entries.map(e=>'<span class="cross-tag" onclick="filterToDrug(\''+esc(drug).replace(/'/g,"\\'")+'\',\''+esc(e.disease).replace(/'/g,"\\'")+'\')">' +esc(e.disease)+' <small>('+e.score.toFixed(2)+')</small></span>').join('')}</div>
    </div>`).join('');
}

function filterToDrug(drug,disease){
  document.getElementById('f-disease').value=disease;
  document.getElementById('f-search').value=drug;
  applyFilters();
  document.getElementById('predictions').scrollIntoView({behavior:'smooth'});
  // Open the panel for this drug+disease
  setTimeout(()=>{
    const idx=DATA.findIndex(c=>c.drug_name===drug&&c.disease===disease);
    if(idx>=0)openPanel(idx);
  },300);
}

// ---- BREAKTHROUGHS ----
function renderBT(){
  const bt=DATA.filter(c=>c.novelty_level==='BREAKTHROUGH').sort((a,b)=>b.combined_score-a.combined_score).slice(0,10);
  const container=document.getElementById('bt-cards');
  container.innerHTML=bt.map(c=>{
    const idx=DATA.indexOf(c);
    // Mini pillar bars
    const pillars=[
      {v:c.transe_rank>0?Math.max(0,1-c.transe_rank/500):0,c:'#3b82f6'},
      {v:c.mol_similarity,c:'#8b5cf6'},
      {v:c.mr_score,c:'#059669'},
      {v:c.signature_reversal?0.8:0,c:'#d97706'},
      {v:c.faers_signal==='strong'?0.9:c.faers_signal==='moderate'?0.5:c.faers_signal==='weak'?0.2:0,c:'#dc2626'},
      {v:Math.min(c.shared_targets/5,1),c:'#0891b2'},
      {v:Math.min(c.pubmed_articles/20,1),c:'#6366f1'},
      {v:Math.min(c.clinical_trials/5,1),c:'#059669'},
    ];
    return `<div class="bt-card" onclick="openPanel(${idx})">
      <h3>${esc(c.drug_name)}</h3>
      <div class="bt-disease">${esc(c.disease)}</div>
      <div class="bt-meta">Score: <strong>${c.combined_score.toFixed(2)}</strong> &middot; ${c.pillars_hit} pillars &middot; MR: ${c.mr_score>0?c.mr_score.toFixed(2):'-'} &middot; <span class="b b-${c.confidence.toLowerCase()}" style="font-size:.68rem">${c.confidence}</span></div>
      <div class="bt-pillars">${pillars.map(p=>'<div class="pillar-bar-mini" style="background:'+(p.v>0?p.c:'#e2e8f0')+'"></div>').join('')}</div>
    </div>`
  }).join('');
}

// ---- CSV EXPORT ----
function exportCSV(){
  const headers=['Disease','Drug','DrugBank ID','Score','Confidence','Novelty','Pillars','MR Score','PubMed Articles','Clinical Trials','Mol Similarity','Similar To','FAERS Signal','TransE Rank'];
  const rows=filtered.map(c=>[c.disease,c.drug_name,c.drug_id,c.combined_score,c.confidence,c.novelty_level,c.pillars_hit,c.mr_score,c.pubmed_articles,c.clinical_trials,c.mol_similarity,c.similar_to,c.faers_signal,c.transe_rank].map(v=>'"'+(v+'').replace(/"/g,'""')+'"').join(','));
  const csv=headers.join(',')+'\n'+rows.join('\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='opencure_predictions.csv';a.click();
}
"""


def main():
    print("Loading database + disease evidence files...")
    candidates = load_data()
    print(f"  {len(candidates)} candidates loaded with rich evidence")

    print("Computing cross-disease analysis...")
    cross = compute_cross_disease(candidates)
    print(f"  {len(cross)} drugs across multiple diseases")

    stats = compute_stats(candidates)
    print(f"  Stats: {stats}")

    print("Building HTML...")
    html_content = build_html(candidates, cross, stats)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html_content)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  Saved: {OUT_PATH} ({size_kb:.0f} KB)")
    print("\nDone! Open docs/index.html in a browser to preview.")


if __name__ == "__main__":
    main()
