#!/usr/bin/env python3
"""Generate the OpenCure Explorer — an interactive static HTML dashboard.

Reads experiments/results/opencure_database.json and produces docs/index.html,
a self-contained single-file dashboard with search, filters, cross-disease
network analysis, and radar charts. Deployed via GitHub Pages.
"""

from __future__ import annotations

import json
import html
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("experiments/results/opencure_database.json")
OUT_PATH = Path("docs/index.html")


def load_data():
    raw = json.loads(DB_PATH.read_text())
    return raw["candidates"]


def compute_cross_disease(candidates):
    """Find drugs predicted for multiple diseases (novel/breakthrough only)."""
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
    return {
        "total": len(candidates),
        "diseases": len(diseases),
        "novel": novel,
        "breakthrough": bt,
        "high_confidence": high,
    }


def build_html(candidates, cross_disease, stats):
    diseases_list = sorted(set(c["disease"] for c in candidates))

    # Prepare data for JS — strip large text fields to keep size reasonable
    js_candidates = []
    for c in candidates:
        js_candidates.append({
            "disease": c["disease"],
            "drug_name": c["drug_name"],
            "drug_id": c["drug_id"],
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
            "faers_signal": c.get("faers_signal", "none"),
            "faers_cooccurrences": c.get("faers_cooccurrences", 0),
            "signature_reversal": c.get("signature_reversal", False),
            "signature_rank": c.get("signature_rank", 0),
            "novelty_interpretation": c.get("novelty_interpretation", ""),
            "confidence_reasons": c.get("confidence_reasons", ""),
            "is_known_treatment": c.get("is_known_treatment", False),
        })

    js_cross = {}
    for drug, entries in cross_disease.items():
        js_cross[drug] = entries

    data_json = json.dumps(js_candidates, separators=(",", ":"))
    cross_json = json.dumps(js_cross, separators=(",", ":"))
    diseases_json = json.dumps(diseases_list, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenCure Explorer - AI Drug Repurposing Dashboard</title>
<style>
{CSS}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <div class="hero-text">
      <h1>Open<span class="accent">Cure</span> Explorer</h1>
      <p class="subtitle">Interactive AI Drug Repurposing Dashboard</p>
      <p class="tagline">Open-source multi-pillar computational predictions for 25 underserved diseases</p>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-num">{stats['total']}</div><div class="stat-label">Candidates</div></div>
      <div class="stat-card"><div class="stat-num">{stats['diseases']}</div><div class="stat-label">Diseases</div></div>
      <div class="stat-card accent-card"><div class="stat-num">{stats['novel']}</div><div class="stat-label">Novel / Breakthrough</div></div>
      <div class="stat-card"><div class="stat-num">{stats['high_confidence']}</div><div class="stat-label">High Confidence</div></div>
    </div>
  </div>
</header>

<nav class="nav-bar" id="navbar">
  <a href="#predictions">All Predictions</a>
  <a href="#cross-disease">Cross-Disease Network</a>
  <a href="#breakthroughs">Top Breakthroughs</a>
  <a href="#methodology">Methodology</a>
  <a href="https://github.com/SimonBartosDev/opencure" target="_blank" rel="noopener">GitHub</a>
</nav>

<main>

<!-- PREDICTIONS TABLE -->
<section id="predictions" class="section">
  <h2>All Predictions</h2>
  <div class="filters">
    <select id="filter-disease"><option value="">All Diseases</option></select>
    <select id="filter-confidence">
      <option value="">All Confidence</option>
      <option value="HIGH">HIGH</option>
      <option value="MEDIUM">MEDIUM</option>
      <option value="LOW">LOW</option>
    </select>
    <select id="filter-novelty">
      <option value="">All Novelty</option>
      <option value="BREAKTHROUGH">BREAKTHROUGH</option>
      <option value="NOVEL">NOVEL</option>
      <option value="EMERGING">EMERGING</option>
      <option value="KNOWN">KNOWN</option>
      <option value="ESTABLISHED">ESTABLISHED</option>
    </select>
    <input type="text" id="filter-search" placeholder="Search drug name...">
    <span id="result-count" class="result-count"></span>
  </div>
  <div class="table-wrap">
    <table id="pred-table">
      <thead>
        <tr>
          <th data-sort="disease">Disease <span class="sort-arrow"></span></th>
          <th data-sort="drug_name">Drug <span class="sort-arrow"></span></th>
          <th data-sort="combined_score">Score <span class="sort-arrow"></span></th>
          <th data-sort="confidence">Confidence <span class="sort-arrow"></span></th>
          <th data-sort="novelty_level">Novelty <span class="sort-arrow"></span></th>
          <th data-sort="pillars_hit">Pillars <span class="sort-arrow"></span></th>
          <th data-sort="mr_score">MR Score <span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody id="pred-body"></tbody>
    </table>
  </div>
</section>

<!-- CROSS-DISEASE NETWORK -->
<section id="cross-disease" class="section">
  <h2>Cross-Disease Drug Network</h2>
  <p class="section-desc">These drugs are predicted as novel repurposing candidates for <strong>multiple diseases</strong>, suggesting shared biological mechanisms. Cross-disease convergence increases confidence — if independent AI pillars predict the same drug across different diseases via different pathways, the signal is likely real.</p>
  <div id="cross-cards" class="cross-grid"></div>
</section>

<!-- TOP BREAKTHROUGHS -->
<section id="breakthroughs" class="section">
  <h2>Top 10 Breakthrough Predictions</h2>
  <p class="section-desc">Highest-scoring predictions with <strong>no prior published evidence</strong> — genuinely novel computational discoveries supported by multiple independent AI pillars.</p>
  <div id="bt-cards" class="bt-grid"></div>
</section>

<!-- METHODOLOGY -->
<section id="methodology" class="section">
  <h2>Methodology: 8-Pillar AI Scoring</h2>
  <div class="method-grid">
    <div class="method-card"><div class="method-icon">1</div><h3>TransE Embeddings</h3><p>Knowledge graph link prediction using DRKG (5.87M biological relationships)</p></div>
    <div class="method-card"><div class="method-icon">2</div><h3>RotatE / PyKEEN</h3><p>Complex relation pattern scoring via rotation-based graph embeddings</p></div>
    <div class="method-card"><div class="method-icon">3</div><h3>TxGNN</h3><p>Graph neural network for therapeutic use prediction (Harvard)</p></div>
    <div class="method-card"><div class="method-icon">4</div><h3>Molecular Fingerprints</h3><p>RDKit ECFP similarity to known treatments for each disease</p></div>
    <div class="method-card"><div class="method-icon">5</div><h3>ChemBERTa Embeddings</h3><p>Transformer-learned molecular representations capturing deep structural patterns</p></div>
    <div class="method-card"><div class="method-icon">6</div><h3>Gene Signatures</h3><p>L1000CDS2 transcriptomic reversal — drugs that reverse disease gene expression</p></div>
    <div class="method-card"><div class="method-icon">7</div><h3>Network Proximity</h3><p>STRING protein-protein interaction shortest paths between drug targets and disease genes</p></div>
    <div class="method-card"><div class="method-icon">8</div><h3>Mendelian Randomization</h3><p>Causal genetic evidence from Open Targets — strongest form of drug target validation</p></div>
  </div>
  <div class="method-note">
    <p>Each drug receives a <strong>dynamic weighted score</strong> across all applicable pillars, with convergence bonuses when multiple independent methods agree. Candidates are then enriched with evidence from PubMed, ClinicalTrials.gov, FDA FAERS, Semantic Scholar, and L1000CDS2.</p>
  </div>
</section>

<footer>
  <p>OpenCure is open-source (Apache 2.0) and free for all researchers.</p>
  <p><a href="https://github.com/SimonBartosDev/opencure">GitHub</a></p>
</footer>

</main>

<div class="detail-modal" id="detail-modal">
  <div class="modal-content" id="modal-content"></div>
</div>

<script>
const DATA = {data_json};
const CROSS = {cross_json};
const DISEASES = {diseases_json};
{JS}
</script>
</body>
</html>"""


CSS = """
:root {
  --bg: #f8f9fb;
  --surface: #ffffff;
  --border: #e2e8f0;
  --text: #1a202c;
  --text2: #4a5568;
  --text3: #718096;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --green: #059669;
  --green-light: #d1fae5;
  --orange: #d97706;
  --orange-light: #fef3c7;
  --red: #dc2626;
  --red-light: #fee2e2;
  --purple: #7c3aed;
  --purple-light: #ede9fe;
  --radius: 10px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-lg: 0 4px 14px rgba(0,0,0,0.1);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}

.hero {
  background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
  color: white;
  padding: 3rem 2rem 2rem;
}

.hero-inner { max-width: 1200px; margin: 0 auto; }
.hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; }
.accent { color: #60a5fa; }
.subtitle { font-size: 1.15rem; color: #93c5fd; margin-top: 0.3rem; font-weight: 500; }
.tagline { font-size: 0.95rem; color: #94a3b8; margin-top: 0.2rem; }

.stats-row {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
  flex-wrap: wrap;
}

.stat-card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: var(--radius);
  padding: 1rem 1.5rem;
  min-width: 130px;
  text-align: center;
}

.accent-card {
  background: rgba(96,165,250,0.15);
  border-color: rgba(96,165,250,0.3);
}

.stat-num { font-size: 2rem; font-weight: 800; }
.stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

.nav-bar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
  display: flex;
  gap: 0;
  max-width: 100%;
  overflow-x: auto;
  box-shadow: var(--shadow);
}

.nav-bar a {
  padding: 0.85rem 1.2rem;
  text-decoration: none;
  color: var(--text2);
  font-size: 0.9rem;
  font-weight: 500;
  white-space: nowrap;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
}

.nav-bar a:hover { color: var(--accent); border-bottom-color: var(--accent); }

main { max-width: 1200px; margin: 0 auto; padding: 0 1.5rem 3rem; }

.section { margin-top: 2.5rem; }
.section h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: var(--text);
}

.section-desc { color: var(--text2); margin-bottom: 1.2rem; font-size: 0.95rem; }

.filters {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
  align-items: center;
}

.filters select, .filters input {
  padding: 0.55rem 0.8rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 0.88rem;
  background: var(--surface);
  color: var(--text);
  outline: none;
}

.filters select:focus, .filters input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-light); }

.result-count { font-size: 0.85rem; color: var(--text3); margin-left: 0.5rem; }

.table-wrap {
  overflow-x: auto;
  background: var(--surface);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
}

table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
thead { background: #f1f5f9; }
th {
  padding: 0.75rem 1rem;
  text-align: left;
  font-weight: 600;
  color: var(--text2);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

th:hover { color: var(--accent); }

.sort-arrow { font-size: 0.7rem; margin-left: 0.2rem; }

td {
  padding: 0.65rem 1rem;
  border-top: 1px solid var(--border);
  vertical-align: middle;
}

tbody tr { cursor: pointer; transition: background 0.1s; }
tbody tr:hover { background: #f8fafc; }

.badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.badge-high { background: var(--green-light); color: var(--green); }
.badge-medium { background: var(--orange-light); color: var(--orange); }
.badge-low { background: var(--red-light); color: var(--red); }
.badge-breakthrough { background: var(--purple-light); color: var(--purple); }
.badge-novel { background: var(--accent-light); color: var(--accent); }
.badge-emerging { background: var(--orange-light); color: var(--orange); }
.badge-known { background: #f1f5f9; color: var(--text3); }
.badge-established { background: #f1f5f9; color: var(--text3); }

.score-bar {
  display: inline-block;
  width: 60px;
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
  vertical-align: middle;
  margin-right: 0.4rem;
}

.score-fill {
  height: 100%;
  border-radius: 3px;
  background: var(--accent);
}

.cross-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1rem;
}

.cross-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem;
  box-shadow: var(--shadow);
}

.cross-card h3 { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; }
.cross-card .disease-count {
  font-size: 0.82rem;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.6rem;
}

.cross-disease-tag {
  display: inline-block;
  background: #f1f5f9;
  color: var(--text2);
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  font-size: 0.78rem;
  margin: 0.15rem 0.15rem;
}

.bt-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 1rem;
}

.bt-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem;
  box-shadow: var(--shadow);
  display: flex;
  gap: 1.2rem;
  align-items: flex-start;
}

.bt-info { flex: 1; }
.bt-info h3 { font-size: 1rem; font-weight: 700; margin-bottom: 0.15rem; }
.bt-disease { font-size: 0.88rem; color: var(--accent); font-weight: 500; }
.bt-meta { font-size: 0.82rem; color: var(--text3); margin-top: 0.4rem; }

.radar-wrap { width: 110px; height: 110px; flex-shrink: 0; }
.radar-wrap svg { width: 100%; height: 100%; }

.method-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}

.method-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem;
  box-shadow: var(--shadow);
}

.method-icon {
  width: 32px;
  height: 32px;
  background: var(--accent);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.85rem;
  margin-bottom: 0.6rem;
}

.method-card h3 { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.3rem; }
.method-card p { font-size: 0.85rem; color: var(--text2); }

.method-note {
  margin-top: 1.2rem;
  padding: 1rem 1.2rem;
  background: var(--accent-light);
  border-radius: var(--radius);
  font-size: 0.9rem;
  color: #1e40af;
}

footer {
  text-align: center;
  padding: 2rem;
  color: var(--text3);
  font-size: 0.85rem;
  border-top: 1px solid var(--border);
  margin-top: 3rem;
}

footer a { color: var(--accent); text-decoration: none; }

.detail-modal {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4);
  z-index: 1000;
  overflow-y: auto;
  padding: 2rem;
}

.detail-modal.active { display: flex; justify-content: center; align-items: flex-start; }

.modal-content {
  background: var(--surface);
  border-radius: 14px;
  padding: 2rem;
  max-width: 700px;
  width: 100%;
  box-shadow: var(--shadow-lg);
  position: relative;
  margin-top: 2rem;
}

.modal-close {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text3);
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}

.modal-close:hover { background: #f1f5f9; }

.modal-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; }
.modal-disease { font-size: 1rem; color: var(--accent); font-weight: 500; margin-bottom: 1rem; }

.pillar-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  margin: 1rem 0;
}

.pillar-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  font-size: 0.85rem;
}

.pillar-label { font-weight: 600; color: var(--text2); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
.pillar-value { margin-top: 0.2rem; color: var(--text); font-weight: 500; }

.evidence-box {
  background: #f8fafc;
  border-radius: 8px;
  padding: 0.8rem 1rem;
  margin-top: 0.8rem;
  font-size: 0.85rem;
  color: var(--text2);
  line-height: 1.6;
}

.evidence-box strong { color: var(--text); }

@media (max-width: 768px) {
  .hero { padding: 2rem 1rem 1.5rem; }
  .hero h1 { font-size: 1.8rem; }
  .stats-row { gap: 0.5rem; }
  .stat-card { padding: 0.7rem 1rem; min-width: 100px; }
  .stat-num { font-size: 1.5rem; }
  .filters { flex-direction: column; }
  .filters select, .filters input { width: 100%; }
  .pillar-grid { grid-template-columns: 1fr; }
  .bt-card { flex-direction: column; }
  .radar-wrap { width: 100%; max-width: 160px; height: 160px; margin: 0 auto; }
  .cross-grid, .bt-grid { grid-template-columns: 1fr; }
}
"""

JS = """
// ---- STATE ----
let sortCol = 'combined_score';
let sortAsc = false;
let filtered = [...DATA];

// ---- INIT ----
(function init() {
  const sel = document.getElementById('filter-disease');
  DISEASES.forEach(d => {
    const o = document.createElement('option');
    o.value = d; o.textContent = d;
    sel.appendChild(o);
  });

  document.getElementById('filter-disease').addEventListener('change', applyFilters);
  document.getElementById('filter-confidence').addEventListener('change', applyFilters);
  document.getElementById('filter-novelty').addEventListener('change', applyFilters);
  document.getElementById('filter-search').addEventListener('input', applyFilters);

  document.querySelectorAll('#pred-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (sortCol === col) sortAsc = !sortAsc;
      else { sortCol = col; sortAsc = col === 'disease' || col === 'drug_name'; }
      renderTable();
    });
  });

  // Modal close
  document.getElementById('detail-modal').addEventListener('click', e => {
    if (e.target.id === 'detail-modal' || e.target.classList.contains('modal-close'))
      document.getElementById('detail-modal').classList.remove('active');
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('detail-modal').classList.remove('active');
  });

  applyFilters();
  renderCrossDisease();
  renderBreakthroughs();
})();

function applyFilters() {
  const disease = document.getElementById('filter-disease').value;
  const conf = document.getElementById('filter-confidence').value;
  const nov = document.getElementById('filter-novelty').value;
  const search = document.getElementById('filter-search').value.toLowerCase();

  filtered = DATA.filter(c => {
    if (disease && c.disease !== disease) return false;
    if (conf && c.confidence !== conf) return false;
    if (nov && c.novelty_level !== nov) return false;
    if (search && !c.drug_name.toLowerCase().includes(search)) return false;
    return true;
  });

  document.getElementById('result-count').textContent = filtered.length + ' of ' + DATA.length + ' candidates';
  renderTable();
}

function renderTable() {
  filtered.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  // Update sort arrows
  document.querySelectorAll('#pred-table th[data-sort]').forEach(th => {
    const arrow = th.querySelector('.sort-arrow');
    arrow.textContent = th.dataset.sort === sortCol ? (sortAsc ? '\\u25B2' : '\\u25BC') : '';
  });

  const tbody = document.getElementById('pred-body');
  tbody.innerHTML = filtered.map((c, i) => `
    <tr onclick="showDetail(${DATA.indexOf(c)})">
      <td>${esc(c.disease)}</td>
      <td><strong>${esc(c.drug_name)}</strong></td>
      <td><span class="score-bar"><span class="score-fill" style="width:${Math.min(c.combined_score / 1.2 * 100, 100)}%"></span></span>${c.combined_score.toFixed(3)}</td>
      <td><span class="badge badge-${c.confidence.toLowerCase()}">${c.confidence}</span></td>
      <td><span class="badge badge-${c.novelty_level.toLowerCase()}">${c.novelty_level}</span></td>
      <td style="text-align:center">${c.pillars_hit}</td>
      <td>${c.mr_score > 0 ? c.mr_score.toFixed(2) : '-'}</td>
    </tr>
  `).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function showDetail(idx) {
  const c = DATA[idx];
  const modal = document.getElementById('detail-modal');
  const content = document.getElementById('modal-content');

  const faersText = c.faers_signal !== 'none'
    ? c.faers_signal + ' (' + c.faers_cooccurrences.toLocaleString() + ' co-occurrences)'
    : 'No signal';

  const sigText = c.signature_reversal
    ? 'Yes (rank ' + c.signature_rank + ')'
    : 'Not detected';

  content.innerHTML = `
    <button class="modal-close" aria-label="Close">&times;</button>
    <div class="modal-title">${esc(c.drug_name)}</div>
    <div class="modal-disease">${esc(c.disease)}</div>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem">
      <span class="badge badge-${c.confidence.toLowerCase()}">${c.confidence} confidence</span>
      <span class="badge badge-${c.novelty_level.toLowerCase()}">${c.novelty_level}</span>
      <span class="badge" style="background:#f1f5f9;color:var(--text)">${c.pillars_hit} pillars</span>
      <span class="badge" style="background:#f1f5f9;color:var(--text)">Score: ${c.combined_score.toFixed(4)}</span>
      ${c.drug_id ? '<span class="badge" style="background:#f1f5f9;color:var(--text)">' + esc(c.drug_id) + '</span>' : ''}
    </div>

    <div style="display:flex;gap:1.2rem;align-items:flex-start;flex-wrap:wrap">
      <div style="flex:1;min-width:250px">
        <div class="pillar-grid">
          <div class="pillar-item"><div class="pillar-label">TransE Rank</div><div class="pillar-value">#${c.transe_rank || '-'}</div></div>
          <div class="pillar-item"><div class="pillar-label">Mol. Similarity</div><div class="pillar-value">${c.mol_similarity > 0 ? (c.mol_similarity * 100).toFixed(1) + '% to ' + esc(c.similar_to) : '-'}</div></div>
          <div class="pillar-item"><div class="pillar-label">MR Score</div><div class="pillar-value">${c.mr_score > 0 ? c.mr_score.toFixed(4) + ' (' + c.mr_genetic_targets + ' targets)' : '-'}</div></div>
          <div class="pillar-item"><div class="pillar-label">Gene Signature</div><div class="pillar-value">${sigText}</div></div>
          <div class="pillar-item"><div class="pillar-label">FAERS Signal</div><div class="pillar-value">${faersText}</div></div>
          <div class="pillar-item"><div class="pillar-label">Shared Targets</div><div class="pillar-value">${c.shared_targets}</div></div>
          <div class="pillar-item"><div class="pillar-label">PubMed Articles</div><div class="pillar-value">${c.pubmed_articles} (${c.pubmed_treatment} treatment, ${c.pubmed_repurposing} repurposing)</div></div>
          <div class="pillar-item"><div class="pillar-label">Clinical Trials</div><div class="pillar-value">${c.clinical_trials || 'None'}</div></div>
        </div>
      </div>
      <div class="radar-wrap">${makeRadar(c)}</div>
    </div>

    ${c.confidence_reasons ? '<div class="evidence-box"><strong>Confidence reasoning:</strong><br>' + esc(c.confidence_reasons).replace(/;/g, '<br>') + '</div>' : ''}
    ${c.novelty_interpretation ? '<div class="evidence-box"><strong>Novelty interpretation:</strong><br>' + esc(c.novelty_interpretation) + '</div>' : ''}
  `;

  modal.classList.add('active');
}

function makeRadar(c) {
  // 8 pillars normalized to 0-1
  const axes = [
    { label: 'TransE', value: c.transe_rank > 0 ? Math.max(0, 1 - c.transe_rank / 500) : 0 },
    { label: 'MolSim', value: c.mol_similarity || 0 },
    { label: 'MR', value: c.mr_score || 0 },
    { label: 'GeneSig', value: c.signature_reversal ? 0.8 : 0 },
    { label: 'FAERS', value: c.faers_signal === 'strong' ? 0.9 : c.faers_signal === 'moderate' ? 0.5 : c.faers_signal === 'weak' ? 0.2 : 0 },
    { label: 'Targets', value: Math.min(c.shared_targets / 5, 1) },
    { label: 'PubMed', value: Math.min(c.pubmed_articles / 20, 1) },
    { label: 'Trials', value: Math.min(c.clinical_trials / 5, 1) },
  ];

  const cx = 55, cy = 55, r = 42;
  const n = axes.length;
  let polyBg = '', polyData = '', labels = '';

  for (let i = 0; i < n; i++) {
    const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    const xv = cx + r * axes[i].value * Math.cos(angle);
    const yv = cy + r * axes[i].value * Math.sin(angle);
    const xl = cx + (r + 14) * Math.cos(angle);
    const yl = cy + (r + 14) * Math.sin(angle);

    polyBg += `${x},${y} `;
    polyData += `${xv},${yv} `;
    labels += `<text x="${xl}" y="${yl}" text-anchor="middle" dominant-baseline="middle" font-size="6" fill="#718096">${axes[i].label}</text>`;
  }

  // Grid circles
  let grid = '';
  for (const s of [0.25, 0.5, 0.75, 1]) {
    let pts = '';
    for (let i = 0; i < n; i++) {
      const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
      pts += `${cx + r * s * Math.cos(angle)},${cy + r * s * Math.sin(angle)} `;
    }
    grid += `<polygon points="${pts}" fill="none" stroke="#e2e8f0" stroke-width="0.5"/>`;
  }

  return `<svg viewBox="0 0 110 110" xmlns="http://www.w3.org/2000/svg">
    ${grid}
    <polygon points="${polyBg}" fill="none" stroke="#cbd5e1" stroke-width="0.8"/>
    <polygon points="${polyData}" fill="rgba(37,99,235,0.15)" stroke="#2563eb" stroke-width="1.5"/>
    ${labels}
  </svg>`;
}

function renderCrossDisease() {
  const container = document.getElementById('cross-cards');
  const drugs = Object.entries(CROSS).sort((a, b) => b[1].length - a[1].length);

  container.innerHTML = drugs.map(([drug, entries]) => `
    <div class="cross-card">
      <h3>${esc(drug)}</h3>
      <div class="disease-count">${entries.length} diseases predicted</div>
      <div>${entries.map(e =>
        '<span class="cross-disease-tag">' + esc(e.disease) +
        ' <small>(' + e.score.toFixed(2) + ', ' + e.novelty + ')</small></span>'
      ).join('')}</div>
    </div>
  `).join('');
}

function renderBreakthroughs() {
  const bt = DATA.filter(c => c.novelty_level === 'BREAKTHROUGH')
    .sort((a, b) => b.combined_score - a.combined_score)
    .slice(0, 10);

  const container = document.getElementById('bt-cards');
  container.innerHTML = bt.map(c => `
    <div class="bt-card" onclick="showDetail(${DATA.indexOf(c)})" style="cursor:pointer">
      <div class="bt-info">
        <h3>${esc(c.drug_name)}</h3>
        <div class="bt-disease">${esc(c.disease)}</div>
        <div class="bt-meta">
          Score: <strong>${c.combined_score.toFixed(3)}</strong> &middot;
          ${c.pillars_hit} pillars &middot;
          MR: ${c.mr_score > 0 ? c.mr_score.toFixed(2) : '-'} &middot;
          <span class="badge badge-${c.confidence.toLowerCase()}" style="font-size:0.72rem">${c.confidence}</span>
        </div>
      </div>
      <div class="radar-wrap">${makeRadar(c)}</div>
    </div>
  `).join('');
}
"""


def main():
    print("Loading database...")
    candidates = load_data()
    print(f"  {len(candidates)} candidates loaded")

    print("Computing cross-disease analysis...")
    cross = compute_cross_disease(candidates)
    print(f"  {len(cross)} drugs predicted across multiple diseases")

    stats = compute_stats(candidates)
    print(f"  Stats: {stats}")

    print("Building HTML...")
    html_content = build_html(candidates, cross, stats)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html_content)
    print(f"  Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print("\nDone! Open docs/index.html in a browser to preview.")


if __name__ == "__main__":
    main()
