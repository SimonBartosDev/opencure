# OpenCure v4: Breakthrough-Credibility Implementation Plan

## Goal

Transform OpenCure from "looks reasonable" → "publishable with honest numbers + genuine breakthrough narrative".

Three concrete, defensible claims this plan unlocks:
1. **"OpenCure's held-out Hit@10 is X %"** — publishable metric, comparable to TxGNN / Hetionet.
2. **"OpenCure identifies N pandemic-preparedness polypharmacology clusters"** — novel framing.
3. **"X % of OpenCure's Jan-2026 predictions were independently validated within 6 months"** — rolling prospective precision.

## Problems being solved (from v3 audit)

1. **Hub bias** — Cimetidine(5/7), Dex(4/7), Tacrolimus(4/7) dominate top-10s because of DRKG degree (1,400–3,400 triplets).
2. **Metabolite leakage** — Uric Acid, Cordycepin-TP, Androstene-ol, Glutathione, L-Alanine, Folic Acid, Creatinine + IUPAC research chemicals still pass filter.
3. **No held-out evaluation** — cannot cite AUROC / Hit@10 defensibly.
4. **DRKG is 2020-era** — missing 5 years of biology.
5. **No mechanism explanations** — `kg_paths_text` empty for every candidate.
6. **Cross-disease signal untapped** — 32 cross-disease drugs but no cluster reasoning.
7. **No prospective validation loop** — predictions never checked against later literature.
8. **ADMET multiplier too soft** for non-FDA compounds.

## Phase 0 — Preflight
- Don't disturb running v3 screening (PID 91225).
- Feature branch `v4-breakthrough`.
- New scaffolding:
  - `opencure/eval/` (heldout_benchmark, ground_truth)
  - `opencure/scoring/hub_normalize.py`, `mechanism_cluster.py`
  - `opencure/filters/metabolite_blacklist.py`, `name_heuristics.py`
  - `opencure/evidence/path_explainer.py`
  - `opencure/data/open_targets.py`
  - `scripts/precompute_degree.py`, `build_metabolite_blacklist.py`, `build_open_targets_kg.py`, `run_heldout_eval.py`, `prospective_monitor.py`

## Phase 1 — Held-out evaluation (CRITICAL, 1 d)
- Load DrugCentral + RepoDB ground-truth → `data/eval/ground_truth.tsv`
- Hold out 20%, strip `treats` edges from DRKG → `data/drkg/drkg_holdout.tsv`
- Rebuild embeddings on holdout graph (~1 hr)
- Runner: Hit@10, Hit@50, MRR, AUROC
- Acceptance: v3 ≥ 0.20, v4 ≥ 0.35

## Phase 2 — Hub-degree normalization (4 h)
- `scripts/precompute_degree.py` — count triplets per drug → `data/drkg/drug_degree.json`
- `opencure/scoring/hub_normalize.py::degree_penalty(drug_id, alpha=0.5)` — log-damping, median drug ≈ 1.0, hub ≈ 0.55
- Multiply `kg_group_score`, `network_group_score` by penalty in `pillar_groups.py`
- Validation: Cimetidine drops from ≥3 of 7 disease top-10s

## Phase 3 — Metabolite + research-chemical filter (1 d)
- Download HMDB endogenous metabolite set (~4k InChIKeys)
- Parse DrugBank `<groups>` for "experimental"/"nutraceutical"
- `metabolite_blacklist.py::is_endogenous_metabolite(drug_id, smiles)` (ChEMBL phase ≥4 bypass)
- `name_heuristics.py::looks_like_research_chemical(name)` — IUPAC regex, length, dashes
- Wire into `drug_filter.is_therapeutic_candidate` as Gate 1.5
- Validation: Uric Acid, Cordycepin TP, Androstene-ol, Glutathione, L-Alanine, Creatinine, Folic Acid all rejected; Aspirin/Donepezil/Hydroxyurea pass.

## Phase 4 — Two-stage ADMET multiplier (15 m)
```python
if is_fda_approved: return 0.8 + 0.2 * admet_score   # [0.8, 1.0]
else:               return 0.3 + 0.7 * admet_score   # [0.3, 1.0]
```

## Phase 5 — Open Targets 24.09 KG (3 d, the biology refresh)
- `opencure/data/open_targets.py` downloads OT 24.09 parquet (~3 GB)
- Emit triplets: `(drug, mechanism, target)`, `(target, associates, disease)`, `(drug, indication, disease)`
- Merge DRKG + PrimeKG + OT → `data/unified_kg/unified.tsv`
- Retrain RotatE via PyKEEN (~6 h)
- Extend `kg_fusion.fuse_kg_scores` to 4-way RRF
- Validation: +0.10 Hit@10 over DRKG-only

## Phase 6 — Path-based mechanism explanations (2 d)
- `opencure/evidence/path_explainer.py::explain_path(drug, disease, max=3, cutoff=3)`
- `networkx.all_simple_paths` on unified subgraph
- Natural-language rendering: "Clarithromycin → inhibits → 50S ribosome → essential for → P. falciparum"
- Populate `report.kg_paths_text` + `mechanistic_hypothesis`
- Dashboard: expand top-10 row to show 3 paths

## Phase 7 — Cross-disease mechanism clustering (1 d)
- After 61-disease scoring, group hits by shared pathway (MR targets + DTI targets + KEGG)
- `cluster_strength = #diseases × mean_score × pathway_coherence`
- New dashboard page `/clusters`
- Headline: "OpenCure identifies polypharmacology clusters for pandemic preparedness"

## Phase 8 — Prospective validation loop (0.5 d)
- `scripts/prospective_monitor.py` runs monthly
- For every prediction ≥3 months old, re-query PubMed after prediction date
- Log to `data/prospective/validation_log.jsonl`
- Rolling precision@10 shown on dashboard footer

## Phase 9 — Integration, full re-screening, release (0.5 d + 12 h compute)
- After v3 screening finishes, regenerate v3 baseline first
- Switch to v4, run full 61-disease screening
- Head-to-head metrics card on dashboard
- bioRxiv preprint addendum with held-out metrics + cluster screenshot

## Dependency graph
```
Phase 1 (eval) ────────────────────────── validates everything
Phase 2 (hub)  ─┐
Phase 3 (meta) ─┼─► Phase 4 ─► Phase 5 ─┬─► Phase 6 (paths) ─► Phase 7 (clusters) ─► Phase 9
Phase 5 (OT)   ─┘                       │
                      Phase 8 (prospective) ─────────────────────────────────────────┘
```

Phase 1 runs in parallel with v3 screening (no conflict).
Phases 2–4 land as one commit on `v4-breakthrough`.

## Total estimate
~10 working days + ~20 hrs compute.

## Acceptance metrics (head-to-head v3 → v4)
- Hit@10 ≥ 2×
- Metabolites in top-10 = 0
- Hub-drug cross-disease dominance = 0
- ≥5 mechanism clusters identified
- Dashboard prospective precision@10 widget live
