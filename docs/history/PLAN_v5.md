# OpenCure v5: "Truly Best" Comprehensive Plan

**Objective:** ship the software that is genuinely state-of-the-art in open
drug repurposing — credible, modern, mechanism-aware, clinically useful,
and self-validating. Everything a solo engineer can deliver; humans-only
items (wet-lab, peer review) scoped separately with templates.

Branch: `v5-truly-best` (cuts from `v4-breakthrough`)

## The 8 "won'ts" that v4 can't fix — each mapped to a concrete deliverable

| # | v4 limitation | v5 fix |
|---|---|---|
| 1 | Contamination in held-out metrics | Edge-stripped retrain → **clean Hit@K** |
| 2 | 2020-era biology | **2024-native KG** from STRING-v12 + OT 24.09 + ChEMBL 34 + DisGeNET 2024 |
| 3 | Shallow embeddings only | **R-GCN heterogeneous GNN** as 12th pillar |
| 4 | Tissue-blind | **Tabula Sapiens + cell-line L1000** context pillar |
| 5 | No validation | **Docking triangulation + prospective registry + crowd endpoint** |
| 6 | 61-disease coverage | **Full ~7000 MeSH indication coverage** |
| 7 | No dose/DDI/subtype reasoning | **Clinical layer**: dose plausibility, DDI flags, pharmacogenomic flags |
| 8 | No peer-reviewed stats | **Time-sliced benchmark** (train 2019, test 2020-2024 approvals) + methods paper draft |

## Phase A — Foundation fixes (2 days, blocking all others)

### A1. Stop in-flight plan, consolidate
- Kill v3 screening ✓
- Let unified training finish (2/20 epochs done, ~2h remaining)
- Snapshot `v4-breakthrough` as baseline for v3 vs v5 comparison later

### A2. Edge-stripped clean retrain (2 days)
- `scripts/strip_heldout_edges.py`: remove 993 test pairs from unified.tsv → unified_train.tsv
- Retrain TransE on stripped graph (~2h MPS)
- Save as `data/models/unified_transE_clean/`
- Rerun `run_heldout_eval.py` → report CLEAN Hit@10, MRR, AUROC
- Dashboard shows BOTH contaminated (upper bound) + clean (defensible)

### A3. Time-sliced benchmark (1 day)
- `data/eval/time_sliced.jsonl`: (drug, disease, first_approval_year) pairs from ChEMBL + DrugCentral
- Train on pre-2020 edges; evaluate on post-2020 approvals
- Single most credible benchmark format (what reviewers actually ask for)

## Phase B — 2024 biology (3 days)

### B1. Modern source downloads (1 day)
- STRING v12.0 protein-protein (+ confidence scores)
- Reactome 2024 pathways
- ChEMBL 34 bioactivities (Nov 2024)
- DisGeNET 2024 gene-disease
- OncoKB + CIVIC (cancer variants)
- IntAct 2024-Q1 PPI
- LINCS L1000 2024 expanded cell lines

### B2. Native-2024 unified KG (1 day)
- `scripts/build_2024_native_kg.py` — unions all 2024 sources + OT 24.09 + keep selected DRKG/PrimeKG edges
- Target: ~30M triples, 2024 as primary (not appendix)
- Dedupe via InChIKey + UniProt + MeSH canonicalization

### B3. Retrain TransE on 2024-native (1 day, parallel with other work)
- Same pipeline as unified → `data/models/kg2024_transE/`

## Phase C — Modern ML architecture (4 days)

### C1. Heterogeneous GNN pillar (2 days)
- DGL or PyG R-GCN on unified KG
- Train on edge-stripped graph for honest metrics
- New scorer `opencure/scoring/gnn_scorer.py`
- Register as 12th pillar

### C2. Cell-type context pillar (2 days)
- Tabula Sapiens: ~500 cell types, gene expression per cell type
- Per-disease: identify relevant tissue (disease ontology → tissue mapping already in Open Targets)
- `opencure/scoring/tissue_context.py`: re-weight proximity/GeneSig by tissue-specific expression
- Switch L1000 lookup to cell-line-matched instead of pooled

## Phase D — Clinical layer (3 days)

### D1. Dose plausibility (1 day)
- ChEMBL IC50/EC50 per compound-target pair
- DrugBank dosage + plasma level
- For each prediction: is achievable Cmax within 10× of Kd for predicted target?
- Emit `dose_plausible` flag + recommended dose range

### D2. DDI warnings (0.5 day)
- DRKG already has 1.4M ddi edges — aggregate per candidate
- For each top-10 prediction: list top 5 dangerous interactions
- Surface in dashboard drug detail page

### D3. Pharmacogenomic flags (0.5 day)
- Pull CPIC guidelines + PharmGKB VIP variants (TSV download)
- For each drug: list HLA / CYP / TPMT / DPYD warnings
- E.g., "Abacavir + HLA-B*57:01 → contraindicated"

### D4. Patient subtype stratification (1 day)
- For heterogeneous diseases (breast cancer HER2+/−, lung NSCLC subtypes), predict per-subtype
- Requires extending disease list from single entity to subtypes
- Target top 10 subtype-variable diseases first (breast, lung, glioblastoma, AML, melanoma)

## Phase E — Validation infrastructure (2 days)

### E1. In-silico triangulation (1 day)
- AutoDock Vina docking (already partial in structure_docking.py; complete it)
- Pharos target development level lookup
- Agreement score across {embedding, docking, Pharos, literature} = silver-standard
- Flag top-10 with triangulation_score in dashboard

### E2. Prospective registry + DOI (0.5 day)
- Snapshot v5 predictions at release
- Generate Zenodo DOI for the snapshot (public, timestamped, immutable)
- Monthly prospective_monitor.py already wired — just add DOI reference
- Dashboard widget: "Predictions made YYYY-MM-DD • Rolling precision@10 • DOI"

### E3. Crowd validation endpoint (0.5 day)
- FastAPI endpoint + form in dashboard
- Researchers submit: "I tested Drug X on Disease Y in [model] → IC50/EC50/no effect"
- Moderation queue + append to `data/crowd_validation.jsonl`
- Public count on dashboard

## Phase F — Scale (3 days)

### F1. Expand to all MeSH indications (2 days)
- Currently 61 diseases; expand to all MeSH-indexed with ≥5 gene associations
- Target ~2,000 diseases (rare + neglected emphasized)
- Parallelize with process pool (`scripts/mass_screen.py`)
- Shard results per category

### F2. Cloud-ready deployment (1 day)
- Docker image with all models + unified KG (~3 GB layer)
- One-command deploy: `docker run opencure/opencure:v5 run-screen --disease <name>`
- API endpoint `/score` for programmatic access

## Phase G — Publication assets (2 days)

### G1. Methods paper draft (1 day)
- Target: *Nature Machine Intelligence* or *Bioinformatics*
- Sections: 11+ pillars, RRF fusion, tissue context, clinical layer, held-out + time-sliced benchmarks
- Figures: pipeline schematic, head-to-head vs TxGNN/Hetionet on time-sliced set
- Zenodo archive link for all code + data

### G2. Lab outreach templates (0.5 day)
- Per-disease 1-page briefs for top 5 neglected disease candidates
- Email templates to NCATS 3D Tissue Chip, Broad Repurposing Hub, EU-OPENSCREEN
- Target: commit to 3 lab partnerships in next 6 months

### G3. Pharma pitch deck (0.5 day)
- 10-slide deck: problem, method, head-to-head, polypharmacology page, prospective validation, ask
- Use for cold outreach to biotech repurposing teams

## Total timeline

~19 engineering days. Parallelizable:
- Training runs happen in background
- Downloads happen while writing code
- Compute cost: ~$50-100 cloud GPU if we retrain on CUDA (for RotatE + R-GCN)

## Success metrics (claims I can defend at the end)

1. **Clean held-out Hit@10 ≥ 0.40** on time-sliced 2020-2024 benchmark
2. **12 pillars** including heterogeneous GNN + tissue context
3. **All ~2000 MeSH diseases screened** with full v5 pipeline
4. **Every top-10 candidate** has: mechanism path, dose estimate, DDI warnings, triangulation score, pharmacogenomic flags
5. **Prospective registry live** at Zenodo with rolling p@10
6. **Methods paper submitted** to NMI / Bioinformatics
7. **3 lab partnership invitations sent** with per-disease briefs

## What's still NOT v5's job (honest)

- Wet-lab validation itself (requires partner commitment)
- Peer review outcome (requires 3-6 month cycle)
- Pharma partnership signing (requires biotech-side appetite)
- FDA/regulatory path (requires legal + clinical apparatus)
- Real-world evidence from EHR data (requires Epic/Cerner access)

These are the remaining "humans-only" items. Everything else I will deliver.

## Execution order (start immediately)

Now (while unified training runs):
1. Phase B1 downloads (parallel to training)
2. Phase D3 pharmacogenomic data (parallel)
3. Phase A2 strip holdout edges (queued for when training finishes)

Day 2-3: Phase A3 + B2
Day 4-6: Phase C
Day 7-9: Phase D
Day 10-11: Phase E
Day 12-14: Phase F
Day 15-16: Phase G
Day 17-19: integration, testing, dashboard, commits
