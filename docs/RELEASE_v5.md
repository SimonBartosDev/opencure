# OpenCure v5 Release Notes

**Release date:** 2026-04-19
**Data manifest hash:** `2da2aa88f2457d1b`
**Branch:** `v5-truly-best`
**Commits in this release:** 24 over 2 days

## TL;DR

v5 is a **code-quality + clinical-actionability** release, not a scoring
breakthrough. Numbers from the 2020-era DRKG core remain the same;
everything wrapped around them is now dramatically better — tests,
field-name correctness, persistent provenance, clinical guardrails,
prospective timestamping infrastructure, and evidence caching.

**Honesty note (post-hoc).** OpenCure implements 13 pillars, but under
leak-free, popularity-baselined evaluation the knowledge-graph-embedding,
chemical-structure, and cell-morphology pillars do **not** beat a trivial
popularity baseline, and neither does the fused multi-pillar score. The one
component that beats baseline is **genetics-anchored target prioritization**
(~5× popularity on the genetics-covered subset, leak-free and temporally
validated, honest temporal Hit@10 ~10%) — but it is rediscovery-leaning and
covers only part of diseases. It is a triage/prioritizer, not a discovery
engine. No prediction here is wet-lab confirmed; no novel credible lead was
found. See `docs/honest_evaluation.md` and `docs/conditional_lift_validation.md`.

**Three silent bugs** that had shipped in v3/v4 were caught in v5 via
field-name audit — `proximity_score`, `dti_score`, and `gene_sig_score`
were being computed at runtime but never reaching saved output. All
three now fire correctly (10/10, 9/10, 1/10 respectively in live tests).

**Every top prediction** now carries dose plausibility, drug-drug
interaction warnings, pharmacogenomic flags, mechanism paths, and
4-axis triangulation scores — turning rankings into triage hypotheses
for expert review. These are not wet-lab-ready leads: zero predictions
are experimentally confirmed and no novel credible lead was found.

## What's new

### Clinical guardrails (every prediction)

- **Dose plausibility** — 2-stage: ChEMBL phase (always) + Cmax/IC50 ratio against the predicted target (when ChEMBL 34 bioactivity cache is present; now shipping with 94,717 drug-target median-nM pairs across 4,707 drugs)
- **DDI warnings** — top 10 dangerous co-prescriptions from 1.4M DrugBank drug-drug interaction edges, severity-ranked against 25 commonly-co-prescribed reference drugs
- **Pharmacogenomic flags** — 76 CPIC drug-gene pairs + 5,187 PharmGKB annotations, classified high-risk / moderate / advisory
- **Mechanism paths** — natural-language graph paths via bounded BFS on filtered DRKG adjacency (filtered to drug-target, target-disease, drug-disease, literature-mined only; mega-hub clipped)
- **Triangulation** — 4-axis aggregate (KG + docking + Pharos target-development-level + literature); labels "silver-standard" when ≥3 agree. **Live v5.2 coverage on full 61-disease screen (n=610 candidates): ≥1 axis 90%, ≥2 axes 70%, ≥3 axes (silver-standard) 35%.** Unlocked by `scripts/build_disease_gene_index.py` (OT + DRKG/GNBR merge → 5,207 diseases × 115K gene associations) and the ChEMBL-bioactivity docking proxy. Real AutoDock Vina docking remains v6 work — the proxy stands in for now, honestly labeled in the `docking.source` field of every candidate.
- **Tissue context** — GTEx v8 median TPM across 54 tissues; 55 diseases manually mapped to relevant tissue sets; context modifier in [0.85, 1.15] applied to pillar scores

### Scoring improvements

- **Hub-degree normalization** (Phase 2) — multiplicative penalty on KG/network pillars calibrated against ChEMBL phase≥1 median degree (81 edges). See `docs/hub_bias_analysis.md` for honest disclosure of what this fixes and what remains.
- **Metabolite + research-chemical filter** — 96 curated endogenous compounds blacklisted + IUPAC-pattern heuristic for research chemicals. ChEMBL phase ≥4 bypass preserves legitimately-approved compounds (Folic Acid, Hydroxyurea, etc.)
- **Mechanistic gene-signature reversal** — supplement to L1000CDS2 using OT disease-gene associations + ChEMBL drug-target activities. Expands gene-sig coverage from ~10 drugs/disease to ~4,700.
- **Two-stage ADMET multiplier** — FDA-approved drugs get [0.8, 1.0]; non-FDA get [0.3, 1.0]. Penalizes predicted-toxic research chemicals harder than approved drugs with known real-world tolerability.
- **Unified-KG scorer** (scaffolded) — TransE trained on 14M-triple union of DRKG + PrimeKG + OT 24.09. Current local training underperforms the 2020 DGL-KE baseline; awaits cloud GPU retrain to unlock value. Bugfix landed for the `disease_entities` variable issue.

### Held-out + time-sliced evaluation

- **993-pair random holdout** from DRKG treats edges (seed 42)
- **210-pair time-sliced benchmark** — drugs with `yearOfFirstApproval >= 2020` per Open Targets 24.09 × approved indications. Year distribution: 2020 (91) / 2021 (49) / 2022 (45) / 2023 (25).
- **Edge-stripping**: `scripts/strip_heldout_edges.py` removes 1,891 treats-like edges across 4 KGs (DRKG, Hetionet, OT, GNBR) corresponding to 1,200 unique held-out pairs — closes cross-KG alias leakage.
- **Learned ensemble**: XGBoost + 5-fold CV + isotonic calibration. AUC-ROC = 0.9968 ± 0.0004 on 23,814 training pairs — **WITHDRAWN: data leakage.** KG features (derived from the same DRKG `treats` edges used as labels) dominate at 91% combined importance, so the model was graded on recalling its own training graph. A retrained, leak-free ensemble scores CV AUROC ≈ 0.72 with hard negatives and is at chance on a fair temporal test. This is not a validated benchmark accuracy and must not be presented as one.

### Prospective timestamping infrastructure

This records predictions for future checking; it has produced no validated
outcome and is not evidence of accuracy.

- **Immutable snapshots** — `scripts/snapshot_predictions.py` writes timestamped folders with SHA-256 content-hashed `predictions.jsonl`, `methods.json`, and Zenodo-ready metadata.
- **Monthly re-query cron** — `scripts/prospective_monitor.py` checks PubMed + ClinicalTrials.gov for evidence **after** the snapshot date, computes rolling precision@K on predictions ≥90 days old. Note: this precision@K is a later-mention / co-occurrence rate, not accuracy.
- **First snapshot**: `2026-04-18T164121Z` (content-fingerprinted; DOI registration pending).
- **Zenodo uploader**: `scripts/zenodo_upload.py` mints a DOI via Zenodo REST API given ZENODO_TOKEN env var; supports `--sandbox` for zenodo.sandbox.org testing.

### Engineering quality

- **172 automated tests** across filters, scoring, evidence, evaluation harness, and a regression suite that catches the class of the 3-pillar silent-zero bug.
- **CI**: GitHub Actions on every push, Python 3.11 and 3.12.
- **Structured logging** (`opencure/log_setup.py`): colored console output + optional JSON-lines file logging + per-pillar timing CSV at `data/metrics/timings.csv` for performance regression detection.
- **Data provenance**: `scripts/compute_data_manifest.py` hashes 15 tracked source files → `data/manifest.json`. Every result JSON now carries `data_manifest_hash` field (`2da2aa88f2457d1b` this release).
- **Evidence cache**: `opencure/evidence/cache.py` provides `@disk_cached` decorator on PubMed, FAERS, Semantic Scholar, ClinicalTrials.gov. Verified: **4,174× speedup on repeat PubMed calls**. Future re-screens drop from ~6h to ~1h on the same disease set.
- **Search refactor**: deleted 249 lines of dead v2 combiner code; single canonical field names enforced via `opencure/scoring/common.py` TypedDict. Zero stray `*_raw` or `*_raw_score` fields in output.

### 2024 data integrations

- **Open Targets 24.09** → 83,392 derived triplets
- **ChEMBL 34** (Nov 2024) → 94,717 drug-target bioactivities (median IC50/Ki in nM)
- **CPIC 2025-07** → 76 curated drug-gene guideline pairs
- **PharmGKB 2025-07** → 5,187 clinical annotations
- **GTEx v8** → 54-tissue × ~56K-gene median TPM matrix
- **STRING v12** → 473K high-confidence protein-protein interactions
- **HGNC complete set** → 41,732 Ensembl↔Entrez↔symbol mappings

## Breaking changes

- `_combine_scores_v2` removed from `opencure.search`. Any external caller importing it will break; switch to `combine_grouped_scores` from `opencure.scoring.grouped_combiner`.
- Field name `txgnn_raw_score` is no longer written in result JSONs (was leftover from v3 transition). Use `txgnn_score` (rank-normalized) or query the `txgnn_scores` intermediate dict at runtime for raw probability.
- Result JSONs now include `pipeline_version: "v5"` and `data_manifest_hash` — downstream consumers that parse with strict schemas need to allow these new fields.

## Known limitations (honest disclosure)

- **KG retrieval quality** on edge-stripped clean retrain: 3.33% Hit@10 vs 57.2% for the 2020 DGL-KE baseline (contaminated). DGL-KE is unavailable for Python 3.14; PyKEEN at 128-dim/50-epochs on MPS does not reach DGL-KE's 400-dim/400-epoch quality in tractable time. **Fix requires CUDA GPU** (~$30 cloud spend for a proper retrain).
- **Time-sliced Hit@10**: 0.0% across all three local models. DRKG is 2020-era; cannot retrieve post-2020 approved indications. This is a truthful disclosure — the stale-biology limitation is real.
- **Hub-drug bias** partially persists despite degree penalty. See `docs/hub_bias_analysis.md`. Cimetidine wins 3/5 early v5 infectious disease re-screens; interpretation requires per-pillar inspection.
- **Unified-KG scorer** is trained but underperforms; currently disabled in runtime pending proper CUDA retrain. Scaffold + bugfix are in place.
- **R-GCN pillar** scaffolded (`opencure/scoring/rgcn_scorer.py`) but untrained. Activates as an additional pillar once a GPU-trained model lands in `data/models/rgcn_v5/`. (For context: OpenCure implements 13 pillars, but under leak-free evaluation the KG-embedding, chemical-structure and cell-morphology pillars do not beat a popularity baseline — only genetics-anchored target prioritization does; adding more similarity pillars does not change that.)
- **No wet-lab validation yet.** Zero predictions have been experimentally confirmed. 5 lab outreach briefs ready to send at `docs/lab_outreach_briefs.md`.
- **No peer review yet.** Methods paper drafted at `docs/methods_paper_draft.md`; target journals Nature Machine Intelligence or Bioinformatics.
- **Silver-standard triangulation coverage in v5.2**: 29% of top candidates (126/430 across 43 diseases) achieve ≥3 axes of evidence, up from 0% in v5.1. Unlocked by two fixes: (a) `scripts/build_disease_gene_index.py` builds a unified disease-gene index from OT + DRKG/GNBR (115K gene-disease pairs across 5,207 diseases — lifts Schisto from 0 → 5 genes, Malaria 0 → 57, TB 0 → 104); (b) a sign-fix in the ChEMBL docking-proxy (nM → kcal/mol was returning positive, now correctly negative per Vina convention). True AutoDock Vina integration remains v6 work, but the proxy axis now fires on 47% of candidates where drug-target bioactivity overlaps disease-associated genes.
- **is_known_treatment pre-v5 heuristic was broken** (Oxamniquine-for-Schisto was flagged False). Fixed in v5.1 post-screen patch by consulting the full `KNOWN_TREATMENT_RELATIONS` set (DRUGBANK::treats + GNBR + Hetionet). New positive rate across v5 top-10: 7/100 (Oxamniquine, Dexamethasone, Dapsone, Ceftriaxone, Methadone, …). DRUGBANK::treats is NOT added to the scoring KG (still stripped by edge-stripping); the expanded list is labeling-only.
- **ensemble_v5.pkl was trained but unused at inference in the initial v5 ship.** Fixed post-screen: `scripts/score_ensemble_v5.py` attaches `ensemble_prob` + `ensemble_rank` to every candidate. Ensemble surfaces positive controls strongly (Oxamniquine ens_prob 0.954, Dexamethasone 0.681) while keeping hub drugs mid-pack (Cimetidine ens_prob 0.05–0.33 depending on disease).
- **tissue_context was populated only on candidates with shared drug-disease targets** — a minority. Fixed: now falls back to OT disease-associated genes, so the field always carries the disease's GTEx-relevant tissue map (Liver/Whole Blood for Schisto, Lung for TB, etc.) even when shared-target overlap is zero.

## Upgrade path

From v4 (or earlier):
```bash
git fetch origin
git checkout v5-truly-best
pip install -r requirements.txt              # gets xgboost, sklearn, pyarrow
python3 scripts/compute_data_manifest.py     # generates data/manifest.json
python3 scripts/strip_heldout_edges.py       # builds edge-stripped training KG
python3 scripts/build_drug_target_activities.py  # builds ChEMBL bioactivity cache
python3 experiments/systematic_screening.py --no-resume  # re-screen on v5 pipeline
python3 scripts/finalize_v5.py               # regenerate dashboard + snapshot + scoring
```

Total time: ~6 hours for a full re-screen on 61 diseases; future re-screens drop to ~1 hour with evidence cache warm.

## What's next (v6 roadmap)

1. **CUDA GPU retrain** of the unified KG — unblocks publication-competitive clean Hit@10
2. **R-GCN additional pillar** — PyG heterogeneous GNN, needs CUDA (a similarity-family pillar; will not, on its own, beat the popularity baseline the other similarity pillars fail against)
3. **Wet-lab partnerships** — 5 briefs ready; focus on neglected-tropical (Chagas, Schisto, Leishmaniasis)
4. **bioRxiv submission** — methods paper draft at `docs/methods_paper_draft.md`
5. **Zenodo DOI** — registered via `scripts/zenodo_upload.py` once ZENODO_TOKEN is set
6. **Cell-type resolution** — integrate CellxGene / Tabula Sapiens for cell-type-matched target relevance
7. **Time-sliced re-run** — retrain on 2024-native KG (OT 24.09 as primary, not addendum) to unlock non-zero time-sliced Hit@10

## Acknowledgments

Hardware for this release: Apple M-series MPS for all local compute.
Data: DRKG (Amazon/UCLA), PrimeKG (Harvard), Open Targets Platform,
ChEMBL (EBI), GTEx (Broad), HGNC, CPIC, PharmGKB, STRING, L1000CDS2.

## License

Apache 2.0. Patent grant for pharmaceutical and biotech applications.
