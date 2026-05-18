# OpenCure result JSON schema

Every disease screened lands at `experiments/results/<Disease>.json`. This
document is the **canonical** reference for what fields are in that file
and what they mean. The schema is defined in code at
`opencure/scoring/common.py` — if this doc ever drifts from that file,
the code is authoritative.

Validation is automated: `opencure.scoring.common.validate_result_file(data)`
returns a list of warning strings; empty list = conformant. The test
suite runs this against every result JSON.

---

## Top-level keys

| Key | Type | Meaning |
|---|---|---|
| `disease` | str | Human-readable disease name. |
| `disease_entity` | str | DRKG entity (`Disease::MESH:Dxxxxxx`). Added by the v5.1 post-processor `refresh_known_treatment_labels.py` so downstream tools can skip name→entity resolution. |
| `status` | str | `"success"` or `"failed"`. |
| `timestamp` | str | ISO-8601 UTC, when the screen emitted this file. |
| `elapsed_seconds` | float | Wall-clock per-disease screen time. |
| `total_searched` | int | Candidate compounds scored before ranking. |
| `evidence_generated` | int | Number with full evidence populated. |
| `confidence_counts` | dict | Histogram of `{"HIGH": n, "MEDIUM": n, "LOW": n}` across top candidates. |
| `pipeline_version` | str | `"v5"` … `"v7"` depending on screen vintage. |
| `data_manifest_hash` | str | SHA-256 fingerprint of the input-data manifest at screen time. |
| `candidates` | list | Top-ranked drug candidates (typically 10). Each conforms to the Candidate schema below. |
| `ensemble_version` | str (opt) | Set once `scripts/score_ensemble_v5.py` has run. |
| `ensemble_model_path` | str (opt) | Path to the model pickle used. |
| `docking_axis` | dict (opt) | Metadata for the docking-axis backfill (source, n_disease_genes, note). |
| `mechanism_confidence` | float (opt) | v7 — 0–1 heuristic for how well the disease's molecular mechanism is mapped (`opencure/evidence/mechanism_uncertainty.py`). Below 0.4, wet-lab briefs flag every prediction as speculative. |
| `error` | str (opt) | Present with `status="failed"` when the disease name could not be resolved to a DRKG entity. |

---

## Candidate schema

Every entry in `candidates` must carry the required fields below; everything
else is optional but must come from the canonical names — no legacy field
names are permitted (the validator will flag them).

### Required

| Field | Type | Notes |
|---|---|---|
| `drug_id` | str | DrugBank ID (`DBxxxxx`). |
| `drug_name` | str | Human-readable name. |
| `disease_name` | str | Human-readable disease name. |
| `combined_score` | float | Final rank score in [0, 1]. |
| `pillars_hit` | int | How many pillars fired (non-zero). |
| `confidence` | str | `"HIGH"`, `"MEDIUM"`, or `"LOW"`. |

### Identity

`rank`, `drug_entity` (`Compound::DBxxxxx`), `disease_entity`, `relation_type`.

### Pillar scores (13 active)

KG embeddings: `transe_score`, `transe_rank`, `pykeen_score`, `pykeen_rank`,
`primekg_score`, `unified_score`, `unified_rank`, `txgnn_score`, `txgnn_rank`.

R-GCN (v6.1+, heterogeneous GNN, DistMult head): `rgcn_score`, `rgcn_rank`,
`rgcn_relation`.

Structural: `mol_similarity`, `similar_to`, `mol_emb_similarity`,
`mol_emb_similar_to` (`mol_emb_*` now sourced from MoLFormer-XL when the
artifact is present, ChemBERTa fallback otherwise).

JUMP Cell Painting (v7, 13th pillar): `jump_score`, `jump_rank`,
`jump_similar_to` — morphological-profile similarity to known treatments.

Network / transcriptomic: `proximity_score`, `proximity_distance`,
`gene_sig_score`, `gene_sig_rank`.

Causal / binding / ADMET: `mr_score`, `mr_genetic_targets`, `dti_score`,
`dti_best_target`, `admet_score`, `admet_flags`.

### Grouped scores (grouped_combiner output)

`kg_group_score`, `structural_group_score`, `network_group_score`,
`txgnn_group_score`, `mr_group_score`, `admet_multiplier`, `efficacy_score`,
`degree_penalty`, `groups_hit`.

### Evidence

`pubmed_total`, `pubmed_treatment_total`, `pubmed_repurposing_total`,
`key_papers`, `most_cited_paper`, `max_citations`, `semantic_scholar_papers`,
`abstract_analyses`, `repurposing_papers`, `clinical_trials`,
`clinical_trials_total`, `trial_phases`, `failed_trial_*`, `faers_*`,
`shared_targets`, `shared_target_count`, `direct_relations`,
`validation_experiments`, `ai_hypothesis`, `mechanistic_hypothesis`
(single-path NL), `kg_paths_text` (up to 3 NL paths),
`signature_reversal_*`, `signature_interpretation`.

### Labels

- `is_known_treatment` (bool) — True when the (drug, disease) pair has a
  DRKG treats edge from any relation in `KNOWN_TREATMENT_RELATIONS`
  (DRUGBANK::treats + GNBR + Hetionet). Labeling-only; never used for
  pillar scoring.
- `novelty_score` (float 0–1), `novelty_level` (str).
- `confidence`, `confidence_reasons`.

### v5 clinical guardrails (always present)

- `dose_plausibility`: dict with `plausibility`, `dose_range`, `confidence`,
  `rationale`, `stage_2` (Cmax/IC50 when ChEMBL bioactivity cache is
  present).
- `ddi_warnings`: dict with `n_interactions`, `has_warnings`, `top_interactions`.
- `pharmacogenomics`: dict with `has_flags`, `highest_risk`, `summary`,
  `cpic`, `pharmgkb`. Replaces pre-v5 `pgx_flags` (now a legacy name).
- `triangulation`: dict with `n_axes_agree` (0–4), `axes`
  (`{kg, docking, pharos, literature}`), `triangulation_score`, `label`
  (`"silver-standard"` when `n_axes_agree >= 3`), `axis_values`.
  Replaces pre-v5 flat `triangulation_score` field.
- `tissue_context`: dict from GTEx v8 + curated disease-tissue map.
  Keys: `tissues`, `n_genes`, `n_expressed`, `mean_tpm_in_tissue`,
  `max_tpm_in_tissue`, `tissue_specificity`, `context_modifier` in
  `[0.85, 1.15]`.

### v5.1 post-processor outputs

- `ensemble_prob` (float 0–1) — calibrated probability from
  `ensemble_v5.pkl` (`scripts/score_ensemble_v5.py`).
- `ensemble_rank` (int) — secondary rank by `ensemble_prob`.
- `ensemble_features` (dict) — the 6-feature vector used at inference.
- `docking`: dict with `kcal_per_mol`, `target_symbol`, `source`
  (`"chembl_bioactivity_proxy"` or `"not_wired"`), `note`, `hit`.
- `structural_alerts` — PAINS / Brenk / NIH structural-alert annotations.

### v7 post-processor outputs

Attached by the v7 layers; all fail-open (absent when the relevant
artifact hasn't been produced).

- `ensemble_prob_lower`, `ensemble_prob_upper` (float 0–1) — conformal
  interval at 90 % coverage, from `opencure/scoring/conformal.py`.
- `prediction_set_at_90` (list[int]) — conformally-consistent labels:
  `[1]` confident-positive, `[0]` confident-negative, `[0, 1]` genuinely
  uncertain.
- `ensemble_head` (str) — which ensemble head scored the candidate:
  one of `parasitic`, `viral`, `bacterial`, `oncology`, `rare_metabolic`,
  `chronic_systemic`, or `"shared"` when the disease's class fell back.
- `selectivity_score` (float 0–1), `n_off_targets` (int),
  `primary_target` (str), `primary_nM` (float) — ChEMBL off-target panel
  (`opencure/scoring/selectivity_panel.py`). High score = selective.
- `target_essentiality` (float | null), `essentiality_warning` (bool) —
  DepMap CRISPR essentiality of the primary target. The warning trips
  when the gene is essential in ≥ 80 % of cell lines.
- `red_team_assessment` (str) — adversarial critique covering single-
  pillar artifacts, low selectivity, essentiality, hub-bias, low
  mechanism confidence, evidence shortage, failed-trial history.

The full v7 field tuple is `V7_FIELDS` in `opencure/scoring/common.py`.

---

## Legacy names (forbidden)

The validator rejects any candidate that carries these field names. They
are leftovers from v1/v2/v3 and must be migrated to the canonical names
above.

| Legacy name | Canonical replacement |
|---|---|
| `txgnn_raw_score` | `txgnn_score` (rank-normalized) |
| `proximity_raw_score` | `proximity_score` |
| `dti_raw_score` | `dti_score` |
| `gene_sig_raw_score` | `gene_sig_score` |
| `pgx_flags` | `pharmacogenomics` |
| `triangulation_score` (flat) | `triangulation.triangulation_score` (nested) |
| `mechanism_path` | `mechanistic_hypothesis` or `kg_paths_text` |
| `top_candidates` (top-level) | `candidates` |

---

## Stability contract

Fields in this document do not break between patch releases. New fields
may be added at any time (v5 → v5.1 → v6 → v7 have all been additive);
existing ones keep their type and semantics. Removed fields go through
one release-cycle of deprecation (listed in `LEGACY_FIELDS`) before the
validator rejects them.
