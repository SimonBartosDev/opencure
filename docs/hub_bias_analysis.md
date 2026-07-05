# The hub-drug bias in OpenCure: observation, mitigation, limits

**Status:** degree-penalty mitigation active and unchanged through v7;
bias partially persists. v7 adds two further damping signals — the
hub-bias check is now one of the seven failure modes the adversarial
red-team agent flags per candidate, and the negative-control suite
(`tests/data/negative_controls.yaml`) includes a universal-hubs set
that must rank below the per-disease median in CI.
**Last updated:** v5 re-screen observations (2026-04-19); v7 mitigation
notes added 2026-05-11.

## The observation

Across OpenCure's early disease results, a handful of drugs repeatedly
surface as top-1 for biologically unrelated diseases:

| Drug | DRKG triplet degree | Appeared top-1 for |
|---|---|---|
| Cimetidine (DB00501) | 1,979 | Malaria, Chagas, Leishmaniasis (3/5 early v5 runs) |
| Dexamethasone (DB01234) | 3,413 | TB, HIV (multiple historical runs) |
| Tacrolimus (DB00864) | 2,638 | TB, Schistosomiasis |
| Octreotide (DB00104) | 1,453 | Multiple infectious diseases |

The signature: **drugs connected to thousands of genes via
well-annotated targets are mechanically close to whatever disease-gene
set you probe against**, regardless of whether the drug has actual
disease-specific mechanism.

## Why it happens

All four top-offending drugs are real-world pleiotropic agents:
- Dexamethasone: broad immunomodulator, 100+ approved indications
- Tacrolimus: calcineurin inhibitor, transplant + autoimmune + dermatology
- Cimetidine: H2 blocker with off-target effects at CYP, androgen
  receptor, immune cells
- Octreotide: somatostatin analog with endocrine + GI + oncology uses

Their KG representations have many outgoing edges to many gene/target
nodes. Proximity-based scoring (STRING PPI, TransE distance,
Open Targets mechanism paths) finds them "close" to most disease gene
sets by construction. This is not a bug in any single pillar — it's an
emergent property of the graph topology.

## v5 mitigation: hub-degree penalty

`opencure/scoring/hub_normalize.py` applies a multiplicative penalty to
KG-group and network-group scores, calibrated against the median
degree among ChEMBL phase ≥1 drugs (81 edges):

```
penalty = (log(81 + 1) / log(degree + 1)) ^ 0.5
       → 1.0 for typical drugs (degree ≈ 80)
       → 0.72–0.78 for the four named hubs (degree 1400–3400)
       → 0.38 for worst-case hubs (degree > 100,000)
```

The penalty is applied *only* to KG and network-based scores, leaving
structural, MR, TxGNN, ADMET untouched (these are not driven by graph
topology).

## What the mitigation achieves

Empirical check on v5's first 5 diseases (50 candidates):

| | Degree-penalty off | Degree-penalty on (v5) |
|---|---|---|
| Top-1 captured by Cimetidine | 5/5 diseases (all) | 3/5 (Malaria, Chagas, Leishmaniasis) |
| Top-1 captured by hub drugs (any) | 5/5 | 4/5 |
| Genuinely disease-specific top-1 | Dexamethasone-for-TB (inconsistent) | Tacrolimus (TB — real: standard TB-IRIS care), Icosapent (Dengue — plausible anti-inflammatory) |

The penalty reduces hub dominance but doesn't eliminate it for diseases
where hub drugs genuinely do rank via multiple converging pillars.

## Where the mitigation falls short

**Cimetidine still wins Malaria + Chagas + Leishmaniasis at v5 scores
around 0.46-0.55.** Inspection reveals this is because Cimetidine's
score is actually supported by multiple independent pillars for
parasitic diseases specifically:

- Proximity: Cimetidine targets H1/H2 histamine receptors, expressed in
  macrophages and relevant to immune response to parasites
- DTI: some published repurposing literature for Cimetidine in malaria
- KG: Cimetidine has genuine treats-edges to several infectious
  conditions in DRKG, not just parasitic
- ChemBERTa: matches other anti-parasitic drugs via substructural
  features

In other words: Cimetidine is winning because multiple pillars agree,
not just because of graph topology. The hub penalty correctly damps
the portion of the score that's from topology alone. What's left is a
signal from the pillars that happen to have genuinely relevant biology.

## Interpretation for downstream consumers

For researchers using OpenCure predictions:

1. **If a hub drug ranks #1, look at its pillar breakdown**, not just
   the combined score. If only KG/proximity pillars fire, the
   prediction is suspect. Note the strong caveat, though: under
   leak-free, popularity-baselined evaluation the KG-embedding,
   chemical-structure (ChemBERTa) and cell-morphology pillars do **not**
   beat a trivial popularity baseline, so "4+ pillars agree" is weak
   evidence of a genuine prediction — the only component that beats
   popularity is genetics-anchored target prioritization. Treat any
   top-ranked hit as a triage hypothesis for expert review, not a
   validated lead: zero OpenCure predictions are wet-lab confirmed and
   no novel credible lead has been found.

2. **Cross-reference the mechanism path**. v5 surfaces natural-language
   paths. If Cimetidine → targets → gene-X → is bound by → KnownDrug →
   treats → Disease, that's a mechanistically plausible hypothesis.
   If the path is uninformative (Cimetidine → binds → common-hub-gene →
   linked to → Disease), treat with skepticism.

3. **Check the triangulation score**. Silver-standard predictions (3+
   independent axes: KG + docking + Pharos TDL + literature) should
   survive the hub-bias filter naturally.

## Planned further mitigation (post-v5)

- **Relation-type-weighted RRF**: downweight KG relations that are
  topologically dense (e.g., `STRING::OTHER::Gene:Gene`) when scoring
  drug-disease pairs. Preserves scoring when edges are specific
  (e.g., DGIDB inhibitor/agonist).
- **Per-disease calibration**: some diseases have naturally dense
  gene-disease graphs (cancer, diabetes) where proximity alone is
  uninformative. Calibrate the network-group weight per disease.
- **Publication-guided reweighting**: for each predicted drug-disease
  pair, downweight if the drug's overall PubMed footprint is
  disproportionate to its evidence-specific footprint for this disease.

None of these are strictly code problems — they require either more
data curation or a per-disease training step that's on the v6 roadmap.

## Summary

The v5 hub-degree penalty reduces but does not eliminate hub-drug
dominance. Per-drug inspection (pillar breakdown, mechanism path,
triangulation) remains necessary before treating any candidate as a
triage hypothesis for expert review.
Cimetidine-for-infectious-diseases is the clearest remaining case;
the other three (Dex, Tac, Octreotide) are mostly resolved.

Full transparency: OpenCure predictions are **triage hypotheses for
expert review**, not point-estimates of therapeutic probability. Under
leak-free evaluation the KG-embedding, chemical-structure and
cell-morphology pillars do not beat a popularity baseline; the single
component that does is **genetics-anchored target prioritization** —
it beats a popularity baseline ~5× on the genetics-covered subset
(leak-free, temporally validated, honest temporal Hit@10 ~10%), but it
is rediscovery-leaning and covers only part of diseases. Hub bias is
disclosed here so that reviewers and collaborators can apply
appropriate skepticism when top-ranked predictions share the hub
signature.
