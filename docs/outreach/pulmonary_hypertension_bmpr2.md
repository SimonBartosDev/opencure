# OpenCure — Heritable Pulmonary Arterial Hypertension Repurposing Brief

*A lab-outreach brief for the heritable-PAH / BMPR2 research community. It
follows the honest structure of OpenCure's Chagas template
(`docs/outreach/chagas_dndi.md`): one mechanistically-grounded hypothesis,
an explicit account of what is genuinely novel versus already-explored, and a
frank triage of the screen's false positives. Every factual claim is cited;
every hypothesis is labelled as a hypothesis.*

---

## 1. Who we are — and what we are not

**OpenCure** is an open-source, non-profit computational drug-repurposing
platform. All code is Apache-2.0; all results are content-hashed and
reproducible.

We want to be precise about what OpenCure is **not**, because the field is full
of tools that overclaim:

- OpenCure is **not a validated predictor.** We recently found, and have
  publicly withdrawn, an inflated headline figure (an ensemble "AUC-ROC 0.997"
  that turned out to be data leakage — knowledge-graph features were scored
  from a graph that still contained the test edges).
- This brief comes from a deliberately narrower, **leak-free** method:
  a genetics-anchored screen that links a disease to a human-genetics causal
  gene and then to drugs with a curated ChEMBL mechanism on that gene. On the
  subset of diseases this method covers, its honest, leak-controlled evaluation
  performs at roughly **5× a trivial popularity baseline** for retrieving known
  drug–disease pairs — useful as a triage signal, **not** a prediction of
  efficacy.
- The genetics-anchored ranker **does not check direction of effect.** Its raw
  output mixes genuine leads with rediscoveries (a disease's own approved drug)
  and direction-flipped artefacts. This brief is the product of *manual*
  direction and novelty filtering on top of the screen — we say so plainly.

What OpenCure **does** genuinely offer: a systematic, reproducible,
fully-auditable screen, with the human-genetics rationale for every candidate
made explicit and checkable.

We are not writing to claim we have found a treatment for pulmonary arterial
hypertension. We are writing to offer a transparent screening infrastructure
and to ask whether a specific, genotype-stratified validation question is worth
a bench experiment to a lab already working on BMPR2 biology.

## 2. The unmet need

- Pulmonary arterial hypertension (PAH) is a progressive obliterative
  vasculopathy of the small pulmonary arteries. Despite three approved
  vasodilator drug classes, it remains a disease with a **median survival of
  roughly 7 years** — the approved drugs relieve symptoms but do not reverse
  the underlying vascular remodelling.
  ([BMPR2 mutations and survival, individual-participant meta-analysis, *Lancet Respir Med*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4737700/))
- **Heritable PAH** is the genetically defined core of the disease.
  Loss-of-function mutations in **BMPR2** (bone morphogenetic protein receptor
  type 2) are found in **>80% of familial** and **~20% of sporadic** PAH.
  Carriers are diagnosed younger (mean ~35 vs ~42 years) and carry a **~27%
  higher all-cause mortality** and **~42% higher hazard of death or transplant**
  than non-carriers.
  ([BMPR2 survival meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4737700/);
  [FK506 activates BMPR2 — Spiekerkoetter et al., *JCI* 2013](https://www.jci.org/articles/view/65592))
- No approved PAH therapy is directed at the **BMPR2 pathway itself.** The one
  pathway-targeted biologic to reach the clinic — sotatercept, an
  activin-ligand trap — validates the broader idea that *restoring*
  BMP/activin balance is disease-modifying, but it is a new, costly biologic.
  An orally available, off-patent small molecule acting on the same axis would
  be of obvious value to patients in lower-resource settings.

## 3. What our screen produced — honestly triaged

We screened drugs with curated ChEMBL mechanisms against the human-genetics
causal genes for PAH. Here is our candid read of what the genetics-anchored
screen surfaced, in three buckets.

### (a) One hypothesis with a genuine, direction-correct rationale — Tacrolimus (FK506)

- The screen's **top-ranked causal gene for PAH is BMPR2** (genetics score
  0.97), and the highest-scoring drug action it pairs with that gene is a
  **BMPR2 agonist** — i.e. the screen correctly identifies that PAH is a
  **deficiency** of BMPR2 signalling and that the therapeutic direction is to
  **restore** it. (The specific top-listed agonists in our raw output —
  dibotermin alfa / eptotermin alfa, recombinant BMP-2 bone-graft products —
  are not plausible systemic PAH drugs; this is where manual curation is
  required.)
- The literature-validated small molecule that performs exactly this action is
  **tacrolimus (FK506)**, an approved calcineurin-inhibitor immunosuppressant.
  A screen of >3,500 FDA-approved drugs identified FK506 as a strong activator
  of BMP signalling: it releases the repressor **FKBP12** from the type-I BMP
  receptors ALK1/2/3, restoring downstream SMAD1/5 signalling and *ID1* gene
  regulation. In patient-derived pulmonary artery endothelial cells, low-dose
  FK506 reversed dysfunctional BMPR2 signalling, and it prevented and reversed
  experimental PAH in rodents.
  ([FK506 activates BMPR2 — Spiekerkoetter et al., *JCI* 2013](https://www.jci.org/articles/view/65592))
- **The testable, direction-correct hypothesis:** in heritable-PAH cells
  carrying a defined BMPR2 loss-of-function mutation, does low-dose tacrolimus
  restore SMAD1/5 / *ID1* signalling and reverse the disease cellular
  phenotype — and **does the magnitude of rescue depend on mutation class**
  (missense vs truncating)? Missense and truncating BMPR2 mutations produce
  measurably different disease severity
  ([truncating vs missense BMPR2 severity, *Eur Respir J*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2762975/)),
  and FK506's mechanism — de-repressing residual receptor — should, in
  principle, rescue *hypomorphic* (missense) receptors more than *null*
  (truncating) ones.

### (b) Honest caveat — this idea is partly explored; the *novelty* is the stratification

- We must be explicit: **tacrolimus-for-PAH is not a new idea.** It has been
  through a single-centre, randomised, placebo-controlled **Phase IIa safety
  and tolerability trial (TransformPAH; 23 patients, NYHA II/III).** FK506 was
  generally well tolerated; the treatment arm showed **increased BMPR2
  expression** versus placebo — i.e. target engagement — but the trial was
  **not powered for, and did not show, clinical efficacy** on exercise capacity
  or haemodynamics.
  ([Randomised safety/tolerability trial of FK506 for PAH, *Eur Respir J* 2017](https://publications.ersnet.org/content/erj/50/3/1602449);
  [Low-dose FK506 in end-stage PAH, *AJRCCM*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4532822/))
- So this brief does **not** claim a novel drug or a novel target. What is
  genuinely under-explored is whether the **null TransformPAH efficacy result
  reflects an inactive drug, or an unstratified trial**: TransformPAH enrolled
  PAH broadly and did not select or pre-stratify by BMPR2 genotype. The
  mechanistically motivated, still-open question is whether **BMPR2-mutation
  carriers — and specifically missense-mutation carriers — are the responder
  subgroup.** That is a hypothesis a basic-science lab can interrogate at the
  cellular level *before* any further clinical commitment, and it is the
  cheapest, most falsifiable next step.

### (c) Likely false positives — what direction-checking flags

We include these to be honest about the screen's failure modes:

- For PAH, the same raw screen also ranks **KDR/VEGFR2 inhibitors** (sunitinib,
  pazopanib, lenvatinib and ~30 related kinase inhibitors) highly. Anti-VEGF
  tyrosine-kinase inhibitors are associated with *causing* or worsening
  pulmonary hypertension; ranking them as treatments is a direction artefact,
  not a lead.
- Across other diseases the same screen produced clear direction-flipped
  errors — e.g. for **Pompe disease** (an acid-α-glucosidase *deficiency*) its
  top hit was *voglibose*, an *inhibitor* of that enzyme; for
  **Ehlers-Danlos syndrome** (a collagen defect) it surfaced *collagenase*,
  which *degrades* collagen. We report these openly: a screen whose ranker
  ignores direction of effect must be filtered by a human before any brief is
  written, and this document is that filtered output.

## 4. What we are proposing

A **collaboration, not a claim.**

- **OpenCure brings:** the open, reproducible genetics-anchored screen and the
  explicit, checkable human-genetics rationale behind every candidate.
- **The partner lab brings:** BMPR2 disease-cell models (patient-derived or
  iPSC-derived pulmonary artery endothelial / smooth-muscle cells with defined
  BMPR2 genotypes) and the assay expertise that a computational screen cannot
  supply.
- **Concretely — the single cheapest, most falsifiable next step:** treat a
  panel of BMPR2-mutant PAH cells (missense vs truncating) and BMPR2-wild-type
  controls with low-dose tacrolimus, and read out **SMAD1/5 phosphorylation,
  *ID1* expression, and a remodelling phenotype** (e.g. endothelial apoptosis /
  smooth-muscle proliferation). The directional prediction is concrete:
  **rescue should be greatest in missense (hypomorphic) carriers.** A positive
  result re-frames the negative TransformPAH trial as an enrolment problem and
  motivates a genotype-stratified re-trial; a negative result is honest,
  publishable evidence that bounds the hypothesis.
- We are **not** asking for funding or endorsement. We are asking whether an
  open, honest computational collaborator is useful, and whether this one
  stratified question is worth a plate.

## 5. Suggested validation experiment

- **System:** patient-derived or iPSC-derived pulmonary artery endothelial
  cells (PAECs) and/or smooth-muscle cells, with **defined BMPR2 genotypes**
  spanning missense and truncating loss-of-function variants, plus
  BMPR2-wild-type controls.
- **Intervention:** low-dose tacrolimus across the sub-immunosuppressive
  concentration range used in the prior PAH work (target trough analogues of
  the TransformPAH <2 / 2–3 / 3–5 ng·mL⁻¹ bands).
- **Primary readout:** restoration of canonical BMP signalling — SMAD1/5
  phosphorylation and *ID1* / *ID3* transcript induction.
- **Secondary readout:** reversal of a disease-relevant cellular phenotype —
  PAEC apoptosis / monolayer integrity, or pulmonary artery smooth-muscle
  proliferation.
- **The discriminating analysis:** rescue magnitude as a function of mutation
  class. The hypothesis predicts an ordered effect (missense > truncating),
  which is what a genotype-stratified clinical re-trial would need to justify
  itself.

## 6. Target labs / foundations

These are real, independently verifiable groups already working on BMPR2 and
PAH. OpenCure has **no affiliation with, and no endorsement from, any of them**;
they are listed as the natural expert audience for this hypothesis.

- **Rabinovitch Laboratory for Cardiopulmonary Research, Stanford University** —
  Prof. Marlene Rabinovitch (Dwight & Vera Dunlevie Professor of Pediatrics,
  Cardiology). Long-standing programme on BMPR2 signalling, patient-specific
  iPSC-derived endothelial cells, and pathways that protect BMPR2-mutation
  carriers.
  ([Rabinovitch Lab, Stanford Medicine](https://med.stanford.edu/rabinovitchbland/research/individual-projects.html);
  [Stanford Profiles](https://profiles.stanford.edu/marlene-rabinovitch))
- **Prof. Nicholas Morrell, University of Cambridge / Royal Papworth Hospital** —
  Research Director of the National Pulmonary Hypertension Service; group
  defines the genetic architecture of PAH and the molecular consequences of
  BMPR-II loss of function.
  ([Cambridge Cardiovascular — Nick Morrell](https://www.cardiovascular.cam.ac.uk/directory/nmorrell))
- **Pulmonary Hypertension Association (PHA) research programme** — funds
  innovative adult and paediatric PH research (Innovation in PH Grant; Pediatric
  PH Research Award), and is the natural route to the wider clinical-research
  community and to patient-stratified cohorts.
  ([PHA Research Programs](https://phassociation.org/research/pha-research-programs/))

The originating mechanistic work on FK506 / BMPR2 (Spiekerkoetter, Rabinovitch
and colleagues, *JCI* 2013) and the TransformPAH trial are the direct prior art
this brief builds on, and either group above is well placed to judge whether
the genotype-stratified question is worth pursuing.

## 7. Draft outreach email

> **Subject:** Open-source genetics-anchored screen for PAH — a stratified
> tacrolimus/BMPR2 question
>
> Dear Prof. [name],
>
> I maintain OpenCure, a non-profit open-source drug-repurposing platform. I am
> writing with a deliberately modest, mechanistically specific proposal, and I
> want to be upfront about its limits before anything else.
>
> OpenCure is **not** a validated predictor — we recently found and publicly
> withdrew an inflated benchmark figure. The method behind this note is a
> narrower, leak-free, genetics-anchored screen: it links a disease to a
> human-genetics causal gene and then to drugs with a curated mechanism on that
> gene. For PAH it correctly identifies **BMPR2** as the causal gene and a
> **BMPR2 agonist** as the therapeutic direction.
>
> The literature-validated small molecule for that action is **tacrolimus
> (FK506)**, via FKBP12 release at the type-I BMP receptors. I know this is not
> a new idea — TransformPAH already showed FK506 engages BMPR2 but did not show
> clinical efficacy. What I think is genuinely under-explored is whether that
> null result reflects an *unstratified* trial: TransformPAH did not select or
> pre-stratify by BMPR2 genotype. The falsifiable, bench-scale question is
> whether **BMPR2-missense (hypomorphic) carriers are the responder subgroup** —
> testable in your BMPR2-genotyped PAH cell models via SMAD1/5 / *ID1* rescue
> before any clinical commitment.
>
> I am not asking for funding or endorsement — only whether an open, honest
> computational collaborator would be useful, and whether this one stratified
> question is worth a plate. A negative result would be just as valuable to us
> as a positive one.
>
> All code, data, and our honest evaluation are public at
> github.com/SimonBartosDev/opencure.
>
> With respect for your work,
> [Maintainer name] — OpenCure — imon.bartos@gmail.com

## Honest status of this document

This is an **outreach brief built on a deliberately filtered lead.** Read it
with these caveats explicit:

- **The genetics is strong; the drug is approved; the direction is correct.**
  BMPR2 loss of function is the best-established genetic cause of PAH,
  tacrolimus is an approved immunosuppressant (a true repurposing), and
  restoring BMP signalling is the established therapeutic direction.
- **The core idea is not novel.** Tacrolimus-for-PAH has a published mechanism
  (Spiekerkoetter et al., *JCI* 2013) and a completed Phase IIa trial
  (TransformPAH). This brief does not claim otherwise. The only genuinely
  under-explored element is the **BMPR2-genotype-stratified responder
  hypothesis**, and that is presented strictly as a hypothesis.
- **OpenCure's screen did not "discover" this.** The screen surfaced the right
  gene and the right direction; manual literature curation supplied the
  specific drug and the stratification idea. We say this plainly rather than
  dressing up curation as algorithmic discovery.
- We considered several alternative covered diseases and rejected them as
  outreach leads because, on honest checking, each was a rediscovery,
  direction-uncertain, or already definitively tried and failed (e.g.
  galunisertib was directly tested in Marfan-syndrome mice and showed no
  benefit; verapamil failed six trials in bipolar disorder). PAH/BMPR2 is the
  most defensible lead the genetics-anchored screen produced — and even it is
  honestly a *re-framing* of prior work, not a fresh discovery.

*Sources are linked inline. Drafted from OpenCure's leak-controlled
genetics-anchored screen and verified public literature; no trial result,
citation, or affiliation in this document is fabricated. Contact details are
placeholders — the maintainer name should be filled in, and the real labs
reached via their published institutional channels, before sending.*
