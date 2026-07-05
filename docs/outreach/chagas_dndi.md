# OpenCure — Chagas Disease Repurposing Brief

*A lab-outreach brief for the DNDi Chagas programme. This is the **template**
for OpenCure's honest outreach: one mechanistically-grounded hypothesis, one
cautionary signal, and a frank account of the screen's false positives. Every
factual claim is cited; every hypothesis is labelled as a hypothesis.*

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
- Our own **leak-free** evaluation shows the screen's ranking performs at
  roughly the level of a trivial popularity baseline. We state this openly at
  the top of our README and architecture page.

What OpenCure **does** genuinely offer:

- a systematic, reproducible, fully-auditable screen of ~10,500 approved and
  investigational drugs against a disease;
- a **calibrated uncertainty interval** and an automated **adversarial
  red-team critique** on every candidate;
- complete transparency — open code, open data, open evaluation.

We are not writing to claim we have found a treatment for Chagas disease. We
are writing to offer an honest, transparent screening infrastructure, and to
ask whether a collaboration — in which DNDi's disease expertise and assays
anchor and validate the screen — would be useful to your programme.

## 2. The unmet need

- Chagas disease (*Trypanosoma cruzi* infection) affects an estimated
  6–7 million people, predominantly across Latin America.
- Only two drugs exist — **benznidazole** and **nifurtimox** — both decades
  old, both with significant toxicity and limited efficacy against the chronic
  phase of the disease.
- DNDi's own Chagas programme is built around exactly this gap: the Shionogi
  collaboration begun January 2025, the natural-products project with Kitasato,
  Nagasaki and Tokyo universities, and screening with Institut Pasteur Korea
  and the University of Dundee — with the stated aim of a Phase III trial by
  2028. ([DNDi Chagas programme](https://dndi.org/research-development/portfolio/open-chagas/))

## 3. What our screen produced — honestly triaged

We screened ~10,500 drugs against Chagas disease. Rather than present a ranked
list as "predictions" — which, given our honest evaluation, we cannot justify —
here is our candid read of what the screen surfaced, in three buckets.

### (a) One hypothesis with a genuine mechanistic rationale — Forodesine

- **Forodesine** (immucillin-H / BCX-1777; approved in Japan as a treatment
  for relapsed peripheral T-cell lymphoma) is a transition-state-analogue
  inhibitor of **purine nucleoside phosphorylase (PNP)**.
- *T. cruzi* is a **purine auxotroph**: it cannot synthesise the purine ring
  *de novo* and depends entirely on the **purine salvage pathway** to acquire
  purines from its host. Salvage enzymes — HGPRT, the 6-oxopurine
  phosphoribosyltransferases, IMPDH — are an established anti-trypanosomal
  target class.
  ([HGPRT in *T. cruzi*, PMC9536471](https://pmc.ncbi.nlm.nih.gov/articles/PMC9536471/);
  [6-oxopurine salvage as a drug target, PLOS NTD](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0006301);
  [*T. cruzi* IMPDH, AAC 2025](https://journals.asm.org/doi/10.1128/aac.01210-25))
- **The testable hypothesis:** does Forodesine engage the *T. cruzi* purine
  salvage pathway with an anti-parasitic effect? The *pathway* is a validated
  target class; the *specific drug* against *T. cruzi* is, to our knowledge,
  untested. This is a clean, falsifiable in-vitro assay question.
- **Honest caveat:** we have found no evidence that Forodesine has been tested
  against *T. cruzi*. This is a hypothesis, not a finding.

### (b) Structural neighbours of azoles — a cautionary signal, not a positive one

- Two candidates (Phosphoramidon, an etheno-NAD analogue) were surfaced by
  molecular-structure similarity to the antifungal azoles **posaconazole** and
  **itraconazole**.
- We flag this **against itself.** Posaconazole (CHAGASAZOL trial; Molina et
  al., *NEJM* 2014) and the ravuconazole prodrug E1224 (STOP-CHAGAS trial)
  **both failed** in Chagas clinical trials: 81–92 % of posaconazole-treated
  patients were PCR-positive for *T. cruzi* during follow-up, versus 38 % on
  benznidazole.
  ([Randomized trial of posaconazole and benznidazole, NEJM](https://www.nejm.org/doi/full/10.1056/NEJMoa1313122);
  [STOP-CHAGAS, JACC](https://www.jacc.org/doi/abs/10.1016/j.jacc.2016.12.023))
- A drug that resembles a *failed* drug is a reason for **caution**, not
  optimism. We include this bucket precisely to show that our screen surfaces —
  and our brief reports — inconvenient context rather than hiding it.

### (c) Likely false positives — what our own critique flags

- Several top-ranked candidates are **immunosuppressants** (tacrolimus,
  hydrocortisone). For an infectious disease in which immunosuppression risks
  *T. cruzi* **reactivation**, these are mechanistically implausible as
  treatments.
- We list them not as leads, but as an honest demonstration of the kind of
  artefact our adversarial red-team layer is designed to catch. A screen that
  cannot recognise its own false positives is not worth a wet lab's time.

## 4. What we are proposing

A **collaboration, not a claim.**

- **OpenCure brings:** the screening infrastructure, the calibrated-uncertainty
  and adversarial-critique layers, and fully open, reproducible code and data.
- **DNDi brings:** *T. cruzi* assay capacity and disease expertise — the ground
  truth that a computational screen cannot supply itself.
- **Concretely:** the single cheapest, most falsifiable next step is the
  Forodesine / purine-salvage question in bucket (a) — one in-vitro assay
  against *T. cruzi*. A result either way is informative: a positive is a lead;
  a negative is honest, publishable evidence that calibrates the platform.
- We are **not** asking for funding or endorsement. We are asking whether an
  honest, open, computational collaborator is useful to your Chagas pipeline.

## 5. Draft outreach email

> **Subject:** Open-source repurposing screen for Chagas — collaboration enquiry
>
> Dear DNDi Chagas R&D team,
>
> I maintain OpenCure, a non-profit open-source drug-repurposing platform. I am
> writing with a deliberately modest proposal.
>
> OpenCure has screened ~10,500 approved and investigational drugs against
> Chagas disease. I want to be upfront: OpenCure is **not** a validated
> predictor — we recently found and publicly withdrew an inflated benchmark
> figure, and our honest evaluation shows the ranking is no better than a
> popularity baseline. What the platform genuinely provides is a transparent,
> reproducible screen with calibrated uncertainty and an automated adversarial
> critique on every candidate.
>
> From that screen, one hypothesis has a genuine mechanistic rationale worth a
> single in-vitro assay: **Forodesine**, an approved PNP inhibitor, against the
> purine-salvage pathway that *T. cruzi*, as a purine auxotroph, depends on.
> The pathway is a validated target class; Forodesine against *T. cruzi* is, as
> far as I can find, untested.
>
> I am not asking for funding or endorsement — only whether an open, honest
> computational collaborator would be useful to your programme, and whether
> this one hypothesis is worth a plate. A negative result would be just as
> valuable to us as a positive one.
>
> All code, data, and our honest evaluation are public at
> github.com/SimonBartosDev/opencure.
>
> With respect for your work,
> [Maintainer name] — OpenCure — imon.bartos@gmail.com

## Honest status of this document

This is a **template** outreach brief. It deliberately does not oversell: it
presents exactly one mechanistically-grounded hypothesis, one cautionary
signal, and a frank account of the screen's false positives. The contact
details are placeholders — the real DNDi contact should be reached via their
published Chagas-programme channels, and the maintainer name filled in before
sending.

*Sources are linked inline. Drafted from OpenCure's leak-controlled screen and
verified public literature; no trial result or citation in this document is
fabricated.*
