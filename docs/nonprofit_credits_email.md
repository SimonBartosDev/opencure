# Nonprofit / research credits — outreach drafts

Two emails to send before paying out-of-pocket for the v7 push:

1. **Modal** — covers compute (~$53 for the full first-time push, ~$50/quarter for retraining).
2. **Anthropic** — covers Claude API tokens for the LLM-narrated layers
   (red-team prose + wet-lab brief mechanism paragraphs). Optional —
   the deterministic fallbacks always run, but Claude makes the briefs
   readable for cold outreach.

Both organisations have public research-credits programs.
Application turnaround is ~1-2 weeks.

---

## 1. Modal — Research / Open-source credits

**To:** `support@modal.com` (cc `partnerships@modal.com` if known)

**Subject:** Modal credit application — OpenCure (open-source drug
repurposing for neglected tropical and rare diseases)

```
Hi Modal team,

I'm building OpenCure (https://github.com/SimonBartosDev/opencure),
an open-source drug-repurposing platform that ranks existing FDA-
approved compounds against neglected tropical diseases (Schistosomiasis,
Chagas, Leishmaniasis, etc.) and rare genetic diseases (Niemann-Pick,
Sickle Cell, Gaucher). The codebase is Apache-2.0 and explicitly
mission-locked — every prediction is publicly browsable, every model
is deposited to Zenodo with a content hash, and we will not pivot to
monetisation.

The architecture currently runs at version 7: 13 orthogonal scoring
pillars (KG embeddings, MoLFormer-XL chemistry embeddings, ESM-2
protein embeddings, JUMP Cell Painting morphological similarity,
DeepPurpose DTI, gene-signature reversal, network proximity, R-GCN
heterogeneous GNN, Mendelian Randomization, ADMET, etc.) combined
into a calibrated ensemble with conformal prediction intervals,
adversarial red-team critiques, and per-disease wet-lab briefs.

Modal already runs our v6 GPU retrain (RotatE, R-GCN, 93-disease
screen) — typical cost ~$15/cycle. The v7 push adds:

  - MoLFormer-XL embedding precompute (A10G, ~$2)
  - ESM-2 150M protein-embedding precompute (A100 40GB, ~$11)
  - Conformal calibration + per-class ensemble heads (CPU, ~$1)
  - Updated 93-disease screen + finalize (~$11)
  - Quarterly retraining cycle (~$50/quarter ongoing)

We're requesting USD 1,000 of credit to:
  1. Run the full v7 first-time push (~$53)
  2. Cover four quarters of retraining (~$200)
  3. Leave headroom (~$700) for ablation runs, head-to-head benchmarks
     against TxGNN/NetMed baselines, and the bioRxiv methods-paper
     reproducibility submission.

Beyond the compute, we'd attribute Modal in the methods paper,
the public dashboard footer, and the methods-paper acknowledgements
section.

Happy to share:
  - Application code (modal_app.py wires the full chain)
  - Running benchmark numbers from v6.1 (357 regression tests passing)
  - The methods-paper draft (docs/methods_paper_draft.md) on request

Thanks,
Simon Bartos
imon.bartos@gmail.com
github.com/SimonBartosDev/opencure
```

---

## 2. Anthropic — API research credits

**To:** Apply via https://console.anthropic.com (Settings → Plans →
Research / nonprofit) or `support@anthropic.com`.

**Subject:** API credit application — OpenCure (open-source drug
repurposing for neglected and rare diseases)

```
Hi Anthropic team,

OpenCure is an open-source drug-repurposing platform
(github.com/SimonBartosDev/opencure, Apache-2.0) targeting neglected
tropical diseases and rare genetic disorders. We're requesting USD 200
of API credit to:

  1. Generate adversarial "red-team" critiques per top-K prediction
     across 93 screened diseases. Output: a structured 3-risk
     argument-against-the-prediction string attached to each
     candidate, forming part of the per-disease wet-lab briefs.
     Estimated tokens: ~5K output × 93 diseases × 20 candidates =
     ~9M output tokens. Using Haiku 4.5 batch API (50% off):
     ~$22.

  2. Generate mechanism-of-action paragraphs for the per-disease
     wet-lab briefs handed to academic / nonprofit lab partners
     (DNDi, MMV, FIND, CureSCi, NPUK). One paragraph per top-5
     candidate × 41 diseases (NTDs + rare): ~10M tokens, ~$25 with
     Haiku batch.

  3. One-time methods-paper revision pass on the bioRxiv submission
     (Sonnet 4.5 / Opus 4.7): ~$50.

We have local-Llama-3.1-8B-via-MLX fallbacks for both red-team and
brief generation, so Claude is not on the critical path — but the
brief quality difference matters for cold-outreach to wet-lab PIs.

Happy to attribute Claude in the methods paper and the per-disease
brief footer ("LLM-narrated mechanism paragraph generated using
Claude {model_version}, {date}").

Thanks,
Simon Bartos
imon.bartos@gmail.com
github.com/SimonBartosDev/opencure
```

---

## Process

1. Send Modal email today. Their credits team typically replies within
   one week.
2. Send Anthropic application via console.anthropic.com (the Research
   tier has a built-in form). Approval typically 1-2 weeks.
3. While waiting, run the cheapest precomputes on the existing $30
   free tier:
       modal run scripts/modal_app.py::v7_precomputes_cheap
   (Estimated ~$13, leaves $17 headroom for the rescreen + finalize.)
4. Once Modal credits land, run the full retrain cycle:
       modal run scripts/modal_app.py::full_chain
5. After v7 numbers land, submit bioRxiv preprint and revisit
   Anthropic credits for any LLM-heavy brief regeneration.
