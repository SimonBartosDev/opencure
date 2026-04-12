# OpenCure Roadmap: From Predictions to Impact

## What's Done (v1.0)

- 8-pillar AI scoring engine (TransE, RotatE, TxGNN, Fingerprints, ChemBERTa, MR, Gene Signatures, Network Proximity)
- 25 diseases screened, 245 candidates (84 HIGH, 151 novel/breakthrough)
- Ensemble validation: AUC-ROC 0.998
- Per-disease PDF reports with full evidence breakdowns
- Database: JSON + CSV exports
- GitHub repo public (Apache 2.0): github.com/SimonBartosDev/opencure
- bioRxiv preprint drafted and formatted
- 243 researcher outreach emails sent
- Dockerfile + deployment config ready

---

## Phase 1: Immediate (This Week)

### 1A. Submit bioRxiv Preprint
- Upload OpenCure_Preprint.pdf to biorxiv.org
- Subject area: Bioinformatics
- License: CC-BY 4.0
- Check AI disclosure box
- Result: Citable DOI within 48 hours

### 1B. Deploy Web App to Railway
- Create Railway account, link GitHub repo
- Add persistent volume for DRKG data (~576MB)
- Set env vars: OPENCURE_DATA_DIR, PORT
- Result: Public URL researchers can search from browser

### 1C. Monitor Researcher Responses
- Check project email daily
- Prioritize replies from: Harvard, Johns Hopkins, CDC, Fiocruz, NIH
- Any researcher willing to run a cell assay = highest priority

---

## Phase 2: Validation Sprint (Weeks 2-4)

### 2A. Deep-Dive the Top 3 Breakthrough Predictions
Pick 3 candidates where: MR score > 0.5 (causal genetic evidence), drug is cheap/off-patent, disease has no good treatment. Current top picks:

1. **Sirolimus for IPF** (MR: 0.64, 5 pillars) — mTOR pathway, fibroblast proliferation
2. **S-Adenosyl-L-Homocysteine for Parkinson's** (MR: 0.76) — methylation pathway
3. **Everolimus for Gaucher disease** (BREAKTHROUGH) — mTOR, no prior literature

For each: write a 2-page deep-dive with full mechanism hypothesis, proposed experiments, and estimated cost. Send directly to the most relevant researcher from the outreach list.

### 2B. Contact Disease-Specific Organizations
- **DNDi** (Drugs for Neglected Diseases initiative) — Chagas, Leishmaniasis, Malaria
- **WIPO Re:Search** — IP-free drug repurposing partnerships
- **NORD** (National Organization for Rare Disorders) — Gaucher, Fabry, Duchenne
- **CureDuchenne** — Duchenne muscular dystrophy specifically
- **Michael J. Fox Foundation** — Parkinson's disease
- **ALS Association** — ALS candidates

### 2C. Add More Scoring Pillars (v2 Engine)
- **Real molecular docking** (AutoDock Vina) — actual binding affinity, not proxy
- **DTI prediction** (DeepPurpose) — deep learning drug-target interaction
- **Clinical trial outcome predictor** — ML model on historical trial success/failure

---

## Phase 3: Agent Orchestration (Weeks 4-8)

### Why Agents?

Once OpenCure has validated predictions and researcher partnerships, the bottleneck shifts from "making predictions" to "managing the operation": monitoring new literature, updating predictions, coordinating with researchers, tracking validation progress, writing reports. This is where agent orchestration becomes valuable — not before.

### Architecture: Paperclip vs OpenClaw vs Custom

| Framework | Strengths | Weaknesses | Verdict |
|-----------|-----------|------------|---------|
| **Paperclip** | Multi-agent orchestration, tool use, memory | Early stage, limited docs | Good for prototyping |
| **OpenClaw** | Agent-to-agent communication, workflows | Very new, small community | Promising but risky |
| **Custom (Claude API + cron)** | Full control, simple, reliable | More code to write | Best for production |
| **n8n** | Visual workflows, 400+ integrations | Not AI-native, more for automation | Good for email/notification flows |

**Recommendation: Hybrid approach**
- Use **Claude API + scheduled tasks** for core AI agents (reliable, full control)
- Use **n8n** for notification/email/monitoring workflows (proven, visual)
- Evaluate **Paperclip** for multi-agent research coordination once it matures

### Proposed Agent Roster

#### Agent 1: Literature Monitor
- **Schedule**: Daily
- **Task**: Search PubMed/bioRxiv for new papers mentioning OpenCure predictions
- **Action**: If a paper validates/invalidates a prediction → email alert + update database
- **Implementation**: Cron job + Claude API + PubMed E-utilities

#### Agent 2: Database Updater
- **Schedule**: Weekly
- **Task**: Check for new DRKG releases, new TxGNN models, updated Open Targets data
- **Action**: Re-run affected scoring pillars, update predictions
- **Implementation**: Python script + data version checking

#### Agent 3: Outreach Manager
- **Schedule**: Weekly
- **Task**: Find new researchers publishing on our diseases, generate personalized emails
- **Action**: Draft emails for review, track response rates
- **Implementation**: PubMed search + email drafting (human reviews before sending)

#### Agent 4: Validation Tracker
- **Schedule**: On-demand
- **Task**: Track which predictions are being experimentally tested
- **Action**: Maintain a public dashboard of validation status
- **Implementation**: Google Sheet / GitHub issues + status updates

#### Agent 5: Report Generator
- **Schedule**: Monthly
- **Task**: Regenerate all PDF reports with latest data and any new evidence
- **Action**: Push updated reports to GitHub, notify followers
- **Implementation**: Existing generate_reports.py + git automation

#### Agent 6: Grant Writer
- **Schedule**: On-demand
- **Task**: Draft grant applications for disease-specific funding
- **Action**: Generate grant drafts targeting NIH R21 (exploratory), Gates Foundation, Wellcome Trust
- **Implementation**: Claude API with grant templates + our prediction data

### Implementation Order
1. **Literature Monitor** (highest value, simplest) — 1 day to build
2. **Report Generator** automation — 0.5 day
3. **Outreach Manager** — 1 day
4. **Validation Tracker** — 1 day
5. **Database Updater** — 2 days
6. **Grant Writer** — 2 days

---

## Phase 4: Scale (Months 2-6)

### 4A. Expand Disease Coverage
- Screen ALL ~4,000 diseases in DRKG (not just 25)
- Prioritize by: disease burden (DALYs), treatment gap, MR data availability
- Automated pipeline: add disease → score → evidence → report → publish

### 4B. Build Public Dashboard
- Web app showing all predictions, sortable/filterable
- Real-time validation status for each prediction
- Researcher collaboration board (who's testing what)
- Public API for other tools to query predictions

### 4C. Establish Nonprofit Entity
- Register OpenCure as a 501(c)(3) or equivalent
- Apply for grants (NIH R21, Gates Grand Challenges, Wellcome Trust)
- Accept donations for wet-lab validation costs
- Partner with university labs for experimental validation

### 4D. First Experimental Validation
- Fund a cell-based assay for the top prediction ($5K-20K)
- If positive → animal model ($50K-200K)
- If positive → seek clinical trial partnership
- One validated prediction = proof of concept that changes everything

---

## Phase 5: Moonshot (Year 1+)

### 5A. Autonomous Drug Discovery Pipeline
Full agent-orchestrated pipeline:
```
New disease identified
    → Agent scores all 10K drugs (8 pillars)
    → Agent gathers evidence (6 databases)
    → Agent generates PDF report
    → Agent identifies top researchers
    → Agent drafts outreach emails (human approves)
    → Agent monitors for validation results
    → Agent updates public dashboard
    → Agent writes grant applications for validated hits
```

### 5B. Wet Lab Partnerships
- Partner with CROs (contract research organizations) for automated validation
- Use OpenCure predictions to prioritize which drugs to test
- Create a "prediction → validation → publication" pipeline

### 5C. Regulatory Pathway
- For validated repurposing candidates: work with FDA on 505(b)(2) pathway
- Repurposed drugs skip Phase I (safety already established)
- Phase II proof-of-concept trials are feasible for ~$1-5M

---

## Priority Matrix

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Submit bioRxiv preprint | HIGH | 10 min | DO NOW |
| Deploy web app | HIGH | 1 hour | DO NOW |
| Monitor researcher replies | CRITICAL | Ongoing | DO NOW |
| Contact DNDi/NORD/foundations | VERY HIGH | 2 hours | THIS WEEK |
| Deep-dive top 3 predictions | HIGH | 1 day | THIS WEEK |
| Literature Monitor agent | MEDIUM | 1 day | WEEK 2 |
| Add real docking pillar | MEDIUM | 3 days | WEEK 3 |
| Expand to 100+ diseases | MEDIUM | 2 days | WEEK 4 |
| Build public dashboard | MEDIUM | 3 days | MONTH 2 |
| Register nonprofit | HIGH | 1 week | MONTH 2 |
| Fund first cell assay | CRITICAL | $5-20K | MONTH 3 |

---

## The One Thing That Matters Most

Every item on this roadmap is secondary to ONE thing: **getting a single prediction experimentally validated.**

If Sirolimus shows activity against IPF cell models, or Everolimus affects glucocerebrosidase in Gaucher cell lines, that single result transforms OpenCure from "an AI tool that makes predictions" into "an AI tool that discovered a treatment." That's the difference between a GitHub repo and a breakthrough.

Everything else — agents, dashboards, more pillars, more diseases — is infrastructure. The breakthrough is in the wet lab.
