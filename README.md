# OpenCure 🧬

**Open-source AI drug repurposing platform for neglected and rare diseases.**

OpenCure uses 8 independent AI scoring methods + 8 evidence sources to identify new therapeutic uses for existing FDA-approved drugs. Mission-driven: prioritize human need over profit.

## What Makes OpenCure Different

Most drug repurposing tools use 1-2 methods. OpenCure combines **8 scoring pillars** that each capture different biological signals:

| Pillar | Method | What It Measures |
|--------|--------|-----------------|
| TransE | Knowledge graph embeddings | Statistical patterns in 5.87M biomedical relationships |
| RotatE | Complex rotation embeddings | Symmetric/asymmetric relation patterns |
| TxGNN | Graph neural network (Harvard) | Zero-shot drug-disease predictions (49% better than baselines) |
| Fingerprints | Morgan/ECFP molecular similarity | Structural similarity to known treatments |
| ChemBERTa | Transformer molecular embeddings | Learned functional drug similarity |
| Gene Signatures | L1000CDS2 reversal | Drugs that reverse disease gene expression |
| Network Proximity | STRING PPI shortest paths | Drug target to disease gene distance |
| Mendelian Randomization | Open Targets genetic evidence | Causal genetic support for drug targets |

Plus **8 evidence sources**: PubMed, ClinicalTrials.gov, Semantic Scholar, FDA FAERS, Open Targets, L1000CDS2, PharmGKB, and LLM-powered explanations.

## Key Features

- **Novelty scoring**: Separates genuine NEW discoveries from rediscovering known treatments
- **Failed trial detection**: Penalizes drugs that already failed for a disease
- **LLM explainability**: Generates mechanistic hypotheses via knowledge graph path analysis
- **Drug combinations**: Predicts synergistic drug pairs via target complementarity
- **Pharmacogenomics**: Identifies which patient populations benefit most

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (DRKG, STRING, embeddings)
bash scripts/setup_data.sh

# Run the web app
python -m opencure.web.run

# Search from command line
python -m opencure.cli "Alzheimer's disease"

# Run full 25-disease screening
python experiments/systematic_screening.py
```

## Results

Screening across 25 neglected/rare diseases identified:
- **39 BREAKTHROUGH candidates** (zero published literature)
- **179 total novel candidates**
- **217 total candidates** with confidence assessments

## Architecture

```
Disease name → 8 AI pillars (parallel scoring) → Per-drug dynamic weighting
  → Convergence bonus → MR causal bonus → Ranked candidates
    → On-demand evidence reports (8 APIs) → Confidence + Novelty scoring
```

## Documentation

- `OPENCURE_STUDY_GUIDE.md` — Technical architecture (formulas, parameters, algorithms)
- `OPENCURE_FOUNDATIONS.md` — Everything explained from scratch (biology, ML, chemistry)

## License

MIT License. Open source, nonprofit mission.

## Citation

If you use OpenCure in your research, please cite:
```
OpenCure: Multi-pillar AI platform for drug repurposing in neglected and rare diseases.
https://github.com/opencure-ai/opencure
```
