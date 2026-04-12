# Contributing to OpenCure

OpenCure is an open-source drug repurposing platform. We welcome contributions from researchers, developers, and anyone passionate about making medicine more accessible.

## Ways to Contribute

### Validate a Prediction
The highest-impact contribution is experimentally validating one of our predictions. Browse the [Explorer dashboard](https://simonbartosdev.github.io/opencure/) to find BREAKTHROUGH predictions in your area of expertise. If you validate or refute a prediction, please open an issue with your results — we will update the database and credit your work.

### Report Issues
- Incorrect drug classifications or novelty labels
- Bugs in the scoring pipeline
- Dashboard display issues
- Missing disease synonyms in PubMed queries

### Add Disease Synonyms
Our novelty scoring depends on comprehensive disease synonym coverage. To add synonyms, edit the `DISEASE_SYNONYMS` dictionary in `opencure/evidence/pubmed.py` and submit a pull request.

### Improve Scoring Pillars
Each pillar in `opencure/scoring/` is a self-contained module. Improvements to individual scoring methods (better embeddings, new data sources, improved normalization) are welcome.

### Add New Diseases
To add a new disease for screening:
1. Find its MESH or DOID identifier
2. Add an entry to `DISEASE_NAME_MAP` in `opencure/data/drkg.py`
3. Add synonyms to `DISEASE_SYNONYMS` in `opencure/evidence/pubmed.py`
4. Test with `python -m opencure.cli "Your Disease Name"`

## Development Setup

```bash
git clone https://github.com/SimonBartosDev/opencure.git
cd opencure
pip install -r requirements.txt
bash setup_data.sh
python -m opencure.cli "Alzheimer's disease"  # Verify it works
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Test your changes locally
3. Update documentation if needed
4. Submit a pull request with a clear description of what changed and why

## Code of Conduct

Be respectful, collaborative, and focused on the mission: getting treatments to patients who need them. This project exists to save lives, not to publish papers or build careers — though both are welcome side effects.

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
