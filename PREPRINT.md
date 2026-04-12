# OpenCure: An Open-Source Multi-Pillar AI Platform for Systematic Drug Repurposing in Neglected and Rare Diseases

## Authors

Simon Bartos^1*

^1 Independent Researcher

*Corresponding author: opencure.research@gmail.com

## Abstract

Drug repurposing -- identifying new therapeutic uses for existing approved drugs -- offers a faster, cheaper path to treatment than de novo drug development. Yet most computational repurposing tools rely on one or two methods and lack systematic validation. Here we present OpenCure, an open-source platform that integrates eight independent AI scoring pillars spanning knowledge graph embeddings, graph neural networks, molecular similarity, gene expression reversal, protein network proximity, and causal genetic evidence. Each pillar captures a distinct biological signal; candidates are ranked by per-drug dynamic weighting with convergence bonuses for multi-pillar agreement. We screened 10,551 FDA-approved drugs across 25 neglected and rare diseases, generating 245 drug-disease predictions with comprehensive evidence reports. A GradientBoosting ensemble trained on 54,775 known drug-disease pairs achieved an AUC-ROC of 0.998 on held-out test data. Among the 245 candidates, 78 are classified as BREAKTHROUGH (no existing published literature) and 73 as NOVEL (minimal literature), representing potential new therapeutic hypotheses. Post-scoring evidence gathering from PubMed, ClinicalTrials.gov, Semantic Scholar, and FDA FAERS provides confidence assessments and novelty scoring to distinguish genuine discoveries from rediscoveries of known treatments. OpenCure is freely available under Apache 2.0 at https://github.com/SimonBartosDev/opencure, with per-disease PDF reports designed for direct researcher use.

**Keywords:** drug repurposing, drug repositioning, knowledge graph, graph neural network, neglected diseases, rare diseases, computational pharmacology, open-source

## 1. Introduction

The average cost of developing a new drug exceeds $2.6 billion and takes 10-15 years from discovery to market approval (DiMasi et al., 2016). For neglected tropical diseases affecting over 1 billion people -- and rare diseases with patient populations too small to attract commercial investment -- this development pipeline is effectively inaccessible. Drug repurposing, the strategy of finding new therapeutic applications for existing approved drugs, circumvents much of this cost and timeline because safety and pharmacokinetic profiles are already established (Pushpakom et al., 2019).

Computational approaches to drug repurposing have accelerated in recent years, driven by large biomedical knowledge graphs (Himmelstein et al., 2017; Drkg, 2020), molecular representation learning (Chithrananda et al., 2020), and graph neural networks (Huang et al., 2024). However, existing tools typically employ a single method -- knowledge graph embeddings alone (Mohamed et al., 2021), molecular similarity alone (Luo et al., 2021), or network proximity alone (Gysi et al., 2021) -- and thus capture only one facet of the complex biology underlying drug-disease relationships.

We hypothesized that integrating multiple orthogonal computational methods, each measuring a different biological signal, would produce more robust predictions than any single method. Here we present OpenCure, an open-source platform that combines eight AI scoring pillars with six evidence databases to systematically identify and validate drug repurposing candidates.

OpenCure differs from existing tools in three key ways: (1) it integrates eight independent scoring methods spanning four distinct computational paradigms (graph-based, structure-based, expression-based, and genetics-based); (2) it includes a novelty scoring system that separates genuine new discoveries from rediscoveries of known treatments; and (3) it is designed as a nonprofit, open-source tool specifically targeting diseases neglected by the pharmaceutical industry.

## 2. Methods

### 2.1 Data Sources

**Drug Repurposing Knowledge Graph (DRKG).** We use DRKG (Ioannidis et al., 2020), which integrates six databases (DrugBank, Hetionet, GNBR, STRING, IntAct, and DGIdb) into a unified knowledge graph of 97,238 entities, 5,874,261 triplets, and 107 relation types. Entity types include compounds (10,551 DrugBank drugs), diseases, genes, anatomical structures, and biological processes.

**Molecular structures.** SMILES representations were obtained from PubChem for 9,425 of 10,551 DrugBank compounds (89.3% coverage). Structures were processed using RDKit for fingerprint generation and conformer computation.

**Protein-protein interactions.** The STRING database (v12, Szklarczyk et al., 2023) provides 236,930 high-confidence (combined score > 700) interactions among 16,201 human proteins.

**TxGNN predictions.** Pre-computed drug-disease predictions from TxGNN (Huang et al., 2024) for 60 diseases were obtained from the published model.

**Genetic associations.** Open Targets Platform (Ochoa et al., 2023) genetic_association scores from genome-wide association studies (GWAS) were queried via GraphQL API for Mendelian randomization analysis.

**Gene expression signatures.** Disease gene expression signatures were queried from L1000CDS2 (Duan et al., 2016) for drug-induced gene expression reversal analysis.

### 2.2 Multi-Pillar Scoring Architecture

Each drug-disease pair is evaluated by up to eight independent scoring pillars:

**Pillar 1: TransE knowledge graph embeddings.** TransE (Bordes et al., 2013) learns 400-dimensional vector representations of all DRKG entities such that for a triplet (head, relation, tail), the embedding satisfies h + r ~ t. For a query disease d and the treatment relation r, we compute the target vector t = d - r and rank all compounds by cosine distance to t. Base weight: 0.10.

**Pillar 2: RotatE complex rotation embeddings.** RotatE (Sun et al., 2019) represents relations as rotations in complex vector space (200 dimensions), capturing asymmetric relation patterns that TransE cannot model. Trained via PyKEEN (Ali et al., 2021) on DRKG. Base weight: 0.10.

**Pillar 3: TxGNN graph neural network.** TxGNN (Huang et al., 2024) uses heterogeneous message-passing on a biomedical graph to predict drug-disease associations, achieving 49% improvement over prior baselines. We use pre-computed dot-product predictions from the published drug and disease embeddings for 60 diseases. Base weight: 0.20.

**Pillar 4: Molecular fingerprint similarity.** Morgan/ECFP circular fingerprints (radius=2, 2048 bits) are computed via RDKit for each candidate drug and compared to fingerprints of drugs known to treat the query disease using Tanimoto similarity. Base weight: 0.05.

**Pillar 5: ChemBERTa transformer embeddings.** ChemBERTa (Chithrananda et al., 2020; seyonec/ChemBERTa-zinc-base-v1) encodes SMILES strings into 768-dimensional contextual embeddings. Cosine similarity between candidate embeddings and known treatment embeddings captures functional molecular similarity beyond 2D structural fingerprints. Base weight: 0.08.

**Pillar 6: Gene signature reversal.** The L1000CDS2 API (Duan et al., 2016) is queried for each disease to identify drugs whose gene expression signature reverses the disease transcriptome. Drug-disease matching uses exact name matching with salt form stripping and fuzzy string matching (rapidfuzz, threshold 85%). Base weight: 0.12.

**Pillar 7: Network proximity.** Following the approach validated by Gysi et al. (2021), we measure the shortest-path distance between drug target proteins and disease gene proteins in the STRING PPI network. Distance matrix computation uses single-source BFS from each disease gene node via scipy sparse graph routines (cutoff=4 hops). Proximity score = max(0, 1 - avg_distance / 4). Base weight: 0.15.

**Pillar 8: Mendelian randomization (causal genetics).** For each drug target gene, we query Open Targets for genetic_association scores from GWAS studies. A positive MR score indicates that genome-wide genetic evidence supports a causal link between the drug's molecular target and the disease -- evidence stronger than observational correlation. MR is applied as an additive bonus: +0.15 x mr_score, not as a weighted pillar, to avoid penalizing drugs that lack genetic data.

### 2.3 Score Combination

Scores are combined using per-drug dynamic weighting. For each drug, only pillars that produced a non-zero score contribute; their weights are renormalized to sum to 1.0. A convergence bonus of +0.05 is added for each additional pillar beyond 1 that scores the drug. The MR bonus is added separately as an additive term. This approach avoids penalizing drugs that lack data for certain pillars (e.g., drugs without SMILES cannot be scored by fingerprint or ChemBERTa pillars).

### 2.4 Ensemble Validation

A GradientBoosting classifier (scikit-learn; max_depth=6, n_estimators=200, learning_rate=0.1) was trained on 54,775 positive drug-disease pairs (DRKG treatment-type relations) and 54,775 randomly sampled negative pairs. Features include percentile ranks from each scoring pillar plus the number of pillars hit. Evaluation on a held-out 20% test set yielded AUC-ROC = 0.998.

### 2.5 Evidence Gathering and Confidence Assessment

The top 10 candidates per disease undergo evidence gathering from six external databases:

1. **PubMed**: article counts for drug+disease co-occurrence, treatment context, and repurposing mentions
2. **ClinicalTrials.gov**: active and completed trials with phase distribution
3. **Semantic Scholar**: citation-weighted paper search (200M+ corpus)
4. **FDA FAERS**: adverse event report co-occurrence between drug and disease
5. **Open Targets**: genetic association data for MR scoring
6. **L1000CDS2**: gene expression reversal evidence

Confidence is assessed on a point system: +3 for clinical trials, +2 for extensive literature (>50 PubMed articles), +1 for repurposing mentions, +1 for direct knowledge graph relations, +1 for multi-pillar support, +2 for strong FAERS signal, +2 for MR > 0.7, -3 for failed/terminated trials. Classifications: HIGH (>=5 points), MEDIUM (>=3), LOW (>=1), NOVEL (strong computation but no published evidence).

### 2.6 Novelty Scoring

To distinguish new discoveries from rediscoveries, we compute: novelty = computational_score x (1 - knowledge_score), where knowledge_score aggregates PubMed article counts, clinical trial counts, citation metrics, and FAERS co-occurrences. Classifications: BREAKTHROUGH (knowledge_score = 0), NOVEL (<0.1), EMERGING (0.1-0.3), KNOWN (0.3-0.6), ESTABLISHED (>0.6). Candidates with MR > 0.5 receive a 20% novelty boost reflecting the added credibility of causal genetic support.

### 2.7 Disease Selection

We selected 25 diseases across four categories representing conditions underserved by pharmaceutical development:

- **Neglected tropical diseases** (8): Malaria, Tuberculosis, Dengue, Chagas disease, Leishmaniasis, Schistosomiasis, HIV, Hepatitis C
- **Rare genetic diseases** (8): Sickle cell disease, Fragile X syndrome, Duchenne muscular dystrophy, Neurofibromatosis, Marfan syndrome, Ehlers-Danlos syndrome, Gaucher disease, Fabry disease
- **Neurodegenerative diseases** (5): Alzheimer's disease, Parkinson's disease, Huntington's disease, Amyotrophic lateral sclerosis, Multiple sclerosis
- **Other underserved conditions** (4): Idiopathic pulmonary fibrosis, Cystic fibrosis, Pulmonary hypertension, Sepsis

## 3. Results

### 3.1 Screening Overview

We screened all 10,551 DrugBank compounds against 25 diseases using all eight scoring pillars. The top 10 candidates per disease (250 total) underwent comprehensive evidence gathering. After excluding 5 candidates with unresolved drug names, 245 candidates were retained for analysis.

### 3.2 Confidence Distribution

Of 245 candidates: 84 (34.3%) were classified HIGH confidence, 40 (16.3%) MEDIUM, 108 (44.1%) LOW, and 13 (5.3%) NOVEL (strong computational prediction with no published evidence). Diseases with the most HIGH confidence candidates were Hepatitis C (9), Alzheimer's disease (8), HIV (8), Parkinson's disease (8), and Tuberculosis (8).

### 3.3 Novelty Distribution

78 candidates (31.8%) were classified BREAKTHROUGH (zero published literature linking drug to disease), 73 (29.8%) NOVEL (minimal literature), 12 (4.9%) EMERGING, 55 (22.4%) KNOWN, and 27 (11.0%) ESTABLISHED. The 27 ESTABLISHED candidates represent known treatments that serve as positive controls, validating the scoring pipeline. After filtering known treatments, 218 candidates remain as actual repurposing hypotheses.

### 3.4 Pillar Coverage

Of the 245 candidates, the median number of active pillars was 3 (range: 1-6). 110 candidates (44.9%) had positive Mendelian randomization scores, providing causal genetic support for the drug-target-disease link. Molecular similarity data was available for 186 candidates (75.9%), and gene signature reversal matches were found for 38 candidates (15.5%).

### 3.5 Ensemble Validation

The GradientBoosting ensemble achieved AUC-ROC = 0.998 on the held-out test set (10,955 positive and 10,955 negative pairs), indicating that the multi-pillar feature representation strongly discriminates known treatments from random drug-disease pairs.

### 3.6 Notable Predictions

Several BREAKTHROUGH candidates with strong multi-pillar support and causal genetic evidence warrant further investigation:

**Sirolimus for Idiopathic Pulmonary Fibrosis** (AI score: 1.157, MR: 0.64, 5 pillars). Sirolimus (rapamycin) is an mTOR inhibitor currently used as an immunosuppressant. The high MR score indicates that GWAS evidence supports a causal role for mTOR pathway targets in IPF pathogenesis. mTOR signaling has been implicated in fibroblast proliferation and extracellular matrix deposition, the hallmarks of pulmonary fibrosis.

**S-Adenosyl-L-Homocysteine for Parkinson's Disease** (AI score: 1.147, MR: 0.76, 2 pillars). The high MR score (0.76) represents one of the strongest causal genetic signals in our dataset, suggesting that methylation pathway dysregulation has GWAS-supported causal involvement in Parkinson's disease.

**Everolimus for Gaucher Disease** (AI score: 1.120, BREAKTHROUGH). Everolimus is an mTOR inhibitor currently used in oncology. No published literature links it to Gaucher disease. The prediction is driven by knowledge graph embedding similarity to known Gaucher treatments and shared molecular pathway signatures.

**Histamine Dihydrochloride for HIV** (AI score: 1.119, MR: 0.46, BREAKTHROUGH). Histamine dihydrochloride is approved for acute myeloid leukemia maintenance. Its potential anti-HIV mechanism may involve immune modulation through histamine H2 receptor-mediated enhancement of NK cell and T cell function.

## 4. Discussion

### 4.1 Multi-Pillar Advantage

The integration of eight scoring pillars provides several advantages over single-method approaches. First, different pillars capture complementary biological signals: knowledge graph embeddings encode statistical patterns from published relationships, molecular similarity identifies structurally related compounds, network proximity measures functional pathway overlap, and Mendelian randomization provides causal genetic evidence. Second, per-drug dynamic weighting ensures that missing data for one pillar does not penalize a candidate -- only pillars with available data contribute to the final score. Third, the convergence bonus rewards multi-source agreement, increasing confidence in predictions supported by multiple independent lines of evidence.

### 4.2 Novelty Scoring as a Discovery Filter

A key challenge in drug repurposing is distinguishing genuinely novel predictions from rediscoveries of known treatments. Our novelty scoring system addresses this by quantifying the ratio of computational support to existing literature. Candidates with high computational scores but zero or minimal published evidence (BREAKTHROUGH and NOVEL) represent the most scientifically interesting predictions. The 78 BREAKTHROUGH candidates identified in this study have no published PubMed articles linking the drug to the disease, suggesting they represent entirely unexplored therapeutic hypotheses.

### 4.3 Mendelian Randomization as Causal Evidence

The inclusion of Mendelian randomization distinguishes OpenCure from purely correlation-based approaches. MR uses genetic variants as instrumental variables to assess whether modulating a drug's target gene causally affects disease risk, independent of confounding factors (Davey Smith & Hemani, 2014). Among our candidates, 110 (44.9%) had positive MR scores, providing causal genetic support that strengthens the biological plausibility of the computational predictions.

### 4.4 Limitations

Several limitations should be noted. First, all predictions are computational hypotheses that require experimental validation. The AUC-ROC of 0.998 reflects the ensemble's ability to discriminate known treatments from random pairs, not its ability to predict clinically successful repurposing. Second, the DRKG knowledge graph reflects published biomedical knowledge and may encode existing biases. Third, gene signature reversal data from L1000CDS2 has limited coverage for many drugs and diseases. Fourth, the network proximity analysis is limited by the completeness of the STRING PPI network. Fifth, the Mendelian randomization analysis uses genetic_association scores from Open Targets as a proxy, rather than performing formal two-sample MR with individual-level GWAS summary statistics.

### 4.5 Open-Source and Nonprofit Mission

OpenCure is released under the Apache 2.0 license with the explicit goal of accelerating drug repurposing for diseases neglected by the pharmaceutical industry. All code, data processing pipelines, trained models, screening results, and per-disease PDF reports are freely available. We encourage researchers with wet-lab capabilities to validate the BREAKTHROUGH predictions identified in this study.

## 5. Data and Code Availability

OpenCure is available at https://github.com/SimonBartosDev/opencure under Apache 2.0 license. The repository includes:
- Complete source code for all eight scoring pillars
- Screening results for 25 diseases (JSON and CSV)
- Per-disease PDF reports with full evidence breakdowns
- Trained ensemble model
- Data download and setup scripts

Contact: opencure.research@gmail.com

## References

Ali, M., Berrendorf, M., Hoyt, C. T., et al. (2021). PyKEEN 1.0: A Python library for training and evaluating knowledge graph embeddings. *Journal of Machine Learning Research*, 22(82), 1-6.

Bordes, A., Usunier, N., Garcia-Duran, A., et al. (2013). Translating embeddings for modeling multi-relational data. *Advances in Neural Information Processing Systems*, 26.

Chithrananda, S., Grand, G., & Ramsundar, B. (2020). ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. *arXiv preprint arXiv:2010.09885*.

Davey Smith, G., & Hemani, G. (2014). Mendelian randomization: genetic anchors for causal inference in epidemiological studies. *Human Molecular Genetics*, 23(R1), R89-R98.

DiMasi, J. A., Grabowski, H. G., & Hansen, R. W. (2016). Innovation in the pharmaceutical industry: new estimates of R&D costs. *Journal of Health Economics*, 47, 20-33.

Duan, Q., Reid, S. P., Clark, N. R., et al. (2016). L1000CDS2: LINCS L1000 characteristic direction signatures search engine. *NPJ Systems Biology and Applications*, 2, 16015.

Gysi, D. M., do Valle, I., Zitnik, M., et al. (2021). Network medicine framework for identifying drug-repurposing opportunities for COVID-19. *Proceedings of the National Academy of Sciences*, 118(19), e2025581118.

Himmelstein, D. S., Lizee, A., Hessler, C., et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726.

Huang, K., Chandak, P., Wang, Q., et al. (2024). A foundation model for clinician-centered drug repurposing. *Nature Medicine*, 30, 3601-3613.

Ioannidis, V. N., Song, X., Manchanda, S., et al. (2020). DRKG - Drug Repurposing Knowledge Graph. *GitHub repository*.

Luo, Y., Zhao, X., Zhou, J., et al. (2021). A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. *Nature Communications*, 8(1), 573.

Mohamed, S. K., Nounu, A., & Jarada, T. N. (2021). Drug target discovery using knowledge graph embeddings. *SAC '21: Proceedings of the 36th Annual ACM Symposium on Applied Computing*.

Ochoa, D., Hercules, A., Brber, B. M., et al. (2023). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. *Nucleic Acids Research*, 51(D1), D1353-D1359.

Pushpakom, S., Iorio, F., Eyers, P. A., et al. (2019). Drug repurposing: progress, challenges and recommendations. *Nature Reviews Drug Discovery*, 18(1), 41-58.

Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR 2019*.

Szklarczyk, D., Kirsch, R., Koutrouli, M., et al. (2023). The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. *Nucleic Acids Research*, 51(D1), D D483-D488.

---

## Supplementary Information

### Table S1: Disease Categories and Candidate Counts

| Category | Disease | HIGH | MEDIUM | LOW | NOVEL | Total |
|----------|---------|------|--------|-----|-------|-------|
| Neglected Tropical | Malaria | 5 | 4 | 1 | 0 | 10 |
| Neglected Tropical | Tuberculosis | 8 | 1 | 1 | 0 | 10 |
| Neglected Tropical | Dengue | 3 | 5 | 2 | 0 | 10 |
| Neglected Tropical | Chagas disease | 6 | 1 | 3 | 0 | 10 |
| Neglected Tropical | Leishmaniasis | 5 | 4 | 1 | 0 | 10 |
| Neglected Tropical | Schistosomiasis | 1 | 2 | 7 | 0 | 10 |
| Neglected Tropical | HIV | 8 | 1 | 1 | 0 | 10 |
| Neglected Tropical | Hepatitis C | 9 | 0 | 1 | 0 | 10 |
| Rare | Sickle cell disease | 7 | 3 | 0 | 0 | 10 |
| Rare | Fragile X syndrome | 0 | 3 | 7 | 0 | 10 |
| Rare | Duchenne muscular dystrophy | 4 | 1 | 5 | 0 | 10 |
| Rare | Neurofibromatosis | 0 | 0 | 6 | 3 | 9 |
| Rare | Marfan syndrome | 1 | 3 | 6 | 0 | 10 |
| Rare | Ehlers-Danlos syndrome | 0 | 1 | 3 | 3 | 7 |
| Rare | Gaucher disease | 1 | 0 | 8 | 0 | 9 |
| Rare | Fabry disease | 2 | 0 | 5 | 3 | 10 |
| Neurodegenerative | Alzheimer's disease | 8 | 2 | 0 | 0 | 10 |
| Neurodegenerative | Parkinson's disease | 8 | 2 | 0 | 0 | 10 |
| Neurodegenerative | Huntington's disease | 2 | 1 | 4 | 3 | 10 |
| Neurodegenerative | ALS | 0 | 0 | 10 | 0 | 10 |
| Neurodegenerative | Multiple sclerosis | 6 | 4 | 0 | 0 | 10 |
| Other | IPF | 0 | 2 | 8 | 0 | 10 |
| Other | Cystic fibrosis | 0 | 0 | 10 | 0 | 10 |
| Other | Pulmonary hypertension | 0 | 0 | 10 | 0 | 10 |
| Other | Sepsis | 0 | 0 | 9 | 1 | 10 |

### Figure S1: Multi-Pillar Architecture

```
                         Disease Query
                              |
              +---------------+---------------+
              |               |               |
        Knowledge Graph   Molecular      Biological
              |            Methods         Methods
         +---------+     +--------+     +----------+
         |  TransE |     |  FP    |     | Gene Sig |
         | RotatE  |     |ChemBERTa|    | Net Prox |
         | TxGNN   |     +--------+     | MR       |
         +---------+          |         +----------+
              |               |              |
              +-------+-------+------+-------+
                      |              |
                Per-Drug Dynamic Weighting
                      |
                Convergence Bonus (+0.05/pillar)
                      |
                MR Additive Bonus (+0.15 x score)
                      |
                Ranked Candidates
                      |
              Evidence Gathering
           (PubMed, Trials, FAERS, S2)
                      |
              Confidence + Novelty
                      |
               Final Report (PDF)
```
