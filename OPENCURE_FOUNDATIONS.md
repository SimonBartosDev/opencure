# OpenCure: Foundations — Every Concept Explained From Scratch
## If You Understand This Document, You Understand Everything

---

# CHAPTER 1: MATHEMATICS FOUNDATIONS

## 1.1 What Is a Number?

A **scalar** is a single number: `5`, `3.14`, `-0.7`. It has magnitude but no direction.

A **vector** is a list of numbers: `[0.3, 0.7, 0.1]`. It represents a point in space OR a direction.

- A 2D vector `[3, 4]` is a point on a flat plane (like x,y coordinates on a map)
- A 3D vector `[1, 2, 3]` is a point in 3D space (like x,y,z coordinates in a room)
- A 400D vector `[0.3, 0.7, ..., 0.1]` (400 numbers) is a point in 400-dimensional space

**Why 400 dimensions?** We can't visualize it, but the math works identically to 2D/3D. Each dimension represents some abstract "feature" that the AI learned. Dimension 1 might (loosely) encode "how anti-inflammatory is this drug", dimension 2 might encode "how related to cancer", etc. The AI decides what each dimension means during training.

A **matrix** is a grid of numbers (a table). An embedding matrix of shape (97238, 400) means 97,238 rows (one per entity) and 400 columns (one per dimension). Each row IS the vector for that entity.

## 1.2 Distance Between Points (Norms)

**Euclidean distance** (L2 norm): How far apart are two points?

In 2D: distance between `[1, 2]` and `[4, 6]`:
```
d = sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5
```

In general (any number of dimensions):
```
d = sqrt(sum of (a_i - b_i)^2 for all dimensions i)
```

In code: `np.linalg.norm(a - b)` — numpy computes this in one call.

**Why we use distance**: If two drug vectors are CLOSE in 400D space, the AI thinks they're similar. If a drug vector + "treats" vector lands CLOSE to a disease vector, the AI predicts that drug treats that disease.

## 1.3 Dot Product and Cosine Similarity

**Dot product**: Multiply corresponding elements and sum:
```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Cosine similarity**: Dot product of UNIT vectors (vectors scaled to length 1):
```
cos(A, B) = (A · B) / (|A| × |B|)
```

- cos = 1.0: Vectors point in exactly the same direction (identical)
- cos = 0.0: Vectors are perpendicular (unrelated)
- cos = -1.0: Vectors point in opposite directions (opposites)

**Unit vector** (normalization): Divide each element by the vector's length:
```
v = [3, 4]
length = sqrt(3^2 + 4^2) = 5
unit_v = [3/5, 4/5] = [0.6, 0.8]
```

In code:
```python
normalized = vector / np.linalg.norm(vector)
cosine_sim = np.dot(normalized_a, normalized_b)
```

**Why we use cosine similarity**: For comparing molecular embeddings (ChemBERTa). Two drugs with similar biological behavior will have similar vector DIRECTIONS, regardless of vector magnitude.

## 1.4 Complex Numbers

A **complex number** has two parts: `a + bi`, where:
- `a` = real part
- `b` = imaginary part
- `i` = the imaginary unit (i^2 = -1)

**Multiplying complex numbers**:
```
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```

**Euler's formula**: `e^(iθ) = cos(θ) + i×sin(θ)`

This means multiplying by `e^(iθ)` ROTATES a complex number by angle θ.

**Why RotatE uses complex numbers**: By modeling each relation as a rotation angle per dimension, RotatE can capture patterns that TransE cannot (like symmetry: rotating by 180° maps A to B AND B back to A).

## 1.5 What Is a Hash Function?

A **hash function** takes any input and produces a fixed-size output:
```
hash("hello") → 5d41402abc4b2a76b9719d911017c592
hash("hello!") → completely different output
```

**Molecular fingerprints use hashing**: Take a molecular substructure (like "benzene ring attached to hydroxyl group") and hash it to a bit position (0-2047 in a 2048-bit vector). Set that bit to 1. Repeat for all substructures. The final 2048-bit vector IS the fingerprint.

## 1.6 Percentile Rank

If you score 85 out of 100 on a test and 70% of students scored below you, your **percentile rank** is 0.70 (or 70th percentile).

Formula for ranking N items by score:
```
percentile(item) = 1.0 - (rank_position / N)
```
- Best item: position 0 → percentile = 1.0
- Worst item: position N-1 → percentile ≈ 0.0

**Why we use percentiles**: Different AI pillars produce scores on wildly different scales (TransE: -20 to 0, fingerprints: 0 to 1). Converting to percentiles puts everything on the same 0-1 scale so we can fairly combine them.

---

# CHAPTER 2: MACHINE LEARNING FOUNDATIONS

## 2.1 What Is a Neural Network?

A neural network is a function that takes numbers in and produces numbers out, with adjustable parameters (**weights**) that are tuned during training.

**A single neuron**:
```
output = activation(weight1 × input1 + weight2 × input2 + ... + bias)
```

- **Inputs**: Numbers (e.g., features of a drug molecule)
- **Weights**: Learnable parameters (initially random, adjusted during training)
- **Bias**: A constant offset (also learnable)
- **Activation function**: Adds non-linearity (e.g., ReLU: `max(0, x)`)

**A layer**: Many neurons operating in parallel, each with their own weights:
```
Layer with 100 neurons: takes N inputs → produces 100 outputs
```

**A deep network**: Multiple layers stacked:
```
Input (400 dims) → Layer 1 (200 neurons) → Layer 2 (100 neurons) → Output (1 number)
```

Each layer transforms the data into a more useful representation. Early layers detect simple patterns; deeper layers detect complex combinations.

## 2.2 What Is Training?

Training = adjusting all the weights so the network produces correct outputs.

**Step 1: Forward pass** — Run an input through the network, get a prediction
**Step 2: Loss computation** — Compare prediction to the correct answer using a **loss function**:
```
loss = (prediction - correct_answer)^2    ← squared error
```
**Step 3: Backward pass (backpropagation)** — Compute the **gradient** (which direction each weight should change to reduce the loss)
**Step 4: Update weights** — Move each weight a small step in the gradient direction:
```
new_weight = old_weight - learning_rate × gradient
```

**Learning rate**: How big each step is (e.g., 0.001). Too big = overshoots, too small = takes forever.

**Epoch**: One complete pass through all training data. OpenCure's RotatE trains for 100 epochs.

**Batch size**: How many examples to process before updating weights. RotatE uses batch size 8192 (process 8192 triplets, compute average gradient, update weights).

## 2.3 What Is an Embedding?

An **embedding** is a learned vector representation of a discrete entity.

**Problem**: How do you feed "Aspirin" to a neural network? It's a word, not a number.

**Solution**: Assign every entity a unique vector. Initially random, but during training, similar entities end up with similar vectors because the training process pushes them together.

```
Before training:
  Aspirin   = [0.72, -0.31, 0.44, ...]   ← random
  Ibuprofen = [-0.15, 0.88, 0.02, ...]   ← random

After training:
  Aspirin   = [0.45, 0.68, 0.21, ...]   ← similar to Ibuprofen!
  Ibuprofen = [0.43, 0.71, 0.19, ...]   ← because both are NSAIDs
```

The embedding matrix IS the learned knowledge. It encodes relationships between all entities implicitly in the geometry of the vector space.

## 2.4 What Is a Transformer?

A transformer is a neural network architecture that processes sequences using **self-attention**.

### Self-Attention Explained

Given a sequence like a SMILES string: `C C ( = O ) O c 1 c c c c c 1`

Each token needs to "understand" its context. Self-attention lets every token look at every other token and decide what's relevant:

1. Each token is projected into three vectors: **Query** (Q), **Key** (K), **Value** (V)
2. Attention score = dot product of one token's Query with all other tokens' Keys
3. These scores are normalized (softmax) to sum to 1
4. Each token's output = weighted sum of all Values, weighted by attention scores

```
Token "O" asks: "Who is relevant to me?"
Attention scores: C=0.1, C=0.1, (=0.05, ==0.3, O=0.15, )=0.05, O=0.1, c1=0.05, ...
                              ↑ The "=" token is highly relevant (double-bonded to O)
```

### Multi-Head Attention
Instead of one set of Q/K/V, use multiple "heads" (e.g., 12). Each head learns to attend to different types of relationships (one head learns bond patterns, another learns ring structures, etc.).

### Why Transformers Work for Molecules
A SMILES string is a sequence of chemical tokens. The transformer learns which atoms interact with which, even across long distances in the string (e.g., two ends of a ring system).

**ChemBERTa** is a BERT-style transformer:
- Pre-trained on 77 million SMILES strings from ZINC database
- 12 attention heads, 6 layers, 768-dimensional embeddings
- The **[CLS] token** (a special token prepended to every input) aggregates information from the entire molecule
- This [CLS] embedding IS the molecule's learned representation

## 2.5 What Is a Graph Neural Network (GNN)?

A GNN operates on graph-structured data (nodes connected by edges) instead of sequences.

### Message Passing

In each layer, every node updates its embedding by aggregating messages from its neighbors:

```
Round 1: Drug node collects info from its target gene nodes
Round 2: Drug node's embedding now contains info about its targets
         Gene nodes collect info from disease nodes they're connected to
Round 3: Drug node (via updated gene nodes) now knows about diseases
         connected to its targets — even though there's no direct drug→disease edge!
```

**Formula**:
```
new_embedding(node) = UPDATE(
    old_embedding(node),
    AGGREGATE(embedding(neighbor) for each neighbor of node)
)
```

Where AGGREGATE could be: sum, mean, attention-weighted sum
And UPDATE could be: a neural network layer

### TxGNN Specifics
- **Input graph**: PrimeKG (7,957 drugs + 17,080 diseases + genes/proteins/pathways)
- **Architecture**: Multi-layer GNN with 100-dimensional hidden representations
- **Prototypes**: 3 learnable "disease archetype" vectors for zero-shot prediction
- **Output**: For any drug-disease pair, compute dot product of their embeddings → higher = more likely to treat

### How GNNs Differ from TransE
- **TransE**: Each entity gets ONE fixed embedding regardless of context
- **GNN**: Embeddings are COMPUTED dynamically through message passing. A drug's embedding changes depending on what disease you're predicting for (via attention mechanisms)

## 2.6 What Is a Loss Function?

A loss function measures how wrong the model's predictions are. Training minimizes this function.

**For knowledge graph embeddings (TransE/RotatE)**:

Margin-based ranking loss:
```
L = max(0, score(negative_triplet) - score(positive_triplet) + margin)
```

Where:
- Positive triplet: a real fact from DRKG (e.g., Aspirin→treats→Pain)
- Negative triplet: a corrupted fact (e.g., Aspirin→treats→Happiness) created by replacing the tail with a random entity
- Margin (γ): how much better the positive should score (typically 1.0-10.0)

The model learns to score real facts HIGH and fake facts LOW.

**Negative sampling**: For each real triplet, generate 32 fake triplets by randomly replacing the head or tail. This teaches the model what ISN'T true.

## 2.7 What Is an Optimizer?

An optimizer is the algorithm that updates weights during training.

**Adam** (used by OpenCure): Combines momentum (smoothing out noisy gradients) with adaptive learning rates (different step sizes for different weights):

```
For each weight:
  m = 0.9 × old_momentum + 0.1 × gradient           ← smoothed gradient
  v = 0.999 × old_variance + 0.001 × gradient^2       ← gradient magnitude tracker
  weight -= learning_rate × m / (sqrt(v) + 1e-8)      ← adaptive step
```

Parameters in OpenCure: learning_rate = 0.001

## 2.8 What Is Overfitting?

When a model memorizes the training data instead of learning general patterns.

**Example**: If a model learns "Aspirin→treats→Pain" by memorizing this specific fact, it won't generalize to predict "Aspirin→treats→Inflammation". But if it learns that "Aspirin is near other anti-inflammatory drugs in embedding space", it generalizes.

**How OpenCure prevents overfitting**:
- Large dataset (5.87M triplets)
- Multiple independent pillars (if one overfits, others correct it)
- Percentile normalization (prevents any single pillar from dominating)
- External validation (12 known repurposing successes benchmark)

---

# CHAPTER 3: BIOLOGY AND CHEMISTRY FOUNDATIONS

## 3.1 What Is a Protein?

A **protein** is a molecular machine made of amino acids (20 types) chained together. The chain folds into a 3D shape that determines its function.

- **Enzymes**: Proteins that speed up chemical reactions (e.g., COX-2 produces prostaglandins)
- **Receptors**: Proteins on cell surfaces that detect signals (e.g., dopamine receptor D2)
- **Transporters**: Proteins that move molecules across cell membranes
- **Ion channels**: Proteins that let charged atoms (ions) flow through membranes
- **Structural proteins**: Build cell scaffolding (e.g., collagen)

**Drug targets**: Most drugs work by binding to a specific protein and changing its behavior:
- **Inhibitor**: Blocks the protein's function (e.g., Aspirin inhibits COX-2)
- **Agonist**: Activates the protein (e.g., morphine activates opioid receptors)
- **Antagonist**: Blocks the protein without activating it (e.g., beta-blockers block adrenaline receptors)

## 3.2 What Is a Gene?

A **gene** is a section of DNA that encodes instructions for making a specific protein.

```
DNA → (transcription) → mRNA → (translation) → Protein
Gene BRCA1 → mRNA copy → BRCA1 protein (DNA repair enzyme)
```

**Gene expression**: How much protein a gene produces. Can be measured as mRNA levels.

- **Upregulated**: Gene is producing MORE protein than normal (overexpressed)
- **Downregulated**: Gene is producing LESS protein than normal (underexpressed)

**Disease gene signatures**: In a disease, certain genes are abnormally up/down:
- Alzheimer's: APP (amyloid precursor) is UPREGULATED → excess amyloid beta → plaques
- Cancer: TP53 (tumor suppressor) is DOWNREGULATED → cells can't stop dividing

## 3.3 What Is Gene Expression Profiling?

Measuring which genes are turned on/off in a cell. Technologies:

- **Microarray**: Measure mRNA levels for ~20,000 genes simultaneously
- **RNA-seq**: Sequence ALL mRNA molecules, count reads per gene (more accurate)
- **L1000**: Measure 978 "landmark" genes (cheaper, faster, enough to infer the rest)

**How OpenCure uses it**: Get the disease's gene expression signature (which genes are up/down), then find drugs that REVERSE it using L1000CDS2.

## 3.4 What Is a Protein-Protein Interaction (PPI) Network?

Proteins don't work alone — they physically bind to and regulate each other:

```
Insulin binds to Insulin Receptor → activates IRS1 → activates PI3K → activates AKT → cell growth
```

A PPI network is a graph where:
- **Nodes** = proteins (16,201 human proteins in STRING)
- **Edges** = physical or functional interactions (236,930 in STRING)

**STRING database** scores interactions from 0-1000 based on evidence:
- Experimental (physical binding confirmed in lab)
- Co-expression (genes expressed together)
- Text mining (mentioned together in papers)
- Genomic context (found near each other in genomes)

OpenCure uses score ≥ 700 (high confidence only).

## 3.5 What Is GWAS?

**Genome-Wide Association Study**: Scan millions of genetic variants (SNPs) across thousands of people to find which variants are associated with a disease.

**SNP** (Single Nucleotide Polymorphism): A position in DNA where people differ:
```
Person A: ...ACTG...
Person B: ...AGTG...
              ↑ SNP (C→G)
```

**GWAS logic**: If people with variant G at position X get Alzheimer's more often than people with variant C:
- That region of DNA is associated with Alzheimer's
- The nearby gene is likely involved in the disease
- If a drug targets that gene's protein, it might treat the disease

**eQTL** (Expression Quantitative Trait Locus): A SNP that affects how much protein a gene produces:
```
People with SNP rs429358-C4 (APOE ε4) produce more APOE4 protein → higher Alzheimer's risk
```

## 3.6 What Is Mendelian Randomization?

**The problem with observational studies**: People who take Vitamin D supplements have lower cancer rates. But is it because Vitamin D prevents cancer? Or because healthy, wealthy people take more supplements?

**MR solution**: Use genetic variants as "natural experiments":
```
Genetic variant → determines Vitamin D level → affects cancer risk?

If people GENETICALLY predisposed to high Vitamin D still have lower cancer:
  → Vitamin D CAUSALLY prevents cancer (not just correlation)

If genetically high Vitamin D people have SAME cancer rates:
  → The correlation was just confounding (healthy lifestyle, not the vitamin)
```

**In drug repurposing**: If a genetic variant that increases Protein X also increases Disease Y risk → drugs that reduce Protein X should treat Disease Y.

## 3.7 What Is a Molecular Fingerprint?

A binary (0/1) encoding of a molecule's substructures.

**Morgan Fingerprint algorithm** (also called ECFP — Extended Connectivity Fingerprint):

Step 1: Give each atom an initial ID based on: atomic number, bond count, charge
```
Carbon with 3 bonds → ID: 6_3_0
Oxygen with 1 bond  → ID: 8_1_0
```

Step 2: For radius=1, combine each atom's ID with its neighbor IDs:
```
Central carbon + [neighbor oxygen, neighbor carbon, neighbor hydrogen]
→ new_ID = hash(6_3_0 + 8_1_0 + 6_3_0 + 1_1_0) = 4729182
```

Step 3: For radius=2, repeat using radius=1 IDs (now capturing 2-bond neighborhoods)

Step 4: Fold all atom IDs into a 2048-bit vector:
```
bit_position = atom_ID modulo 2048
fingerprint[bit_position] = 1
```

**Result**: A 2048-bit binary vector where each bit represents a specific molecular substructure. Two similar molecules will have many bits in common.

## 3.8 What Is Tanimoto Similarity?

The standard way to compare molecular fingerprints:
```
T(A, B) = c / (a + b - c)
```
Where:
- a = number of bits set to 1 in fingerprint A
- b = number of bits set to 1 in fingerprint B
- c = number of bits set to 1 in BOTH A and B

**Example**:
```
Aspirin:     1 0 1 1 0 1 0 0 1 1  (a = 6 bits set)
Ibuprofen:   1 0 1 0 1 1 0 0 1 0  (b = 5 bits set)
Both:        1 0 1 0 0 1 0 0 1 0  (c = 4 bits shared)

T = 4 / (6 + 5 - 4) = 4/7 = 0.571
```

Both are NSAIDs (non-steroidal anti-inflammatory drugs), so moderate similarity makes sense.

## 3.9 What Is SMILES?

A text encoding of molecular structure (Simplified Molecular-Input Line-Entry System):

```
C           → methane (CH4)
CC          → ethane (C-C)
C=C         → ethylene (C=C double bond)
C#C         → acetylene (C≡C triple bond)
c1ccccc1    → benzene ring (6 aromatic carbons)
O           → water
CC(=O)O     → acetic acid (vinegar): C-C(=O)-O
```

**Rules**:
- Uppercase: aliphatic atoms (C, N, O, S)
- Lowercase: aromatic atoms (c, n, o)
- `()`: branches (side chains)
- Numbers: ring closures (c**1**ccccc**1** = 6-membered ring)
- `=`: double bond, `#`: triple bond
- `@`/`@@`: chirality (3D arrangement)
- `[NH]`: explicit hydrogen, charges, isotopes

**Example**: Aspirin
```
CC(=O)Oc1ccccc1C(=O)O

Decoded:
C-C(=O)-O-c1-c-c-c-c-c1-C(=O)-O
acetyl   ester  benzene    carboxyl
group    bond   ring       group
```

## 3.10 What Is LogP?

**LogP** (partition coefficient) measures how much a molecule prefers oil vs water:
```
LogP = log10(concentration in oil / concentration in water)
```

- LogP < 0: Hydrophilic (water-loving) — stays in blood
- LogP 0-5: Good drug range — can cross cell membranes but still dissolves in blood
- LogP > 5: Hydrophobic (fat-loving) — may get stuck in fat tissue, poor bioavailability

**Lipinski's Rule of 5** (drug-likeness criteria):
- Molecular weight ≤ 500 Da
- LogP ≤ 5
- H-bond donors ≤ 5
- H-bond acceptors ≤ 10

Most approved drugs follow these rules. OpenCure's docking module uses them for binding compatibility scoring.

## 3.11 What Is an Adverse Event Report (FAERS)?

When a patient experiences a side effect from a drug, their doctor or the patient can report it to the FDA's **Adverse Event Reporting System**:

```
Report #12345:
  Patient: 65-year-old female
  Drug: Metformin (for diabetes)
  Events reported: Nausea, Weight loss, Improved cognition ← interesting!
```

OpenCure mines 28+ million FAERS reports looking for **positive signals** — unexpected beneficial effects. If many patients taking Drug X for Disease A also report improvement in Disease B, that's a repurposing signal.

## 3.12 What Is a Clinical Trial Phase?

Drug development goes through phases of clinical testing:

- **Phase 1** (20-100 patients): Is it safe? What dose?
- **Phase 2** (100-300 patients): Does it work? What's the optimal dose?
- **Phase 3** (300-3000 patients): Is it better than existing treatments?
- **Phase 4** (post-approval): Long-term monitoring of approved drugs

A drug that FAILED Phase 3 is strong evidence it doesn't work (penalty -3 points in OpenCure's confidence system). Phase 1 failure may just be a dosing issue (penalty -1 point).

---

# CHAPTER 4: SOFTWARE AND API FOUNDATIONS

## 4.1 What Is an API?

An **Application Programming Interface** is a way for software to talk to other software.

**REST API**: Send HTTP requests, get JSON responses:
```
GET https://api.fda.gov/drug/event.json?search=Aspirin
→ Returns: {"results": [{"patient": {...}, "reactions": [...]}]}
```

**GraphQL API**: Send structured queries, get exactly what you ask for:
```
POST https://api.platform.opentargets.org/api/v4/graphql
Body: { disease(efoId: "MONDO_0004975") { name associatedTargets { ... } } }
→ Returns: {"data": {"disease": {"name": "Alzheimer", "associatedTargets": [...]}}}
```

OpenCure uses 8 external APIs:

| API | Purpose | Cost | Auth |
|-----|---------|------|------|
| PubMed E-utilities | Research papers | Free | None |
| ClinicalTrials.gov | Trial data | Free | None |
| Semantic Scholar | 200M+ papers | Free | None |
| FDA FAERS | Adverse events | Free | None |
| Open Targets | Genetic evidence | Free | None |
| L1000CDS2 | Gene signatures | Free | None |
| PharmGKB | Pharmacogenomics | Free | None |
| Anthropic Claude | LLM explanations | Paid | API key |

## 4.2 What Is NumPy?

NumPy is a Python library for fast math on arrays (vectors, matrices).

Instead of:
```python
# Slow Python loop
result = []
for i in range(10000):
    result.append(a[i] + b[i])
```

NumPy does:
```python
# Fast C-level vectorized operation
result = a + b  # operates on all 10,000 elements at once
```

OpenCure's TransE scores 10,551 drugs in ~0.1 seconds using NumPy vectorization vs ~30 seconds with Python loops.

## 4.3 What Is PyTorch?

PyTorch is a deep learning framework. It provides:
- **Tensors**: Like NumPy arrays but can run on GPU and track gradients
- **Autograd**: Automatic differentiation (computes gradients for backpropagation)
- **nn.Module**: Building blocks for neural networks

OpenCure uses PyTorch for:
- RotatE model (PyKEEN wraps PyTorch)
- ChemBERTa transformer (HuggingFace wraps PyTorch)
- TxGNN (DGL wraps PyTorch)

## 4.4 What Is Caching?

Storing expensive results so you don't recompute them:

```python
_cache = {}

def expensive_function(input):
    if input in _cache:
        return _cache[input]  # Return stored result (instant)

    result = ... # Compute (slow)
    _cache[input] = result   # Store for next time
    return result
```

OpenCure caches:
- DRKG embeddings (148 MB, loaded once)
- PPI network (16K nodes, built once)
- Drug name mappings
- API responses (avoid repeated calls)

## 4.5 What Is BFS?

**Breadth-First Search**: An algorithm for finding shortest paths in a graph.

```
Start at node A. Want to reach node F.

      A
     / \
    B   C
   / \   \
  D   E   F  ← Found! Distance = 2 hops

BFS visits: A → B, C (distance 1) → D, E, F (distance 2) → FOUND F at distance 2
```

**With cutoff**: Stop after N hops (OpenCure: cutoff=4). Don't explore the entire graph, just the local neighborhood. Much faster.

## 4.6 What Is FastAPI?

A Python web framework for building APIs. OpenCure's web server:

```python
@app.get("/api/search")
async def api_search(disease: str, top_k: int = 25):
    results = search(disease, top_k=top_k)
    return {"results": results}
```

When you visit `http://localhost:8000/api/search?disease=Alzheimer`, FastAPI:
1. Parses the URL parameters (disease="Alzheimer", top_k=25)
2. Calls the `api_search` function
3. Returns the results as JSON

---

# CHAPTER 5: HOW IT ALL FITS TOGETHER

## 5.1 The Complete Pipeline (Every Step)

### Step 0: Data Loading (happens once at startup)

```
Load data/drkg/embed/DRKG_TransE_l2_entity.npy → 97,238 × 400 float32 matrix
Load data/drkg/embed/DRKG_TransE_l2_relation.npy → 107 × 400 float32 matrix
Load data/drkg/embed/entities.tsv → mapping: "Compound::DB00945" → row index 45231
Load data/drkg/embed/relations.tsv → mapping: "GNBR::T::Compound:Disease" → row index 73
Load data/drkg/drkg.tsv → 5,874,261 triplets into pandas DataFrame
Load data/drkg/drug_names_cache.tsv → "DB00945" → "Aspirin"
Load data/drkg/compound_smiles.tsv → "DB00945" → "CC(=O)Oc1ccccc1C(=O)O"
Load data/models/pykeen/rotate/trained_model.pkl → PyTorch RotatE model
Load data/drkg/embeddings/chemberta_embeddings.npz → 2,268 × 768 float32 matrix
Load data/ncbi_gene_info.gz → 193,862 Entrez ID → gene symbol mappings
```

### Step 1: Disease Entity Matching

User inputs: "Alzheimer's disease"

```python
# Search the curated DISEASE_NAME_MAP dictionary (90 entries)
# "alzheimer" matches "Alzheimer's disease" → "MESH:D000544"
# Construct entity ID: "Disease::MESH:D000544"
# Verify it exists in entity_to_id mapping
# Return: [("Disease::MESH:D000544", 1.00)]
```

### Step 2: TransE Scoring (Pillar 1)

For EACH of the 6 treatment relations:
```python
relation = "GNBR::T::Compound:Disease"  # "treats"
relation_index = 73                       # from relations.tsv
r_emb = relation_emb[73]                  # shape: (400,) — the "treats" vector

disease_index = entity_to_id["Disease::MESH:D000544"]  # e.g., 85432
disease_emb = entity_emb[85432]           # shape: (400,) — Alzheimer's vector

# Get ALL 10,551 compound embeddings at once
compound_indices = [entity_to_id[c] for c in compounds]  # list of 10,551 ints
compound_embs = entity_emb[compound_indices]              # shape: (10551, 400)

# Vectorized scoring: score = -||h + r - t||
diff = compound_embs + r_emb - disease_emb    # shape: (10551, 400) — broadcasting
scores = -np.linalg.norm(diff, axis=1)         # shape: (10551,) — one score per drug
```

Keep the BEST score per drug across all 6 relations. Result: dict of 10,551 compound→(score, relation, disease_entity).

### Step 3: RotatE Scoring (Pillar 2)

```python
# Build batch tensor
heads = torch.tensor([45231, 7891, 3456, ...], dtype=torch.long)  # 500 drug indices
relations = torch.full_like(heads, 73)                             # all "treats"
tails = torch.full_like(heads, 85432)                              # all "Alzheimer's"
batch = torch.stack([heads, relations, tails], dim=1)              # shape: (500, 3)

# PyKEEN internally:
# 1. Look up complex embeddings for heads, relations, tails
# 2. Compute: tail ≈ head ⊙ relation (element-wise complex multiplication)
# 3. Score = -||head ⊙ relation - tail|| in complex space
scores = model.score_hrt(batch).squeeze(-1)  # shape: (500,)
```

### Step 4: TxGNN Scoring (Pillar 3)

```python
# Load pre-computed predictions
predictions = pd.read_csv("data/txgnn_predictions.tsv", sep="\t")
# Filter for this disease (fuzzy match)
# "Alzheimer's disease" matches "Alzheimer disease" in TxGNN
alz_predictions = predictions[predictions["disease"] == "Alzheimer disease"]
# Returns: 100 drugs with scores and ranks
# Map drug names back to Compound:: entities via drug_names dict
```

### Step 5: Fingerprint Scoring (Pillar 4)

```python
# Find known treatments for Alzheimer's in DRKG
known_treatments = triplets[
    (triplets["tail"] == "Disease::MESH:D000544") &
    (triplets["relation"].isin(TREATMENT_RELATIONS))
]["head"]  # e.g., ["Compound::DB00843", "Compound::DB01043", ...]

# Compute Morgan fingerprints for known treatments
for drug in known_treatments:
    smiles = smiles_map[drug.split("::")[1]]  # e.g., "COC1=CC(=CC(=C1)OC)C2=CN=C(N=C2N)N"
    mol = Chem.MolFromSmiles(smiles)           # Parse SMILES into molecular graph
    fp = mfpgen.GetFingerprint(mol)            # 2048-bit fingerprint

# For each OTHER drug:
candidate_fp = mfpgen.GetFingerprint(candidate_mol)
similarity = BulkTanimotoSimilarity(candidate_fp, known_treatment_fps)
max_sim = max(similarity)  # Best match to any known treatment
```

### Step 6: ChemBERTa Scoring (Pillar 5)

```python
# Pre-computed embeddings loaded at startup (2268 drugs × 768 dims)
# Get known treatment embeddings
known_embs = chemberta_emb[known_treatment_indices]  # shape: (K, 768)
# Get all candidate embeddings
candidate_embs = chemberta_emb[candidate_indices]     # shape: (M, 768)

# Normalize to unit vectors
known_norm = known_embs / np.linalg.norm(known_embs, axis=1, keepdims=True)
cand_norm = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)

# Cosine similarity matrix
sim_matrix = cand_norm @ known_norm.T  # shape: (M, K)
max_sims = sim_matrix.max(axis=1)       # Best match per candidate
```

### Step 7: Gene Signature Reversal (Pillar 6)

```python
# 1. Get Alzheimer's disease genes from Open Targets
response = requests.post("https://api.platform.opentargets.org/api/v4/graphql",
    json={"query": "{ disease(efoId: 'MONDO_0004975') { associatedTargets { ... } } }"})
# Extract target genes, classify as up or down
up_genes = ["APP", "BACE1", "APOE", ...]    # Overexpressed in Alzheimer's
down_genes = ["BDNF", "SYP", "CHAT", ...]   # Underexpressed

# 2. Query L1000CDS2 for reversers
response = requests.post("https://maayanlab.cloud/L1000CDS2/query",
    json={"data": {"genes": up_genes + down_genes,
                    "vals": [1.0]*len(up_genes) + [-1.0]*len(down_genes)},
           "config": {"aggravate": False}})  # False = find REVERSERS
# Returns: ranked list of drugs that reverse this signature
# e.g., [{"pert_desc": "valproic acid", "score": 0.85}, ...]
```

### Step 8: Network Proximity (Pillar 7)

```python
# Load STRING PPI network (16,201 proteins, 236,930 edges)
G = nx.Graph()
# ... add edges with score >= 700

# For each drug (top 200 by TransE score):
drug_targets = ["ENSP00000361405", "ENSP00000284981"]  # Drug's target proteins
disease_proteins = ["ENSP00000354360", ...]             # Disease-associated proteins

for target in drug_targets[:10]:
    # BFS from drug target, max 4 hops
    distances = nx.single_source_shortest_path_length(G, target, cutoff=4)
    # Find minimum distance to ANY disease protein
    min_dist = min(distances[dp] for dp in disease_proteins if dp in distances)

avg_distance = mean(all_min_distances)
proximity_score = max(0, 1.0 - avg_distance / 4.0)  # Closer = higher score
```

### Step 9: Mendelian Randomization (Pillar 8)

```python
# 1. Map disease to EFO ID
response = requests.post(OPENTARGETS_URL,
    json={"query": '{ search(queryString: "alzheimer") { hits { id } } }'})
efo_id = "MONDO_0004975"

# 2. Get targets with genetic evidence
response = requests.post(OPENTARGETS_URL,
    json={"query": f'{{ disease(efoId: "{efo_id}") {{ associatedTargets {{ rows {{ target {{ approvedSymbol }} datatypeScores {{ id score }} }} }} }} }}'})
# Extract: {"APP": 0.924, "PSEN1": 0.954, "APOE": 0.896, ...}

# 3. For each drug: check if its targets overlap with disease genetic targets
drug_targets = get_drug_target_genes(drug_entity, triplets)  # ["348", "351", ...]
gene_symbols = [entrez_map[t] for t in drug_targets]          # ["APOE", "APP", ...]
overlap = set(gene_symbols) & set(genetic_targets.keys())     # {"APOE", "APP"}
mr_score = 0.6 * max_score + 0.4 * mean(other_scores)
```

### Step 10: Score Combination

```python
# For Drug X scored by TransE (pct=0.85), TxGNN (pct=0.92), proximity (pct=0.70):
# Active base weights: transe=0.10, txgnn=0.20, proximity=0.15
# Total = 0.45
# Normalized: transe=0.222, txgnn=0.444, proximity=0.333

weighted_sum = 0.222 * 0.85 + 0.444 * 0.92 + 0.333 * 0.70
             = 0.189 + 0.409 + 0.233
             = 0.831

convergence_bonus = 0.05 * (3 - 1) = 0.10  # 3 pillars
mr_bonus = 0.15 * 0.896 = 0.134            # APOE genetic evidence

final_score = 0.831 + 0.10 + 0.134 = 1.065
```

### Step 11: Return Results

```python
# Sort all drugs by final_score, return top K
results = [
    {"rank": 1, "drug_name": "Donepezil", "combined_score": 1.23, "pillars_hit": 5, "mr_score": 0.92},
    {"rank": 2, "drug_name": "Memantine", "combined_score": 1.18, "pillars_hit": 4, "mr_score": 0.85},
    ...
]
```

---

# CHAPTER 5C: MODULES NOT YET EXPLAINED

## What Is GraphRAG?

**RAG** (Retrieval Augmented Generation) means: before asking an LLM to generate text, first RETRIEVE relevant information, then include it in the prompt. The LLM generates better answers because it has specific context.

**GraphRAG** applies this to graph-structured data:
1. **Retrieve**: Extract paths from the knowledge graph between the drug and disease
2. **Augment**: Format those paths as human-readable text
3. **Generate**: Feed paths + evidence to Claude, ask for a mechanistic hypothesis

**Why this matters**: Without RAG, an LLM would generate generic text ("Metformin may affect metabolic pathways"). With GraphRAG, it gets specific paths ("Metformin → targets AMPK → regulates mTOR pathway → linked to tau phosphorylation in Alzheimer's") and generates precise mechanistic hypotheses.

## What Is Drug Synergy?

Two drugs work **synergistically** when their combined effect is greater than the sum of individual effects:
- Drug A kills 20% of cancer cells
- Drug B kills 30% of cancer cells
- Together they kill 80% (not just 50%)

**Why?** They hit different parts of the same disease pathway. Drug A blocks the main pathway, Drug B blocks the escape route that cancer cells would otherwise use.

OpenCure predicts synergy by measuring **target complementarity**: if Drug A's gene targets and Drug B's gene targets together cover MORE disease genes than either alone, they're likely complementary.

## What Is Pharmacogenomics?

The study of how genetic differences between patients affect their response to drugs.

**Example**: The gene CYP2D6 encodes an enzyme that metabolizes ~25% of all drugs. Some people have variants that make this enzyme work faster (ultra-rapid metabolizers) or slower (poor metabolizers):
- **Poor metabolizer** taking codeine: drug doesn't work (can't convert to active form)
- **Ultra-rapid metabolizer** taking codeine: drug is too strong (dangerous overdose risk)

OpenCure queries PharmGKB to find which genetic variants affect each drug's response, helping identify which patients would benefit most from a repurposed drug.

## What Is a Web API Server?

A program that listens on a network port and responds to HTTP requests:
```
Browser sends:  GET http://localhost:8000/api/search?disease=Alzheimer
Server runs:    search("Alzheimer's disease", top_k=25)
Server returns: {"results": [{rank: 1, drug: "Donepezil", score: 1.23}, ...]}
Browser shows:  Formatted results page
```

**FastAPI** is the Python framework that handles this. It automatically:
- Parses URL parameters (`disease=Alzheimer`)
- Validates types (`top_k` must be integer)
- Returns JSON responses
- Serves HTML templates for the web UI

## What Is a Screening Pipeline?

Running the analysis systematically across MANY diseases rather than one at a time:

```
For each of 25 diseases:
  1. Search top 50 drug candidates (fast, ~1 min)
  2. For top 10: gather full evidence reports (slow, ~10 min each)
     - Query PubMed for publications
     - Query ClinicalTrials.gov for trials
     - Query FAERS for adverse event signals
     - Check for failed trials
     - Run gene signature reversal
     - Generate LLM hypothesis
  3. Save results as JSON
  4. Move to next disease
```

The screening took ~12 hours for all 25 diseases and produced 217 candidates, including 39 BREAKTHROUGH predictions with zero published literature.

## What Is a BREAKTHROUGH Prediction?

A drug-disease pair where:
1. The AI pillars give it a HIGH score (strong computational prediction)
2. There is ZERO published literature about this specific drug-disease combination
3. No clinical trials exist for it

This means: the AI found a pattern that no human researcher has published about yet. It could be a genuine new discovery — or it could be noise. That's why experimental validation is critical.

---

# CHAPTER 5B: v4 IMPROVEMENTS EXPLAINED

## What Changed and Why

After building the initial 8-pillar system, an honest audit revealed serious gaps. Here are the fixes explained at the foundations level:

### Improvement 1: scipy Sparse Matrix for Network Proximity

**Before**: For each drug, we ran BFS (Breadth-First Search) through a 16,201-node graph using Python's NetworkX library. Each BFS is O(V+E) = ~250,000 operations, and Python's function call overhead made it ~50,000 operations per second. For 200 drugs: 200 × 10 targets × 1 BFS = 2,000 BFS calls = ~40 seconds.

**After**: We convert the entire graph into a **scipy sparse matrix** — a memory-efficient representation of the adjacency matrix where only non-zero entries are stored. Then we call `shortest_path()` once, which computes ALL distances between ALL nodes using compiled C code running at ~5,000,000 operations per second.

**What is a sparse matrix?** A regular 16,201 × 16,201 matrix would have 262 million entries, but only ~473,000 are non-zero (edges exist). A sparse matrix stores only the non-zero entries: `(row_index, col_index, value)` — using ~100x less memory and enabling vectorized C-level operations.

**Result**: One 20-second precomputation replaces thousands of individual BFS calls. Per-drug lookup is O(1) — just reading a number from a pre-computed table.

### Improvement 2: Fuzzy String Matching for Gene Signatures

**Before**: When L1000CDS2 returned "metformin hydrochloride", we compared it to our database which stores "Metformin". Exact match failed because the strings are different.

**What is fuzzy matching?** It measures how similar two strings are, even if they're not identical:
- **Edit distance** (Levenshtein): Count minimum edits (insert/delete/replace) to transform one string into another
  - "metformin" → "metformin hydrochloride" = 14 insertions = far apart
- **Ratio** (what we use): `2 × matching_chars / total_chars` scaled to 0-100
  - "metformin" vs "metformin hydrochloride" = ~70 (below our 85 threshold)
  - But "metformin" vs "metformin" (after stripping " hydrochloride") = 100 (perfect match!)

**Our three-tier approach**:
1. Build alias map: strip known salt suffixes ("hydrochloride", "sulfate", etc.) to create alternate names
2. Try exact match against expanded aliases
3. If no match: use `rapidfuzz.process.extractOne()` with 85% threshold

**What is rapidfuzz?** A C-compiled library for fast fuzzy string matching. It can compare one string against 10,000 candidates in milliseconds (vs seconds in pure Python).

### Improvement 3: RotatE Candidate Cap

**The problem explained simply**: Imagine two judges ranking 500 contestants. Judge A (TransE) has seen all 500. Judge B (RotatE) brings 500 more contestants that Judge A hasn't seen. Now the combined ranking has 1,000 people, but Judge A's top picks are diluted among Judge B's extras.

**The fix**: Judge B can comment on Judge A's 500 contestants (improving/adjusting their scores), but can only ADD up to 100 new contestants of its own. This keeps the pool manageable while still benefiting from Judge B's expertise.

### Improvement 4: Learned Ensemble

**Before**: Hand-picked weights (TransE: 0.10, TxGNN: 0.20, etc.) — these were educated guesses.

**After**: Train a machine learning model to learn the optimal combination:
- **Training data**: 54,775 known drug→disease treatment pairs (positive) + 54,775 random pairs (negative)
- **Task**: Given pillar scores for a drug-disease pair, predict: is this a real treatment?
- **Model**: GradientBoosting (an ensemble of decision trees that corrects each other's mistakes)
- **Result**: AUC-ROC 0.998 — the model nearly perfectly separates real treatments from random pairs

**What is AUC-ROC?** Area Under the Receiver Operating Characteristic curve. It measures how well a binary classifier separates two classes:
- AUC = 0.5: Random guessing (no better than flipping a coin)
- AUC = 0.8: Good classifier
- AUC = 0.95: Excellent classifier
- AUC = 1.0: Perfect separation

Our 0.998 means: if you pick a random real treatment and a random fake pair, the model correctly ranks the real one higher 99.8% of the time.

---

# CHAPTER 6: GLOSSARY

| Term | Definition |
|------|-----------|
| **Agonist** | Drug that activates a protein receptor |
| **Antagonist** | Drug that blocks a receptor without activating it |
| **API** | Interface for software to communicate with other software |
| **Backpropagation** | Algorithm for computing gradients in neural networks |
| **Batch size** | Number of training examples processed before updating weights |
| **BFS** | Breadth-First Search — explores graph level by level |
| **Bias** | Constant offset added in a neural neuron |
| **BRCA1/2** | Breast cancer susceptibility genes (DNA repair proteins) |
| **ChemBERTa** | BERT transformer trained on 77M molecules |
| **ClinicalTrials.gov** | US government registry of all clinical studies |
| **CLS token** | Special token in transformers that aggregates sequence-level information |
| **Complex number** | Number with real and imaginary parts: a + bi |
| **Convergence bonus** | Extra score when multiple independent methods agree |
| **Complementarity** | Two drugs targeting DIFFERENT disease genes (better than same targets) |
| **Cosine similarity** | Dot product of unit vectors — measures directional alignment |
| **CYP2D6** | Liver enzyme that metabolizes ~25% of drugs — genetic variants affect drug response |
| **COX-2** | Cyclooxygenase-2 — enzyme targeted by aspirin and NSAIDs |
| **DRKG** | Drug Repurposing Knowledge Graph (97K entities, 5.87M relationships) |
| **Dot product** | Sum of element-wise products of two vectors |
| **DrugBank** | Database of FDA-approved drugs with targets and pharmacology |
| **DGL** | Deep Graph Library — framework for graph neural networks |
| **Embedding** | Learned vector representation of a discrete entity |
| **Entrez ID** | NCBI's numeric gene identifier (e.g., 348 = APOE gene) |
| **Epoch** | One complete pass through all training data |
| **eQTL** | Genetic variant that affects gene expression levels |
| **Euclidean distance** | Straight-line distance: sqrt(sum of squared differences) |
| **FAERS** | FDA Adverse Event Reporting System (28M+ reports) |
| **FastAPI** | Python web framework for building REST APIs |
| **Fingerprint** | Binary encoding of molecular substructures |
| **GNN** | Graph Neural Network — operates on graph-structured data |
| **Gradient** | Direction of steepest increase of a function |
| **GraphQL** | Query language for APIs that returns exactly what you request |
| **GraphRAG** | Graph-based Retrieval Augmented Generation — retrieve KG paths, then feed to LLM |
| **GWAS** | Genome-Wide Association Study — scans genome for disease variants |
| **Hetionet** | Curated biomedical knowledge graph |
| **Hydrophilic** | Water-loving (LogP < 0) |
| **Hydrophobic** | Water-repelling (LogP > 5) |
| **Inhibitor** | Drug that blocks a protein's function |
| **JSON** | JavaScript Object Notation — structured data format |
| **Knowledge graph** | Database of entity-relationship-entity triplets |
| **L1000** | Gene expression profiling assay measuring 978 genes |
| **L1000CDS2** | API for finding drugs that reverse gene signatures |
| **L2 norm** | Euclidean distance / vector length |
| **Learning rate** | Step size for weight updates during training |
| **LogP** | Oil/water partition coefficient (drug lipophilicity) |
| **Loss function** | Measures how wrong predictions are (training minimizes this) |
| **Matrix** | Grid of numbers (rows × columns) |
| **MedDRA** | Medical Dictionary for Regulatory Activities (standardized terms) |
| **Mendelian Randomization** | Using genetic variants as natural experiments for causality |
| **MESH** | Medical Subject Headings — controlled vocabulary for diseases |
| **Message passing** | GNN technique: nodes aggregate information from neighbors |
| **Molecular weight** | Mass of a molecule in Daltons (drugs: typically 150-900 Da) |
| **Morgan fingerprint** | Circular fingerprint based on atom neighborhoods (radius + hash) |
| **mRNA** | Messenger RNA — copy of a gene used to make protein |
| **Negative sampling** | Creating fake data to teach models what ISN'T true |
| **Neural network** | Function with learnable weights, trained on data |
| **Node** | A point in a graph (entity in a knowledge graph) |
| **NumPy** | Python library for fast numerical computing |
| **Overfitting** | Model memorizes training data instead of learning patterns |
| **Percentile rank** | Fraction of values below a given score (0-1 scale) |
| **PPI** | Protein-Protein Interaction |
| **PrimeKG** | Knowledge graph used by TxGNN (17K diseases, 8K drugs) |
| **Protein** | Molecular machine made of amino acids, folded into 3D shape |
| **PyKEEN** | Python Knowledge Embedding framework |
| **PyTorch** | Deep learning framework (tensors + automatic gradients) |
| **RDKit** | Open-source cheminformatics toolkit |
| **Relation** | Connection type between entities in a knowledge graph |
| **ReLU** | Rectified Linear Unit: activation function max(0, x) |
| **REST API** | HTTP-based API (GET/POST requests, JSON responses) |
| **RotatE** | KG embedding model using complex rotations |
| **Scalar** | A single number |
| **Self-attention** | Each element in a sequence looks at all others for context |
| **SLCWA** | Stochastic Local Closed World Assumption (negative sampling strategy) |
| **SMILES** | Text encoding of molecular structure |
| **SNP** | Single Nucleotide Polymorphism — a genetic variant |
| **STRING** | Database of protein-protein interactions (string-db.org) |
| **Tanimoto coefficient** | Similarity measure for binary fingerprints |
| **Tensor** | Multi-dimensional array (generalization of vectors and matrices) |
| **Transformer** | Neural network using self-attention for sequences |
| **TransE** | KG embedding model: head + relation ≈ tail |
| **Triplet** | (head, relation, tail) fact in a knowledge graph |
| **TxGNN** | Graph neural network for drug repurposing (Harvard, 2024) |
| **UniProt** | Database of protein sequences and functions |
| **Vector** | An ordered list of numbers representing a point in space |
| **Vectorization** | Computing on entire arrays at once (fast) vs element-by-element (slow) |
| **Weight** | Learnable parameter in a neural network |
| **AlphaFold** | Google DeepMind's AI that predicts 3D protein structures from amino acid sequences |
| **Anthropic API** | Cloud API for Claude LLM — used for generating mechanistic explanations |
| **AUC-ROC** | Area Under ROC Curve — measures classifier quality (0.5=random, 1.0=perfect) |
| **BREAKTHROUGH** | OpenCure novelty level: zero published evidence + strong AI prediction = potential new discovery |
| **AUC-PR** | Area Under Precision-Recall Curve — like AUC-ROC but better for imbalanced data |
| **Biologics** | Large protein-based drugs (antibodies, peptides) — no SMILES representation |
| **Edit distance** | Minimum insertions/deletions/substitutions to transform one string into another |
| **Fuzzy matching** | Finding similar (not identical) strings using edit distance or character overlap |
| **GradientBoosting** | Ensemble of decision trees where each tree corrects the previous one's errors |
| **Joblib** | Python library for saving/loading scikit-learn models to disk |
| **rapidfuzz** | C-compiled library for fast fuzzy string matching (100x faster than Python) |
| **Salt form** | Drug + a salt (e.g., "metformin hydrochloride") — same drug, different formulation |
| **Sparse matrix** | Matrix that stores only non-zero entries — saves memory for mostly-empty matrices |
| **scipy.sparse** | Library for sparse matrix operations (creation, arithmetic, graph algorithms) |
| **shortest_path** | scipy function that computes all-pairs shortest paths on sparse graphs |
| **PharmGKB** | Pharmacogenomics Knowledge Base — genetic variants affecting drug response |
| **RAG** | Retrieval Augmented Generation — give LLMs specific context before generating |
| **Screening pipeline** | Automated analysis of many diseases sequentially (25 in OpenCure) |
| **Semantic Scholar** | Academic search engine with 200M+ papers and citation data |
| **Supplementary compounds** | RotatE candidates not in TransE pool, capped at 100 to prevent flooding |
| **Synergy** | Combined drug effect greater than sum of individual effects |
| **Zero-shot** | Predicting for cases not seen during training |
