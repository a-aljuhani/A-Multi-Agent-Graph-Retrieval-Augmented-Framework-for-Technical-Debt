# GraphRAG-Grounded Multi-Agent Reasoning for Explainable Self-Admitted Technical Debt Analysis

This repository contains the implementation and reproducibility materials for the study:

**GraphRAG-Grounded Multi-Agent Reasoning for Explainable Self-Admitted Technical Debt Analysis**

The project implements a multi-agent framework for analyzing **Self-Admitted Technical Debt (SATD)** using supervised classification, graph-based retrieval, evidence-grounded explanation, and developer-oriented recommendation.

The framework separates SATD prediction from downstream reasoning so that retrieved evidence does not modify the original classification decision.

---

## 1. Project Description

The proposed framework consists of three main layers:

1. **Classification Backbone**
   - Detection Agent: predicts SATD vs. non-SATD.
   - Category Agent: classifies detected SATD into one of five categories:
     - Design
     - Defect
     - Requirement
     - Documentation
     - Test

2. **Graph-Grounded Reasoning**
   - Builds a Neo4j evidence graph using training artifacts only.
   - Performs lexical retrieval using BM25.
   - Performs dense semantic retrieval using sentence-transformer embeddings.
   - Combines rankings using Reciprocal Rank Fusion (RRF).
   - Applies training-derived lexical cue reranking.
   - Retrieves supporting and contrasting evidence.
   - Generates an evidence-grounded explanation for the frozen classification result.

3. **Action**
   - Generates developer-oriented remediation recommendations for artifacts classified as SATD.
   - Recommendations are conditioned on the predicted SATD category, retrieved evidence, and generated explanation.

The framework is designed so that GraphRAG acts as an **evidence and provenance layer**, rather than replacing the supervised SATD classifier.

---

## 2. Dataset Information

### Multi-Source SATD Dataset

The experiments use the publicly available **multi-source SATD dataset** introduced by Li et al. (2023).

The dataset contains SATD examples from four types of software-development artifacts:

- Source code comments
- Commit messages
- Issue trackers
- Pull requests

### Dataset source

https://github.com/yikun-li/satd-different-sources-data

### Dataset publication

Li, Y., Soliman, M., & Avgeriou, P. (2023).  
*Automatic identification of self-admitted technical debt from four different sources.*  
Empirical Software Engineering, 28(3), 65.

DOI:

https://doi.org/10.1007/s10664-023-10297-9

The source-code-comment component originates from the SATD dataset of Maldonado et al.:

https://github.com/maldonado/tse.satd.data

### Important note about third-party data

**No third-party dataset rows are redistributed in this repository.**

Users should obtain the original datasets from the public repositories above and construct the local experimental database using the preparation steps provided in this repository.

The local SQLite database is generated from the public dataset and is used as the working data store for the experimental pipeline.

---

## 3. Evaluation Split

The experiments use the **project-disjoint Fold-2 split**.

Each software project occurs in only one of the training, validation, or test partitions.

| Split | Projects | non-SATD | SATD | Total |
|---|---:|---:|---:|---:|
| Training | 82 | 44,034 | 5,390 | 49,424 |
| Validation | 18 | 9,365 | 1,224 | 10,589 |
| Test | 16 | 9,423 | 1,167 | 10,590 |
| **Total** | **116** | **62,822** | **7,781** | **70,603** |

The corresponding six-class training distribution is:

| Class | Training |
|---|---:|
| non-SATD | 44,034 |
| Design | 3,765 |
| Defect | 295 |
| Requirement | 543 |
| Documentation | 447 |
| Test | 340 |

The training partition is used for:

- supervised fine-tuning,
- construction of the Neo4j evidence graph,
- lexical-cue extraction, and
- retrieval evidence.

The validation partition is used for model and component selection.

The locked test partition is used only after the final architecture and configurations are frozen.

**Validation and test artifacts are not inserted into the evidence graph.**

---

## 4. Code Information

The repository contains the implementation required to reproduce the main experimental pipeline, including code for:

- dataset preparation,
- SQLite database construction,
- project-disjoint data handling,
- SATD binary classification,
- SATD category classification,
- Neo4j evidence-graph construction,
- lexical BM25 retrieval,
- dense semantic retrieval,
- Reciprocal Rank Fusion,
- lexical-cue extraction and reranking,
- supporting and contrasting evidence selection,
- explanation generation,
- recommendation generation,
- classification evaluation,
- retrieval evaluation,
- reasoning-quality evaluation, and
- bootstrap confidence-interval estimation.

Configuration files are provided to document the settings used in the reported experiments.

---

## 5. Methodology

### Step 1 — Dataset Preparation

Download the multi-source SATD dataset from:

https://github.com/yikun-li/satd-different-sources-data

The source-code-comment data can also be traced to:

https://github.com/maldonado/tse.satd.data

Prepare the data using the project-disjoint Fold-2 partition described above.

The prepared artifacts are stored locally in SQLite for downstream processing.

No third-party dataset rows are included in this repository.

---

### Step 2 — Classification Backbone

The framework uses two independently fine-tuned GPT-4.1-mini models.

#### Detection Agent

The Detection Agent performs binary classification:

```text
artifact -> SATD / non-SATD
```

The binary fine-tuning dataset contains:

- 5,390 SATD examples
- 5,390 non-SATD examples
- 10,780 total examples

The model was fine-tuned for 2 epochs.

#### Category Agent

Artifacts predicted as SATD are passed to the Category Agent:

```text
SATD artifact -> design / defect / requirement / documentation / test
```

The Category Agent uses all 5,390 SATD training artifacts and preserves the observed class distribution.

The model was fine-tuned for 3 epochs.

The classification output is frozen before GraphRAG retrieval and downstream reasoning.

---

### Step 3 — Neo4j Evidence Graph

The evidence graph is constructed **only from the Fold-2 training partition**.

The graph represents relationships among:

- Artifacts
- Projects
- Source types
- SATD categories
- Issue threads
- Pull-request threads
- Lexical cues
- Provenance information

The final training evidence graph contains:

- **55,845 nodes**
- **466,768 relationships**
- **49,424 artifact nodes**

Validation and test examples are external queries and are never inserted into the evidence graph.

---

### Step 4 — Lexical Retrieval

Lexical retrieval is implemented using Neo4j full-text search.

Query preprocessing includes:

- lowercasing,
- token normalization,
- removal of tokens shorter than two characters,
- escaping Lucene special characters, and
- joining the remaining terms with `OR`.

Each retrieval method initially returns the top 20 candidates.

---

### Step 5 — Dense Semantic Retrieval

Dense retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings.

Embeddings are L2-normalized and compared using cosine similarity.

---

### Step 6 — Hybrid Retrieval

The lexical and dense rankings are combined using equal-weight Reciprocal Rank Fusion:

```text
RRF k = 60
```

For candidate artifact `d`:

```text
RRF(d) =
1 / (60 + rank_BM25(d))
+
1 / (60 + rank_dense(d))
```

---

### Step 7 — Lexical-Cue Reranking

High-support lexical cues are extracted from the training partition only.

The cue layer uses:

- unigrams and bigrams,
- lowercase normalization,
- stop-word filtering, and
- minimum support of 100 training artifacts.

The cues are used only for reranking retrieved candidates.

They are **not** used as an additional classification mechanism and do not introduce validation or test labels into retrieval.

The selected configuration applies the lexical-cue signal to the hybrid top-20 candidates.

---

### Step 8 — Evidence Packaging

The final evidence package contains up to:

- **3 supporting training artifacts**, and
- **2 contrasting training artifacts**.

Evidence identifiers and provenance information are retained so that generated explanations can be traced back to retrieved training examples.

---

### Step 9 — Evidence-Grounded Explanation

The Evidence and Explanation Agent receives:

- the raw artifact,
- the frozen classification result, and
- the retrieved GraphRAG evidence package.

The agent produces an evidence-grounded explanation without modifying the original classification result.

---

### Step 10 — Developer Recommendation

For artifacts classified as SATD, the Recommendation Agent receives:

- the raw artifact,
- predicted SATD category,
- retrieved evidence,
- generated explanation.

It then produces developer-oriented remediation guidance.

The agent is instructed to avoid unsupported assumptions when implementation details are unavailable.

---

## 6. Frozen Experimental Configuration

The final architecture evaluated on the locked test set is:

| Stage | Model / Configuration |
|---|---|
| Detection | Fine-tuned GPT-4.1-mini |
| Category classification | Fine-tuned GPT-4.1-mini |
| Retrieval | Hybrid BM25 + dense RRF + support-100 cue reranking |
| Explanation | DeepSeek V4 Pro |
| Recommendation | DeepSeek V4 Pro |
| Independent quality judge | GPT-5.4 |

### Detection and Category inference

```text
Temperature = 0
Maximum output = 20 tokens
Output format = JSON
```

### Explanation and Recommendation

```text
Temperature = 0
Top-p = 1
Maximum output = 384 tokens
```

---

## 7. Requirements

The experiments reported in the study used the following environment:

- Python 3.13.5
- Neo4j Community Edition 5.26.24
- OpenJDK 21.0.10
- PyTorch 2.9.1 (CPU)
- Transformers 4.57.3
- SentenceTransformers 5.2.2
- NumPy 2.3.5
- scikit-learn 1.7.2
- Neo4j Python Driver 6.1.0
- ranx 0.3.21
- OpenAI SDK 2.8.1

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

A local Neo4j instance is required for graph construction and retrieval.

---

## 8. External Model/API Requirements

Some stages of the experimental pipeline use externally hosted language models.

Users reproducing these stages must provide their own valid credentials for the corresponding providers.

Depending on the experiment being reproduced, this may include:

- OpenAI API credentials
- DeepSeek API credentials

**API keys are not included in this repository and should never be committed to source control.**

Set the required credentials through environment variables or the configuration mechanism provided in the repository.

---

## 9. Reproduction Workflow

A full reproduction follows this general order:

```text
1. Download the public SATD datasets
           |
           v
2. Prepare the Fold-2 project-disjoint split
           |
           v
3. Build the local SQLite database
           |
           v
4. Prepare classification data
           |
           v
5. Configure the fine-tuned classification models
           |
           v
6. Build the Neo4j training evidence graph
           |
           v
7. Build lexical and semantic retrieval indexes
           |
           v
8. Run hybrid retrieval and cue reranking
           |
           v
9. Generate supporting/contrasting evidence packages
           |
           v
10. Run evidence-grounded explanation generation
           |
           v
11. Run SATD recommendation generation
           |
           v
12. Compute classification, retrieval, and reasoning metrics
```

To reproduce the reported results, use the frozen settings and configuration files supplied in this repository.

The locked test partition should not be used for model selection or configuration tuning.

---

## 10. Evaluation

### Classification Evaluation

The Classification Backbone is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- Specificity
- Balanced accuracy
- Confusion matrices

Macro-F1 is the primary metric for the six-class end-to-end classification task because of class imbalance.

### Retrieval Evaluation

The GraphRAG retrieval component is evaluated using:

- Precision@k
- Recall@k
- Hit@k
- Mean Reciprocal Rank (MRR)
- MAP@k
- NDCG@k

for:

```text
k = 1, 3, 5, 10, 20
```

Six-class macro-NDCG@10 is used as the primary retrieval-selection metric.

### Explanation Evaluation

Explanation quality is evaluated according to:

- Faithfulness
- Grounding
- Classification relevance
- Clarity
- Overall quality

### Recommendation Evaluation

Recommendation quality is evaluated according to:

- Actionability
- Relevance
- Category consistency
- Grounding
- Clarity
- Overall quality

The generated outputs are evaluated post hoc using an independent model.

Deterministic checks are additionally used for evidence-reference validity and output consistency.


---

## 13. Repository and Code Availability

The implementation and reproducibility materials are publicly available at:

https://github.com/a-aljuhani/A-Multi-Agent-Graph-Retrieval-Augmented-Framework-for-Technical-Debt

The repository provides the implementation, configuration information, dataset references, software requirements, and instructions required to reproduce the principal experimental pipeline.

---

Dataset:

https://github.com/yikun-li/satd-different-sources-data

DOI:

https://doi.org/10.1007/s10664-023-10297-9

---

## 15. License

This repository is provided for academic research and reproducibility purposes.

Third-party datasets remain subject to the licenses and terms specified by their original authors and repositories and are **not redistributed here**.

If a separate `LICENSE` file is included in this repository, use and redistribution of the implementation code are governed by that license.

---

## 16. Contributions

Contributions that improve reproducibility, documentation, or implementation quality are welcome.

For substantial methodological changes or extensions, please open an issue or contact the repository maintainers before submitting a pull request.

When contributing:

1. Clearly describe the proposed change.
2. Keep dataset provenance intact.
3. Do not upload third-party dataset rows unless redistribution is explicitly permitted.
4. Do not introduce validation or test artifacts into the training evidence graph.
5. Document any dependency or configuration changes.
6. Preserve reproducible experimental settings whenever possible.

---

## 17. Contact

For questions regarding the implementation or reproduction of the experiments, please contact the authors through the repository or the contact information provided in the associated manuscript.
