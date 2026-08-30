# A Multi-Agent Graph Retrieval-Augmented Framework for Technical Debt

This repository provides a compact, implementatio of a research framework for detecting self-admitted technical debt (SATD), classifying its type, retrieving supporting and opposing evidence from a Neo4j graph, and generating evidence-grounded explanations and recommendations.

It is intended for architecture review and independent reimplementation. 

## Research objective

The framework investigates whether a leakage-safe, project-disjoint evidence graph can make SATD analysis more traceable by combining specialized classifiers with retrieval-grounded explanation and recommendation agents.

## Pipeline overview

```text
Raw software artifact
        |
        v
Binary Detection Agent --------> SATD / non-SATD
        |
        +-- if SATD -----------> Category Agent
        |                        design | defect | requirement |
        |                        documentation | test
        v
Training-only GraphRAG retrieval
  full text + vector search + high-support cue reranking
        |
        v
Evidence package
  supportive examples + opposite-binary examples + lexical cues
        |
        +----------------------> Explanation Agent
        |
        +-- if SATD -----------> Recommendation Agent
```

The Explanation and Recommendation Agents consume frozen predictions. They provide context and guidance but cannot change the binary decision or SATD category.

## Leakage-safe evaluation design

The study uses a project-disjoint Fold 2 of the Li/Maldonado multi-source Java SATD dataset:

| Split | Projects | Artifacts | Purpose |
|---|---:|---:|---|
| Train | 82 | 49,424 | Model development, graph construction, and evidence corpus |
| Validation | 18 | 10,589 | Architecture selection and ablation studies |
| Test | 16 | 10,590 | Locked final evaluation |

Each of the 116 projects occurs in exactly one split. Only training artifacts may become graph nodes or retrieved evidence. Validation and test artifacts are passed as external queries, and their labels are unavailable during retrieval and inference.

## Evidence graph composition

| Element | Count |
|---|---:|
| Artifact nodes | 49,424 |
| Project nodes | 82 |
| Category nodes | 6 |
| Source-type nodes | 4 |
| Issue-thread nodes | 2,354 |
| Pull-request-thread nodes | 3,052 |
| Lexical-cue nodes | 923 |
| **Total nodes** | **55,845** |
| `FROM_PROJECT` relationships | 49,424 |
| `HAS_SOURCE_TYPE` relationships | 49,424 |
| `LABELED_AS` relationships | 49,424 |
| Issue `PART_OF_THREAD` relationships | 15,150 |
| Pull-request `PART_OF_THREAD` relationships | 3,129 |
| `CONTAINS_CUE` relationships | 300,217 |
| **Total relationships** | **466,768** |

Thread identifiers are parsed exactly and scoped by project. Category relationships store training metadata but are not used as a category-sharing retrieval shortcut. Thread expansion was rejected by validation ablation, and semantic-similarity edges were not used in the final architecture.

## Frozen retrieval abstraction

The selected retrieval configuration has three stages:

1. **Full-text retrieval:** Neo4j full-text search over artifact text using lowercased, escaped, OR-connected Lucene terms.
2. **Vector retrieval:** `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with 384-dimensional L2-normalized embeddings and cosine similarity.
3. **Hybrid cue reranking:** equal-weight reciprocal-rank fusion with constant 60 combines the top 20 lexical and vector candidates. `hybrid_cue_rerank_w100` then reranks only this candidate set using training-derived unigram and bigram cues with support at least 100 and `lambda=1.0`.

```text
final_score(query, document)
  = rrf_score(query, document)
  + (1.0 / 61) * weighted_cue_jaccard(query_cues, document_cues)
```

Cue statistics are derived only from Fold-2 training labels. Cue reranking does not introduce new candidates, traverse category nodes, or use validation/test labels, project priors, or source priors.

## Evidence and agent interfaces

For each query, the retrieval layer forms an evidence package containing up to three training examples matching the predicted binary side and up to two examples representing the opposite side. Matching high-support lexical cues are attached to the evidence.

The abstract interfaces in `pipeline.py` represent five components:

- **Binary Detection Agent:** predicts SATD or non-SATD from raw text.
- **Category Agent:** assigns design, defect, requirement, documentation, or test when SATD is predicted.
- **Training-only Retriever:** returns supportive and opposite-binary graph evidence.
- **Explanation Agent:** reports whether the frozen decision is supported, challenged, or insufficiently supported and cites supplied evidence.
- **Recommendation Agent:** returns a priority, suggested remediation, rationale, and evidence references for predicted SATD.

## Repository contents

```text
.
|-- README.md
|-- pipeline.py
|-- frozen-settings.example.yaml


- `pipeline.py` documents component inputs, outputs, authority, and execution order without implementing models or database access.
- `frozen-settings.example.yaml` records the non-secret frozen retrieval and evidence settings using model placeholders for local reimplementation.



This repository is an architectural abstraction, not the private executable research system.
