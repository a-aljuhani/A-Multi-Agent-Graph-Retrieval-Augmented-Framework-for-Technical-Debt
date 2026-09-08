# GraphRAG-grounded multi-agent reasoning for explainable self-admitted technical debt analysis

A self-contained, reviewer-checkable implementation of the Self-Admitted
Technical Debt (SATD) evidence-graph pipeline described in the accompanying
paper. It detects SATD, classifies its type, retrieves supporting and opposing
evidence from a Neo4j graph built only from training data, and generates
evidence-grounded explanations and recommendations.

The retrieval math, cue reranking, agent prompts, and metrics implement exactly
the configuration used to produce the reported results. Every frozen numeric
constant is read from `frozen-settings.yaml` at runtime; changing a value there
changes the program's behavior on the next run. The repository is fully
self-contained and contains no API keys, model weights, or dataset rows.

## What this is

- **Is**: the GraphRAG retrieval pipeline (full-text + vector + reciprocal-rank
  fusion + high-support cue reranking), the evidence-package composition, the
  explanation and recommendation agent prompts and validators, and the metrics
  used to report results (macro-F1, per-class precision/recall/F1, confusion
  matrix).
-  You supply your own SQLite database (see
  [Expected SQLite schema](#expected-sqlite-schema)), your own Neo4j instance,
  your own API keys, and your own fine-tuned classifier model IDs (or retrain
  them — see [Reproducing the classifiers](#reproducing-the-classifiers)).

## Repository contents

```text
pipeline.py                  # all pipeline logic + CLI (build-graph / run / evaluate / prepare-finetune-data)
prompts.py                   # verbatim agent prompt templates
frozen-settings.example.yaml # frozen constants + model-id placeholders (copy to frozen-settings.yaml)
requirements.txt             # pinned dependencies
```

## Pipeline overview

```text
Raw artifact text
    |
    v
Binary Detection Agent (fine-tuned OpenAI model)  -> SATD / non-SATD
    |
    +-- if SATD --> Category Agent (fine-tuned OpenAI model) --> one of 5 categories
    |
    v
GraphRAG retrieval over Neo4j (TRAINING split only)
  full-text (Lucene OR-query) + MiniLM vector search -> Reciprocal Rank Fusion
  -> high-support lexical-cue reranking
    |
    v
Evidence package (up to 3 supportive + 2 opposite training examples, plus matching cues)
    |
    +--> Explanation Agent (DeepSeek)     -- cannot change the frozen label
    +--> Recommendation Agent (DeepSeek)  -- cannot change the frozen label/category
```

## Dataset

The pipeline is evaluated on the publicly available multi-source SATD dataset of
Li et al. (2023), which aggregates SATD from four artifact sources (source code
comments, commit messages, pull requests, and issue trackers) across 103
open-source Java projects. The five-type SATD taxonomy (design, defect,
requirement, documentation, test) follows Maldonado & Shihab (2015).

- Source: https://github.com/yikun-li/satd-different-sources-data
- Li, Y., Soliman, M., & Avgeriou, P. (2023). *Automatic identification of
  self-admitted technical debt from four different sources.* Empirical Software
  Engineering, 28(3), 65. https://doi.org/10.1007/s10664-023-10297-9

No dataset rows are distributed here. You build the SQLite database yourself from
the public dataset, using the project-disjoint Fold-2 split (82 train / 18 val /
16 test projects; 49,424 / 10,589 / 10,590 artifacts).

## Requirements

```
pip install -r requirements.txt
```

You also need:
- A running Neo4j instance (5.13+ / 5.x with vector-index support), reachable via
  `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`.
- An `OPENAI_API_KEY` (for the binary and category fine-tuned classifiers).
- A `DEEPSEEK_API_KEY` (for the explanation and recommendation agents).
- Your own SQLite database matching the schema below.

Put credentials in a local `.env` file (never committed):

```
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

## Frozen settings

Copy `frozen-settings.example.yaml` to `frozen-settings.yaml` and fill in your
own fine-tuned model IDs. **Every** frozen constant used by the pipeline (RRF
`k`, retrieval depths, cue support threshold, cue rerank lambda/denominator,
embedding model + revision + similarity, evidence-package composition,
retry/backoff policy, batch sizes) is read from this file at runtime — none of
it is hardcoded in `pipeline.py`.

## Expected SQLite schema

`pipeline.py` expects a SQLite database with (at minimum) these tables:

- `samples(sample_id, project_id, raw_text, source_type, language, external_ref, ground_truth_binary, ground_truth_category, dataset_id)`
- `splits(sample_id, fold, split)` — `split` in `{train, val, test}`
- `projects(project_id, project_name, language)`
- `satd_categories(category_id, name, definition, taxonomy_source)`

This mirrors the project-disjoint Fold-2 split described under [Dataset](#dataset).
No dataset rows are included in this release.

## CLI usage

```
# 1. Build the training-only evidence graph (nodes, embeddings, vector + full-text
#    indexes, and the high-support cue layer). Run once per graph version.
python pipeline.py build-graph --db path/to/research.sqlite3 --settings frozen-settings.yaml

# 2. Classify + explain + recommend for a single artifact or a JSONL file.
python pipeline.py run --db path/to/research.sqlite3 --settings frozen-settings.yaml --query "TODO: fix this hack later"
python pipeline.py run --db path/to/research.sqlite3 --settings frozen-settings.yaml --queries-file queries.jsonl --output results.json

# 3. Evaluate on a held-out split and write metrics.
python pipeline.py evaluate --db path/to/research.sqlite3 --settings frozen-settings.yaml --split test --output metrics.json

# 4. Build the balanced binary fine-tuning JSONL from the Fold-2 TRAIN split.
python pipeline.py prepare-finetune-data --db path/to/research.sqlite3 --settings frozen-settings.yaml --output finetune_binary.jsonl
```

`build-graph` creates the vector and full-text indexes; on a fresh Neo4j they
populate asynchronously, so allow them to come online before running `evaluate`.

## Reproducing the classifiers

The binary detection and category classification agents are **fine-tuned OpenAI
models**. Their model IDs are account-specific and are not distributed here:
`frozen-settings.yaml` ships with placeholder values that `run` and `evaluate`
refuse to execute with. Retrain equivalents as follows.

### Base model and hyperparameters

- **Base model**: `gpt-4.1-mini-2025-04-14` (OpenAI fine-tuning API).
- **Binary detector fine-tune**: 2 epochs, batch size 14, ~2.87M trained tokens.
  These are the hyperparameters used to produce the reported binary detection
  results; OpenAI auto-selects a learning-rate multiplier unless overridden, and
  it was not overridden.
- **Category classifier fine-tune**: use the binary detector's hyperparameters as
  a starting point and report whatever you settle on, for full reproducibility of
  your own run.

### Building the fine-tuning data from Fold-2 TRAIN

`pipeline.py prepare-finetune-data` builds the **binary** JSONL:

1. Take every SATD (`ground_truth_binary=1`) artifact in Fold-2 TRAIN.
2. Sample an equal number of non-SATD artifacts using minimum-one-then-
   largest-remainder proportional allocation stratified by
   `(project_id, source_type)`, with a fixed seed (`finetune_data.seed`), so
   every non-empty project/source stratum in TRAIN is represented and the
   sample is deterministic.
3. Shuffle the combined set with the same seed.
4. Emit one OpenAI chat-format JSONL record per artifact: `system` = the exact
   binary-detector system prompt (`prompts.BINARY_SYSTEM`), `user` = the raw
   artifact text, `assistant` = `{"is_satd": true|false}`.

The **category** JSONL is built analogously over the SATD-labeled artifacts of
Fold-2 TRAIN: `system` = `prompts.CATEGORY_SYSTEM`, `user` = the raw artifact
text, `assistant` = `{"category": "<ground-truth category>"}`.

### Fine-tuning and wiring it up

1. Upload the JSONL and launch an OpenAI fine-tuning job against
   `gpt-4.1-mini-2025-04-14`.
2. Paste the resulting `ft:...` model IDs into `frozen-settings.yaml` under
   `models.binary_detector.model_id` and `models.category_classifier.model_id`.
3. Run `pipeline.py evaluate --split val` to check your reproduction against your
   own labels before running `--split test`.

## Metrics

`evaluate` reports six-class macro-F1 (the primary metric, given the ~11% SATD
class prevalence), per-class precision/recall/F1, a confusion matrix, and binary
(SATD vs. non-SATD) accuracy/F1.

