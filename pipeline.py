#!/usr/bin/env python3
"""Self-contained pipeline for the SATD Evidence-Graph framework.

Every frozen numeric constant, model identifier, index name, and structural
threshold is read from a YAML settings file (frozen-settings.yaml) at runtime
-- none of it is hardcoded here. Prompt TEXT templates are kept as Python
constants in prompts.py.

Subcommands:
  build-graph          Build the Neo4j evidence graph from the TRAINING split only.
  run                  Run inference (detect -> category -> retrieve -> explain ->
                       recommend) on a single query or a file of queries.
  evaluate             Run inference over a val/test split and compute metrics.
  prepare-finetune-data
                       Build the balanced binary fine-tuning JSONL from the
                       Fold-2 TRAIN split (see README.md "Reproducing the
                       classifiers").

No API keys, model weights, or dataset rows are embedded in this file.
Credentials come only from environment variables (OPENAI_API_KEY,
DEEPSEEK_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from dotenv import load_dotenv

import prompts

# ---------------------------------------------------------------------------
# Settings / environment
# ---------------------------------------------------------------------------

PLACEHOLDER_PREFIX = "REPLACE_WITH_"


def load_settings(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise RuntimeError(f"Invalid settings file: {path}")
    return settings


def setting(settings: dict, dotted_path: str) -> Any:
    """Fetch a nested settings value, e.g. setting(s, 'retrieval.rrf_k')."""
    node = settings
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required setting: {dotted_path}")
        node = node[key]
    return node


def require_model_id(settings: dict, agent_key: str) -> str:
    model_id = setting(settings, f"models.{agent_key}.model_id")
    if not isinstance(model_id, str) or model_id.startswith(PLACEHOLDER_PREFIX):
        raise RuntimeError(
            f"models.{agent_key}.model_id is a placeholder. Fill in your own "
            "fine-tuned model ID in frozen-settings.yaml (see README.md "
            "'Reproducing the classifiers')."
        )
    return model_id


def load_env(env_file: Path) -> None:
    load_dotenv(env_file, override=False)


def require_env(*names: str) -> dict[str, str]:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


# ---------------------------------------------------------------------------
# Cue tokenizer
# ---------------------------------------------------------------------------

_BASE_STOPWORDS = set("""a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could couldn't did didn't do
does doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves""".split())
_PRESERVE = {"no", "nor", "not", "can't", "cannot", "couldn't", "didn't",
             "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't",
             "mustn't", "shan't", "shouldn't", "wasn't", "weren't", "won't",
             "wouldn't", "should", "must", "will", "would", "can", "could"}
STOPWORDS = _BASE_STOPWORDS - _PRESERVE
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def extract_cues(text: str) -> set[str]:
    tokens = [x.strip("'") for x in _TOKEN_RE.findall((text or "").lower())]
    tokens = [x for x in tokens if x]
    kept = {x for x in tokens if len(x) >= 2 and x not in STOPWORDS}
    kept.update(
        f"{a} {b}" for a, b in zip(tokens, tokens[1:])
        if len(a) >= 2 and len(b) >= 2 and a not in STOPWORDS and b not in STOPWORDS
    )
    return kept


def cue_id(cue_text: str) -> str:
    return f"{2 if ' ' in cue_text else 1}|{cue_text}"


# ---------------------------------------------------------------------------
# SQLite access (expects the schema documented in README.md)
# ---------------------------------------------------------------------------

def connect_db(db_path: Path, read_only: bool = True) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro" if read_only else str(db_path)
    conn = sqlite3.connect(uri, uri=read_only)
    conn.row_factory = sqlite3.Row
    return conn


def load_split_rows(conn: sqlite3.Connection, settings: dict, split: str) -> list[dict]:
    dataset_id = setting(settings, "dataset.dataset_id")
    fold = setting(settings, "dataset.fold")
    rows = conn.execute(
        """
        SELECT s.sample_id, s.project_id, s.raw_text, s.source_type, s.language,
               s.external_ref, s.ground_truth_binary, s.ground_truth_category,
               c.name AS category_name
        FROM samples s
        JOIN splits sp ON sp.sample_id = s.sample_id
        JOIN satd_categories c ON c.category_id = s.ground_truth_category
        WHERE s.dataset_id=? AND sp.fold=? AND sp.split=?
        ORDER BY s.sample_id
        """,
        (dataset_id, fold, split),
    ).fetchall()
    return [dict(row) for row in rows]


def load_train_lookup(conn: sqlite3.Connection, settings: dict) -> dict[int, dict]:
    rows = load_split_rows(conn, settings, setting(settings, "dataset.train_split"))
    return {
        int(r["sample_id"]): {
            "text": r["raw_text"],
            "binary": int(r["ground_truth_binary"]),
            "category": r["category_name"],
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

_ISSUE_REF = re.compile(r"^issue:(\d+):(summary|description|comment_\d+)$")
_PR_REF = re.compile(r"^pr:(\d+):(?:summary:0|description:0|comment:\d+|review:\d+)$")


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _neo4j_driver(settings: dict):
    from neo4j import GraphDatabase
    creds = require_env("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    return GraphDatabase.driver(creds["NEO4J_URI"], auth=(creds["NEO4J_USER"], creds["NEO4J_PASSWORD"]))


def _neo4j_database() -> str:
    return os.environ.get("NEO4J_DATABASE", "neo4j")


class GraphBuilder:
    """Builds the training-only evidence graph: artifacts, projects, categories,
    source types, issue/PR threads, and the high-support lexical-cue layer."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.graph_version = setting(settings, "neo4j.graph_version")
        self.cue_graph_version = setting(settings, "neo4j.cue_graph_version")
        self.batch_size = setting(settings, "neo4j.build_batch_size")
        self.cue_support_threshold = setting(settings, "retrieval.cue_support_threshold")

    def create_schema(self, session) -> None:
        constraints = (
            ("satd_artifact_sample_id_unique", "SATDArtifact", "sample_id"),
            ("satd_project_project_id_unique", "SATDProject", "project_id"),
            ("satd_category_category_id_unique", "SATDCategory", "category_id"),
            ("satd_source_type_name_unique", "SATDSourceType", "name"),
            ("satd_issue_thread_id_unique", "SATDIssueThread", "thread_id"),
            ("satd_pr_thread_id_unique", "SATDPRThread", "thread_id"),
            ("satd_cue_graph_version_cue_id_unique", "SATDCue", "cue_id"),
        )
        for name, label, prop in constraints:
            session.run(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            ).consume()
        session.run(
            f"CREATE FULLTEXT INDEX {setting(self.settings, 'neo4j.fulltext_index_name')} IF NOT EXISTS "
            "FOR (n:SATDArtifact) ON EACH [n.raw_text]"
        ).consume()
        vector_dim = int(setting(self.settings, "embedding.dim"))
        vector_similarity = setting(self.settings, "embedding.similarity")
        session.run(
            f"CREATE VECTOR INDEX {setting(self.settings, 'neo4j.vector_index_name')} IF NOT EXISTS "
            "FOR (n:SATDArtifact) ON (n.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {vector_dim}, "
            f"`vector.similarity_function`: '{vector_similarity}'}}}}"
        ).consume()

    def _prepare_artifacts(self, rows: list[dict]) -> list[dict]:
        prepared = []
        for row in rows:
            ref = row["external_ref"] or ""
            issue = _ISSUE_REF.fullmatch(ref)
            pull_request = _PR_REF.fullmatch(ref)
            item = dict(row)
            item["issue_thread_id"] = f"issue:{row['project_id']}:{issue.group(1)}" if issue else None
            item["pr_thread_id"] = f"pr:{row['project_id']}:{pull_request.group(1)}" if pull_request else None
            prepared.append(item)
        return prepared

    def build_artifact_graph(self, session, conn: sqlite3.Connection) -> None:
        train_split = setting(self.settings, "dataset.train_split")
        artifacts = self._prepare_artifacts(load_split_rows(conn, self.settings, train_split))
        dataset_id = setting(self.settings, "dataset.dataset_id")
        fold = setting(self.settings, "dataset.fold")
        projects = [dict(r) for r in conn.execute(
            """
            SELECT DISTINCT p.project_id, p.project_name, p.language
            FROM projects p JOIN samples s ON s.project_id = p.project_id
            JOIN splits sp ON sp.sample_id = s.sample_id
            WHERE s.dataset_id=? AND sp.fold=? AND sp.split=?
            ORDER BY p.project_id
            """,
            (dataset_id, fold, train_split),
        )]
        categories = [dict(r) for r in conn.execute(
            "SELECT category_id, name, definition, taxonomy_source FROM satd_categories ORDER BY category_id"
        )]
        source_types = [{"name": r[0]} for r in conn.execute(
            """
            SELECT DISTINCT s.source_type FROM samples s JOIN splits sp ON sp.sample_id=s.sample_id
            WHERE s.dataset_id=? AND sp.fold=? AND sp.split=? ORDER BY s.source_type
            """,
            (dataset_id, fold, train_split),
        )]

        session.run(
            "UNWIND $rows AS row MERGE (p:SATDProject {project_id: row.project_id}) "
            "SET p.project_name=row.project_name, p.language=row.language, p.graph_version=$version",
            rows=projects, version=self.graph_version,
        ).consume()
        session.run(
            "UNWIND $rows AS row MERGE (c:SATDCategory {category_id: row.category_id}) "
            "SET c.name=row.name, c.definition=row.definition, c.taxonomy_source=row.taxonomy_source",
            rows=categories,
        ).consume()
        session.run(
            "UNWIND $rows AS row MERGE (:SATDSourceType {name: row.name})", rows=source_types,
        ).consume()

        artifact_query = """
            UNWIND $rows AS row
            MERGE (a:SATDArtifact {sample_id: row.sample_id})
            SET a.raw_text=row.raw_text, a.source_type=row.source_type,
                a.language=row.language, a.external_ref=row.external_ref,
                a.ground_truth_binary=row.ground_truth_binary,
                a.ground_truth_category=row.ground_truth_category,
                a.fold=$fold, a.split=$split, a.graph_version=$version
            WITH a, row
            MATCH (p:SATDProject {project_id: row.project_id})
            MATCH (s:SATDSourceType {name: row.source_type})
            MATCH (c:SATDCategory {category_id: row.ground_truth_category})
            MERGE (a)-[:FROM_PROJECT]->(p)
            MERGE (a)-[:HAS_SOURCE_TYPE]->(s)
            MERGE (a)-[:LABELED_AS]->(c)
        """
        issue_query = """
            UNWIND $rows AS row MATCH (a:SATDArtifact {sample_id: row.sample_id})
            MERGE (t:SATDIssueThread {thread_id: row.issue_thread_id}) SET t.graph_version=$version
            MERGE (a)-[:PART_OF_THREAD]->(t)
        """
        pr_query = """
            UNWIND $rows AS row MATCH (a:SATDArtifact {sample_id: row.sample_id})
            MERGE (t:SATDPRThread {thread_id: row.pr_thread_id}) SET t.graph_version=$version
            MERGE (a)-[:PART_OF_THREAD]->(t)
        """
        for batch in _chunks(artifacts, self.batch_size):
            session.run(artifact_query, rows=batch, fold=fold, split=train_split, version=self.graph_version).consume()
            issue_rows = [row for row in batch if row["issue_thread_id"] is not None]
            pr_rows = [row for row in batch if row["pr_thread_id"] is not None]
            if issue_rows:
                session.run(issue_query, rows=issue_rows, version=self.graph_version).consume()
            if pr_rows:
                session.run(pr_query, rows=pr_rows, version=self.graph_version).consume()

    def build_embeddings(self, session, conn: sqlite3.Connection) -> None:
        """Encode every TRAINING artifact with the frozen embedding model and
        write the vector to its node so the vector index can serve the vector
        retrieval arm. Model, revision, normalization, and dimension all come
        from the YAML -- the same settings used at query time."""
        from sentence_transformers import SentenceTransformer

        train_split = setting(self.settings, "dataset.train_split")
        rows = load_split_rows(conn, self.settings, train_split)
        model = SentenceTransformer(
            setting(self.settings, "embedding.model_name"),
            revision=setting(self.settings, "embedding.model_revision"),
        )
        normalize = setting(self.settings, "embedding.normalize")
        encode_batch_size = setting(self.settings, "embedding.encode_batch_size")
        ids = [int(r["sample_id"]) for r in rows]
        texts = [r["raw_text"] for r in rows]
        written = 0
        for start in range(0, len(rows), self.batch_size):
            id_chunk = ids[start:start + self.batch_size]
            text_chunk = texts[start:start + self.batch_size]
            vectors = model.encode(
                text_chunk, convert_to_numpy=True,
                normalize_embeddings=normalize, batch_size=encode_batch_size,
            )
            payload = [
                {"sample_id": sid, "embedding": [float(x) for x in vec]}
                for sid, vec in zip(id_chunk, vectors)
            ]
            session.run(
                "UNWIND $rows AS row MATCH (a:SATDArtifact {sample_id: row.sample_id}) "
                "SET a.embedding = row.embedding",
                rows=payload,
            ).consume()
            written += len(payload)
        print(json.dumps({"embeddings_written": written}))

    def build_cue_layer(self, session, conn: sqlite3.Connection) -> None:
        train_split = setting(self.settings, "dataset.train_split")
        dataset_id = setting(self.settings, "dataset.dataset_id")
        fold = setting(self.settings, "dataset.fold")
        rows = conn.execute(
            """
            SELECT s.sample_id, s.raw_text, s.ground_truth_binary
            FROM samples s JOIN splits sp ON sp.sample_id=s.sample_id
            WHERE s.dataset_id=? AND sp.fold=? AND sp.split=? ORDER BY s.sample_id
            """,
            (dataset_id, fold, train_split),
        ).fetchall()

        totals: Counter = Counter()
        satd_totals: Counter = Counter()
        artifact_cues: dict[int, set[str]] = {}
        for row in rows:
            sample_id, text, label = int(row[0]), row[1], int(row[2])
            cues = extract_cues(text)
            artifact_cues[sample_id] = cues
            totals.update(cues)
            if label == 1:
                satd_totals.update(cues)

        eligible = {c for c, support in totals.items() if support >= self.cue_support_threshold}
        cue_nodes = []
        for text in sorted(eligible, key=lambda v: (2 if " " in v else 1, v)):
            support = totals[text]
            satd_count = satd_totals[text]
            probability = satd_count / support
            signed = 2.0 * probability - 1.0
            cue_nodes.append({
                "cue_id": cue_id(text), "cue_text": text,
                "ngram_order": 2 if " " in text else 1,
                "support_train": support, "satd_count_train": satd_count,
                "non_satd_count_train": support - satd_count,
                "p_satd_train": probability, "signed_signal": signed,
                "cue_strength": abs(signed), "support_threshold": self.cue_support_threshold,
                "graph_version": self.cue_graph_version,
            })
        relationships = [
            {"sample_id": sid, "cue_id": cue_id(text)}
            for sid, texts in artifact_cues.items() for text in sorted(texts & eligible)
        ]
        relationships.sort(key=lambda r: (r["sample_id"], r["cue_id"]))

        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:SATDCue {cue_id: row.cue_id})
            SET c += row
            """,
            rows=cue_nodes,
        ).consume()
        rel_query = """
            UNWIND $rows AS row
            MATCH (a:SATDArtifact {sample_id: row.sample_id})
            MATCH (c:SATDCue {cue_id: row.cue_id, graph_version: $version})
            MERGE (a)-[r:CONTAINS_CUE {graph_version: $version}]->(c)
            SET r.presence = true
        """
        for batch in _chunks(relationships, self.batch_size):
            session.run(rel_query, rows=batch, version=self.cue_graph_version).consume()

        print(json.dumps({"cue_nodes": len(cue_nodes), "relationships": len(relationships)}))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

_LUCENE_SPECIAL = re.compile(r"(&&|\|\||[+\-!(){}\[\]^\"~*?:\\/])")


def lucene_or_query(text: str) -> str:
    collapsed = " ".join(text.lower().split())
    terms = [term for term in collapsed.split(" ") if len(term) >= 2]
    escaped = [_LUCENE_SPECIAL.sub(r"\\\1", term) for term in terms]
    return " OR ".join(escaped)


class Retriever:
    def __init__(self, settings: dict, session):
        self.settings = settings
        self.session = session
        self.graph_version = setting(settings, "neo4j.graph_version")
        self.fulltext_index = setting(settings, "neo4j.fulltext_index_name")
        self.vector_index = setting(settings, "neo4j.vector_index_name")
        self.component_depth = setting(settings, "retrieval.component_depth")
        self.rrf_k = setting(settings, "retrieval.rrf_k")
        self.ranking_depth = setting(settings, "retrieval.ranking_depth")
        self.cue_lambda = setting(settings, "retrieval.cue_lambda")
        self.cue_denominator = setting(settings, "retrieval.cue_denominator")
        train_split = setting(settings, "dataset.train_split")
        self.train_split = train_split

    def fulltext_retrieve(self, query_text: str) -> list[tuple[int, float]]:
        lucene = lucene_or_query(query_text)
        if not lucene:
            return []
        cypher = """
            CALL db.index.fulltext.queryNodes($index_name, $lucene, {limit: $limit})
            YIELD node, score
            WHERE node.graph_version=$version AND node.split=$split
            RETURN node.sample_id AS sample_id, score
        """
        return [
            (int(r["sample_id"]), float(r["score"]))
            for r in self.session.run(
                cypher, index_name=self.fulltext_index, lucene=lucene,
                limit=self.component_depth, version=self.graph_version, split=self.train_split,
            )
        ]

    def vector_retrieve(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        cypher = """
            CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
            YIELD node, score
            WHERE node.graph_version=$version AND node.split=$split
            RETURN node.sample_id AS sample_id, score
        """
        return [
            (int(r["sample_id"]), float(r["score"]))
            for r in self.session.run(
                cypher, index_name=self.vector_index, embedding=embedding.tolist(),
                limit=self.component_depth, version=self.graph_version, split=self.train_split,
            )
        ]

    def rrf(self, first: list[tuple[int, float]], second: list[tuple[int, float]]) -> list[tuple[int, float]]:
        scores: defaultdict = defaultdict(float)
        for ranking in (first, second):
            for rank, (sample_id, _) in enumerate(ranking, start=1):
                scores[sample_id] += 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[: self.component_depth]

    def load_cue_layer(self) -> tuple[dict[str, float], dict[str, list[int]], dict[int, set[str]], dict[int, float]]:
        cue_version = setting(self.settings, "neo4j.cue_graph_version")
        strengths: dict[str, float] = {}
        postings: defaultdict = defaultdict(list)
        artifact_cues: defaultdict = defaultdict(set)
        artifact_weights: defaultdict = defaultdict(float)
        for row in self.session.run(
            """
            MATCH (a:SATDArtifact)-[:CONTAINS_CUE {graph_version:$version}]->(c:SATDCue {graph_version:$version})
            RETURN a.sample_id AS sample_id, c.cue_id AS cue_id, c.cue_strength AS strength
            """,
            version=cue_version,
        ):
            cue = str(row["cue_id"])
            strength = float(row["strength"])
            sample_id = int(row["sample_id"])
            strengths[cue] = strength
            postings[cue].append(sample_id)
            artifact_cues[sample_id].add(cue)
            artifact_weights[sample_id] += strength
        return strengths, dict(postings), dict(artifact_cues), dict(artifact_weights)

    def cue_rerank(
        self,
        base: list[tuple[int, float]],
        query_text: str,
        strengths: dict[str, float],
        artifact_cues: dict[int, set[str]],
        artifact_weights: dict[int, float],
    ) -> list[tuple[int, float]]:
        query_cue_ids = sorted(cue_id(x) for x in extract_cues(query_text) if cue_id(x) in strengths)
        query_set = set(query_cue_ids)
        query_weight = sum(strengths[c] for c in query_cue_ids)
        base_scores = dict(base)
        base_ranks = {sid: rank for rank, (sid, _) in enumerate(base, 1)}
        stats: dict[int, dict[str, float | int]] = {}
        for sample_id, _ in base:
            shared = query_set & artifact_cues.get(sample_id, set())
            intersection = sum(strengths[c] for c in shared)
            denominator = query_weight + artifact_weights.get(sample_id, 0.0) - intersection
            stats[sample_id] = {
                "cue_similarity": intersection / denominator if denominator else 0.0,
                "weighted_intersection": intersection,
                "shared_cues": len(shared),
            }
        ranked = []
        for sample_id in base_scores:
            values = stats[sample_id]
            final = base_scores[sample_id] + self.cue_lambda * float(values["cue_similarity"]) / self.cue_denominator
            ranked.append((sample_id, final))
        ranked.sort(key=lambda item: (
            -item[1], -base_scores.get(item[0], 0.0),
            -float(stats.get(item[0], {}).get("cue_similarity", 0.0)),
            base_ranks.get(item[0], sys.maxsize),
            -float(stats.get(item[0], {}).get("weighted_intersection", 0.0)),
            -int(stats.get(item[0], {}).get("shared_cues", 0)),
            item[0],
        ))
        return ranked[: self.ranking_depth]

    def build_evidence_package(
        self,
        query_text: str,
        predicted_label: str,
        ranked_ids: list[int],
        train_lookup: dict[int, dict],
        cues: dict[str, dict],
    ) -> dict:
        package = setting(self.settings, "retrieval.evidence_package")
        if predicted_label != "non_satd":
            satd_n, non_satd_n = package["satd_count_when_predicted_satd"], package["non_satd_count_when_predicted_satd"]
        else:
            satd_n, non_satd_n = package["satd_count_when_predicted_non_satd"], package["non_satd_count_when_predicted_non_satd"]
        satd_ids = [x for x in ranked_ids if train_lookup[x]["binary"]][:satd_n]
        non_satd_ids = [x for x in ranked_ids if not train_lookup[x]["binary"]][:non_satd_n]
        satd_evidence = [
            {"ref": f"S{i}", "sample_id": x, "rank": ranked_ids.index(x) + 1,
             "category": train_lookup[x]["category"], "text": train_lookup[x]["text"]}
            for i, x in enumerate(satd_ids, 1)
        ]
        non_satd_evidence = [
            {"ref": f"N{i}", "sample_id": x, "rank": ranked_ids.index(x) + 1, "text": train_lookup[x]["text"]}
            for i, x in enumerate(non_satd_ids, 1)
        ]
        matched = sorted(
            (cues[t] for t in extract_cues(query_text) if t in cues),
            key=lambda c: (-c["cue_strength"], -c["support_train"], c["cue_id"]),
        )
        return {
            "artifact": query_text, "classification": predicted_label,
            "satd_evidence": satd_evidence, "non_satd_evidence": non_satd_evidence,
            "lexical_cues": matched,
        }

    def load_cue_definitions(self) -> dict[str, dict]:
        cue_version = setting(self.settings, "neo4j.cue_graph_version")
        cues = {}
        for row in self.session.run(
            "MATCH (c:SATDCue {graph_version:$version}) RETURN c.cue_id AS cue_id, c.cue_text AS cue_text, "
            "c.support_train AS support_train, c.p_satd_train AS p_satd_train, c.cue_strength AS cue_strength",
            version=cue_version,
        ):
            cues[row["cue_text"]] = {
                "cue_id": row["cue_id"], "text": row["cue_text"],
                "support_train": int(row["support_train"]), "p_satd_train": float(row["p_satd_train"]),
                "cue_strength": float(row["cue_strength"]),
            }
        return cues


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    pass


async def _call_with_retries(call_fn, max_attempts: int, backoff: list[int]):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call_fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller after retries
            last_error = exc
            if attempt < max_attempts:
                await asyncio.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    raise RuntimeError(f"Exhausted {max_attempts} attempts: {last_error}")


class BinaryDetector:
    def __init__(self, settings: dict, client):
        self.client = client
        self.model = require_model_id(settings, "binary_detector")
        self.temperature = setting(settings, "models.binary_detector.temperature")
        self.max_tokens = setting(settings, "models.binary_detector.max_tokens")
        self.max_attempts = setting(settings, "execution.max_attempts")
        self.backoff = setting(settings, "execution.retry_backoff_seconds")

    async def classify(self, raw_text: str) -> bool:
        async def attempt():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompts.BINARY_SYSTEM}, {"role": "user", "content": raw_text}],
                temperature=self.temperature, max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            if set(payload) != {"is_satd"} or not isinstance(payload["is_satd"], bool):
                raise ValidationError("binary schema")
            return bool(payload["is_satd"])
        return await _call_with_retries(attempt, self.max_attempts, self.backoff)


class CategoryClassifier:
    def __init__(self, settings: dict, client):
        self.client = client
        self.model = require_model_id(settings, "category_classifier")
        self.temperature = setting(settings, "models.category_classifier.temperature")
        self.max_tokens = setting(settings, "models.category_classifier.max_tokens")
        self.categories = set(setting(settings, "labels.satd_categories"))
        self.max_attempts = setting(settings, "execution.max_attempts")
        self.backoff = setting(settings, "execution.retry_backoff_seconds")

    async def classify(self, raw_text: str) -> str:
        async def attempt():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompts.CATEGORY_SYSTEM}, {"role": "user", "content": raw_text}],
                temperature=self.temperature, max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            if set(payload) != {"category"} or payload["category"] not in self.categories:
                raise ValidationError("category schema")
            return payload["category"]
        return await _call_with_retries(attempt, self.max_attempts, self.backoff)


def render_evidence_package(package: dict) -> str:
    lines = ["Artifact:", "<artifact>", package["artifact"], "</artifact>", "",
             f"Frozen classification: {package['classification']}", "", "SATD training evidence:"]
    if package["satd_evidence"]:
        for e in package["satd_evidence"]:
            lines += [f"{e['ref']} (frozen rank {e['rank']}, training category {e['category']}):", e["text"]]
    else:
        lines.append("[missing]")
    lines += ["", "Non-SATD training evidence:"]
    if package["non_satd_evidence"]:
        for e in package["non_satd_evidence"]:
            lines += [f"{e['ref']} (frozen rank {e['rank']}):", e["text"]]
    else:
        lines.append("[missing]")
    lines += ["", "Matching frozen train-derived lexical cues:"]
    if package["lexical_cues"]:
        for c in package["lexical_cues"]:
            lines.append(f"- {c['text']} (train support {c['support_train']}, train P(SATD|cue) {c['p_satd_train']:.6f})")
    else:
        lines.append("[none]")
    return "\n".join(lines)


def validate_explanation(raw: str, package: dict) -> dict:
    try:
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"json:{exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"classification", "explanation", "evidence_refs"}:
        raise ValidationError("fields")
    if payload["classification"] != package["classification"]:
        raise ValidationError("classification_changed")
    if not isinstance(payload["explanation"], str) or not payload["explanation"].strip():
        raise ValidationError("explanation")
    valid_refs = {e["ref"] for e in package["satd_evidence"] + package["non_satd_evidence"]}
    refs = payload["evidence_refs"]
    if not isinstance(refs, list) or len(refs) != len(set(refs)) or any(not isinstance(r, str) or r not in valid_refs for r in refs):
        raise ValidationError("refs")
    return payload


def validate_recommendation(raw: str, package: dict) -> dict:
    try:
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"json:{exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"priority", "recommended_action", "rationale", "evidence_refs"}:
        raise ValidationError("exact_fields")
    if payload["priority"] not in {"low", "medium", "high"}:
        raise ValidationError("priority")
    if not isinstance(payload["recommended_action"], str) or not payload["recommended_action"].strip():
        raise ValidationError("action")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValidationError("rationale")
    valid_refs = {e["ref"] for e in package["satd_evidence"] + package["non_satd_evidence"]}
    refs = payload["evidence_refs"]
    if not isinstance(refs, list) or len(refs) != len(set(refs)) or any(not isinstance(r, str) or r not in valid_refs for r in refs):
        raise ValidationError("refs")
    text = (payload["recommended_action"] + " " + payload["rationale"]).lower()
    if any(phrase in text for phrase in prompts.RECOMMENDATION_FORBIDDEN_PHRASES):
        raise ValidationError("classification_language")
    return payload


class ExplanationAgent:
    def __init__(self, settings: dict, client):
        self.client = client
        self.model = setting(settings, "models.explanation_agent.model_id")
        self.temperature = setting(settings, "models.explanation_agent.temperature")
        self.top_p = setting(settings, "models.explanation_agent.top_p")
        self.max_tokens = setting(settings, "models.explanation_agent.max_tokens")
        self.max_attempts = setting(settings, "execution.max_attempts")
        self.backoff = setting(settings, "execution.retry_backoff_seconds")

    async def explain(self, package: dict) -> dict:
        user = prompts.EXPLANATION_INSTRUCTIONS + "\n\n" + render_evidence_package(package)

        async def attempt():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompts.EXPLANATION_SYSTEM}, {"role": "user", "content": user}],
                temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return validate_explanation(response.choices[0].message.content, package)
        return await _call_with_retries(attempt, self.max_attempts, self.backoff)


class RecommendationAgent:
    def __init__(self, settings: dict, client):
        self.client = client
        self.model = setting(settings, "models.recommendation_agent.model_id")
        self.temperature = setting(settings, "models.recommendation_agent.temperature")
        self.top_p = setting(settings, "models.recommendation_agent.top_p")
        self.max_tokens = setting(settings, "models.recommendation_agent.max_tokens")
        self.max_attempts = setting(settings, "execution.max_attempts")
        self.backoff = setting(settings, "execution.retry_backoff_seconds")

    async def recommend(self, package: dict, explanation: str) -> dict:
        base = render_evidence_package(package)
        user = prompts.RECOMMENDATION_INSTRUCTIONS + "\n\n" + base + "\n\nFrozen explanation:\n" + explanation

        async def attempt():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompts.RECOMMENDATION_SYSTEM}, {"role": "user", "content": user}],
                temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return validate_recommendation(response.choices[0].message.content, package)
        return await _call_with_retries(attempt, self.max_attempts, self.backoff)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_single_query(
    raw_text: str,
    settings: dict,
    session,
    embedding_model,
    train_lookup: dict[int, dict],
    cue_definitions: dict[str, dict],
    openai_client,
    deepseek_client,
) -> dict:
    binary_detector = BinaryDetector(settings, openai_client)
    category_classifier = CategoryClassifier(settings, openai_client)
    explanation_agent = ExplanationAgent(settings, deepseek_client)
    recommendation_agent = RecommendationAgent(settings, deepseek_client)
    retriever = Retriever(settings, session)

    is_satd = await binary_detector.classify(raw_text)
    label = await category_classifier.classify(raw_text) if is_satd else "non_satd"

    embedding = embedding_model.encode(
        [raw_text], convert_to_numpy=True, normalize_embeddings=setting(settings, "embedding.normalize"),
    )[0]
    fulltext_hits = retriever.fulltext_retrieve(raw_text)
    vector_hits = retriever.vector_retrieve(embedding)
    fused = retriever.rrf(fulltext_hits, vector_hits)
    strengths, _, artifact_cues, artifact_weights = retriever.load_cue_layer()
    reranked = retriever.cue_rerank(fused, raw_text, strengths, artifact_cues, artifact_weights)
    ranked_ids = [sample_id for sample_id, _ in reranked]

    package = retriever.build_evidence_package(raw_text, label, ranked_ids, train_lookup, cue_definitions)
    explanation = await explanation_agent.explain(package)
    recommendation = None
    if label != "non_satd":
        recommendation = await recommendation_agent.recommend(package, explanation["explanation"])

    return {
        "raw_text": raw_text, "is_satd": is_satd, "classification": label,
        "ranked_evidence_ids": ranked_ids, "evidence_package": package,
        "explanation": explanation, "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix_order": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[False, True], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "non_satd": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
        "satd": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
    }


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_build_graph(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    load_env(args.env_file)
    conn = connect_db(args.db, read_only=True)
    driver = _neo4j_driver(settings)
    try:
        with driver.session(database=_neo4j_database()) as session:
            builder = GraphBuilder(settings)
            builder.create_schema(session)
            builder.build_artifact_graph(session, conn)
            builder.build_embeddings(session, conn)
            builder.build_cue_layer(session, conn)
    finally:
        driver.close()
        conn.close()
    print("build-graph complete")


def _make_openai_client():
    from openai import AsyncOpenAI
    creds = require_env("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=creds["OPENAI_API_KEY"])


def _make_deepseek_client(settings: dict):
    from openai import AsyncOpenAI
    creds = require_env("DEEPSEEK_API_KEY")
    base_url = setting(settings, "models.explanation_agent.base_url")
    return AsyncOpenAI(api_key=creds["DEEPSEEK_API_KEY"], base_url=base_url)


def _load_queries(args: argparse.Namespace) -> list[dict]:
    if args.query is not None:
        return [{"id": "query-1", "raw_text": args.query}]
    lines = Path(args.queries_file).read_text(encoding="utf-8").splitlines()
    queries = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        queries.append({"id": record.get("id", f"query-{i}"), "raw_text": record["raw_text"]})
    return queries


async def _run_queries(queries: list[dict], settings: dict, args: argparse.Namespace) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    load_env(args.env_file)
    conn = connect_db(args.db, read_only=True)
    train_lookup = load_train_lookup(conn, settings)
    driver = _neo4j_driver(settings)
    openai_client = _make_openai_client()
    deepseek_client = _make_deepseek_client(settings)
    embedding_model = SentenceTransformer(
        setting(settings, "embedding.model_name"),
        revision=setting(settings, "embedding.model_revision"),
    )
    results = []
    try:
        with driver.session(database=_neo4j_database()) as session:
            retriever = Retriever(settings, session)
            cue_definitions = retriever.load_cue_definitions()
            for query in queries:
                outcome = await run_single_query(
                    query["raw_text"], settings, session, embedding_model, train_lookup,
                    cue_definitions, openai_client, deepseek_client,
                )
                outcome["id"] = query["id"]
                results.append(outcome)
    finally:
        driver.close()
        conn.close()
    return results


def cmd_run(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    queries = _load_queries(args)
    results = asyncio.run(_run_queries(queries, settings, args))
    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    load_env(args.env_file)
    conn = connect_db(args.db, read_only=True)
    rows = load_split_rows(conn, settings, args.split)
    conn.close()

    queries = [{"id": str(row["sample_id"]), "raw_text": row["raw_text"]} for row in rows]
    ground_truth_binary = {str(row["sample_id"]): bool(row["ground_truth_binary"]) for row in rows}
    ground_truth_category = {str(row["sample_id"]): row["category_name"] for row in rows}

    results = asyncio.run(_run_queries(queries, settings, args))

    y_true_labels = [ground_truth_category[r["id"]] for r in results]
    y_pred_labels = [r["classification"] for r in results]
    y_true_binary = [ground_truth_binary[r["id"]] for r in results]
    y_pred_binary = [r["is_satd"] for r in results]

    metrics = {
        "six_class": compute_metrics(y_true_labels, y_pred_labels, setting(settings, "labels.six_class")),
        "binary": compute_binary_metrics(y_true_binary, y_pred_binary),
        "n": len(results),
    }
    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


def cmd_prepare_finetune_data(args: argparse.Namespace) -> None:
    """Builds the balanced binary fine-tuning JSONL from the Fold-2 TRAIN
    split. See README.md 'Reproducing the classifiers' for how the category
    fine-tuning data is constructed analogously."""
    settings = load_settings(args.settings)
    seed = setting(settings, "finetune_data.seed")
    conn = connect_db(args.db, read_only=True)
    rows = load_split_rows(conn, settings, setting(settings, "dataset.train_split"))
    conn.close()

    positives = [r for r in rows if int(r["ground_truth_binary"]) == 1]
    negatives = [r for r in rows if int(r["ground_truth_binary"]) == 0]
    target = len(positives)

    strata: defaultdict = defaultdict(list)
    for row in negatives:
        strata[(row["project_id"], row["source_type"])].append(row)
    quotas = _allocate_quotas(strata, target)
    rng = random.Random(seed)
    selected_negatives = []
    for key in sorted(strata):
        selected_negatives.extend(rng.sample(strata[key], quotas[key]))

    selected = positives + selected_negatives
    random.Random(seed).shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            record = {
                "messages": [
                    {"role": "system", "content": prompts.BINARY_SYSTEM},
                    {"role": "user", "content": row["raw_text"]},
                    {"role": "assistant", "content": json.dumps({"is_satd": bool(row["ground_truth_binary"])}, separators=(",", ":"))},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(selected), "seed": seed}, indent=2))


def _allocate_quotas(strata: dict, target: int) -> dict:
    """Minimum-one plus largest-remainder proportional allocation of the
    non-SATD sample across (project, source) strata."""
    keys = sorted(strata)
    if target < len(keys):
        raise RuntimeError("Target cannot cover every non-empty project/source stratum")
    quotas = {key: 1 for key in keys}
    remaining = target - len(keys)
    capacities = {key: len(strata[key]) - 1 for key in keys}
    capacity_total = sum(capacities.values())
    exact = {key: remaining * capacities[key] / capacity_total for key in keys}
    for key in keys:
        quotas[key] += min(capacities[key], int(exact[key]))
    left = target - sum(quotas.values())
    order = sorted(keys, key=lambda key: (-(exact[key] - int(exact[key])), key))
    while left:
        progressed = False
        for key in order:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                left -= 1
                progressed = True
                if not left:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate requested non-SATD sample")
    return quotas


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="pipeline.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--settings", type=Path, default=root / "frozen-settings.yaml")
        p.add_argument("--env-file", type=Path, default=root / ".env")
        p.add_argument("--db", type=Path, required=True, help="Path to the SQLite research database (see README.md)")

    build_graph = subparsers.add_parser("build-graph", help="Build the Neo4j evidence graph from the TRAINING split only")
    add_common(build_graph)
    build_graph.set_defaults(func=cmd_build_graph)

    run_parser = subparsers.add_parser("run", help="Run inference on a query or a file of queries")
    add_common(run_parser)
    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="A single artifact text to classify")
    group.add_argument("--queries-file", type=Path, help="JSONL file with {\"id\":..., \"raw_text\":...} per line")
    run_parser.add_argument("--output", type=Path, default=None, help="Write JSON results here instead of stdout")
    run_parser.set_defaults(func=cmd_run)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run inference over a val/test split and compute metrics")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--split", choices=["val", "test"], required=True)
    evaluate_parser.add_argument("--output", type=Path, default=None, help="Write JSON metrics here in addition to stdout")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    finetune_parser = subparsers.add_parser(
        "prepare-finetune-data",
        help="Build the balanced binary fine-tuning JSONL from the Fold-2 TRAIN split",
    )
    finetune_parser.add_argument("--settings", type=Path, default=root / "frozen-settings.yaml")
    finetune_parser.add_argument("--db", type=Path, required=True)
    finetune_parser.add_argument("--output", type=Path, required=True)
    finetune_parser.set_defaults(func=cmd_prepare_finetune_data)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
