"""Abstract SATD GraphRAG pipeline.

This file documents component order and interfaces only. It contains no model
calls, prompts, credentials, database connection, data, or research code.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol


BinaryLabel = Literal["satd", "non_satd"]
Category = Literal["design", "defect", "requirement", "documentation", "test"]


@dataclass(frozen=True)
class Prediction:
    binary: BinaryLabel
    category: Category | None = None


@dataclass(frozen=True)
class Evidence:
    reference: str
    text: str
    rank: int
    binary_side: BinaryLabel
    matching_cues: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvidencePackage:
    supportive: tuple[Evidence, ...]
    opposite: tuple[Evidence, ...]


@dataclass(frozen=True)
class Explanation:
    binary_assessment: Literal["supported", "challenged", "insufficient"]
    category_assessment: Literal[
        "supported", "challenged", "insufficient", "not_applicable"
    ]
    evidence_references: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Recommendation:
    priority: Literal["low", "medium", "high"]
    suggested_action: str
    rationale: str
    evidence_references: tuple[str, ...]


@dataclass(frozen=True)
class PipelineResult:
    prediction: Prediction
    evidence: EvidencePackage
    explanation: Explanation
    recommendation: Recommendation | None


class BinaryDetector(Protocol):
    def predict(self, raw_text: str) -> BinaryLabel: ...


class CategoryClassifier(Protocol):
    def predict(self, raw_text: str) -> Category: ...


class TrainingOnlyRetriever(Protocol):
    def retrieve(self, raw_text: str, prediction: Prediction) -> EvidencePackage: ...


class ExplanationAgent(Protocol):
    def explain(
        self, raw_text: str, prediction: Prediction, evidence: EvidencePackage
    ) -> Explanation: ...


class RecommendationAgent(Protocol):
    def recommend(
        self,
        raw_text: str,
        prediction: Prediction,
        evidence: EvidencePackage,
        explanation: Explanation,
    ) -> Recommendation: ...


class SATDGraphRAGPipeline:
    """Coordinates frozen components without implementing them."""

    def __init__(
        self,
        detector: BinaryDetector,
        category_classifier: CategoryClassifier,
        retriever: TrainingOnlyRetriever,
        explanation_agent: ExplanationAgent,
        recommendation_agent: RecommendationAgent,
    ) -> None:
        self.detector = detector
        self.category_classifier = category_classifier
        self.retriever = retriever
        self.explanation_agent = explanation_agent
        self.recommendation_agent = recommendation_agent

    def run(self, raw_text: str) -> PipelineResult:
        binary = self.detector.predict(raw_text)
        category = self.category_classifier.predict(raw_text) if binary == "satd" else None
        prediction = Prediction(binary=binary, category=category)

        evidence = self.retriever.retrieve(raw_text, prediction)
        explanation = self.explanation_agent.explain(raw_text, prediction, evidence)

        recommendation = None
        if binary == "satd":
            recommendation = self.recommendation_agent.recommend(
                raw_text, prediction, evidence, explanation
            )

        # Downstream agents provide context but cannot modify the prediction.
        return PipelineResult(prediction, evidence, explanation, recommendation)
