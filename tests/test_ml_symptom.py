"""Tests for the embedding-backed symptom matcher and its agent integration."""

from __future__ import annotations

import numpy as np

from app.agents.symptom.agent import run_symptom_agent
from app.agents.symptom.schemas import SymptomRequest
from app.knowledge.symptoms import SYMPTOM_CONCEPTS, concept_description
from app.ml.base import ModelInfo
from app.ml.symptom_semantic import SemanticMatch, SymptomSemanticModel

_PROMPT = "It stings down there when I go."
_DESCRIPTIONS = [concept_description(concept) for concept in SYMPTOM_CONCEPTS]
_CODES = [concept.code for concept in SYMPTOM_CONCEPTS]


class FakeEmbeddings:
    """Deterministic stand-in for MiniLM: no download, no torch."""

    name = "fake-embeddings"

    @property
    def ready(self) -> bool:
        return True

    def encode(self, texts):
        texts = list(texts)
        if texts == _DESCRIPTIONS:
            return np.eye(len(_DESCRIPTIONS), dtype=np.float32)
        rows = np.zeros((len(texts), len(_DESCRIPTIONS)), dtype=np.float32)
        for index, text in enumerate(texts):
            if "stings" in text.lower():
                rows[index, _CODES.index("E_DYSURIA")] = 0.9
        return rows


class AmbiguousEmbeddings(FakeEmbeddings):
    """Two near-equal candidates so the margin rule has something to reject."""

    def encode(self, texts):
        texts = list(texts)
        if texts == _DESCRIPTIONS:
            return np.eye(len(_DESCRIPTIONS), dtype=np.float32)
        rows = np.zeros((len(texts), len(_DESCRIPTIONS)), dtype=np.float32)
        rows[:, _CODES.index("E_DYSURIA")] = 0.70
        rows[:, _CODES.index("E_FEVER")] = 0.68
        return rows


class LowConfidenceEmbeddings(FakeEmbeddings):
    def encode(self, texts):
        texts = list(texts)
        if texts == _DESCRIPTIONS:
            return np.eye(len(_DESCRIPTIONS), dtype=np.float32)
        return np.full((len(texts), len(_DESCRIPTIONS)), 0.10, dtype=np.float32)


class FakeSemantic:
    """Minimal semantic model object for agent-level tests."""

    def __init__(self, *, enabled: bool = True, ready: bool = True) -> None:
        self.info = ModelInfo(
            name="symptom-semantic",
            version="symptom-semantic@test",
            framework="fake",
            enabled=enabled,
        )
        self.ready = ready
        self.enabled = enabled
        self._prompt = _PROMPT

    def match(self, text: str):
        if not self.enabled or not self.ready:
            return []
        start = text.lower().find(self._prompt.lower())
        if start < 0:
            return []
        return [
            SemanticMatch(
                code="E_DYSURIA",
                text=self._prompt,
                start=start,
                end=start + len(self._prompt),
                score=0.91,
                runner_up=0.12,
            )
        ]


def test_semantic_model_adds_a_concept_for_a_lay_phrase():
    model = SymptomSemanticModel(FakeEmbeddings())
    assert model.ready is False
    matches = model.match(_PROMPT)
    assert model.ready is True
    assert [match.code for match in matches] == ["E_DYSURIA"]
    assert matches[0].score == 0.9
    assert matches[0].text == _PROMPT.rstrip(".")


def test_semantic_model_requires_confidence_and_runner_up_margin():
    assert SymptomSemanticModel(AmbiguousEmbeddings()).match(_PROMPT) == []
    assert SymptomSemanticModel(LowConfidenceEmbeddings()).match(_PROMPT) == []


def test_disabled_semantic_model_is_inert():
    model = SymptomSemanticModel(FakeEmbeddings(), enabled=False)
    assert model.ready is False
    assert model.match(_PROMPT) == []


def test_agent_uses_semantic_match_when_lexicon_misses():
    result = run_symptom_agent(
        SymptomRequest(text=_PROMPT),
        semantic=FakeSemantic(),
    )
    assert result.evidence_codes == ["E_DYSURIA"]
    assert result.ml is not None and result.ml.ready is True
    assert result.ml.fallback is False
    entity = result.entities[-1]
    assert entity.source == "ml"
    assert entity.code == "E_DYSURIA"
    assert entity.score == 0.91


def test_agent_does_not_duplicate_codes_already_found_by_lexicon():
    result = run_symptom_agent(
        SymptomRequest(text="Dysuria. It stings down there when I go."),
        semantic=FakeSemantic(),
    )
    assert result.evidence_codes.count("E_DYSURIA") == 1
    assert all(entity.source != "ml" for entity in result.entities)


def test_agent_respects_negation_for_semantic_matches():
    text = f"Denies {_PROMPT.lower()}"
    result = run_symptom_agent(SymptomRequest(text=text), semantic=FakeSemantic())
    assert result.evidence_codes == []
    assert result.negated_codes == ["E_DYSURIA"]


def test_agent_falls_back_when_semantic_model_is_disabled():
    result = run_symptom_agent(
        SymptomRequest(text=_PROMPT),
        semantic=FakeSemantic(enabled=False),
    )
    assert result.evidence_codes == []
    assert result.ml is not None
    assert result.ml.ready is False
    assert result.ml.fallback is True


def test_agent_reports_unavailable_semantic_model_without_failing():
    result = run_symptom_agent(
        SymptomRequest(text=_PROMPT),
        semantic=FakeSemantic(ready=False),
    )
    assert result.status == "ok"
    assert result.ml is not None
    assert result.ml.ready is False
    assert "unavailable" in (result.ml.detail or "")
