"""Tests for the learned differential re-ranker and its rule-preserving blend."""

from __future__ import annotations

import pytest

from app.agents.differential.agent import run_differential_agent
from app.agents.differential.schemas import DifferentialRequest
from app.ml.base import ModelInfo
from app.ml.differential_logreg import DifferentialLogRegModel
from app.schemas.common import PatientContext

UTI_EVIDENCE = ["E_DYSURIA", "E_FREQUENCY", "E_FEVER", "E_LEUKOCYTOSIS"]
ACS_EVIDENCE = ["E_CHEST_PAIN", "E_SWEATING", "E_TROPONIN_ELEVATED"]


@pytest.fixture(scope="module")
def reranker(tmp_path_factory) -> DifferentialLogRegModel:
    artifact_dir = tmp_path_factory.mktemp("differential")
    return DifferentialLogRegModel(artifact_dir=artifact_dir, enabled=True)


class FakeReranker:
    def __init__(
        self,
        scores: dict[str, float] | None,
        *,
        ml_weight: float = 0.5,
        enabled: bool = True,
        ready: bool = True,
    ) -> None:
        self._scores = scores
        self.ml_weight = ml_weight
        self._enabled = enabled
        self.ready = ready

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="differential-reranker",
            version="differential-reranker@test",
            framework="fake",
            enabled=self._enabled,
        )

    def predict(self, evidence_codes):
        return self._scores


def test_reranker_ranks_classic_presentations(reranker: DifferentialLogRegModel):
    assert reranker.ready is False
    uti = reranker.predict(UTI_EVIDENCE)
    acs = reranker.predict(ACS_EVIDENCE)
    assert reranker.ready is True
    assert uti is not None and acs is not None
    assert max(uti, key=uti.get) == "COND_UTI"
    assert max(acs, key=acs.get) == "COND_ACS"
    assert uti["COND_UTI"] > 0.5
    assert acs["COND_ACS"] > 0.5
    assert reranker.predict([]) == {}


def test_agent_blends_rule_and_model_scores():
    request = DifferentialRequest(
        evidence_codes=UTI_EVIDENCE,
        patient=PatientContext(age_years=28, sex="female"),
    )
    rules_only = run_differential_agent(request)
    rule_score = next(
        item.score for item in rules_only.considerations if item.condition_id == "COND_UTI"
    )

    blended = run_differential_agent(
        request,
        reranker=FakeReranker({"COND_UTI": 1.0}, ml_weight=0.35),
    )
    top = blended.considerations[0]
    assert top.condition_id == "COND_UTI"
    assert top.ml_score == 1.0
    assert top.score == pytest.approx(0.65 * rule_score + 0.35 * 1.0, abs=1e-4)
    assert blended.method == "blended-weighted-logreg@1"
    assert blended.ml is not None and blended.ml.ready is True
    assert "differential-reranker@test" in blended.model_version
    assert "Blended score" in top.rationale


def test_ml_weight_is_configurable():
    request = DifferentialRequest(evidence_codes=UTI_EVIDENCE)
    result = run_differential_agent(
        request,
        reranker=FakeReranker({"COND_UTI": 1.0}, ml_weight=1.0),
    )
    top = result.considerations[0]
    assert top.condition_id == "COND_UTI"
    assert top.score == 1.0
    assert top.ml_score == 1.0


def test_agent_uses_real_reranker_end_to_end(reranker: DifferentialLogRegModel):
    result = run_differential_agent(
        DifferentialRequest(
            evidence_codes=UTI_EVIDENCE,
            patient=PatientContext(age_years=34, sex="female"),
        ),
        reranker=reranker,
    )
    assert result.considerations[0].condition_id == "COND_UTI"
    assert result.considerations[0].ml_score is not None
    assert result.method == "blended-weighted-logreg@1"
    assert result.ml is not None and result.ml.ready is True


def test_disabled_reranker_leaves_rule_output_unchanged():
    request = DifferentialRequest(evidence_codes=UTI_EVIDENCE)
    rules_only = run_differential_agent(request)
    result = run_differential_agent(
        request,
        reranker=FakeReranker({"COND_UTI": 1.0}, enabled=False),
    )
    assert result.method == "weighted-log-odds@1"
    assert result.considerations[0].score == rules_only.considerations[0].score
    assert all(item.ml_score is None for item in result.considerations)
    assert result.ml is not None and result.ml.fallback is True


def test_unavailable_reranker_falls_back_to_rules():
    result = run_differential_agent(
        DifferentialRequest(evidence_codes=UTI_EVIDENCE),
        reranker=FakeReranker(None, ready=False),
    )
    assert result.method == "weighted-log-odds@1"
    assert all(item.ml_score is None for item in result.considerations)
    assert result.ml is not None and result.ml.fallback is True
    assert "unavailable" in (result.ml.detail or "")
