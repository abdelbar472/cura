"""Tests for the IsolationForest lab-pattern model and its agent integration."""

from __future__ import annotations

import pytest

from app.agents.labs.agent import run_lab_agent
from app.agents.labs.schemas import LabRequest, LabValue
from app.ml.base import ModelInfo
from app.ml.lab_anomaly import LabAnomalyModel, LabAnomalyResult

NORMAL_PANEL = {
    "wbc": 7.0,
    "hemoglobin": 14.0,
    "platelets": 250.0,
    "crp": 1.0,
    "lactate": 1.0,
    "creatinine": 0.9,
    "egfr": 100.0,
    "sodium": 140.0,
    "potassium": 4.2,
    "glucose": 95.0,
    "troponin": 0.01,
    "ddimer": 0.2,
    "bnp": 30.0,
    "alt": 25.0,
    "bilirubin": 0.6,
    "inr": 1.0,
    "ph": 7.4,
    "lipase": 40.0,
}

ABNORMAL_PANEL = {
    "wbc": 22.0,
    "hemoglobin": 8.0,
    "platelets": 800.0,
    "crp": 180.0,
    "lactate": 5.0,
    "creatinine": 3.5,
    "egfr": 20.0,
    "sodium": 125.0,
    "potassium": 6.0,
    "glucose": 350.0,
    "troponin": 0.3,
    "ddimer": 4.0,
    "bnp": 900.0,
    "alt": 300.0,
    "bilirubin": 6.0,
    "inr": 3.5,
    "ph": 7.2,
    "lipase": 900.0,
}


@pytest.fixture(scope="module")
def anomaly_model(tmp_path_factory) -> LabAnomalyModel:
    artifact_dir = tmp_path_factory.mktemp("lab-anomaly")
    return LabAnomalyModel(artifact_dir=artifact_dir, enabled=True)


class FakeAnomalyModel:
    def __init__(self, result: LabAnomalyResult | None, *, enabled: bool = True) -> None:
        self._result = result
        self._enabled = enabled
        self.ready = True

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="lab-anomaly",
            version="lab-anomaly@test",
            framework="fake",
            enabled=self._enabled,
        )

    def score(self, values):
        return self._result


def test_model_separates_normal_and_abnormal_panels(anomaly_model: LabAnomalyModel):
    normal = anomaly_model.score(NORMAL_PANEL)
    abnormal = anomaly_model.score(ABNORMAL_PANEL)

    assert normal is not None and abnormal is not None
    assert anomaly_model.ready is True
    assert normal.is_anomaly is False
    assert abnormal.is_anomaly is True
    assert abnormal.score > normal.score
    assert normal.features_used == 18
    assert abnormal.features_used == 18


def test_model_only_counts_supplied_values(anomaly_model: LabAnomalyModel):
    result = anomaly_model.score({"wbc": 7.0, "crp": 1.0})
    assert result is not None
    assert result.features_used == 2


def test_empty_panel_and_disabled_model_return_none(tmp_path):
    assert anomaly_model_disabled(tmp_path).score(NORMAL_PANEL) is None
    assert LabAnomalyModel(artifact_dir=tmp_path).score({}) is None


def anomaly_model_disabled(tmp_path) -> LabAnomalyModel:
    return LabAnomalyModel(artifact_dir=tmp_path, enabled=False)


def test_agent_adds_anomaly_fields_without_changing_rule_flags():
    fake = FakeAnomalyModel(LabAnomalyResult(score=0.81, is_anomaly=True, decision=-0.4, features_used=2))
    result = run_lab_agent(
        LabRequest(
            values=[
                LabValue(name="WBC", value=15.2, unit="10^9/L"),
                LabValue(name="CRP", value=48.0, unit="mg/L"),
            ]
        ),
        anomaly_model=fake,
    )
    assert result.findings[0].flag == "high"
    assert "E_LEUKOCYTOSIS" in result.evidence_codes
    assert result.ml_anomaly is True
    assert result.ml_anomaly_score == 0.81
    assert result.ml_anomaly_features == 2
    assert result.ml is not None and result.ml.ready is True
    assert "lab-anomaly" in result.model_version


def test_agent_falls_back_when_anomaly_model_is_disabled():
    fake = FakeAnomalyModel(
        LabAnomalyResult(score=0.9, is_anomaly=True, decision=-1.0, features_used=2),
        enabled=False,
    )
    result = run_lab_agent(
        LabRequest(values=[LabValue(name="WBC", value=15.2, unit="10^9/L")]),
        anomaly_model=fake,
    )
    assert result.status == "ok"
    assert result.ml_anomaly is None
    assert result.ml_anomaly_score is None
    assert result.ml is not None and result.ml.fallback is True
    assert "lab-anomaly" not in result.model_version


def test_agent_falls_back_when_model_cannot_score():
    result = run_lab_agent(
        LabRequest(values=[LabValue(name="WBC", value=7.0, unit="10^9/L")]),
        anomaly_model=FakeAnomalyModel(None),
    )
    assert result.ml_anomaly_score is None
    assert result.ml is not None and result.ml.fallback is True
    assert "unavailable" in (result.ml.detail or "")
