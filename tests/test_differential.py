from __future__ import annotations

from app.agents.differential.agent import run_differential_agent
from app.agents.differential.schemas import DifferentialRequest
from app.schemas.common import PatientContext


def test_uti_is_top_for_classic_evidence():
    result = run_differential_agent(
        DifferentialRequest(
            evidence_codes=["E_DYSURIA", "E_FREQUENCY", "E_FEVER", "E_LEUKOCYTOSIS"],
            patient=PatientContext(age_years=28, sex="female"),
        )
    )
    assert result.considerations
    assert result.considerations[0].condition_id == "COND_UTI"
    assert result.considerations[0].score > 0.5
    assert result.considerations[0].contributions
    assert "E_DYSURIA" in result.considerations[0].matched_evidence


def test_chest_pain_raises_emergency_condition():
    result = run_differential_agent(
        DifferentialRequest(
            evidence_codes=["E_CHEST_PAIN", "E_SWEATING", "E_TROPONIN_ELEVATED"],
            patient=PatientContext(age_years=67, sex="male"),
        )
    )
    top = result.considerations[0]
    assert top.condition_id == "COND_ACS"
    assert top.acuity == "emergency"


def test_critical_troponin_supports_acs():
    result = run_differential_agent(
        DifferentialRequest(evidence_codes=["E_TROPONIN_CRITICAL"])
    )
    assert result.considerations[0].condition_id == "COND_ACS"


def test_no_matching_evidence_returns_empty_list_with_warning():
    result = run_differential_agent(DifferentialRequest(evidence_codes=["E_UNKNOWN"]))
    assert result.considerations == []
    assert result.status == "ok"
    assert result.warnings


def test_top_k_is_respected():
    result = run_differential_agent(
        DifferentialRequest(
            evidence_codes=["E_FEVER", "E_COUGH", "E_SOB", "E_CHEST_PAIN", "E_NAUSEA"],
            top_k=2,
        )
    )
    assert len(result.considerations) == 2
