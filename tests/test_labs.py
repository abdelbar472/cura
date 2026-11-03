from __future__ import annotations

from app.agents.labs.agent import run_lab_agent
from app.agents.labs.schemas import LabRequest, LabValue


def test_interprets_high_and_critical_values():
    result = run_lab_agent(
        LabRequest(
            values=[
                LabValue(name="WBC", value=15.2, unit="10^9/L"),
                LabValue(name="lactate", value=5.1, unit="mmol/L"),
            ]
        )
    )
    by_name = {finding.name: finding for finding in result.findings}
    assert by_name["White blood cells"].flag == "high"
    assert by_name["Lactate"].flag == "critical_high"
    assert "E_LEUKOCYTOSIS" in result.evidence_codes
    assert "E_LACTATE_CRITICAL" in result.evidence_codes
    assert result.critical_flags == ["Lactate"]
    assert result.status == "ok"


def test_unknown_lab_is_reported_not_interpreted():
    result = run_lab_agent(LabRequest(values=[LabValue(name="mystery", value=3.0)]))
    assert result.findings[0].flag == "unknown"
    assert result.evidence_codes == []
    assert result.status == "partial"
    assert any("Unknown lab" in warning for warning in result.warnings)


def test_caller_supplied_range_is_used():
    result = run_lab_agent(
        LabRequest(
            values=[LabValue(name="custom marker", value=5.0, reference_low=1.0, reference_high=3.0)]
        )
    )
    finding = result.findings[0]
    assert finding.flag == "high"
    assert finding.evidence_code is None


def test_unit_mismatch_skips_interpretation():
    result = run_lab_agent(
        LabRequest(values=[LabValue(name="creatinine", value=1.0, unit="mmol/L")])
    )
    assert result.findings[0].flag == "unknown"
    assert any("Unit mismatch" in warning for warning in result.warnings)


def test_input_hash_ignores_run_metadata():
    first = run_lab_agent(
        LabRequest(values=[LabValue(name="WBC", value=7.0, unit="10^9/L")], run_id="run_a")
    )
    second = run_lab_agent(
        LabRequest(values=[LabValue(name="WBC", value=7.0, unit="10^9/L")], run_id="run_b")
    )
    assert first.input_hash == second.input_hash
    assert first.run_id != second.run_id
