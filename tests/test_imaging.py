from __future__ import annotations

from app.agents.imaging.agent import run_imaging_agent
from app.agents.imaging.schemas import ImagingRequest


def test_null_backend_reports_unavailable_without_fabricating_findings():
    result = run_imaging_agent(ImagingRequest(modality="xray", body_site="chest"))
    assert result.status == "unavailable"
    assert result.findings == []
    assert result.evidence_codes == []
    assert result.backend == "null"
    assert result.notes
