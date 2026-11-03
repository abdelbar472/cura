from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import AgentResultBase, StrictModel, RunMetadata

LabFlag = Literal["normal", "low", "high", "critical_low", "critical_high", "unknown"]


class LabValue(StrictModel):
    name: str = Field(..., min_length=1, max_length=64)
    value: float
    unit: str | None = Field(default=None, max_length=32)
    reference_low: float | None = None
    reference_high: float | None = None
    collected_at: datetime | None = None


class LabRequest(RunMetadata):
    values: list[LabValue] = Field(default_factory=list, max_length=100)


class LabFinding(StrictModel):
    name: str
    raw_name: str
    value: float
    unit: str | None = None
    flag: LabFlag
    reference_low: float | None = None
    reference_high: float | None = None
    evidence_code: str | None = None


class LabResult(AgentResultBase):
    agent: Literal["labs"] = "labs"
    findings: list[LabFinding] = Field(default_factory=list)
    evidence_codes: list[str] = Field(default_factory=list)
    critical_flags: list[str] = Field(default_factory=list)
    ml_anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ml_anomaly: bool | None = None
    ml_anomaly_features: int | None = None
