from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import AgentResultBase, PatientContext, RunMetadata, StrictModel


class DifferentialRequest(RunMetadata):
    evidence_codes: list[str] = Field(default_factory=list, max_length=200)
    patient: PatientContext = Field(default_factory=PatientContext)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceContribution(StrictModel):
    code: str
    weight: float


class ConditionConsideration(StrictModel):
    condition_id: str
    name: str
    score: float = Field(ge=0.0, le=1.0)
    ml_score: float | None = Field(default=None, ge=0.0, le=1.0)
    acuity: Literal["routine", "urgent", "emergency"]
    matched_evidence: list[str] = Field(default_factory=list)
    missing_key_evidence: list[str] = Field(default_factory=list)
    contributions: list[EvidenceContribution] = Field(default_factory=list)
    rationale: str


class DifferentialResult(AgentResultBase):
    agent: Literal["differential"] = "differential"
    method: str = "weighted-log-odds@1"
    evidence_used: list[str] = Field(default_factory=list)
    considerations: list[ConditionConsideration] = Field(default_factory=list)
    note: str = "Scores are heuristic log-odds, not calibrated probabilities."
