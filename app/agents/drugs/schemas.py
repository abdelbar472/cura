from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import AgentResultBase, PatientContext, RunMetadata, Severity, StrictModel

InteractionSeverity = Literal["minor", "moderate", "major", "contraindicated"]


class Medication(StrictModel):
    name: str = Field(..., min_length=1, max_length=128)
    dose: str | None = Field(default=None, max_length=64)
    route: str | None = Field(default=None, max_length=32)
    frequency: str | None = Field(default=None, max_length=64)


class DrugSafetyRequest(RunMetadata):
    medications: list[Medication] = Field(default_factory=list, max_length=100)
    patient: PatientContext = Field(default_factory=PatientContext)


class NormalizedMedication(StrictModel):
    input_name: str
    ingredient: str
    drug_class: str | None = None
    resolution: Literal["exact", "alias", "fuzzy", "embedding", "unresolved"] = "exact"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class InteractionFinding(StrictModel):
    drugs: list[str] = Field(min_length=2, max_length=2)
    severity: InteractionSeverity
    mechanism: str
    recommendation: str
    source: str


class DuplicateClassFinding(StrictModel):
    drug_class: str
    ingredients: list[str]


class DrugSafetyResult(AgentResultBase):
    agent: Literal["drugs"] = "drugs"
    normalized_medications: list[NormalizedMedication] = Field(default_factory=list)
    interactions: list[InteractionFinding] = Field(default_factory=list)
    duplicate_classes: list[DuplicateClassFinding] = Field(default_factory=list)
    unknown_medications: list[str] = Field(default_factory=list)
    overall_risk: Severity = "info"
