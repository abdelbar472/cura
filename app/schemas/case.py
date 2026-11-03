from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.agents.differential.schemas import DifferentialResult
from app.agents.drugs.schemas import DrugSafetyResult, Medication
from app.agents.imaging.schemas import ImagingRequest, ImagingResult
from app.agents.labs.schemas import LabResult, LabValue
from app.agents.symptom.schemas import SymptomResult
from app.schemas.common import AgentStatus, PatientContext, StrictModel
from app.schemas.triage import TriageAssessment, TriageLevel


class CaseRequest(StrictModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Free-text clinical presentation.",
    )
    patient: PatientContext = Field(default_factory=PatientContext)
    labs: list[LabValue] = Field(default_factory=list, max_length=100)
    medications: list[Medication] = Field(default_factory=list, max_length=100)
    imaging: ImagingRequest | None = None
    symptom_model_version: str = Field(default="symptom-ner@0.2.0", max_length=128)
    symptom_backend: Literal["auto", "lexicon", "scispacy"] = "auto"
    client_reference: str | None = Field(default=None, max_length=128)


class CaseResult(StrictModel):
    case_id: str
    run_id: str
    status: AgentStatus
    created_at: datetime
    client_reference: str | None = None
    triage: TriageAssessment
    symptom: SymptomResult
    labs: LabResult
    imaging: ImagingResult | None = None
    differential: DifferentialResult
    drug_safety: DrugSafetyResult
    evidence_codes: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class CaseSummary(StrictModel):
    case_id: str
    run_id: str
    status: AgentStatus
    created_at: datetime
    triage_level: TriageLevel
    red_flags: list[str] = Field(default_factory=list)
    top_consideration: str | None = None
    client_reference: str | None = None

    @classmethod
    def from_result(cls, result: CaseResult) -> "CaseSummary":
        top = result.differential.considerations[0] if result.differential.considerations else None
        return cls(
            case_id=result.case_id,
            run_id=result.run_id,
            status=result.status,
            created_at=result.created_at,
            triage_level=result.triage.level,
            red_flags=result.red_flags,
            top_consideration=top.name if top else None,
            client_reference=result.client_reference,
        )
