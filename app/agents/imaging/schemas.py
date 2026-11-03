from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import AgentResultBase, RunMetadata, StrictModel

ImagingModality = Literal["xray", "ct", "cta", "mri", "ultrasound"]


class ImagingRequest(RunMetadata):
    modality: ImagingModality
    body_site: str = Field(..., min_length=1, max_length=64)
    clinical_question: str | None = Field(default=None, max_length=512)
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    image_base64: str | None = Field(
        default=None,
        max_length=16_000_000,
        description="Base64-encoded image bytes (optionally a data: URI).",
    )


class ImagingFinding(StrictModel):
    code: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImagingResult(AgentResultBase):
    agent: Literal["imaging"] = "imaging"
    modality: ImagingModality
    body_site: str
    backend: str = "null"
    findings: list[ImagingFinding] = Field(default_factory=list)
    evidence_codes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
