from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import AgentResultBase, RunMetadata


class SymptomRequest(RunMetadata):
    text: str = Field(..., min_length=1, max_length=8000)
    model_version: str = Field(default="symptom-ner@0.2.0", max_length=128)
    model_name: str = Field(default="en_core_sci_sm", max_length=64)
    backend: Literal["auto", "lexicon", "scispacy"] = "auto"


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    code: str | None = None
    negated: bool = False
    source: Literal["lexicon", "scispacy", "ml"] = "lexicon"
    score: float | None = None


class SymptomResult(AgentResultBase):
    agent: Literal["symptom"] = "symptom"
    backend: str = "lexicon"
    symptoms: list[str] = Field(default_factory=list)
    negated_symptoms: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    evidence_codes: list[str] = Field(default_factory=list)
    negated_codes: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
