"""Triage assessment shared by the orchestrator and the case API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TriageLevel = Literal["routine", "urgent", "emergency"]

LEVEL_ORDER: dict[str, int] = {"routine": 0, "urgent": 1, "emergency": 2}

ACTIONS: dict[str, str] = {
    "routine": "Arrange a routine appointment with a qualified clinician.",
    "urgent": "Seek same-day clinical assessment (urgent care or primary care).",
    "emergency": "Seek emergency medical evaluation immediately.",
}


class TriageAssessment(BaseModel):
    level: TriageLevel
    action: str
    reasons: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
