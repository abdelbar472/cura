"""Pluggable imaging model interface.

A real deployment would register a validated model here (X-ray first, CTA
later). The shipped default is still the null backend: producing imaging
"findings" without a model would be fabricated data. When an X-ray and a
configured model are both present, the HF chest-X-ray classifier is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.agents.imaging.schemas import ImagingFinding, ImagingRequest


@dataclass
class ImagingPrediction:
    findings: list[ImagingFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ran: bool = False


class ImagingModel(Protocol):
    name: str

    def predict(self, req: ImagingRequest) -> list[ImagingFinding]: ...


class NullImagingModel:
    name = "null"

    def predict(self, req: ImagingRequest) -> list[ImagingFinding]:
        return []

    def predict_detailed(self, req: ImagingRequest) -> ImagingPrediction:
        return ImagingPrediction()


MODEL_REGISTRY: dict[str, ImagingModel] = {"null": NullImagingModel()}


def get_model(backend: str) -> ImagingModel | None:
    return MODEL_REGISTRY.get(backend)
