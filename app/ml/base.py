"""Shared building blocks for the per-agent ML models.

Every model is optional and lazily loaded. Agents keep their deterministic rule
path as the source of truth and use these models to add recall or re-rank; the
response envelope records which path was actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.schemas.common import ModelUsage


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    framework: str
    enabled: bool = True


class MLModel(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    @property
    def ready(self) -> bool: ...


def model_usage(
    info: ModelInfo,
    *,
    ready: bool,
    fallback: bool = False,
    detail: str | None = None,
) -> ModelUsage:
    """Build the small usage envelope embedded in every agent result."""
    return ModelUsage(
        name=info.name,
        version=info.version,
        framework=info.framework,
        ready=ready,
        fallback=fallback,
        detail=detail,
    )
