"""Agent metadata shared by the registry and the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str
    model_version: str
    status: Literal["ready", "experimental", "unavailable"]
    tags: list[str] = Field(default_factory=list)
