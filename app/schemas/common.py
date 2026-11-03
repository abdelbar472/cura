"""Shared request/response primitives used by every agent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

AgentStatus = Literal["ok", "partial", "unavailable", "error"]
Severity = Literal["info", "minor", "moderate", "major", "contraindicated"]

SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "contraindicated": 4,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def hash_payload(payload: BaseModel | dict[str, Any]) -> str:
    """Stable SHA-256 over a canonical JSON representation of an input."""
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    """Base for request models: reject unknown fields to surface typos early."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RunMetadata(StrictModel):
    """Optional correlation identifiers. Missing values are generated per run."""

    run_id: str | None = Field(default=None, max_length=128)
    case_id: str | None = Field(default=None, max_length=128)
    step_id: str | None = Field(default=None, max_length=256)


class ResolvedRun(BaseModel):
    run_id: str
    case_id: str
    step_id: str


def resolve_run(meta: RunMetadata, agent: str) -> ResolvedRun:
    run_id = meta.run_id or new_id("run")
    case_id = meta.case_id or new_id("case")
    step_id = meta.step_id or f"{run_id}:{agent}"
    return ResolvedRun(run_id=run_id, case_id=case_id, step_id=step_id)


class PatientContext(StrictModel):
    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    pregnant: bool | None = None


class ModelUsage(BaseModel):
    """Describes which model path an agent used for a given run."""

    name: str
    version: str
    framework: str
    ready: bool
    fallback: bool = False
    detail: str | None = None


class AgentResultBase(BaseModel):
    """Common envelope returned by every agent."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    case_id: str
    step_id: str
    status: AgentStatus = "ok"
    model_version: str
    input_hash: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    ml: ModelUsage | None = None


class AgentEvent(BaseModel):
    """Event published to the bus after each agent step."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    topic: str
    agent: str
    run_id: str
    case_id: str
    step_id: str
    status: AgentStatus
    produced_at: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
