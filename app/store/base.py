"""Case-store contract shared by the memory and Postgres backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.schemas.case import CaseResult
from app.schemas.common import AgentEvent


@dataclass
class StoredCase:
    result: CaseResult
    events: list[AgentEvent] = field(default_factory=list)


class CaseStore(Protocol):
    """Async persistence for completed cases and their agent events."""

    backend: str
    degraded: bool

    async def save(self, result: CaseResult, events: list[AgentEvent]) -> None: ...

    async def get(self, case_id: str) -> StoredCase | None: ...

    async def summaries(self, limit: int = 20) -> list[StoredCase]: ...

    async def close(self) -> None: ...
