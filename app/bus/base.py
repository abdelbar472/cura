from __future__ import annotations

from typing import Protocol

from app.schemas.common import AgentEvent


class EventBus(Protocol):
    backend: str
    degraded: bool

    async def publish(self, event: AgentEvent) -> None: ...

    async def close(self) -> None: ...
