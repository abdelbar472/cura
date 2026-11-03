from __future__ import annotations

from collections import defaultdict

from app.schemas.common import AgentEvent


class MemoryEventBus:
    """Default bus: keeps a bounded in-process event log."""

    backend = "memory"
    degraded = False

    def __init__(self, max_events_per_case: int = 500) -> None:
        self._max = max_events_per_case
        self._events: dict[str, list[AgentEvent]] = defaultdict(list)

    async def publish(self, event: AgentEvent) -> None:
        bucket = self._events[event.case_id]
        bucket.append(event)
        if len(bucket) > self._max:
            del bucket[: len(bucket) - self._max]

    def events_for(self, case_id: str) -> list[AgentEvent]:
        return list(self._events.get(case_id, []))

    async def close(self) -> None:
        return None
