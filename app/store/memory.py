"""Bounded in-memory case store (oldest entries are evicted)."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from app.schemas.case import CaseResult
from app.schemas.common import AgentEvent
from app.store.base import StoredCase


class MemoryCaseStore:
    backend = "memory"

    def __init__(self, max_cases: int = 1000, degraded: bool = False) -> None:
        self._max = max_cases
        self._cases: OrderedDict[str, StoredCase] = OrderedDict()
        self._lock = asyncio.Lock()
        self.degraded = degraded

    async def save(self, result: CaseResult, events: list[AgentEvent]) -> None:
        async with self._lock:
            self._cases[result.case_id] = StoredCase(
                result=result.model_copy(deep=True),
                events=[event.model_copy(deep=True) for event in events],
            )
            self._cases.move_to_end(result.case_id)
            while len(self._cases) > self._max:
                self._cases.popitem(last=False)

    async def get(self, case_id: str) -> StoredCase | None:
        async with self._lock:
            stored = self._cases.get(case_id)
            if stored is None:
                return None
            return StoredCase(
                result=stored.result.model_copy(deep=True),
                events=[event.model_copy(deep=True) for event in stored.events],
            )

    async def summaries(self, limit: int = 20) -> list[StoredCase]:
        async with self._lock:
            items = list(self._cases.values())[-limit:]
            return list(reversed(items))

    async def close(self) -> None:
        return None


# Backwards-compatible name used by older imports.
CaseStore = MemoryCaseStore
