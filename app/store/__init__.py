"""Case-store factory with a graceful in-memory fallback."""

from __future__ import annotations

import logging

from app.config import Settings
from app.store.base import CaseStore, StoredCase
from app.store.memory import MemoryCaseStore
from app.store.postgres import PostgresCaseStore

logger = logging.getLogger(__name__)


async def create_case_store(settings: Settings) -> CaseStore:
    if settings.store_backend == "memory":
        return MemoryCaseStore(max_cases=settings.store_max_cases)

    store = PostgresCaseStore(settings.database_url, echo=settings.database_echo)
    try:
        await store.connect()
        return store
    except Exception as exc:
        logger.warning(
            "Postgres unavailable (%s); falling back to the in-memory case store",
            exc,
        )
        return MemoryCaseStore(max_cases=settings.store_max_cases, degraded=True)


__all__ = [
    "CaseStore",
    "MemoryCaseStore",
    "PostgresCaseStore",
    "StoredCase",
    "create_case_store",
]
