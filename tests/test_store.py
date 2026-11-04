"""Tests for the async case-store contract, eviction, and Postgres backend."""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import delete

from app.agents.orchestrator.graph import CaseOrchestrator, CaseRun
from app.bus.memory import MemoryEventBus
from app.config import Settings
from app.schemas.case import CaseRequest
from app.store import create_case_store
from app.store.memory import MemoryCaseStore
from app.store.postgres import CaseRow, EventRow, PostgresCaseStore

TEST_DATABASE_URL = os.getenv("CURA_TEST_DATABASE_URL")


async def _run_case(text: str = "Dysuria, fever and urinary frequency.") -> CaseRun:
    settings = Settings(
        environment="test",
        store_backend="memory",
        ml_enabled=False,
        kafka_enabled=False,
    )
    orchestrator = CaseOrchestrator(MemoryEventBus(), settings)
    return await orchestrator.run(
        CaseRequest(
            text=text,
            patient={"age_years": 34, "sex": "female"},
            labs=[{"name": "WBC", "value": 15.2, "unit": "10^9/L"}],
            medications=[{"name": "Ibuprofen"}, {"name": "Coumadin"}],
            client_reference="TEST-1",
        )
    )


@pytest.mark.asyncio
async def test_memory_store_roundtrip():
    run = await _run_case()
    store = MemoryCaseStore()

    await store.save(run.result, run.events)
    stored = await store.get(run.result.case_id)

    assert store.backend == "memory"
    assert store.degraded is False
    assert stored is not None
    assert stored.result.case_id == run.result.case_id
    assert stored.result.client_reference == "TEST-1"
    assert {event.agent for event in stored.events} == {
        "symptom",
        "labs",
        "differential",
        "drugs",
    }


@pytest.mark.asyncio
async def test_memory_store_returns_copies():
    run = await _run_case()
    store = MemoryCaseStore()
    await store.save(run.result, run.events)

    run.result.client_reference = "MUTATED"
    run.events.clear()

    stored = await store.get(run.result.case_id)
    assert stored is not None
    assert stored.result.client_reference == "TEST-1"
    assert stored.events


@pytest.mark.asyncio
async def test_memory_store_replaces_events_on_resave():
    run = await _run_case()
    store = MemoryCaseStore()

    await store.save(run.result, run.events)
    await store.save(run.result, run.events)

    stored = await store.get(run.result.case_id)
    assert stored is not None
    assert len(stored.events) == len(run.events)


@pytest.mark.asyncio
async def test_memory_store_evicts_oldest_and_orders_summaries():
    first = await _run_case("Dysuria and fever.")
    second = await _run_case("Cough and shortness of breath.")
    store = MemoryCaseStore(max_cases=1)

    await store.save(first.result, first.events)
    await store.save(second.result, second.events)

    assert await store.get(first.result.case_id) is None
    summaries = await store.summaries()
    assert [item.result.case_id for item in summaries] == [second.result.case_id]

    limited = await store.summaries(limit=1)
    assert len(limited) == 1
    await store.close()


@pytest.mark.asyncio
async def test_store_factory_uses_memory_when_configured():
    store = await create_case_store(
        Settings(environment="test", store_backend="memory", ml_enabled=False)
    )
    assert isinstance(store, MemoryCaseStore)
    assert store.backend == "memory"
    assert store.degraded is False


@pytest.mark.asyncio
async def test_store_factory_degrades_when_postgres_is_unreachable():
    store = await asyncio.wait_for(
        create_case_store(
            Settings(
                environment="test",
                store_backend="postgres",
                database_url=(
                    "postgresql+psycopg://cura:cura@127.0.0.1:1/cura"
                    "?connect_timeout=2"
                ),
                ml_enabled=False,
            )
        ),
        timeout=10,
    )
    assert isinstance(store, MemoryCaseStore)
    assert store.backend == "memory"
    assert store.degraded is True


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set CURA_TEST_DATABASE_URL to run the Postgres integration test",
)
@pytest.mark.asyncio
async def test_postgres_store_roundtrip_and_idempotent_save():
    assert TEST_DATABASE_URL is not None
    store = await create_case_store(
        Settings(
            environment="test",
            store_backend="postgres",
            database_url=TEST_DATABASE_URL,
            ml_enabled=False,
        )
    )
    assert isinstance(store, PostgresCaseStore)
    assert store.backend == "postgres"
    assert store.degraded is False

    run = await _run_case()
    case_id = run.result.case_id
    try:
        await store.save(run.result, run.events)
        stored = await store.get(case_id)
        assert stored is not None
        assert stored.result.case_id == case_id
        assert stored.result.triage.level == run.result.triage.level
        assert {event.agent for event in stored.events} == {
            "symptom",
            "labs",
            "differential",
            "drugs",
        }

        # Saving the same case again replaces its event rows rather than
        # duplicating them (event_id is the primary key).
        await store.save(run.result, run.events)
        again = await store.get(case_id)
        assert again is not None
        assert len(again.events) == len(run.events)

        summaries = await store.summaries(limit=10)
        assert any(item.result.case_id == case_id for item in summaries)
    finally:
        if store._engine is not None:
            with store._engine.begin() as connection:
                connection.execute(delete(EventRow).where(EventRow.case_id == case_id))
                connection.execute(delete(CaseRow).where(CaseRow.case_id == case_id))
        await store.close()


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set CURA_TEST_DATABASE_URL to run the Postgres integration test",
)
@pytest.mark.asyncio
async def test_postgres_store_returns_none_for_unknown_case():
    assert TEST_DATABASE_URL is not None
    store = PostgresCaseStore(TEST_DATABASE_URL)
    await store.connect()
    try:
        assert await store.get("case_missing") is None
    finally:
        await store.close()
