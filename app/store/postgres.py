"""Postgres-backed case store.

The store exposes an async interface, but uses SQLAlchemy's synchronous engine
inside worker threads. That keeps it usable under any asyncio loop policy: on
Windows the default Proactor loop cannot run psycopg's async driver, and uvicorn
creates its loop before importing the application.

Schema creation is intentionally simple (``create_all``) because this is a
research prototype; a production deployment should use migrations. Case
payloads are stored as JSON so the agent response models remain the source of
truth.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import JSON, DateTime, String, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.schemas.case import CaseResult
from app.schemas.common import AgentEvent
from app.store.base import StoredCase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), index=True)
    triage_level: Mapped[str] = mapped_column(String(32))
    client_reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    red_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class EventRow(Base):
    __tablename__ = "case_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(128), index=True)
    agent: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    produced_at: Mapped[Any] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class PostgresCaseStore:
    backend = "postgres"

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self._database_url = database_url
        self._echo = echo
        self._engine = None
        self._session_factory: sessionmaker[Session] | None = None
        self.degraded = False

    def _connect_sync(self) -> None:
        engine = create_engine(
            self._database_url,
            echo=self._echo,
            pool_pre_ping=True,
        )
        try:
            Base.metadata.create_all(engine)
        except Exception:
            engine.dispose()
            raise
        self._engine = engine
        self._session_factory = sessionmaker(engine, expire_on_commit=False)
        logger.info("Connected to Postgres case store")

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect_sync)

    def _sessions(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            raise RuntimeError("PostgresCaseStore.connect() must be awaited first")
        return self._session_factory

    def _save_sync(self, result: CaseResult, events: list[AgentEvent]) -> None:
        with self._sessions()() as session:
            row = session.get(CaseRow, result.case_id)
            payload = result.model_dump(mode="json")
            if row is None:
                session.add(
                    CaseRow(
                        case_id=result.case_id,
                        run_id=result.run_id,
                        status=result.status,
                        created_at=result.created_at,
                        triage_level=result.triage.level,
                        client_reference=result.client_reference,
                        red_flags=list(result.red_flags),
                        payload=payload,
                    )
                )
            else:
                row.run_id = result.run_id
                row.status = result.status
                row.created_at = result.created_at
                row.triage_level = result.triage.level
                row.client_reference = result.client_reference
                row.red_flags = list(result.red_flags)
                row.payload = payload

            session.execute(delete(EventRow).where(EventRow.case_id == result.case_id))
            session.add_all(
                [
                    EventRow(
                        event_id=event.event_id,
                        case_id=event.case_id,
                        agent=event.agent,
                        status=event.status,
                        produced_at=event.produced_at,
                        payload=event.model_dump(mode="json"),
                    )
                    for event in events
                ]
            )
            session.commit()

    async def save(self, result: CaseResult, events: list[AgentEvent]) -> None:
        await asyncio.to_thread(self._save_sync, result, events)

    def _get_sync(self, case_id: str) -> StoredCase | None:
        with self._sessions()() as session:
            row = session.get(CaseRow, case_id)
            if row is None:
                return None
            result = CaseResult.model_validate(row.payload)
            events_result = session.execute(
                select(EventRow)
                .where(EventRow.case_id == case_id)
                .order_by(EventRow.produced_at, EventRow.event_id)
            )
            events = [
                AgentEvent.model_validate(event.payload)
                for event in events_result.scalars().all()
            ]
            return StoredCase(result=result, events=events)

    async def get(self, case_id: str) -> StoredCase | None:
        return await asyncio.to_thread(self._get_sync, case_id)

    def _summaries_sync(self, limit: int) -> list[StoredCase]:
        with self._sessions()() as session:
            rows = (
                session.execute(
                    select(CaseRow).order_by(CaseRow.created_at.desc()).limit(limit)
                )
                .scalars()
                .all()
            )
            return [
                StoredCase(result=CaseResult.model_validate(row.payload))
                for row in rows
            ]

    async def summaries(self, limit: int = 20) -> list[StoredCase]:
        return await asyncio.to_thread(self._summaries_sync, limit)

    def _close_sync(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)
