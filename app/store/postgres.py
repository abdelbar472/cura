"""Postgres-backed case store using SQLAlchemy's async API.

Schema creation is intentionally simple (``create_all``) because this is a
research prototype; a production deployment should use migrations. Case
payloads are stored as JSON so the agent response models remain the source of
truth.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import JSON, DateTime, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self.degraded = False

    async def connect(self) -> None:
        self._engine = create_async_engine(
            self._database_url,
            echo=self._echo,
            pool_pre_ping=True,
        )
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        logger.info("Connected to Postgres case store")

    def _sessions(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("PostgresCaseStore.connect() must be awaited first")
        return self._session_factory

    async def save(self, result: CaseResult, events: list[AgentEvent]) -> None:
        async with self._sessions()() as session:
            row = await session.get(CaseRow, result.case_id)
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

            await session.execute(delete(EventRow).where(EventRow.case_id == result.case_id))
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
            await session.commit()

    async def get(self, case_id: str) -> StoredCase | None:
        async with self._sessions()() as session:
            row = await session.get(CaseRow, case_id)
            if row is None:
                return None
            result = CaseResult.model_validate(row.payload)
            events_result = await session.execute(
                select(EventRow)
                .where(EventRow.case_id == case_id)
                .order_by(EventRow.produced_at, EventRow.event_id)
            )
            events = [
                AgentEvent.model_validate(event.payload)
                for event in events_result.scalars().all()
            ]
            return StoredCase(result=result, events=events)

    async def summaries(self, limit: int = 20) -> list[StoredCase]:
        async with self._sessions()() as session:
            rows = (
                await session.execute(
                    select(CaseRow).order_by(CaseRow.created_at.desc()).limit(limit)
                )
            ).scalars().all()
            return [
                StoredCase(result=CaseResult.model_validate(row.payload))
                for row in rows
            ]

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
