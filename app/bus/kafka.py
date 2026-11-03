"""Optional Kafka transport for agent events (disabled by default)."""

from __future__ import annotations

import json
import logging

from app.schemas.common import AgentEvent

logger = logging.getLogger(__name__)


class KafkaEventBus:
    backend = "kafka"
    degraded = False

    def __init__(
        self,
        bootstrap_servers: str,
        topic_prefix: str,
        client_id: str,
        max_block_ms: int = 3000,
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._prefix = topic_prefix
        self._client_id = client_id
        self._max_block_ms = max_block_ms
        self._producer = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            client_id=self._client_id,
            acks="all",
            enable_idempotence=True,
            max_block_ms=self._max_block_ms,
        )
        await self._producer.start()

    def _topic(self, agent: str) -> str:
        return f"{self._prefix}.{agent}.result"

    async def publish(self, event: AgentEvent) -> None:
        if self._producer is None:
            self.degraded = True
            logger.warning("Kafka producer is not started; dropping event %s", event.event_id)
            return
        payload = json.dumps(event.model_dump(mode="json")).encode("utf-8")
        try:
            await self._producer.send_and_wait(
                self._topic(event.agent), value=payload, key=event.case_id.encode("utf-8")
            )
        except Exception as exc:  # pragma: no cover - broker dependent
            self.degraded = True
            logger.warning("Kafka publish failed for %s: %s", event.event_id, exc)

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
