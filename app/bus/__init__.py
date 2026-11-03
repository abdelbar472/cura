from __future__ import annotations

import logging

from app.bus.base import EventBus
from app.bus.kafka import KafkaEventBus
from app.bus.memory import MemoryEventBus
from app.config import Settings

logger = logging.getLogger(__name__)


async def create_event_bus(settings: Settings) -> EventBus:
    if not settings.kafka_enabled:
        return MemoryEventBus()

    bus = KafkaEventBus(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic_prefix=settings.kafka_topic_prefix,
        client_id=settings.kafka_client_id,
        max_block_ms=settings.kafka_max_block_ms,
    )
    try:
        await bus.start()
        logger.info("Connected to Kafka at %s", settings.kafka_bootstrap_servers)
        return bus
    except Exception as exc:
        logger.warning("Kafka unavailable (%s); falling back to in-memory bus", exc)
        return MemoryEventBus()


__all__ = ["EventBus", "KafkaEventBus", "MemoryEventBus", "create_event_bus"]
