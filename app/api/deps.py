from __future__ import annotations

from fastapi import Request

from app.agents.orchestrator.graph import CaseOrchestrator
from app.bus.base import EventBus
from app.config import Settings
from app.ml.bundle import ModelBundle
from app.store.base import CaseStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_orchestrator(request: Request) -> CaseOrchestrator:
    return request.app.state.orchestrator


def get_models(request: Request) -> ModelBundle:
    return request.app.state.models


def get_store(request: Request) -> CaseStore:
    return request.app.state.store


def get_bus(request: Request) -> EventBus:
    return request.app.state.bus
