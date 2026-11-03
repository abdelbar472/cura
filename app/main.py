"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.orchestrator.graph import CaseOrchestrator
from app.api.routes import router
from app.bus import create_event_bus
from app.config import Settings, get_settings
from app.ml.bundle import ModelBundle
from app.store import create_case_store

logger = logging.getLogger("cura")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    bus = await create_event_bus(settings)
    store = await create_case_store(settings)
    models = ModelBundle.create(settings)
    app.state.bus = bus
    app.state.store = store
    app.state.models = models
    app.state.orchestrator = CaseOrchestrator(bus, settings, models=models)
    logger.info(
        "Cura %s ready (env=%s, bus=%s, store=%s, graph=%s, ml=%s)",
        settings.version,
        settings.environment,
        bus.backend,
        store.backend,
        app.state.orchestrator.graph_backend,
        "on" if models.enabled else "off",
    )
    try:
        yield
    finally:
        await store.close()
        await bus.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.version,
        description=(
            "Modular clinical decision-support backend: symptom extraction, lab "
            "interpretation, differential diagnosis, drug-safety review, and "
            "chest X-ray inference, sequenced by a LangGraph orchestrator.\n\n"
            f"**{settings.disclaimer}**"
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
