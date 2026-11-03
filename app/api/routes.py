from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.agents.differential.agent import run_differential_agent
from app.agents.differential.schemas import DifferentialRequest, DifferentialResult
from app.agents.drugs.agent import run_drug_agent
from app.agents.drugs.schemas import DrugSafetyRequest, DrugSafetyResult
from app.agents.imaging.agent import run_imaging_agent
from app.agents.imaging.schemas import ImagingRequest, ImagingResult
from app.agents.labs.agent import run_lab_agent
from app.agents.labs.schemas import LabRequest, LabResult
from app.agents.orchestrator.graph import CaseOrchestrator
from app.agents.registry import agent_specs
from app.agents.symptom.agent import run_symptom_agent
from app.agents.symptom.schemas import SymptomRequest, SymptomResult
from app.api.deps import (
    get_bus,
    get_models,
    get_orchestrator,
    get_settings,
    get_store,
)
from app.bus.base import EventBus
from app.config import Settings
from app.knowledge.conditions import CONDITIONS
from app.ml.bundle import ModelBundle
from app.schemas.case import CaseRequest, CaseResult, CaseSummary
from app.schemas.common import AgentEvent
from app.store.base import CaseStore

router = APIRouter()


class ServiceInfo(BaseModel):
    name: str
    version: str
    environment: str
    docs: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    bus: str
    bus_degraded: bool
    store: str
    store_degraded: bool
    graph: str
    ml_enabled: bool
    models: dict[str, dict[str, Any]]
    agents: dict[str, str]
    disclaimer: str


class ConditionSummary(BaseModel):
    condition_id: str
    name: str
    acuity: str
    evidence_count: int


@router.get("/", response_model=ServiceInfo, tags=["meta"])
async def service_info(
    settings: Settings = Depends(get_settings),
) -> ServiceInfo:
    return ServiceInfo(
        name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        docs="/docs",
        disclaimer=settings.disclaimer,
    )


@router.get("/healthz", response_model=HealthResponse, tags=["meta"])
async def health(
    settings: Settings = Depends(get_settings),
    bus: EventBus = Depends(get_bus),
    store: CaseStore = Depends(get_store),
    orchestrator: CaseOrchestrator = Depends(get_orchestrator),
    models: ModelBundle = Depends(get_models),
) -> HealthResponse:
    degraded = bus.degraded or store.degraded
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.version,
        environment=settings.environment,
        bus=bus.backend,
        bus_degraded=bus.degraded,
        store=store.backend,
        store_degraded=store.degraded,
        graph=orchestrator.graph_backend,
        ml_enabled=models.enabled,
        models=models.health(),
        agents={spec.name: spec.status for spec in agent_specs()},
        disclaimer=settings.disclaimer,
    )


@router.get("/v1/agents", tags=["agents"])
async def list_agents():
    return agent_specs()


@router.get("/v1/knowledge/conditions", response_model=list[ConditionSummary], tags=["knowledge"])
async def list_conditions() -> list[ConditionSummary]:
    return [
        ConditionSummary(
            condition_id=rule.condition_id,
            name=rule.name,
            acuity=rule.acuity,
            evidence_count=len(rule.weights),
        )
        for rule in CONDITIONS
    ]


@router.post(
    "/v1/cases",
    response_model=CaseResult,
    status_code=status.HTTP_201_CREATED,
    tags=["cases"],
)
async def create_case(
    payload: CaseRequest,
    orchestrator: CaseOrchestrator = Depends(get_orchestrator),
    store: CaseStore = Depends(get_store),
) -> CaseResult:
    run = await orchestrator.run(payload)
    await store.save(run.result, run.events)
    return run.result


@router.get("/v1/cases", response_model=list[CaseSummary], tags=["cases"])
async def list_cases(
    limit: int = Query(default=20, ge=1, le=100),
    store: CaseStore = Depends(get_store),
) -> list[CaseSummary]:
    return [CaseSummary.from_result(item.result) for item in await store.summaries(limit)]


@router.get("/v1/cases/{case_id}", response_model=CaseResult, tags=["cases"])
async def get_case(
    case_id: str,
    store: CaseStore = Depends(get_store),
) -> CaseResult:
    stored = await store.get(case_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id!r} not found")
    return stored.result


@router.get("/v1/cases/{case_id}/events", response_model=list[AgentEvent], tags=["cases"])
async def get_case_events(
    case_id: str,
    store: CaseStore = Depends(get_store),
) -> list[AgentEvent]:
    stored = await store.get(case_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id!r} not found")
    return stored.events


@router.post("/v1/agents/symptom", response_model=SymptomResult, tags=["agents"])
async def symptom_agent(
    payload: SymptomRequest,
    models: ModelBundle = Depends(get_models),
) -> SymptomResult:
    semantic = models.symptom_semantic if models.enabled else None
    return await asyncio.to_thread(run_symptom_agent, payload, semantic)


@router.post("/v1/agents/labs", response_model=LabResult, tags=["agents"])
async def labs_agent(
    payload: LabRequest,
    models: ModelBundle = Depends(get_models),
) -> LabResult:
    anomaly = models.lab_anomaly if models.enabled else None
    return await asyncio.to_thread(run_lab_agent, payload, anomaly)


@router.post("/v1/agents/differential", response_model=DifferentialResult, tags=["agents"])
async def differential_agent(
    payload: DifferentialRequest,
    models: ModelBundle = Depends(get_models),
) -> DifferentialResult:
    reranker = models.differential if models.enabled else None
    return await asyncio.to_thread(run_differential_agent, payload, reranker)


@router.post("/v1/agents/drugs", response_model=DrugSafetyResult, tags=["agents"])
async def drugs_agent(
    payload: DrugSafetyRequest,
    models: ModelBundle = Depends(get_models),
) -> DrugSafetyResult:
    resolver = models.drug_resolver if models.enabled else None
    return await asyncio.to_thread(run_drug_agent, payload, resolver)


@router.post("/v1/agents/imaging", response_model=ImagingResult, tags=["agents"])
async def imaging_agent(
    payload: ImagingRequest,
    settings: Settings = Depends(get_settings),
    models: ModelBundle = Depends(get_models),
) -> ImagingResult:
    model = models.imaging if models.enabled else None
    return await asyncio.to_thread(
        run_imaging_agent, payload, settings.imaging_backend, model
    )
