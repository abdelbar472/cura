"""LangGraph-based orchestrator for the Cura agent pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.differential.agent import run_differential_agent
from app.agents.differential.schemas import DifferentialRequest
from app.agents.drugs.agent import run_drug_agent
from app.agents.drugs.schemas import DrugSafetyRequest
from app.agents.imaging.agent import run_imaging_agent
from app.agents.imaging.schemas import ImagingRequest
from app.agents.labs.agent import run_lab_agent
from app.agents.labs.schemas import LabRequest
from app.agents.orchestrator.state import CaseState
from app.agents.orchestrator.triage import assess
from app.agents.registry import failure_result
from app.agents.symptom.agent import run_symptom_agent
from app.agents.symptom.schemas import SymptomRequest
from app.bus.base import EventBus
from app.config import Settings
from app.ml.bundle import ModelBundle
from app.schemas.case import CaseRequest, CaseResult
from app.schemas.common import (
    AgentEvent,
    ResolvedRun,
    hash_payload,
    new_id,
    utcnow,
)

logger = logging.getLogger(__name__)


@dataclass
class CaseRun:
    result: CaseResult
    events: list[AgentEvent] = field(default_factory=list)


class CaseOrchestrator:
    def __init__(
        self,
        bus: EventBus,
        settings: Settings,
        models: ModelBundle | None = None,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._models = models
        self._graph = self._build_graph()

    @property
    def models(self) -> ModelBundle | None:
        return self._models

    def _ml(self) -> ModelBundle | None:
        return self._models if self._models is not None and self._models.enabled else None

    # -- graph ------------------------------------------------------------
    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("langgraph is not installed; using the sequential pipeline")
            return None

        builder = StateGraph(CaseState)
        builder.add_node("symptom", self._symptom_node)
        builder.add_node("labs", self._labs_node)
        builder.add_node("imaging", self._imaging_node)
        builder.add_node("differential", self._differential_node)
        builder.add_node("drugs", self._drugs_node)
        builder.add_edge(START, "symptom")
        builder.add_edge("symptom", "labs")
        builder.add_edge("labs", "imaging")
        builder.add_edge("imaging", "differential")
        builder.add_edge("differential", "drugs")
        builder.add_edge("drugs", END)
        return builder.compile()

    @property
    def graph_backend(self) -> str:
        return "langgraph" if self._graph is not None else "sequential"

    @staticmethod
    def _run_for(state: CaseState, agent: str) -> ResolvedRun:
        return ResolvedRun(
            run_id=state["run_id"],
            case_id=state["case_id"],
            step_id=f"{state['run_id']}:{agent}",
        )

    def _topic(self, agent: str) -> str:
        return f"{self._settings.kafka_topic_prefix}.{agent}.result"

    async def _emit(self, agent: str, run: ResolvedRun, result: Any) -> AgentEvent:
        event = AgentEvent(
            topic=self._topic(agent),
            agent=agent,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
            status=result.status,
            payload={"result": result.model_dump(mode="json")},
        )
        try:
            await self._bus.publish(event)
        except Exception as exc:  # pragma: no cover - transport dependent
            logger.warning("Event publish failed for %s: %s", agent, exc)
        return event

    async def _execute(
        self,
        agent: str,
        run: ResolvedRun,
        payload: Any,
        runner: Callable[[Any], Any],
        key: str,
    ) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(runner, payload)
        except Exception as exc:
            logger.exception("Agent %s failed", agent)
            result = failure_result(agent, run, hash_payload(payload), exc)

        event = await self._emit(agent, run, result)
        updates: dict[str, Any] = {key: result, "events": [event]}
        if result.status != "ok":
            updates["warnings"] = [f"{agent}: status={result.status}"]
        if result.status in {"error", "unavailable"}:
            updates["failed_agents"] = [agent]
        if result.warnings:
            updates.setdefault("warnings", []).extend(f"{agent}: {w}" for w in result.warnings)
        return updates

    # -- nodes ------------------------------------------------------------
    async def _symptom_node(self, state: CaseState) -> dict[str, Any]:
        request = state["request"]
        run = self._run_for(state, "symptom")
        payload = SymptomRequest(
            text=request.text,
            model_version=request.symptom_model_version,
            model_name=self._settings.symptom_model,
            backend=request.symptom_backend,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
        )
        ml = self._ml()
        return await self._execute(
            "symptom",
            run,
            payload,
            lambda item: run_symptom_agent(
                item, semantic=ml.symptom_semantic if ml else None
            ),
            "symptom",
        )

    async def _labs_node(self, state: CaseState) -> dict[str, Any]:
        request = state["request"]
        run = self._run_for(state, "labs")
        payload = LabRequest(
            values=request.labs,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
        )
        ml = self._ml()
        return await self._execute(
            "labs",
            run,
            payload,
            lambda item: run_lab_agent(
                item, anomaly_model=ml.lab_anomaly if ml else None
            ),
            "labs",
        )

    async def _imaging_node(self, state: CaseState) -> dict[str, Any]:
        request = state["request"]
        if request.imaging is None:
            return {}
        run = self._run_for(state, "imaging")
        payload = ImagingRequest(
            modality=request.imaging.modality,
            body_site=request.imaging.body_site,
            clinical_question=request.imaging.clinical_question,
            image_sha256=request.imaging.image_sha256,
            image_base64=request.imaging.image_base64,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
        )
        backend = self._settings.imaging_backend
        ml = self._ml()
        return await self._execute(
            "imaging",
            run,
            payload,
            lambda item: run_imaging_agent(
                item, backend=backend, model=ml.imaging if ml else None
            ),
            "imaging",
        )

    async def _differential_node(self, state: CaseState) -> dict[str, Any]:
        request = state["request"]
        run = self._run_for(state, "differential")
        evidence: list[str] = []
        for key in ("symptom", "labs", "imaging"):
            result = state.get(key)
            for code in getattr(result, "evidence_codes", []) or []:
                if code not in evidence:
                    evidence.append(code)
        payload = DifferentialRequest(
            evidence_codes=evidence,
            patient=request.patient,
            top_k=self._settings.differential_top_k,
            min_score=self._settings.differential_min_score,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
        )
        ml = self._ml()
        return await self._execute(
            "differential",
            run,
            payload,
            lambda item: run_differential_agent(
                item, reranker=ml.differential if ml else None
            ),
            "differential",
        )

    async def _drugs_node(self, state: CaseState) -> dict[str, Any]:
        request = state["request"]
        run = self._run_for(state, "drugs")
        payload = DrugSafetyRequest(
            medications=request.medications,
            patient=request.patient,
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
        )
        ml = self._ml()
        return await self._execute(
            "drugs",
            run,
            payload,
            lambda item: run_drug_agent(
                item, resolver=ml.drug_resolver if ml else None
            ),
            "drugs",
        )

    # -- public -----------------------------------------------------------
    async def run(self, request: CaseRequest) -> CaseRun:
        state: CaseState = {
            "request": request,
            "run_id": new_id("run"),
            "case_id": new_id("case"),
            "events": [],
            "warnings": [],
            "failed_agents": [],
        }

        if self._graph is not None:
            final_state: CaseState = await self._graph.ainvoke(state)
        else:
            final_state = state
            nodes = (
                self._symptom_node,
                self._labs_node,
                self._imaging_node,
                self._differential_node,
                self._drugs_node,
            )
            for node in nodes:
                updates = await node(final_state)
                for key, value in updates.items():
                    if key in {"events", "warnings", "failed_agents"}:
                        final_state.setdefault(key, []).extend(value)
                    else:
                        final_state[key] = value

        result = self._finalize(final_state, request)
        return CaseRun(result=result, events=list(final_state.get("events", [])))

    def _finalize(self, state: CaseState, request: CaseRequest) -> CaseResult:
        symptom = state["symptom"]
        labs = state["labs"]
        imaging = state.get("imaging")
        differential = state["differential"]
        drugs = state["drugs"]

        evidence: list[str] = []
        sources = [symptom.evidence_codes, labs.evidence_codes]
        if imaging is not None:
            sources.append(imaging.evidence_codes)
        for codes in sources:
            for code in codes:
                if code not in evidence:
                    evidence.append(code)

        triage = assess(symptom, labs, differential, drugs)

        results = [symptom, labs, differential, drugs]
        if imaging is not None:
            results.append(imaging)
        if symptom.status == "error":
            status = "error"
        elif any(item.status != "ok" for item in results):
            status = "partial"
        else:
            status = "ok"

        warnings = list(dict.fromkeys(state.get("warnings", [])))
        return CaseResult(
            case_id=state["case_id"],
            run_id=state["run_id"],
            status=status,
            created_at=utcnow(),
            client_reference=request.client_reference,
            triage=triage,
            symptom=symptom,
            labs=labs,
            imaging=imaging,
            differential=differential,
            drug_safety=drugs,
            evidence_codes=evidence,
            red_flags=list(symptom.red_flags),
            warnings=warnings,
            disclaimer=self._settings.disclaimer,
        )
