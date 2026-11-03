"""Single place that lists the agents and builds degraded results on failure."""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentSpec
from app.agents.differential.agent import MODEL_VERSION as DIFFERENTIAL_VERSION
from app.agents.differential.schemas import DifferentialResult
from app.agents.drugs.agent import MODEL_VERSION as DRUGS_VERSION
from app.agents.drugs.schemas import DrugSafetyResult
from app.agents.imaging.agent import MODEL_VERSION as IMAGING_VERSION
from app.agents.imaging.schemas import ImagingResult
from app.agents.labs.agent import MODEL_VERSION as LABS_VERSION
from app.agents.labs.schemas import LabResult
from app.agents.symptom.agent import MODEL_VERSION as SYMPTOM_VERSION
from app.agents.symptom.schemas import SymptomResult
from app.schemas.common import ResolvedRun

_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="symptom",
        description="Extracts symptom concepts and red flags from free text.",
        model_version=SYMPTOM_VERSION,
        status="ready",
        tags=["nlp", "evidence"],
    ),
    AgentSpec(
        name="labs",
        description="Interprets laboratory values against reference ranges.",
        model_version=LABS_VERSION,
        status="ready",
        tags=["rules", "evidence"],
    ),
    AgentSpec(
        name="imaging",
        description="Chest X-ray classification via an optional HF ViT checkpoint; other modalities remain unavailable.",
        model_version=IMAGING_VERSION,
        status="experimental",
        tags=["vision", "chest-xray", "ml"],
    ),
    AgentSpec(
        name="differential",
        description="Ranks candidate conditions from combined evidence codes.",
        model_version=DIFFERENTIAL_VERSION,
        status="ready",
        tags=["rules", "knowledge-base"],
    ),
    AgentSpec(
        name="drugs",
        description="Checks curated drug-drug interactions and duplicate therapy.",
        model_version=DRUGS_VERSION,
        status="ready",
        tags=["rules", "medication-safety"],
    ),
    AgentSpec(
        name="orchestrator",
        description="Sequences the agents with LangGraph and publishes events.",
        model_version="orchestrator@0.2.0",
        status="ready",
        tags=["langgraph", "pipeline"],
    ),
)

AGENT_SPECS: dict[str, AgentSpec] = {spec.name: spec for spec in _SPECS}


def agent_specs() -> list[AgentSpec]:
    return list(_SPECS)


def failure_result(
    agent: str, run: ResolvedRun, input_hash: str, exc: Exception
) -> Any:
    """Build an agent-specific error payload so one failure cannot break a case."""
    message = f"{type(exc).__name__}: {exc}"
    common: dict[str, Any] = {
        "run_id": run.run_id,
        "case_id": run.case_id,
        "step_id": run.step_id,
        "status": "error",
        "input_hash": input_hash,
        "errors": [message],
    }
    if agent == "symptom":
        return SymptomResult(model_version=SYMPTOM_VERSION, **common)
    if agent == "labs":
        return LabResult(model_version=LABS_VERSION, **common)
    if agent == "imaging":
        return ImagingResult(
            model_version=IMAGING_VERSION, modality="xray", body_site="unknown", **common
        )
    if agent == "differential":
        return DifferentialResult(model_version=DIFFERENTIAL_VERSION, **common)
    if agent == "drugs":
        return DrugSafetyResult(model_version=DRUGS_VERSION, **common)
    raise ValueError(f"Unknown agent {agent!r}")
