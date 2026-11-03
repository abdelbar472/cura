from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.agents.differential.schemas import DifferentialResult
from app.agents.drugs.schemas import DrugSafetyResult
from app.agents.imaging.schemas import ImagingResult
from app.agents.labs.schemas import LabResult
from app.agents.symptom.schemas import SymptomResult
from app.schemas.case import CaseRequest
from app.schemas.common import AgentEvent


class CaseState(TypedDict, total=False):
    request: CaseRequest
    run_id: str
    case_id: str
    symptom: SymptomResult
    labs: LabResult
    imaging: ImagingResult
    differential: DifferentialResult
    drugs: DrugSafetyResult
    events: Annotated[list[AgentEvent], operator.add]
    warnings: Annotated[list[str], operator.add]
    failed_agents: Annotated[list[str], operator.add]
