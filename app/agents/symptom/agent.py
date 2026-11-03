"""Symptom analyzer agent: free text -> structured symptom evidence.

Two complementary paths run here:

1. deterministic lexicon + optional scispaCy entity extraction (high precision);
2. an optional sentence-embedding matcher that adds lay paraphrases the
   lexicon misses (higher recall).

Lexicon/scispaCy spans always win when they map to a concept; the semantic
matcher only contributes codes that are not already present. That keeps the
agent's output stable while still using the learned model where it helps.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from app.agents.symptom.mapper import (
    EVIDENCE_MAP_VERSION,
    detect_red_flags,
    is_negated,
    map_entities,
)
from app.agents.symptom.model import extract_entities, scispacy_available
from app.agents.symptom.schemas import Entity, SymptomRequest, SymptomResult
from app.knowledge.symptoms import CONCEPTS_BY_CODE
from app.ml.base import model_usage
from app.ml.symptom_semantic import SymptomSemanticModel
from app.schemas.common import resolve_run

logger = logging.getLogger(__name__)

AGENT_NAME = "symptom"
MODEL_VERSION = "symptom-ner@0.2.0"
FEATURE = "symptom"
Backend = Literal["auto", "lexicon", "scispacy"]


def _input_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_symptom_agent(
    req: SymptomRequest,
    semantic: SymptomSemanticModel | None = None,
) -> SymptomResult:
    run = resolve_run(req, AGENT_NAME)
    backend: Backend = req.backend
    raw_entities = extract_entities(req.text, backend=backend, model_name=req.model_name)
    used_backend = "lexicon"
    if backend != "lexicon" and scispacy_available(req.model_name):
        used_backend = "scispacy"

    mapping = map_entities(raw_entities, req.text)

    ml_usage = None
    semantic_matches = []
    if semantic is not None:
        info = semantic.info
        try:
            semantic_matches = semantic.match(req.text)
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Symptom semantic model failed: %s", exc)
            semantic_matches = []

        if not info.enabled:
            ml_usage = model_usage(
                info, ready=False, fallback=True, detail="disabled by configuration"
            )
        elif getattr(semantic, "ready", False):
            ml_usage = model_usage(
                info,
                ready=True,
                detail=f"{len(semantic_matches)} semantic span(s)",
            )
        else:
            ml_usage = model_usage(
                info,
                ready=False,
                fallback=True,
                detail="model unavailable; deterministic rules used",
            )

    for match in semantic_matches:
        concept = CONCEPTS_BY_CODE.get(match.code)
        if concept is None:
            continue
        if concept.code in mapping.evidence_codes or concept.code in mapping.negated_codes:
            continue
        negated = is_negated(req.text, match.start)
        mapping.entities.append(
            Entity(
                text=match.text,
                label="SYMPTOM",
                start=match.start,
                end=match.end,
                code=concept.code,
                negated=negated,
                source="ml",
                score=match.score,
            )
        )
        if negated:
            mapping.negated_codes.append(concept.code)
            mapping.negated_symptoms.append(concept.canonical)
        else:
            mapping.evidence_codes.append(concept.code)
            mapping.symptoms.append(concept.canonical)

    red_flags = detect_red_flags(mapping.evidence_codes)

    warnings: list[str] = []
    if not mapping.evidence_codes and not mapping.negated_codes:
        warnings.append("No recognised symptom concepts were found in the input text.")
    unknown = list(dict.fromkeys(mapping.unknown_entities))
    if unknown:
        warnings.append(f"{len(unknown)} model entities did not map to a symptom code.")

    model_version = f"{req.model_version}+{EVIDENCE_MAP_VERSION}"
    if ml_usage is not None and ml_usage.ready:
        model_version = f"{model_version}+{ml_usage.version.split('+', 1)[0]}"

    return SymptomResult(
        run_id=run.run_id,
        case_id=run.case_id,
        step_id=run.step_id,
        status="ok",
        model_version=model_version,
        input_hash=_input_hash(req.text),
        warnings=warnings,
        ml=ml_usage,
        backend=used_backend,
        symptoms=mapping.symptoms,
        negated_symptoms=mapping.negated_symptoms,
        red_flags=red_flags,
        evidence_codes=mapping.evidence_codes,
        negated_codes=mapping.negated_codes,
        entities=mapping.entities,
    )
