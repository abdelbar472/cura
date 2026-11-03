"""Map raw entities to versioned evidence codes, with negation handling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agents.symptom.model import RawEntity
from app.agents.symptom.schemas import Entity
from app.knowledge.symptoms import (
    AFFIRMATION_CUES,
    ALIAS_TO_CONCEPT,
    CLAUSE_BOUNDARY,
    CONCEPTS_BY_CODE,
    EVIDENCE_MAP_VERSION,
    NEGATION_CUES,
    RED_FLAG_CODES,
    SymptomConcept,
)

__all__ = [
    "EVIDENCE_MAP_VERSION",
    "RED_FLAG_CODES",
    "MappingResult",
    "detect_red_flags",
    "is_negated",
    "lookup_concept",
    "map_entities",
    "normalise_phrase",
]

_NON_WORD = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z']+")


@dataclass
class MappingResult:
    entities: list[Entity] = field(default_factory=list)
    evidence_codes: list[str] = field(default_factory=list)
    negated_codes: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)


def normalise_phrase(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def lookup_concept(text: str) -> SymptomConcept | None:
    key = normalise_phrase(text)
    if not key:
        return None
    concept = ALIAS_TO_CONCEPT.get(key)
    if concept:
        return concept
    for suffix in ("es", "s"):
        if key.endswith(suffix) and len(key) > len(suffix) + 2:
            concept = ALIAS_TO_CONCEPT.get(key[: -len(suffix)])
            if concept:
                return concept
    return None


def is_negated(text: str, start: int, window: int = 80) -> bool:
    """Heuristic negation scope: cues in the same clause, nearest cue wins.

    Handles list negation ("no fever, chills, or sweats") and resets at clause
    boundaries and affirmation cues ("denies chest pain, reports dyspnea").
    """
    snippet = text[max(0, start - window) : start]
    boundaries = list(CLAUSE_BOUNDARY.finditer(snippet))
    if boundaries:
        snippet = snippet[boundaries[-1].end() :]

    tokens = _TOKEN.findall(snippet.lower())
    last_negation = -1
    last_affirmation = -1
    for index, token in enumerate(tokens):
        if token in NEGATION_CUES:
            last_negation = index
        elif token in AFFIRMATION_CUES:
            last_affirmation = index
    return last_negation > last_affirmation


def map_entities(entities: list[RawEntity], text: str) -> MappingResult:
    result = MappingResult()
    seen_codes: set[str] = set()
    seen_negated: set[str] = set()

    for raw in entities:
        # A learned span may already carry its code (semantic matching happens
        # at sentence level, so looking the full span up in the lexicon would
        # fail). Lexicon and scispaCy spans resolve through the alias table.
        concept = CONCEPTS_BY_CODE.get(raw.code) if raw.code else lookup_concept(raw.text)
        negated = is_negated(text, raw.start)
        result.entities.append(
            Entity(
                text=raw.text,
                label=raw.label,
                start=raw.start,
                end=raw.end,
                code=concept.code if concept else None,
                negated=negated,
                source=raw.source,
                score=raw.score,
            )
        )
        if concept is None:
            result.unknown_entities.append(raw.text)
            continue
        if negated:
            if concept.code not in seen_negated:
                seen_negated.add(concept.code)
                result.negated_codes.append(concept.code)
                result.negated_symptoms.append(concept.canonical)
        elif concept.code not in seen_codes:
            seen_codes.add(concept.code)
            result.evidence_codes.append(concept.code)
            result.symptoms.append(concept.canonical)

    return result


def detect_red_flags(codes: list[str]) -> list[str]:
    return [code for code in codes if code in RED_FLAG_CODES]
