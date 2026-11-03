"""Entity extraction for the symptom agent.

Preferred backend is scispaCy (``en_core_sci_sm``). When it is not installed we
fall back to the curated lexicon, and either way lexicon matches are unioned in
so recognised symptom codes never depend on an optional model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.knowledge.symptoms import ALIAS_PATTERN

logger = logging.getLogger(__name__)

Backend = Literal["auto", "lexicon", "scispacy"]

_NLP = None
_NLP_FAILED = False


@dataclass
class RawEntity:
    text: str
    label: str
    start: int
    end: int
    source: Literal["lexicon", "scispacy", "ml"] = "lexicon"
    code: str | None = None
    score: float | None = None


def _load_model(model_name: str):
    """Load scispaCy once; remember failures so we do not retry per request."""
    global _NLP, _NLP_FAILED
    if _NLP is not None or _NLP_FAILED:
        return _NLP
    try:
        import spacy  # imported lazily: optional dependency

        _NLP = spacy.load(model_name)
        logger.info("Loaded scispaCy model %s", model_name)
    except Exception as exc:  # pragma: no cover - depends on local install
        _NLP_FAILED = True
        logger.warning("scispaCy model %r unavailable (%s); using lexicon", model_name, exc)
    return _NLP


def _lexicon_entities(text: str) -> list[RawEntity]:
    matches = sorted(ALIAS_PATTERN.finditer(text), key=lambda m: (m.start(), -(m.end() - m.start())))
    entities: list[RawEntity] = []
    last_end = -1
    for match in matches:
        if match.start() < last_end:
            continue
        last_end = match.end()
        entities.append(RawEntity(match.group(0), "SYMPTOM", match.start(), match.end(), "lexicon"))
    return entities


def _scispacy_entities(text: str, model_name: str) -> list[RawEntity]:
    nlp = _load_model(model_name)
    if nlp is None:
        return []
    doc = nlp(text)
    entities: list[RawEntity] = []
    seen: set[tuple[int, int]] = set()
    for ent in doc.ents:
        span = (ent.start_char, ent.end_char)
        if span in seen or not ent.text.strip():
            continue
        seen.add(span)
        entities.append(
            RawEntity(ent.text, ent.label_ or "ENTITY", ent.start_char, ent.end_char, "scispacy")
        )
    return entities


def _overlaps(a: RawEntity, others: list[RawEntity]) -> bool:
    return any(a.start < b.end and b.start < a.end for b in others)


def scispacy_available(model_name: str = "en_core_sci_sm") -> bool:
    return _load_model(model_name) is not None


def extract_entities(
    text: str,
    backend: Backend = "auto",
    model_name: str = "en_core_sci_sm",
) -> list[RawEntity]:
    """Return deduplicated entities ordered by position in the text."""
    lexicon = _lexicon_entities(text)

    if backend == "lexicon":
        return lexicon

    scispacy = _scispacy_entities(text, model_name)
    if not scispacy:
        return lexicon

    # Keep model spans for provenance, but never let them hide a lexicon match:
    # lexicon spans win on overlap because they carry the evidence code.
    extra = [ent for ent in scispacy if not _overlaps(ent, lexicon)]
    return sorted(lexicon + extra, key=lambda e: e.start)
