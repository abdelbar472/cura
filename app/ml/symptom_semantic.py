"""Embedding-based symptom concept matcher.

This is the learned half of the symptom agent: it catches lay paraphrases the
lexicon misses ("burning when I pee"). Lexicon spans always win on overlap, so
the model can only add recall, never override an exact match.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from app.knowledge.symptoms import SYMPTOM_CONCEPTS, concept_description
from app.ml.base import ModelInfo
from app.ml.embeddings import EmbeddingBackend

logger = logging.getLogger(__name__)

_SENTENCE = re.compile(r"[^.!?\n]+")

DEFAULT_THRESHOLD = 0.55
DEFAULT_MARGIN = 0.05


@dataclass(frozen=True)
class SemanticMatch:
    code: str
    text: str
    start: int
    end: int
    score: float
    runner_up: float


class SymptomSemanticModel:
    def __init__(
        self,
        embeddings: EmbeddingBackend,
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        enabled: bool = True,
    ) -> None:
        self._embeddings = embeddings
        self._threshold = threshold
        self._margin = margin
        self._enabled = enabled
        self._catalog: np.ndarray | None = None
        self._failed = False
        self._lock = threading.RLock()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="symptom-semantic",
            version=f"symptom-semantic@0.1.0+{self._embeddings.name}",
            framework="sentence-transformers",
            enabled=self._enabled,
        )

    @property
    def ready(self) -> bool:
        return self._enabled and self._catalog is not None and not self._failed

    def _ensure_catalog(self) -> bool:
        if self._catalog is not None:
            return True
        if not self._enabled or self._failed:
            return False
        with self._lock:
            if self._catalog is not None:
                return True
            if self._failed:
                return False
            try:
                self._catalog = self._embeddings.encode(
                    [concept_description(concept) for concept in SYMPTOM_CONCEPTS]
                )
                return True
            except Exception as exc:  # pragma: no cover - model download dependent
                self._failed = True
                logger.warning("Symptom semantic model unavailable: %s", exc)
                return False

    def match(self, text: str) -> list[SemanticMatch]:
        if not self._enabled or not text.strip() or not self._ensure_catalog():
            return []

        spans = [
            (match.group(0), match.start(), match.end())
            for match in _SENTENCE.finditer(text)
            if match.group(0).strip()
        ]
        if not spans:
            return []
        try:
            vectors = self._embeddings.encode([span[0] for span in spans])
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Symptom semantic inference failed: %s", exc)
            self._failed = True
            return []

        similarities = vectors @ self._catalog.T  # type: ignore[union-attr]
        matches: list[SemanticMatch] = []
        for index, (sentence, start, end) in enumerate(spans):
            row = similarities[index]
            best_index = int(np.argmax(row))
            best = float(row[best_index])
            if len(row) > 1:
                second = float(np.partition(row, -2)[-2])
            else:  # pragma: no cover - catalog is always larger
                second = 0.0
            if best < self._threshold or (best - second) < self._margin:
                continue
            concept = SYMPTOM_CONCEPTS[best_index]
            matches.append(
                SemanticMatch(
                    code=concept.code,
                    text=sentence,
                    start=start,
                    end=end,
                    score=round(best, 4),
                    runner_up=round(second, 4),
                )
            )
        return matches
