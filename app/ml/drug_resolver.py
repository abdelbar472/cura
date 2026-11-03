"""Medication-name resolver.

Resolution order is intentionally conservative:

1. exact ingredient name;
2. known brand/synonym alias;
3. string similarity (:class:`difflib.SequenceMatcher`);
4. sentence-embedding similarity against ingredient descriptions.

Only steps 3 and 4 are learned/approximate, and both are surfaced on the
response so a clinician can verify the mapping. An unresolved name is never
silently treated as a known drug.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

import numpy as np

from app.agents.drugs.normalize import KNOWN_INGREDIENTS, clean_medication_name
from app.knowledge.interactions import DRUG_ALIASES
from app.ml.base import ModelInfo
from app.ml.embeddings import EmbeddingBackend

logger = logging.getLogger(__name__)

MODEL_VERSION = "drug-resolver@0.1.0"

ResolutionMethod = Literal["exact", "alias", "fuzzy", "embedding", "unresolved"]

_SALT_OR_FORM = {
    "sodium", "potassium", "hydrochloride", "hcl", "tartrate", "succinate",
    "maleate", "citrate", "mesylate", "besylate", "fumarate", "calcium",
    "extended", "release", "er", "xr", "cr", "sr", "ds", "forte",
}


@dataclass(frozen=True)
class DrugMatch:
    ingredient: str | None
    method: ResolutionMethod
    confidence: float


def _alias_catalog() -> dict[str, str]:
    catalog: dict[str, str] = {}
    for ingredient in KNOWN_INGREDIENTS:
        catalog[ingredient] = ingredient
    for alias, ingredient in DRUG_ALIASES.items():
        catalog[alias] = ingredient
    return catalog


_CATALOG = _alias_catalog()


def _strip_trailing_form(candidate: str) -> str:
    tokens = candidate.split()
    while len(tokens) > 1 and tokens[-1] in _SALT_OR_FORM:
        tokens.pop()
    return " ".join(tokens)


def _description(ingredient: str) -> str:
    aliases = sorted(alias for alias, target in DRUG_ALIASES.items() if target == ingredient)
    if not aliases:
        return ingredient
    return f"{ingredient} ({', '.join(aliases)})"


class DrugResolver:
    """Hybrid alias/fuzzy/embedding resolver for medication names."""

    def __init__(
        self,
        embeddings: EmbeddingBackend | None = None,
        *,
        fuzzy_threshold: float = 0.84,
        embedding_threshold: float = 0.72,
        enabled: bool = True,
    ) -> None:
        self._embeddings = embeddings
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold
        self._enabled = enabled
        self._catalog: np.ndarray | None = None
        self._catalog_ids: tuple[str, ...] = tuple(sorted(KNOWN_INGREDIENTS))
        self._failed = False
        self._lock = threading.RLock()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="drug-resolver",
            version=MODEL_VERSION,
            framework="sentence-transformers",
            enabled=self._enabled,
        )

    @property
    def ready(self) -> bool:
        return self._enabled and not self._failed

    def _ensure_catalog(self) -> bool:
        if self._catalog is not None:
            return True
        if not self._enabled or self._embeddings is None or self._failed:
            return False
        with self._lock:
            if self._catalog is not None:
                return True
            try:
                self._catalog = self._embeddings.encode(
                    [_description(item) for item in self._catalog_ids]
                )
                return True
            except Exception as exc:  # pragma: no cover - model download dependent
                self._failed = True
                logger.warning("Drug embedding catalog unavailable: %s", exc)
                return False

    def resolve(self, raw_name: str) -> DrugMatch:
        candidate = clean_medication_name(raw_name)
        if not candidate:
            return DrugMatch(None, "unresolved", 0.0)

        if candidate in KNOWN_INGREDIENTS and candidate not in DRUG_ALIASES:
            return DrugMatch(candidate, "exact", 1.0)
        if candidate in DRUG_ALIASES:
            return DrugMatch(DRUG_ALIASES[candidate], "alias", 1.0)

        # Form/salt suffixes should not defeat an otherwise exact match.
        stripped = _strip_trailing_form(candidate)
        if stripped in KNOWN_INGREDIENTS:
            return DrugMatch(stripped, "fuzzy", 0.98)
        if stripped in DRUG_ALIASES:
            return DrugMatch(DRUG_ALIASES[stripped], "fuzzy", 0.98)

        best_name: str | None = None
        best_score = 0.0
        for alias, ingredient in _CATALOG.items():
            ratio = SequenceMatcher(None, candidate, alias).ratio()
            # Favour a candidate that starts with a known alias followed by a
            # salt/form word ("metoprolol tartrate").
            if candidate.startswith(alias + " "):
                ratio = max(ratio, 0.9)
            if ratio > best_score:
                best_name, best_score = ingredient, ratio
        if best_name is not None and best_score >= self._fuzzy_threshold:
            return DrugMatch(best_name, "fuzzy", round(best_score, 4))

        if self._ensure_catalog():
            try:
                vector = self._embeddings.encode([candidate])  # type: ignore[union-attr]
                similarities = vector @ self._catalog.T  # type: ignore[union-attr]
                index = int(np.argmax(similarities))
                score = float(similarities[0][index])
                if score >= self._embedding_threshold:
                    return DrugMatch(self._catalog_ids[index], "embedding", round(score, 4))
            except Exception as exc:  # pragma: no cover - inference failure
                logger.warning("Drug embedding inference failed: %s", exc)
                self._failed = True

        if best_name is not None and best_score > 0.6:
            return DrugMatch(None, "unresolved", round(best_score, 4))
        return DrugMatch(None, "unresolved", 0.0)

    def clean(self, raw_name: str) -> str:
        """Exposed for callers that need the normalised candidate string."""
        return re.sub(r"\s+", " ", clean_medication_name(raw_name)).strip()
