"""Lazy sentence-embedding backend shared by the semantic models."""

from __future__ import annotations

import logging
import threading
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingBackend(Protocol):
    name: str

    @property
    def ready(self) -> bool: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    """Thread-safe, lazily loaded SentenceTransformer wrapper."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> None:
        self.name = model_name
        self.device = device
        self._model = None
        self._failed = False
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self._model is not None or self._failed:
            return self._model is not None
        with self._lock:
            if self._model is not None or self._failed:
                return self._model is not None
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device=self.device)
                logger.info("Loaded embedding model %s", self.name)
            except Exception as exc:  # pragma: no cover - depends on local install
                self._failed = True
                logger.warning("Embedding model %r unavailable: %s", self.name, exc)
            return self._model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.load():
            raise RuntimeError(f"Embedding model {self.name!r} is not available")
        with self._lock:
            vectors = self._model.encode(  # type: ignore[union-attr]
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)


class DisabledEmbeddings:
    name = "disabled"

    @property
    def ready(self) -> bool:
        return False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("Embeddings are disabled")
