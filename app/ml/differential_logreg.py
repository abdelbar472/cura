"""Learned re-ranker for the differential-diagnosis agent.

A multinomial logistic regression is trained on seeded synthetic panels drawn
from the same curated condition knowledge base. It does not replace the
transparent rule score; the agent blends the two (default 65% rules, 35% ML) so
an unfamiliar model output cannot silently discard a strongly supported rule.

The synthetic training set is deliberately small and derived from reviewed data,
which keeps the model auditable but means its probabilities are not calibrated.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from app.knowledge.conditions import CONDITIONS, all_evidence_codes
from app.ml.base import ModelInfo

logger = logging.getLogger(__name__)

MODEL_VERSION = "differential-logreg@0.1.0"
ARTIFACT_NAME = "differential_logreg.joblib"

EVIDENCE_CODES: tuple[str, ...] = tuple(all_evidence_codes())
_CODE_INDEX = {code: index for index, code in enumerate(EVIDENCE_CODES)}
CONDITION_IDS: tuple[str, ...] = tuple(rule.condition_id for rule in CONDITIONS)


def _vector(codes: Iterable[str]) -> np.ndarray:
    row = np.zeros(len(EVIDENCE_CODES), dtype=np.float32)
    for code in codes:
        index = _CODE_INDEX.get(code)
        if index is not None:
            row[index] = 1.0
    return row


def _synthetic_training_data(seed: int = 42, per_condition: int = 320):
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for label, rule in enumerate(CONDITIONS):
        for _ in range(per_condition):
            row = np.zeros(len(EVIDENCE_CODES), dtype=np.float32)
            for code, weight in rule.weights.items():
                probability = 1.0 / (1.0 + float(np.exp(-weight)))
                if rng.random() < probability:
                    row[_CODE_INDEX[code]] = 1.0
            # A small amount of unrelated evidence, as in real presentations.
            for index in range(len(EVIDENCE_CODES)):
                if row[index] == 0.0 and rng.random() < 0.012:
                    row[index] = 1.0
            if not row.any():  # guarantee a condition-specific finding
                code = max(rule.weights, key=rule.weights.__getitem__)
                row[_CODE_INDEX[code]] = 1.0
            rows.append(row)
            labels.append(label)
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


class DifferentialLogRegModel:
    """Lazily trained multinomial logistic re-ranker with an on-disk cache."""

    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        enabled: bool = True,
        seed: int = 42,
        ml_weight: float = 0.35,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._enabled = enabled
        self._seed = seed
        self._ml_weight = ml_weight
        self._model = None
        self._failed = False
        self._lock = threading.RLock()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="differential-reranker",
            version=MODEL_VERSION,
            framework="scikit-learn",
            enabled=self._enabled,
        )

    @property
    def ready(self) -> bool:
        return self._enabled and self._model is not None and not self._failed

    @property
    def ml_weight(self) -> float:
        return self._ml_weight

    def _artifact_path(self) -> Path:
        return self._artifact_dir / ARTIFACT_NAME

    def _load_or_train(self) -> bool:
        if self._model is not None:
            return True
        if not self._enabled or self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import joblib
                from sklearn.linear_model import LogisticRegression

                path = self._artifact_path()
                if path.exists():
                    bundle = joblib.load(path)
                    if bundle.get("model_version") == MODEL_VERSION:
                        self._model = bundle["model"]
                        logger.info("Loaded differential re-ranker from %s", path)
                        return True

                x, y = _synthetic_training_data(seed=self._seed)
                model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                    random_state=self._seed,
                )
                model.fit(x, y)
                self._model = model
                try:
                    self._artifact_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"model_version": MODEL_VERSION, "model": model}, path)
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    logger.warning("Could not cache differential re-ranker: %s", exc)
                logger.info("Trained differential re-ranker")
                return True
            except Exception as exc:  # pragma: no cover - optional dependency
                self._failed = True
                logger.warning("Differential re-ranker unavailable: %s", exc)
                return False

    def predict(self, evidence_codes: Iterable[str]) -> dict[str, float] | None:
        """Return condition-id -> probability, or ``None`` when unavailable."""
        codes = list(evidence_codes)
        if not codes:
            return {}
        if not self._load_or_train():
            return None
        try:
            probabilities = self._model.predict_proba([_vector(codes)])[0]  # type: ignore[union-attr]
            classes = self._model.classes_  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Differential re-ranker inference failed: %s", exc)
            self._failed = True
            return None
        return {
            CONDITION_IDS[int(label)]: round(float(probability), 6)
            for label, probability in zip(classes, probabilities)
        }
