"""Unsupervised lab-pattern model.

The deterministic lab interpreter flags each value against its reference
range. This model looks at the *pattern across the panel*: an IsolationForest
trained on seeded synthetic panels detects combinations that are unusual even
when no single value is critical (for example mild changes in several
directions at once).

The score is a heuristic anomaly score, not a probability or a diagnosis.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from app.knowledge.labs import REFERENCES, LabReference
from app.ml.base import ModelInfo

logger = logging.getLogger(__name__)

FEATURE_VERSION = "lab-anomaly-isolationforest@0.1.0"
ARTIFACT_NAME = "lab_anomaly.joblib"

LAB_KEYS: tuple[str, ...] = tuple(ref.key for ref in REFERENCES)
_LAB_BY_KEY: dict[str, LabReference] = {ref.key: ref for ref in REFERENCES}


@dataclass(frozen=True)
class LabAnomalyResult:
    score: float
    is_anomaly: bool
    decision: float
    features_used: int


def _zscore(value: float, ref: LabReference) -> float:
    if ref.low is not None and ref.high is not None:
        center = (ref.low + ref.high) / 2.0
        scale = max((ref.high - ref.low) / 2.0, 1e-6)
    elif ref.low is not None:
        center = ref.low
        scale = max(abs(ref.low) * 0.25, 1.0)
    elif ref.high is not None:
        center = ref.high
        scale = max(abs(ref.high) * 0.25, 1.0)
    else:  # pragma: no cover - every reference has at least one bound
        return 0.0
    return float(np.clip((value - center) / scale, -8.0, 8.0))


def _normal_sample(ref: LabReference, rng: np.random.Generator) -> float:
    if ref.low is not None and ref.high is not None:
        center = (ref.low + ref.high) / 2.0
        sd = max((ref.high - ref.low) / 6.0, 1e-6)
        return float(rng.normal(center, sd))
    if ref.low is not None:
        return float(ref.low * rng.uniform(1.05, 1.6) + rng.normal(0.0, max(ref.low * 0.05, 1e-6)))
    return float(ref.high * rng.uniform(0.4, 0.95))  # type: ignore[operator]


def _abnormal_sample(ref: LabReference, rng: np.random.Generator) -> float:
    if ref.low is not None and ref.high is not None:
        width = max(ref.high - ref.low, 1e-6)
        if rng.random() < 0.5:
            return float(ref.low - width * rng.uniform(0.2, 2.0))
        return float(ref.high + width * rng.uniform(0.2, 2.0))
    if ref.low is not None:
        return float(ref.low * rng.uniform(0.1, 0.8))
    return float(ref.high * rng.uniform(1.2, 3.0))  # type: ignore[operator]


def _panel_vector(values: Mapping[str, float]) -> tuple[np.ndarray, int]:
    row = np.zeros(len(LAB_KEYS) * 2, dtype=np.float32)
    used = 0
    for index, key in enumerate(LAB_KEYS):
        value = values.get(key)
        if value is None:
            continue
        row[index] = _zscore(float(value), _LAB_BY_KEY[key])
        row[len(LAB_KEYS) + index] = 1.0
        used += 1
    return row, used


def _synthetic_panels(seed: int = 42, rows: int = 2400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((rows, len(LAB_KEYS) * 2), dtype=np.float32)
    for row_index in range(rows):
        anomaly = row_index % 8 == 0  # ~12.5% abnormal panels
        for feature_index, key in enumerate(LAB_KEYS):
            if rng.random() > 0.85:
                continue
            ref = _LAB_BY_KEY[key]
            value = _abnormal_sample(ref, rng) if anomaly else _normal_sample(ref, rng)
            data[row_index, feature_index] = _zscore(value, ref)
            data[row_index, len(LAB_KEYS) + feature_index] = 1.0
    return data


class LabAnomalyModel:
    """Lazily trained, cached IsolationForest over lab-panel z-scores."""

    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        contamination: float = 0.10,
        enabled: bool = True,
        seed: int = 42,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._contamination = contamination
        self._enabled = enabled
        self._seed = seed
        self._model = None
        self._failed = False
        self._lock = threading.RLock()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="lab-anomaly",
            version=FEATURE_VERSION,
            framework="scikit-learn",
            enabled=self._enabled,
        )

    @property
    def ready(self) -> bool:
        return self._enabled and self._model is not None and not self._failed

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
                from sklearn.ensemble import IsolationForest

                path = self._artifact_path()
                if path.exists():
                    bundle = joblib.load(path)
                    if bundle.get("feature_version") == FEATURE_VERSION:
                        self._model = bundle["model"]
                        logger.info("Loaded lab anomaly model from %s", path)
                        return True

                model = IsolationForest(
                    n_estimators=120,
                    contamination=self._contamination,
                    random_state=self._seed,
                    n_jobs=1,
                )
                model.fit(_synthetic_panels(seed=self._seed))
                self._model = model
                try:
                    self._artifact_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(
                        {"feature_version": FEATURE_VERSION, "model": model},
                        path,
                    )
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    logger.warning("Could not cache lab anomaly model: %s", exc)
                logger.info("Trained lab anomaly model")
                return True
            except Exception as exc:  # pragma: no cover - optional dependency
                self._failed = True
                logger.warning("Lab anomaly model unavailable: %s", exc)
                return False

    def score(self, values: Mapping[str, float]) -> LabAnomalyResult | None:
        """Score a panel keyed by :data:`LAB_KEYS`; ``None`` means unavailable."""
        if not values:
            return None
        if not self._load_or_train():
            return None
        row, used = _panel_vector(values)
        if used == 0:
            return None
        try:
            decision = float(self._model.decision_function([row])[0])  # type: ignore[union-attr]
            flagged = bool(self._model.predict([row])[0] == -1)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Lab anomaly inference failed: %s", exc)
            self._failed = True
            return None
        score = 1.0 / (1.0 + float(np.exp(3.0 * decision)))
        return LabAnomalyResult(
            score=round(score, 4),
            is_anomaly=flagged,
            decision=round(decision, 4),
            features_used=used,
        )
