"""HuggingFace chest X-ray classifier.

This is a deliberately narrow model: a single public ViT checkpoint fine-tuned
for NORMAL vs PNEUMONIA on chest radiographs. It is loaded only when an X-ray
image is supplied, and its output is reported as a probabilistic pattern flag,
never as a diagnosis. Other modalities stay on the null backend.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
from pathlib import Path
from typing import Any

from app.agents.imaging.schemas import ImagingFinding, ImagingRequest
from app.ml.base import ModelInfo

logger = logging.getLogger(__name__)

MODEL_VERSION = "imaging-hf-vit@0.1.0"
DEFAULT_MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
E_PNEUMONIA = "E_PNEUMONIA"


class ImagingDecodeError(ValueError):
    """Raised when ``image_base64`` cannot be decoded into an image."""


class HFImagingModel:
    """Lazy chest-X-ray classifier with a graceful unavailable state."""

    name = "hf-chest-xray-vit"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        threshold: float = 0.55,
        enabled: bool = True,
    ) -> None:
        self.model_id = model_id
        self._device = 0 if device.startswith("cuda") else -1
        self._threshold = threshold
        self._enabled = enabled
        self._pipeline = None
        self._failed = False
        self._lock = threading.RLock()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="imaging-chest-xray",
            version=f"{MODEL_VERSION}+{self.model_id}",
            framework="transformers",
            enabled=self._enabled,
        )

    @property
    def ready(self) -> bool:
        return self._enabled and self._pipeline is not None and not self._failed

    def _load(self) -> bool:
        if self._pipeline is not None:
            return True
        if not self._enabled or self._failed:
            return False
        with self._lock:
            if self._pipeline is not None:
                return True
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "image-classification",
                    model=self.model_id,
                    device=self._device,
                )
                logger.info("Loaded imaging model %s", self.model_id)
                return True
            except Exception as exc:  # pragma: no cover - model download dependent
                self._failed = True
                logger.warning("Imaging model %r unavailable: %s", self.model_id, exc)
                return False

    @staticmethod
    def _decode_image(image_base64: str) -> Any:
        payload = image_base64.strip()
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImagingDecodeError(f"image_base64 is not valid base64: {exc}") from exc
        if len(raw) > 12 * 1024 * 1024:
            raise ImagingDecodeError("Decoded image exceeds the 12 MB limit")
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(raw))
            image.verify()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            return image
        except Exception as exc:
            raise ImagingDecodeError(f"image_base64 is not a readable image: {exc}") from exc

    def predict_detailed(self, req: ImagingRequest):
        from app.agents.imaging.model import ImagingPrediction

        if req.modality != "xray":
            return ImagingPrediction(
                findings=[],
                notes=["The chest X-ray model does not interpret this modality."],
            )
        if not req.image_base64:
            return ImagingPrediction(
                findings=[],
                notes=["No image supplied; no image interpretation was performed."],
            )
        if req.body_site.lower() not in {"chest", "thorax", "chest x-ray", "cxr"}:
            return ImagingPrediction(
                findings=[],
                notes=["The configured model is chest X-ray only; body site was not interpreted."],
            )
        if not self._load():
            return ImagingPrediction(
                findings=[],
                notes=["The imaging model is configured but could not be loaded."],
            )

        try:
            image = self._decode_image(req.image_base64)
            results = self._pipeline(image, top_k=2)  # type: ignore[misc]
        except ImagingDecodeError:
            raise
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Imaging inference failed: %s", exc)
            self._failed = True
            return ImagingPrediction(
                findings=[],
                notes=["Imaging inference failed; no findings were produced."],
            )

        notes: list[str] = []
        findings: list[ImagingFinding] = []
        by_label = {str(item["label"]).upper(): float(item["score"]) for item in results}
        pneumonia = by_label.get("PNEUMONIA", 0.0)
        normal = by_label.get("NORMAL", 0.0)
        if pneumonia >= self._threshold and pneumonia >= normal:
            findings.append(
                ImagingFinding(
                    code=E_PNEUMONIA,
                    label="Pneumonia pattern (model flag)",
                    confidence=round(pneumonia, 4),
                )
            )
            notes.append(
                "Model flagged a pneumonia pattern. This is a screening signal, "
                "not a diagnosis; radiographic interpretation by a qualified "
                "clinician is required."
            )
        else:
            notes.append(
                "Model did not flag pneumonia above threshold "
                f"({self._threshold:.2f}); this does not exclude disease."
            )
        notes.append(
            f"Classifier: {self.model_id} (NORMAL vs PNEUMONIA), "
            "single public checkpoint, not clinically validated."
        )
        return ImagingPrediction(findings=findings, notes=notes, ran=True)

    def predict(self, req: ImagingRequest) -> list[ImagingFinding]:
        return self.predict_detailed(req).findings
