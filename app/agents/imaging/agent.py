"""Imaging agent: validates requests and defers to a configured model."""

from __future__ import annotations

import hashlib
import logging

from app.agents.imaging.model import ImagingModel, ImagingPrediction, get_model
from app.agents.imaging.schemas import ImagingRequest, ImagingResult
from app.ml.base import ModelInfo, model_usage
from app.schemas.common import hash_payload, resolve_run

logger = logging.getLogger(__name__)

AGENT_NAME = "imaging"
MODEL_VERSION = "imaging@0.2.0"
FEATURE = "imaging"


def _image_hash(req: ImagingRequest) -> str | None:
    if req.image_sha256:
        return req.image_sha256
    if req.image_base64:
        return hashlib.sha256(req.image_base64.encode("utf-8")).hexdigest()
    return None


def _input_hash(req: ImagingRequest) -> str:
    return hash_payload(
        {
            "modality": req.modality,
            "body_site": req.body_site,
            "clinical_question": req.clinical_question,
            "image_sha256": _image_hash(req),
        }
    )


def _model_info(model: ImagingModel) -> ModelInfo:
    info = getattr(model, "info", None)
    if info is not None:
        return info
    return ModelInfo(
        name=getattr(model, "name", "unknown"),
        version=getattr(model, "name", "unknown"),
        framework="custom",
        enabled=True,
    )


def run_imaging_agent(
    req: ImagingRequest,
    backend: str = "null",
    model: ImagingModel | None = None,
) -> ImagingResult:
    run = resolve_run(req, AGENT_NAME)
    selected = get_model(backend)
    if selected is None or selected.name == "null":
        selected = None

    # An explicit model wins over the registry, which keeps tests injectable and
    # lets the orchestrator share a single lazily-loaded instance.
    if model is not None and backend not in {"null"}:
        selected = model

    if selected is None:
        return ImagingResult(
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
            status="unavailable",
            model_version=MODEL_VERSION,
            input_hash=_input_hash(req),
            modality=req.modality,
            body_site=req.body_site,
            backend="null",
            findings=[],
            evidence_codes=[],
            notes=[
                "No imaging model is configured. The request was accepted and "
                "validated but no image interpretation was performed.",
                "Configure an X-ray model and supply image_base64 to enable "
                "image interpretation.",
            ],
        )

    prediction: ImagingPrediction
    try:
        if hasattr(selected, "predict_detailed"):
            prediction = selected.predict_detailed(req)  # type: ignore[attr-defined]
        else:
            prediction = ImagingPrediction(findings=selected.predict(req), ran=True)
    except Exception as exc:
        # Bad image bytes are an input problem, not a server fault. Return a
        # typed error result instead of letting the request 500.
        logger.warning("Imaging interpretation failed: %s", exc)
        return ImagingResult(
            run_id=run.run_id,
            case_id=run.case_id,
            step_id=run.step_id,
            status="error",
            model_version=MODEL_VERSION,
            input_hash=_input_hash(req),
            errors=[f"{type(exc).__name__}: {exc}"],
            modality=req.modality,
            body_site=req.body_site,
            backend=getattr(selected, "name", "unknown"),
            findings=[],
            evidence_codes=[],
            notes=[
                "The image could not be interpreted. No findings were produced.",
                "Check that image_base64 contains a valid, readable image.",
            ],
        )

    info = _model_info(selected)
    ready = bool(getattr(selected, "ready", True))
    if not info.enabled:
        usage = model_usage(info, ready=False, fallback=True, detail="disabled by configuration")
    elif prediction.ran and ready:
        usage = model_usage(
            info,
            ready=True,
            detail=f"{len(prediction.findings)} finding(s)",
        )
    elif prediction.ran:
        usage = model_usage(
            info,
            ready=False,
            fallback=True,
            detail="model produced findings but is not marked ready",
        )
    else:
        usage = model_usage(
            info,
            ready=False,
            detail=prediction.notes[0] if prediction.notes else "no interpretation performed",
        )

    status = "ok" if prediction.ran else "unavailable"
    model_version = MODEL_VERSION
    if prediction.ran:
        model_version = f"{MODEL_VERSION}+{info.version.split('+', 1)[0]}"

    return ImagingResult(
        run_id=run.run_id,
        case_id=run.case_id,
        step_id=run.step_id,
        status=status,
        model_version=model_version,
        input_hash=_input_hash(req),
        ml=usage,
        modality=req.modality,
        body_site=req.body_site,
        backend=getattr(selected, "name", "unknown"),
        findings=prediction.findings,
        evidence_codes=[finding.code for finding in prediction.findings],
        notes=prediction.notes,
    )
