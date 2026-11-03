"""Lab interpreter agent: reference-range checks -> evidence codes.

An optional IsolationForest adds a panel-level anomaly signal. The
deterministic per-value flags remain authoritative; the model only enriches the
response and never rewrites a reference-range decision.
"""

from __future__ import annotations

from app.agents.labs.schemas import LabFinding, LabRequest, LabResult
from app.knowledge.labs import (
    REFERENCE_VERSION,
    LabReference,
    lookup_reference,
    normalise_unit,
)
from app.ml.base import model_usage
from app.ml.lab_anomaly import LabAnomalyModel
from app.schemas.common import hash_payload, resolve_run

AGENT_NAME = "labs"
MODEL_VERSION = "lab-rules@0.2.0"
FEATURE = "labs"


def _flag_for(value: float, low: float | None, high: float | None, ref: LabReference | None) -> str:
    if ref and ref.critical_low is not None and value < ref.critical_low:
        return "critical_low"
    if ref and ref.critical_high is not None and value > ref.critical_high:
        return "critical_high"
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    return "normal"


def _code_for(flag: str, ref: LabReference | None) -> str | None:
    if ref is None:
        return None
    if flag == "critical_low":
        return ref.code_critical_low or ref.code_low
    if flag == "critical_high":
        return ref.code_critical_high or ref.code_high
    if flag == "low":
        return ref.code_low
    if flag == "high":
        return ref.code_high
    return None


def run_lab_agent(
    req: LabRequest,
    anomaly_model: LabAnomalyModel | None = None,
) -> LabResult:
    run = resolve_run(req, AGENT_NAME)
    findings: list[LabFinding] = []
    evidence_codes: list[str] = []
    critical_flags: list[str] = []
    warnings: list[str] = []
    interpreted: dict[str, float] = {}

    for lab in req.values:
        ref = lookup_reference(lab.name)
        if ref is None and lab.reference_low is None and lab.reference_high is None:
            findings.append(
                LabFinding(
                    name=lab.name.strip(),
                    raw_name=lab.name,
                    value=lab.value,
                    unit=lab.unit,
                    flag="unknown",
                )
            )
            warnings.append(f"Unknown lab {lab.name!r}; not interpreted.")
            continue

        display = ref.name if ref else lab.name.strip()
        if ref and lab.unit is not None and normalise_unit(lab.unit) != normalise_unit(ref.unit):
            findings.append(
                LabFinding(
                    name=display,
                    raw_name=lab.name,
                    value=lab.value,
                    unit=lab.unit,
                    flag="unknown",
                )
            )
            warnings.append(
                f"Unit mismatch for {display}: got {lab.unit!r}, expected {ref.unit!r}; not interpreted."
            )
            continue
        if ref and lab.unit is None:
            warnings.append(f"No unit supplied for {display}; assumed {ref.unit}.")

        low = lab.reference_low if lab.reference_low is not None else (ref.low if ref else None)
        high = lab.reference_high if lab.reference_high is not None else (ref.high if ref else None)
        flag = _flag_for(lab.value, low, high, ref)
        code = _code_for(flag, ref)

        findings.append(
            LabFinding(
                name=display,
                raw_name=lab.name,
                value=lab.value,
                unit=lab.unit or (ref.unit if ref else None),
                flag=flag,
                reference_low=low,
                reference_high=high,
                evidence_code=code,
            )
        )
        if ref is not None:
            interpreted.setdefault(ref.key, lab.value)
        if code and code not in evidence_codes:
            evidence_codes.append(code)
        if flag in {"critical_low", "critical_high"}:
            critical_flags.append(display)

    ml_usage = None
    anomaly_score: float | None = None
    anomaly_flag: bool | None = None
    anomaly_features: int | None = None
    model_version = f"{MODEL_VERSION}+{REFERENCE_VERSION}"

    if anomaly_model is not None:
        info = anomaly_model.info
        if not info.enabled:
            ml_usage = model_usage(
                info, ready=False, fallback=True, detail="disabled by configuration"
            )
        else:
            try:
                anomaly = anomaly_model.score(interpreted)
            except Exception:  # pragma: no cover - defensive
                anomaly = None
            if anomaly is None:
                ml_usage = model_usage(
                    info,
                    ready=False,
                    fallback=True,
                    detail="model unavailable; rule flags used",
                )
            else:
                anomaly_score = anomaly.score
                anomaly_flag = anomaly.is_anomaly
                anomaly_features = anomaly.features_used
                ml_usage = model_usage(
                    info,
                    ready=bool(getattr(anomaly_model, "ready", True)),
                    detail=f"{anomaly.features_used} interpreted value(s)",
                )
                if getattr(anomaly_model, "ready", False):
                    model_version = f"{model_version}+{info.version}"

    status = "partial" if warnings else "ok"
    return LabResult(
        run_id=run.run_id,
        case_id=run.case_id,
        step_id=run.step_id,
        status=status,
        model_version=model_version,
        input_hash=hash_payload(
            {"values": [value.model_dump(mode="json") for value in req.values]}
        ),
        warnings=warnings,
        ml=ml_usage,
        findings=findings,
        evidence_codes=evidence_codes,
        critical_flags=critical_flags,
        ml_anomaly_score=anomaly_score,
        ml_anomaly=anomaly_flag,
        ml_anomaly_features=anomaly_features,
    )
