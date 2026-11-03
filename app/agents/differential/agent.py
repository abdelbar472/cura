"""Differential diagnosis agent: weighted evidence -> ranked considerations.

The deterministic weighted-log-odds score is always computed. When a learned
re-ranker is available, its probability is blended in with a fixed weight so a
model surprise cannot silently outrank a strongly supported rule.
"""

from __future__ import annotations

import math

from app.agents.differential.schemas import (
    ConditionConsideration,
    DifferentialRequest,
    DifferentialResult,
    EvidenceContribution,
)
from app.knowledge.conditions import (
    CONDITIONS,
    KNOWLEDGE_BASE_VERSION,
    ConditionRule,
)
from app.ml.base import model_usage
from app.ml.differential_logreg import DifferentialLogRegModel
from app.schemas.common import PatientContext, hash_payload, resolve_run

AGENT_NAME = "differential"
MODEL_VERSION = "differential-weighted@0.2.0"
FEATURE = "differential"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.05
DEFAULT_ML_WEIGHT = 0.35


def _age_bonus(rule: ConditionRule, patient: PatientContext) -> float:
    if patient.age_years is None:
        return 0.0
    return sum(
        bonus for low, high, bonus in rule.age_weights if low <= patient.age_years <= high
    )


def _score(rule: ConditionRule, present: set[str], patient: PatientContext) -> float:
    logit = rule.prior + sum(rule.weights[code] for code in rule.weights if code in present)
    logit += rule.sex_weights.get(patient.sex, 0.0)
    logit += _age_bonus(rule, patient)
    return 1.0 / (1.0 + math.exp(-logit))


def run_differential_agent(
    req: DifferentialRequest,
    reranker: DifferentialLogRegModel | None = None,
) -> DifferentialResult:
    run = resolve_run(req, AGENT_NAME)
    present = set(req.evidence_codes)
    top_k = req.top_k or DEFAULT_TOP_K
    min_score = req.min_score if req.min_score is not None else DEFAULT_MIN_SCORE

    ml_scores: dict[str, float] | None = None
    ml_usage = None
    ml_weight = DEFAULT_ML_WEIGHT
    method = "weighted-log-odds@1"
    model_version = f"{MODEL_VERSION}+{KNOWLEDGE_BASE_VERSION}"

    if reranker is not None:
        info = reranker.info
        if not info.enabled:
            ml_usage = model_usage(
                info, ready=False, fallback=True, detail="disabled by configuration"
            )
        else:
            try:
                ml_scores = reranker.predict(req.evidence_codes)
            except Exception:  # pragma: no cover - defensive
                ml_scores = None
            if ml_scores is None:
                ml_usage = model_usage(
                    info,
                    ready=False,
                    fallback=True,
                    detail="model unavailable; rule scores used",
                )
            else:
                ml_weight = float(getattr(reranker, "ml_weight", DEFAULT_ML_WEIGHT))
                ml_usage = model_usage(
                    info,
                    ready=bool(getattr(reranker, "ready", True)),
                    detail=f"{len(ml_scores)} condition probabilities",
                )
                if getattr(reranker, "ready", False):
                    method = "blended-weighted-logreg@1"
                    model_version = f"{model_version}+{info.version}"

    considerations: list[ConditionConsideration] = []
    for rule in CONDITIONS:
        matched = [code for code in rule.weights if code in present]
        if not matched:
            continue

        rule_score = _score(rule, present, req.patient)
        ml_score = ml_scores.get(rule.condition_id) if ml_scores is not None else None
        score = (
            rule_score
            if ml_score is None
            else ((1.0 - ml_weight) * rule_score + ml_weight * ml_score)
        )
        if score < min_score:
            continue

        key_evidence = sorted(rule.weights, key=lambda c: rule.weights[c], reverse=True)[:3]
        missing = [code for code in key_evidence if code not in present]
        contributions = [
            EvidenceContribution(code=code, weight=rule.weights[code])
            for code in sorted(matched, key=lambda c: rule.weights[c], reverse=True)
        ]
        rationale = f"{len(matched)} supporting finding(s): {', '.join(matched)}."
        if missing:
            rationale += f" Key evidence not reported: {', '.join(missing)}."
        if ml_score is not None:
            rationale += (
                f" Blended score ({(1.0 - ml_weight):.0%} rules + "
                f"{ml_weight:.0%} learned re-ranker)."
            )

        considerations.append(
            ConditionConsideration(
                condition_id=rule.condition_id,
                name=rule.name,
                score=round(score, 4),
                ml_score=None if ml_score is None else round(ml_score, 4),
                acuity=rule.acuity,
                matched_evidence=matched,
                missing_key_evidence=missing,
                contributions=contributions,
                rationale=rationale,
            )
        )

    considerations.sort(key=lambda c: (-c.score, c.condition_id))
    considerations = considerations[:top_k]

    warnings: list[str] = []
    if not considerations:
        warnings.append("No knowledge-base condition matched the supplied evidence.")

    return DifferentialResult(
        run_id=run.run_id,
        case_id=run.case_id,
        step_id=run.step_id,
        status="ok",
        model_version=model_version,
        input_hash=hash_payload(
            {
                "evidence_codes": req.evidence_codes,
                "patient": req.patient.model_dump(mode="json"),
                "top_k": req.top_k,
                "min_score": req.min_score,
            }
        ),
        warnings=warnings,
        ml=ml_usage,
        method=method,
        evidence_used=sorted(present),
        considerations=considerations,
    )
