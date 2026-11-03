"""Drug interaction agent: normalises medications and checks a curated table.

Name resolution is hybrid: exact/alias matches come from curated data, while an
optional resolver adds fuzzy and embedding-based matching for misspellings and
brand variants. Approximate matches are always surfaced in ``warnings``.
"""

from __future__ import annotations

from app.agents.drugs.normalize import (
    KNOWN_INGREDIENTS,
    clean_medication_name,
    split_ingredients,
)
from app.agents.drugs.schemas import (
    DrugSafetyRequest,
    DrugSafetyResult,
    DuplicateClassFinding,
    InteractionFinding,
    NormalizedMedication,
)
from app.knowledge.interactions import (
    DRUG_CLASSES,
    INTERACTION_INDEX,
    INTERACTION_VERSION,
    SEVERITY_ORDER,
    interaction_key,
)
from app.ml.base import model_usage
from app.ml.drug_resolver import DrugResolver
from app.schemas.common import hash_payload, resolve_run

AGENT_NAME = "drugs"
MODEL_VERSION = "drug-safety@0.2.0"
FEATURE = "drugs"


def run_drug_agent(
    req: DrugSafetyRequest,
    resolver: DrugResolver | None = None,
) -> DrugSafetyResult:
    run = resolve_run(req, AGENT_NAME)
    normalized: list[NormalizedMedication] = []
    warnings: list[str] = []
    approximate = 0

    for med in req.medications:
        ingredients = split_ingredients(med.name)
        if not ingredients:
            warnings.append(f"Could not parse medication name {med.name!r}.")
            continue
        cleaned = clean_medication_name(med.name)

        for ingredient in ingredients:
            method = "alias" if ingredient not in cleaned else "exact"
            confidence = 1.0
            if ingredient not in KNOWN_INGREDIENTS and resolver is not None:
                match = resolver.resolve(ingredient)
                if match.ingredient:
                    ingredient = match.ingredient
                    method = match.method
                    confidence = match.confidence
                else:
                    method = "unresolved"
                    confidence = match.confidence
            elif ingredient not in KNOWN_INGREDIENTS:
                method = "unresolved"

            if method in {"fuzzy", "embedding"}:
                approximate += 1
                warnings.append(
                    f"Medication {med.name!r} was matched to {ingredient!r} by "
                    f"{method} (confidence {confidence:.2f}); verify the name."
                )

            normalized.append(
                NormalizedMedication(
                    input_name=med.name,
                    ingredient=ingredient,
                    drug_class=DRUG_CLASSES.get(ingredient),
                    resolution=method,
                    confidence=confidence,
                )
            )

    ingredients = list(dict.fromkeys(item.ingredient for item in normalized))
    unknown = [name for name in ingredients if name not in KNOWN_INGREDIENTS]
    for name in unknown:
        warnings.append(f"Medication {name!r} is not in the interaction knowledge base; not checked.")

    interactions: list[InteractionFinding] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, first in enumerate(ingredients):
        for second in ingredients[i + 1 :]:
            key = interaction_key(first, second)
            for rule in INTERACTION_INDEX.get(key, []):
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                interactions.append(
                    InteractionFinding(
                        drugs=[key[0], key[1]],
                        severity=rule.severity,
                        mechanism=rule.mechanism,
                        recommendation=rule.recommendation,
                        source=INTERACTION_VERSION,
                    )
                )

    by_class: dict[str, list[str]] = {}
    for ingredient in ingredients:
        drug_class = DRUG_CLASSES.get(ingredient)
        if drug_class:
            by_class.setdefault(drug_class, []).append(ingredient)
    duplicates = [
        DuplicateClassFinding(drug_class=drug_class, ingredients=members)
        for drug_class, members in sorted(by_class.items())
        if len(members) > 1
    ]

    overall_risk = "info"
    for finding in interactions:
        if SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[overall_risk]:
            overall_risk = finding.severity

    ml_usage = None
    model_version = f"{MODEL_VERSION}+{INTERACTION_VERSION}"
    if resolver is not None:
        info = resolver.info
        if not info.enabled:
            ml_usage = model_usage(
                info, ready=False, fallback=True, detail="disabled by configuration"
            )
        else:
            ml_usage = model_usage(
                info,
                ready=bool(getattr(resolver, "ready", True)),
                fallback=False,
                detail=f"{approximate} approximate match(es)",
            )
            if getattr(resolver, "ready", False):
                model_version = f"{model_version}+{info.version}"

    # The resolver is optional; brand/exact normalisation already gives useful
    # output. Keep "partial" tied to unresolved or approximate meds so a clean
    # exact list is still "ok".
    status = "partial" if warnings else "ok"
    return DrugSafetyResult(
        run_id=run.run_id,
        case_id=run.case_id,
        step_id=run.step_id,
        status=status,
        model_version=model_version,
        input_hash=hash_payload(
            {
                "medications": [med.model_dump(mode="json") for med in req.medications],
                "patient": req.patient.model_dump(mode="json"),
            }
        ),
        warnings=warnings,
        ml=ml_usage,
        normalized_medications=normalized,
        interactions=interactions,
        duplicate_classes=duplicates,
        unknown_medications=unknown,
        overall_risk=overall_risk,
    )
