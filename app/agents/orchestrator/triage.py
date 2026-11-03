"""Rule-based triage derived from every agent's output."""

from __future__ import annotations

from app.agents.differential.schemas import DifferentialResult
from app.agents.drugs.schemas import DrugSafetyResult
from app.agents.labs.schemas import LabResult
from app.agents.symptom.schemas import SymptomResult
from app.schemas.triage import ACTIONS, LEVEL_ORDER, TriageAssessment

_EMERGENCY_SYMPTOMS = {
    "E_CHEST_PAIN",
    "E_CONFUSION",
    "E_SYNCOPE",
    "E_HEMOPTYSIS",
    "E_HYPOXIA",
}
_URGENT_LAB_CODES = {
    "E_AKI",
    "E_AKI_CRITICAL",
    "E_HYPERKALEMIA",
    "E_HYPOKALEMIA",
    "E_HYPONATREMIA",
    "E_TROPONIN_ELEVATED",
    "E_LACTATE_ELEVATED",
    "E_CRP_MARKEDLY_ELEVATED",
    "E_HYPERGLYCEMIA",
    "E_RENAL_IMPAIRMENT",
    "E_COAGULOPATHY",
}


def assess(
    symptom: SymptomResult,
    labs: LabResult,
    differential: DifferentialResult,
    drugs: DrugSafetyResult,
) -> TriageAssessment:
    level = "routine"
    reasons: list[str] = []

    def escalate(target: str, reason: str) -> None:
        nonlocal level
        if LEVEL_ORDER[target] > LEVEL_ORDER[level]:
            level = target
        reasons.append(reason)

    if labs.critical_flags:
        escalate(
            "emergency",
            "Critical laboratory value(s): " + ", ".join(labs.critical_flags) + ".",
        )

    for flag in labs.findings:
        if flag.evidence_code in _URGENT_LAB_CODES and flag.flag in {"low", "high"}:
            escalate("urgent", f"Clinically significant lab abnormality: {flag.name} ({flag.flag}).")

    for interaction in drugs.interactions:
        pair = " + ".join(interaction.drugs)
        if interaction.severity == "contraindicated":
            escalate("emergency", f"Contraindicated drug combination: {pair}.")
        elif interaction.severity == "major":
            escalate("urgent", f"Major drug interaction: {pair}.")

    for code in symptom.red_flags:
        if code in _EMERGENCY_SYMPTOMS:
            escalate("emergency", f"Red-flag symptom reported: {code}.")
        elif code == "E_SOB":
            escalate("urgent", "Shortness of breath requires prompt assessment.")
        elif code == "E_FEVER":
            reasons.append("Fever reported; assess for systemic infection.")

    systemic = {"E_LEUKOCYTOSIS", "E_CRP_ELEVATED", "E_LACTATE_ELEVATED"}
    if "E_FEVER" in symptom.evidence_codes and systemic & set(symptom.evidence_codes + labs.evidence_codes):
        escalate("urgent", "Fever with inflammatory markers suggests possible systemic infection.")

    if differential.considerations:
        top = differential.considerations[0]
        if top.acuity == "emergency" and top.score >= 0.35:
            escalate("emergency", f"Leading consideration is an emergency condition: {top.name}.")
        elif top.acuity == "urgent" and top.score >= 0.3:
            escalate("urgent", f"Leading consideration requires prompt assessment: {top.name}.")

    if not reasons:
        reasons.append("No urgent or emergency features were detected by the rule set.")

    return TriageAssessment(
        level=level,
        action=ACTIONS[level],
        reasons=list(dict.fromkeys(reasons)),
        red_flags=list(symptom.red_flags),
    )
