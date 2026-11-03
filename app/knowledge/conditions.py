"""Weighted evidence knowledge base for the differential-diagnosis agent.

Weights are log-odds contributions, not calibrated probabilities. The priors
are deliberately conservative; a condition only appears when at least one of
its weighted evidence codes is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

KNOWLEDGE_BASE_VERSION = "condition-kb@0.2.0"

Acuity = Literal["routine", "urgent", "emergency"]


@dataclass(frozen=True)
class ConditionRule:
    condition_id: str
    name: str
    acuity: Acuity
    prior: float
    weights: dict[str, float]
    sex_weights: dict[str, float] = field(default_factory=dict)
    age_weights: tuple[tuple[int, int, float], ...] = ()


CONDITIONS: tuple[ConditionRule, ...] = (
    ConditionRule(
        "COND_UTI", "Urinary tract infection", "routine", -1.6,
        {
            "E_DYSURIA": 2.4, "E_FREQUENCY": 1.8, "E_URGENCY": 1.4,
            "E_FLANK_PAIN": 1.0, "E_FEVER": 0.5, "E_LEUKOCYTOSIS": 0.7,
            "E_CRP_ELEVATED": 0.4,
        },
        sex_weights={"female": 0.9},
        age_weights=((65, 121, 0.3),),
    ),
    ConditionRule(
        "COND_PYELONEPHRITIS", "Pyelonephritis", "urgent", -2.6,
        {
            "E_DYSURIA": 1.0, "E_FLANK_PAIN": 2.2, "E_FEVER": 1.3,
            "E_LEUKOCYTOSIS": 1.0, "E_CRP_ELEVATED": 0.8, "E_NAUSEA": 0.4,
            "E_VOMITING": 0.5, "E_CHILLS": 1.0,
        },
    ),
    ConditionRule(
        "COND_PNEUMONIA", "Community-acquired pneumonia", "urgent", -1.8,
        {
            "E_COUGH": 1.9, "E_FEVER": 1.2, "E_SOB": 1.0,
            "E_LEUKOCYTOSIS": 1.1, "E_CRP_ELEVATED": 1.0, "E_CHILLS": 0.6,
            "E_HYPOXIA": 1.2, "E_CHEST_PAIN": 0.3, "E_CONFUSION": 0.5,
            "E_CRP_MARKEDLY_ELEVATED": 0.7,
        },
        age_weights=((65, 121, 0.4),),
    ),
    ConditionRule(
        "COND_ACS", "Acute coronary syndrome", "emergency", -2.2,
        {
            "E_CHEST_PAIN": 2.6, "E_SWEATING": 1.5, "E_NAUSEA": 0.6,
            "E_SOB": 0.9, "E_TROPONIN_ELEVATED": 2.6, "E_TROPONIN_CRITICAL": 2.9,
            "E_FATIGUE": 0.3, "E_SYNCOPE": 0.4,
        },
        sex_weights={"male": 0.4},
        age_weights=((50, 121, 0.7),),
    ),
    ConditionRule(
        "COND_SEPSIS", "Sepsis / septic shock", "emergency", -3.0,
        {
            "E_FEVER": 1.2, "E_CHILLS": 0.8, "E_LEUKOCYTOSIS": 1.0,
            "E_LACTATE_ELEVATED": 1.8, "E_LACTATE_CRITICAL": 2.4,
            "E_CONFUSION": 1.4, "E_HYPOTENSION": 1.6, "E_HYPOXIA": 0.8,
            "E_CRP_ELEVATED": 0.6, "E_CRP_MARKEDLY_ELEVATED": 0.9,
            "E_AKI": 0.6, "E_TACHYCARDIA": 0.6,
        },
        age_weights=((65, 121, 0.5),),
    ),
    ConditionRule(
        "COND_VIRAL_URI", "Viral upper respiratory infection", "routine", -0.9,
        {
            "E_SORE_THROAT": 1.7, "E_COUGH": 1.3, "E_FEVER": 0.7,
            "E_FATIGUE": 0.6, "E_HEADACHE": 0.4, "E_CONGESTION": 1.4,
            "E_NAUSEA": 0.2,
        },
    ),
    ConditionRule(
        "COND_MIGRAINE", "Migraine", "routine", -1.9,
        {
            "E_HEADACHE": 2.3, "E_NAUSEA": 0.7, "E_VOMITING": 0.4,
            "E_DIZZINESS": 0.5, "E_PHOTOPHOBIA": 1.6,
        },
    ),
    ConditionRule(
        "COND_GASTROENTERITIS", "Acute gastroenteritis", "routine", -1.5,
        {
            "E_DIARRHEA": 2.3, "E_VOMITING": 1.6, "E_NAUSEA": 1.1,
            "E_ABDOMINAL_PAIN": 0.9, "E_FEVER": 0.5, "E_APPETITE_LOSS": 0.4,
        },
    ),
    ConditionRule(
        "COND_HEART_FAILURE", "Heart failure decompensation", "urgent", -2.2,
        {
            "E_SOB": 1.6, "E_EDEMA": 1.7, "E_FATIGUE": 0.6,
            "E_ORTHOPNEA": 1.8, "E_BNP_ELEVATED": 2.0,
            "E_PALPITATIONS": 0.4, "E_COUGH": 0.3,
        },
        age_weights=((65, 121, 0.4),),
    ),
    ConditionRule(
        "COND_PE", "Pulmonary embolism", "emergency", -3.0,
        {
            "E_SOB": 1.7, "E_CHEST_PAIN": 0.9, "E_HEMOPTYSIS": 1.3,
            "E_DDIMER_ELEVATED": 1.9, "E_SYNCOPE": 0.6, "E_PALPITATIONS": 0.5,
            "E_HYPOXIA": 1.0, "E_TACHYCARDIA": 0.7,
        },
    ),
    ConditionRule(
        "COND_APPENDICITIS", "Appendicitis", "emergency", -2.6,
        {
            "E_ABDOMINAL_PAIN": 1.9, "E_NAUSEA": 0.9, "E_VOMITING": 0.7,
            "E_FEVER": 0.9, "E_LEUKOCYTOSIS": 0.9, "E_APPETITE_LOSS": 0.8,
        },
    ),
    ConditionRule(
        "COND_PANCREATITIS", "Acute pancreatitis", "urgent", -3.2,
        {
            "E_ABDOMINAL_PAIN": 1.8, "E_NAUSEA": 1.0, "E_VOMITING": 1.0,
            "E_LIPASE_ELEVATED": 2.6, "E_APPETITE_LOSS": 0.3,
        },
    ),
    ConditionRule(
        "COND_HYPERGLYCEMIC_CRISIS", "Hyperglycemic crisis", "emergency", -3.6,
        {
            "E_HYPERGLYCEMIA": 2.4, "E_CONFUSION": 1.0, "E_NAUSEA": 0.5,
            "E_VOMITING": 0.5, "E_FATIGUE": 0.4, "E_POLYURIA": 0.8,
            "E_POLYDIPSIA": 0.8, "E_APPETITE_LOSS": 0.3,
        },
    ),
)

CONDITIONS_BY_ID: dict[str, ConditionRule] = {c.condition_id: c for c in CONDITIONS}


def all_evidence_codes() -> list[str]:
    codes: set[str] = set()
    for condition in CONDITIONS:
        codes.update(condition.weights)
    return sorted(codes)
