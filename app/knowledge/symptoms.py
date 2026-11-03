"""Curated symptom lexicon.

The lexicon is the source of truth for symptom evidence codes. It backs the
deterministic regex matcher and normalises output from the optional scispaCy
backend, so results stay stable even without a downloaded NLP model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EVIDENCE_MAP_VERSION = "symptom-lexicon@0.2.0"


@dataclass(frozen=True)
class SymptomConcept:
    code: str
    canonical: str
    aliases: tuple[str, ...] = ()
    red_flag: bool = False


SYMPTOM_CONCEPTS: tuple[SymptomConcept, ...] = (
    SymptomConcept("E_FEVER", "fever", ("febrile", "high temperature", "pyrexia"), True),
    SymptomConcept("E_CHILLS", "chills", ("rigors", "shivering")),
    SymptomConcept("E_COUGH", "cough", ("coughing", "dry cough", "productive cough")),
    SymptomConcept(
        "E_SOB",
        "shortness of breath",
        ("short of breath", "dyspnea", "dyspnoea", "breathlessness", "difficulty breathing", "sob", "cannot catch my breath", "can't catch my breath"),
        True,
    ),
    SymptomConcept("E_HYPOXIA", "hypoxia", ("low oxygen", "low oxygen saturation", "desaturation"), True),
    SymptomConcept(
        "E_CHEST_PAIN",
        "chest pain",
        ("chest tightness", "chest pressure", "substernal pain", "chest discomfort", "tightness in my chest", "pressure in my chest"),
        True,
    ),
    SymptomConcept("E_PALPITATIONS", "palpitations", ("racing heart", "heart racing", "heart flutter")),
    SymptomConcept("E_TACHYCARDIA", "tachycardia", ("fast heart rate", "rapid heart rate", "high heart rate")),
    SymptomConcept(
        "E_DYSURIA",
        "dysuria",
        (
            "painful urination",
            "burning urination",
            "burning on urination",
            "pain when urinating",
            "burning when urinating",
            "burning when i pee",
            "pain when i pee",
            "hurts when i urinate",
        ),
    ),
    SymptomConcept(
        "E_FREQUENCY",
        "urinary frequency",
        ("frequency", "frequent urination", "urinating often", "peeing a lot", "need to pee often", "have to pee all the time"),
    ),
    SymptomConcept("E_URGENCY", "urinary urgency", ("urgency", "urge to urinate", "sudden urge to urinate")),
    SymptomConcept("E_POLYURIA", "polyuria", ("excessive urination", "urinating excessively")),
    SymptomConcept("E_POLYDIPSIA", "polydipsia", ("excessive thirst", "very thirsty")),
    SymptomConcept("E_OLIGURIA", "oliguria", ("decreased urine output", "low urine output")),
    SymptomConcept("E_FLANK_PAIN", "flank pain", ("flank tenderness", "side pain", "costovertebral pain")),
    SymptomConcept("E_HEADACHE", "headache", ("head pain", "head pounding", "my head is pounding")),
    SymptomConcept("E_PHOTOPHOBIA", "photophobia", ("light sensitivity", "sensitivity to light")),
    SymptomConcept("E_DIZZINESS", "dizziness", ("lightheadedness", "light headed", "light-headed")),
    SymptomConcept("E_NAUSEA", "nausea", ("nauseous", "queasy", "sick to my stomach")),
    SymptomConcept("E_VOMITING", "vomiting", ("vomited", "throwing up", "emesis")),
    SymptomConcept("E_DIARRHEA", "diarrhea", ("diarrhoea", "loose stools", "watery stools")),
    SymptomConcept("E_ABDOMINAL_PAIN", "abdominal pain", ("stomach pain", "belly pain", "abdominal tenderness")),
    SymptomConcept("E_BACK_PAIN", "back pain", ("lower back pain",)),
    SymptomConcept("E_APPETITE_LOSS", "appetite loss", ("loss of appetite", "decreased appetite", "anorexia")),
    SymptomConcept("E_FATIGUE", "fatigue", ("tiredness", "lethargy", "malaise", "low energy", "exhausted", "very tired")),
    SymptomConcept("E_SORE_THROAT", "sore throat", ("throat pain", "pharyngitis")),
    SymptomConcept("E_CONGESTION", "nasal congestion", ("congestion", "runny nose", "stuffy nose", "rhinorrhea")),
    SymptomConcept("E_SWEATING", "sweating", ("diaphoresis", "night sweats")),
    SymptomConcept("E_WEIGHT_LOSS", "weight loss", ("losing weight", "unintentional weight loss")),
    SymptomConcept("E_HEMOPTYSIS", "hemoptysis", ("coughing blood", "blood in sputum"), True),
    SymptomConcept("E_EDEMA", "edema", ("swelling", "swollen legs", "ankle swelling", "leg swelling", "swollen ankles", "ankles are swollen")),
    SymptomConcept("E_ORTHOPNEA", "orthopnea", ("shortness of breath lying flat", "breathless when lying flat")),
    SymptomConcept("E_SYNCOPE", "syncope", ("fainting", "passed out", "loss of consciousness", "blacked out"), True),
    SymptomConcept("E_CONFUSION", "confusion", ("confused", "altered mental status", "disoriented"), True),
    SymptomConcept("E_HYPOTENSION", "hypotension", ("low blood pressure",)),
    SymptomConcept("E_RASH", "rash", ("skin rash", "hives")),
    SymptomConcept("E_JAUNDICE", "jaundice", ("yellow skin", "yellow eyes")),
)

ALIAS_TO_CONCEPT: dict[str, SymptomConcept] = {}
for _concept in SYMPTOM_CONCEPTS:
    for _alias in (_concept.canonical, *_concept.aliases):
        ALIAS_TO_CONCEPT[_alias] = _concept

CONCEPTS_BY_CODE: dict[str, SymptomConcept] = {c.code: c for c in SYMPTOM_CONCEPTS}


def concept_description(concept: SymptomConcept) -> str:
    """Natural-language description used to embed the concept for semantic matching."""
    return "; ".join((concept.canonical, *concept.aliases))

RED_FLAG_CODES = frozenset(c.code for c in SYMPTOM_CONCEPTS if c.red_flag)


def _build_alias_pattern() -> re.Pattern[str]:
    aliases = sorted(ALIAS_TO_CONCEPT, key=len, reverse=True)
    alternatives = "|".join(re.escape(a).replace("\\ ", r"\s+") for a in aliases)
    # Optional plural suffix on the final word; boundaries keep "sob" out of
    # "sobbing" and "frequency" out of unrelated words.
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?:s|es)?(?![a-z0-9])", re.IGNORECASE)


ALIAS_PATTERN = _build_alias_pattern()

NEGATION_CUES = frozenset(
    {"no", "not", "denies", "denied", "without", "never", "negative", "absent", "none", "nil", "neither", "nor"}
)
AFFIRMATION_CUES = frozenset(
    {
        "reports",
        "reported",
        "endorses",
        "has",
        "have",
        "having",
        "with",
        "presents",
        "presenting",
        "developed",
        "positive",
        "notes",
        "admits",
        "experiencing",
        "complains",
        "complaining",
    }
)

CLAUSE_BOUNDARY = re.compile(r"[.;!?\n]|\b(?:but|however|although|though|except)\b", re.IGNORECASE)
