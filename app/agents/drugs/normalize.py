"""Medication-name normalisation for the drug-safety agent."""

from __future__ import annotations

import re

from app.knowledge.interactions import DRUG_ALIASES, DRUG_CLASSES, INTERACTIONS

_SPLIT = re.compile(r"\s*[/+]\s*|\s+and\s+", re.IGNORECASE)
_FORM_SUFFIXES = {"hydrochloride", "hcl", "er", "xr", "cr", "sr", "ds", "forte"}

KNOWN_INGREDIENTS: set[str] = (
    set(DRUG_CLASSES)
    | set(DRUG_ALIASES.values())
    | {name for rule in INTERACTIONS for name in (rule.a, rule.b)}
    | {
        "acetaminophen", "levothyroxine", "amlodipine", "metformin",
        "cetirizine", "fluconazole", "metronidazole", "clarithromycin",
        "ondansetron", "verapamil", "tacrolimus", "ketoconazole",
        "hydrochlorothiazide", "potassium chloride", "sodium chloride",
        "amoxicillin/clavulanate", "cephalexin", "nitroglycerin",
    }
)


def clean_medication_name(raw: str) -> str:
    value = raw.strip().lower()
    value = re.sub(r"\([^)]*\)", " ", value)          # drop brand parentheticals
    value = re.split(r"\d", value, maxsplit=1)[0]     # drop dose and everything after
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z/+ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip(" /")


def _canonical_single(value: str) -> str:
    value = value.strip()
    if value in DRUG_ALIASES:
        return DRUG_ALIASES[value]
    tokens = value.split()
    if len(tokens) > 1 and tokens[-1] in _FORM_SUFFIXES:
        stripped = " ".join(tokens[:-1])
        if stripped in DRUG_ALIASES or stripped in DRUG_CLASSES:
            return DRUG_ALIASES.get(stripped, stripped)
    return DRUG_ALIASES.get(value, value)


def split_ingredients(raw: str) -> list[str]:
    """Return canonical ingredient names, splitting combination products."""
    base = clean_medication_name(raw)
    if not base:
        return []
    if base in DRUG_ALIASES:
        return [DRUG_ALIASES[base]]
    parts = [p for p in _SPLIT.split(base) if p]
    if len(parts) <= 1:
        return [_canonical_single(base)]
    return [_canonical_single(p) for p in parts]
