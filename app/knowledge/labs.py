"""Reference ranges and evidence codes for common laboratory tests.

Ranges are adult, general-population approximations for demonstration only.
Callers can override the range per value when a lab reports its own reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCE_VERSION = "lab-references@2026.1"


def _normalise_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


@dataclass(frozen=True)
class LabReference:
    key: str
    name: str
    unit: str
    aliases: tuple[str, ...]
    low: float | None = None
    high: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None
    code_low: str | None = None
    code_high: str | None = None
    code_critical_low: str | None = None
    code_critical_high: str | None = None


REFERENCES: tuple[LabReference, ...] = (
    LabReference(
        "wbc", "White blood cells", "10^9/L",
        ("wbc", "white blood cell count", "white blood cells", "white count", "leukocytes", "leucocytes"),
        4.0, 11.0, 1.0, 30.0, "E_LEUKOPENIA", "E_LEUKOCYTOSIS",
    ),
    LabReference(
        "hemoglobin", "Hemoglobin", "g/dL",
        ("hb", "hgb", "haemoglobin", "hemoglobin"),
        12.0, 17.5, 7.0, 20.0, "E_ANEMIA", "E_ERYTHROCYTOSIS",
    ),
    LabReference(
        "platelets", "Platelets", "10^9/L",
        ("plt", "platelet count", "thrombocytes"),
        150.0, 450.0, 50.0, 1000.0, "E_THROMBOCYTOPENIA", "E_THROMBOCYTOSIS",
    ),
    LabReference(
        "crp", "C-reactive protein", "mg/L",
        ("crp", "c reactive protein", "c-reactive protein"),
        0.0, 5.0, None, 200.0, None, "E_CRP_ELEVATED", None, "E_CRP_MARKEDLY_ELEVATED",
    ),
    LabReference(
        "lactate", "Lactate", "mmol/L",
        ("lactate", "lactic acid", "blood lactate"),
        0.5, 2.2, None, 4.0, None, "E_LACTATE_ELEVATED", None, "E_LACTATE_CRITICAL",
    ),
    LabReference(
        "creatinine", "Creatinine", "mg/dL",
        ("creatinine", "serum creatinine", "creat"),
        0.6, 1.3, None, 4.0, "E_CREATININE_LOW", "E_AKI", None, "E_AKI_CRITICAL",
    ),
    LabReference(
        "egfr", "Estimated GFR", "mL/min/1.73m2",
        ("egfr", "estimated gfr", "gfr"),
        90.0, None, 15.0, None, "E_RENAL_IMPAIRMENT", None, "E_RENAL_FAILURE", None,
    ),
    LabReference(
        "sodium", "Sodium", "mmol/L",
        ("na", "sodium", "serum sodium"),
        135.0, 145.0, 120.0, 160.0, "E_HYPONATREMIA", "E_HYPERNATREMIA",
    ),
    LabReference(
        "potassium", "Potassium", "mmol/L",
        ("k", "potassium", "serum potassium"),
        3.5, 5.1, 2.5, 6.5, "E_HYPOKALEMIA", "E_HYPERKALEMIA",
    ),
    LabReference(
        "glucose", "Glucose", "mg/dL",
        ("glucose", "blood glucose", "blood sugar", "serum glucose"),
        70.0, 140.0, 40.0, 400.0, "E_HYPOGLYCEMIA", "E_HYPERGLYCEMIA",
    ),
    LabReference(
        "troponin", "Troponin I", "ng/mL",
        ("troponin", "troponin i", "troponin t", "hs troponin", "high sensitivity troponin"),
        0.0, 0.04, None, 0.4, None, "E_TROPONIN_ELEVATED", None, "E_TROPONIN_CRITICAL",
    ),
    LabReference(
        "ddimer", "D-dimer", "mg/L FEU",
        ("d-dimer", "d dimer", "ddimer"),
        0.0, 0.5, None, None, None, "E_DDIMER_ELEVATED",
    ),
    LabReference(
        "bnp", "BNP", "pg/mL",
        ("bnp", "b-type natriuretic peptide", "nt-probnp", "nt probnp"),
        0.0, 100.0, None, None, None, "E_BNP_ELEVATED",
    ),
    LabReference(
        "alt", "ALT", "U/L",
        ("alt", "alanine aminotransferase", "sgpt", "sgot"),
        7.0, 56.0, None, None, None, "E_TRANSAMINITIS",
    ),
    LabReference(
        "bilirubin", "Total bilirubin", "mg/dL",
        ("bilirubin", "total bilirubin", "bili"),
        0.1, 1.2, None, 10.0, None, "E_HYPERBILIRUBINEMIA", None, "E_HYPERBILIRUBINEMIA_CRITICAL",
    ),
    LabReference(
        "inr", "INR", "ratio",
        ("inr", "international normalized ratio", "pt inr"),
        0.8, 1.2, None, 5.0, "E_COAGULOPATHY", None, None, "E_COAGULOPATHY_CRITICAL",
    ),
    LabReference(
        "ph", "Blood pH", "ratio",
        ("ph", "blood ph", "arterial ph"),
        7.35, 7.45, 7.20, 7.60, "E_ACIDEMIA", "E_ALKALEMIA",
    ),
    LabReference(
        "lipase", "Lipase", "U/L",
        ("lipase", "serum lipase"),
        0.0, 160.0, None, None, None, "E_LIPASE_ELEVATED",
    ),
)

LAB_ALIASES: dict[str, LabReference] = {}
for _ref in REFERENCES:
    for _alias in (_ref.key, _ref.name, *_ref.aliases):
        LAB_ALIASES[_normalise_key(_alias)] = _ref

_UNIT_ALIASES = {
    "": "ratio",
    "ratio": "ratio",
    "x10^9/l": "10^9/l",
    "10*9/l": "10^9/l",
    "10^9/l": "10^9/l",
    "k/ul": "10^9/l",
    "thou/ul": "10^9/l",
    "mg/dl": "mg/dl",
    "mg/l": "mg/l",
    "mmol/l": "mmol/l",
    "ng/ml": "ng/ml",
    "pg/ml": "pg/ml",
    "u/l": "u/l",
    "iu/l": "u/l",
    "ml/min/1.73m2": "ml/min/1.73m2",
    "ml/min/1.73m^2": "ml/min/1.73m2",
}


def normalise_lab_name(value: str) -> str:
    return _normalise_key(re.sub(r"[^a-z0-9^./\- ]+", " ", value.lower()))


def normalise_unit(value: str | None) -> str:
    if value is None:
        return "ratio"
    cleaned = _normalise_key(value.lower().replace("µ", "u").replace("μ", "u"))
    cleaned = re.sub(r"\s+", "", cleaned)
    return _UNIT_ALIASES.get(cleaned, cleaned)


def lookup_reference(name: str) -> LabReference | None:
    return LAB_ALIASES.get(normalise_lab_name(name))
