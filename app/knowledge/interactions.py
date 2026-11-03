"""Curated drug-interaction and duplicate-therapy knowledge base.

Small and explicit on purpose: every entry is reviewed data, not a prediction.
It is not a substitute for a clinical interaction database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

INTERACTION_VERSION = "interaction-kb@2026.1"

Severity = Literal["minor", "moderate", "major", "contraindicated"]


@dataclass(frozen=True)
class InteractionRule:
    a: str
    b: str
    severity: Severity
    mechanism: str
    recommendation: str


INTERACTIONS: tuple[InteractionRule, ...] = (
    InteractionRule("warfarin", "ibuprofen", "major", "Additive bleeding risk (anticoagulation plus antiplatelet/GI irritation).", "Avoid NSAIDs; use acetaminophen where appropriate and monitor for bleeding."),
    InteractionRule("warfarin", "naproxen", "major", "Additive bleeding risk (anticoagulation plus antiplatelet/GI irritation).", "Avoid NSAIDs; monitor INR and for bleeding."),
    InteractionRule("warfarin", "aspirin", "major", "Additive bleeding risk and displacement from protein binding.", "Use only with a documented indication; monitor INR and for bleeding."),
    InteractionRule("warfarin", "amiodarone", "major", "Amiodarone inhibits warfarin metabolism, raising INR.", "Reduce warfarin dose and monitor INR closely."),
    InteractionRule("warfarin", "cotrimoxazole", "major", "CYP2C9 inhibition plus protein-binding displacement raises INR.", "Prefer an alternative antibiotic; monitor INR closely."),
    InteractionRule("apixaban", "ketoconazole", "major", "Strong CYP3A4 and P-glycoprotein inhibition increases apixaban exposure.", "Avoid combination; consider a different antifungal or anticoagulant."),
    InteractionRule("apixaban", "ibuprofen", "major", "Additive bleeding risk.", "Avoid NSAIDs; monitor for bleeding."),
    InteractionRule("clopidogrel", "omeprazole", "moderate", "CYP2C19 inhibition reduces clopidogrel activation.", "Consider pantoprazole or an H2 blocker instead."),
    InteractionRule("simvastatin", "clarithromycin", "contraindicated", "CYP3A4 inhibition markedly raises simvastatin exposure and rhabdomyolysis risk.", "Do not co-administer; hold the statin during therapy."),
    InteractionRule("simvastatin", "amiodarone", "major", "CYP3A4 inhibition increases simvastatin exposure.", "Cap simvastatin at 20 mg/day or use a different statin."),
    InteractionRule("atorvastatin", "clarithromycin", "major", "CYP3A4 inhibition increases atorvastatin exposure and myopathy risk.", "Suspend the statin or use a lower dose during therapy."),
    InteractionRule("sertraline", "tramadol", "major", "Serotonergic synergy plus lowered seizure threshold.", "Avoid combination; choose a non-serotonergic analgesic."),
    InteractionRule("fluoxetine", "tramadol", "major", "Serotonergic synergy plus lowered seizure threshold.", "Avoid combination; choose a non-serotonergic analgesic."),
    InteractionRule("sertraline", "ibuprofen", "moderate", "SSRI plus NSAID increases gastrointestinal bleeding risk.", "Consider gastroprotection or a non-NSAID analgesic."),
    InteractionRule("lisinopril", "spironolactone", "major", "Dual renin-angiotensin-aldosterone blockade increases hyperkalemia risk.", "Monitor potassium and renal function closely."),
    InteractionRule("lisinopril", "ibuprofen", "moderate", "NSAID reduces antihypertensive effect and can precipitate kidney injury.", "Limit NSAID use; monitor blood pressure and creatinine."),
    InteractionRule("lisinopril", "potassium chloride", "major", "Additive hyperkalemia risk.", "Avoid routine supplementation; monitor potassium."),
    InteractionRule("spironolactone", "potassium chloride", "major", "Additive hyperkalemia risk.", "Avoid routine supplementation; monitor potassium."),
    InteractionRule("digoxin", "amiodarone", "major", "Amiodarone raises digoxin levels and can cause toxicity.", "Halve the digoxin dose and monitor levels."),
    InteractionRule("digoxin", "furosemide", "moderate", "Diuretic-induced hypokalemia increases digoxin toxicity risk.", "Monitor potassium and magnesium."),
    InteractionRule("methotrexate", "ibuprofen", "major", "NSAIDs reduce methotrexate clearance, increasing toxicity.", "Avoid NSAIDs; monitor blood counts and renal function."),
    InteractionRule("lithium", "ibuprofen", "major", "NSAIDs reduce lithium clearance, raising levels.", "Avoid NSAIDs; monitor lithium levels."),
    InteractionRule("lithium", "hydrochlorothiazide", "major", "Thiazides reduce lithium clearance, raising levels.", "Monitor lithium levels and consider a dose reduction."),
    InteractionRule("sildenafil", "nitroglycerin", "contraindicated", "Additive vasodilation can cause profound hypotension.", "Do not combine; separate doses per guidance and seek emergency care for chest pain."),
    InteractionRule("tadalafil", "nitroglycerin", "contraindicated", "Additive vasodilation can cause profound hypotension.", "Do not combine; seek emergency care for chest pain."),
    InteractionRule("linezolid", "sertraline", "major", "Monoamine oxidase inhibition plus serotonergic drug raises serotonin syndrome risk.", "Avoid combination unless essential; monitor closely."),
    InteractionRule("ciprofloxacin", "tizanidine", "contraindicated", "CYP1A2 inhibition markedly raises tizanidine exposure (hypotension, sedation).", "Do not co-administer."),
    InteractionRule("alprazolam", "oxycodone", "major", "Additive central nervous system and respiratory depression.", "Avoid combination; if unavoidable, use the lowest doses and monitor closely."),
    InteractionRule("alprazolam", "tramadol", "major", "Additive sedation and respiratory depression.", "Avoid combination; monitor closely."),
    InteractionRule("furosemide", "ibuprofen", "moderate", "NSAID reduces diuretic effect and increases kidney injury risk.", "Limit NSAID use; monitor renal function."),
    InteractionRule("colchicine", "clarithromycin", "major", "CYP3A4 and P-glycoprotein inhibition increases colchicine toxicity.", "Avoid combination or reduce colchicine dose substantially."),
    InteractionRule("azithromycin", "ondansetron", "moderate", "Additive QT prolongation.", "Check the ECG and correct electrolytes before combining."),
    InteractionRule("tacrolimus", "ketoconazole", "major", "CYP3A4 inhibition raises tacrolimus levels.", "Avoid or monitor trough levels closely."),
    InteractionRule("metoprolol", "verapamil", "major", "Additive negative chronotropy and AV nodal blockade.", "Monitor heart rate and blood pressure; avoid in severe conduction disease."),
)

# Ingredient -> therapeutic class, used for duplicate-therapy detection.
DRUG_CLASSES: dict[str, str] = {
    "ibuprofen": "nsaid", "naproxen": "nsaid", "diclofenac": "nsaid",
    "celecoxib": "nsaid", "indomethacin": "nsaid",
    "sertraline": "ssri", "fluoxetine": "ssri", "citalopram": "ssri",
    "escitalopram": "ssri", "paroxetine": "ssri",
    "warfarin": "anticoagulant", "apixaban": "anticoagulant",
    "rivaroxaban": "anticoagulant", "dabigatran": "anticoagulant",
    "aspirin": "antiplatelet", "clopidogrel": "antiplatelet",
    "ticagrelor": "antiplatelet",
    "lisinopril": "ace_inhibitor", "enalapril": "ace_inhibitor",
    "ramipril": "ace_inhibitor",
    "losartan": "arb", "valsartan": "arb", "telmisartan": "arb",
    "alprazolam": "benzodiazepine", "lorazepam": "benzodiazepine",
    "diazepam": "benzodiazepine",
    "oxycodone": "opioid", "hydrocodone": "opioid", "morphine": "opioid",
    "tramadol": "opioid",
    "atorvastatin": "statin", "simvastatin": "statin", "rosuvastatin": "statin",
    "omeprazole": "ppi", "esomeprazole": "ppi", "pantoprazole": "ppi",
    "metoprolol": "beta_blocker", "carvedilol": "beta_blocker",
    "atenolol": "beta_blocker",
    "furosemide": "loop_diuretic", "bumetanide": "loop_diuretic",
}

# Brand, synonym, and abbreviation -> canonical ingredient.
DRUG_ALIASES: dict[str, str] = {
    "coumadin": "warfarin", "jantoven": "warfarin",
    "eliquis": "apixaban", "xarelto": "rivaroxaban", "pradaxa": "dabigatran",
    "plavix": "clopidogrel", "brilinta": "ticagrelor",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aleve": "naproxen", "voltaren": "diclofenac", "celebrex": "celecoxib",
    "indocin": "indomethacin",
    "bayer": "aspirin", "ecotrin": "aspirin", "asa": "aspirin",
    "zocor": "simvastatin", "lipitor": "atorvastatin", "crestor": "rosuvastatin",
    "glucophage": "metformin",
    "lasix": "furosemide", "bumex": "bumetanide",
    "zestril": "lisinopril", "prinivil": "lisinopril", "vasotec": "enalapril",
    "altace": "ramipril", "cozaar": "losartan", "diovan": "valsartan",
    "norvasc": "amlodipine", "lopressor": "metoprolol", "toprol": "metoprolol",
    "coreg": "carvedilol", "tenormin": "atenolol",
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "prilosec": "omeprazole", "nexium": "esomeprazole", "protonix": "pantoprazole",
    "zoloft": "sertraline", "prozac": "fluoxetine", "celexa": "citalopram",
    "lexapro": "escitalopram", "paxil": "paroxetine",
    "ultram": "tramadol", "tylenol": "acetaminophen", "panadol": "acetaminophen",
    "xanax": "alprazolam", "ativan": "lorazepam", "valium": "diazepam",
    "oxycontin": "oxycodone", "percocet": "oxycodone", "vicodin": "hydrocodone",
    "bactrim": "cotrimoxazole", "septra": "cotrimoxazole",
    "trimethoprim/sulfamethoxazole": "cotrimoxazole",
    "trimethoprim sulfamethoxazole": "cotrimoxazole", "tmp-smx": "cotrimoxazole",
    "tmp smx": "cotrimoxazole", "sulfamethoxazole/trimethoprim": "cotrimoxazole",
    "amoxicillin/clavulanate": "amoxicillin/clavulanate",
    "flagyl": "metronidazole", "diflucan": "fluconazole",
    "cipro": "ciprofloxacin", "zithromax": "azithromycin", "z-pak": "azithromycin",
    "augmentin": "amoxicillin/clavulanate", "keflex": "cephalexin",
    "viagra": "sildenafil", "cialis": "tadalafil",
    "nitrostat": "nitroglycerin", "ntg": "nitroglycerin",
    "lanoxin": "digoxin", "cordarone": "amiodarone",
    "zanaflex": "tizanidine", "zyrtec": "cetirizine",
    "hydrochlorothiazide": "hydrochlorothiazide", "hctz": "hydrochlorothiazide",
    "potassium chloride": "potassium chloride", "kcl": "potassium chloride",
    "sodium chloride": "sodium chloride",
}

SEVERITY_ORDER: dict[str, int] = {
    "info": 0, "minor": 1, "moderate": 2, "major": 3, "contraindicated": 4,
}


def interaction_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


INTERACTION_INDEX: dict[tuple[str, str], list[InteractionRule]] = {}
for _rule in INTERACTIONS:
    INTERACTION_INDEX.setdefault(interaction_key(_rule.a, _rule.b), []).append(_rule)
