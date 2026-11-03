from __future__ import annotations

from app.agents.drugs.agent import run_drug_agent
from app.agents.drugs.schemas import DrugSafetyRequest, Medication


def _request(*names: str) -> DrugSafetyRequest:
    return DrugSafetyRequest(medications=[Medication(name=name) for name in names])


def test_brand_names_are_normalised_and_interaction_detected():
    result = run_drug_agent(_request("Coumadin 5 mg", "Advil 400 mg"))
    assert {item.ingredient for item in result.normalized_medications} == {"warfarin", "ibuprofen"}
    assert result.interactions
    assert result.interactions[0].severity == "major"
    assert set(result.interactions[0].drugs) == {"warfarin", "ibuprofen"}
    assert result.overall_risk == "major"


def test_duplicate_class_detection():
    result = run_drug_agent(_request("ibuprofen", "naproxen"))
    assert len(result.duplicate_classes) == 1
    assert result.duplicate_classes[0].drug_class == "nsaid"
    assert set(result.duplicate_classes[0].ingredients) == {"ibuprofen", "naproxen"}


def test_combination_product_is_split():
    result = run_drug_agent(_request("Bactrim DS"))
    assert result.normalized_medications[0].ingredient == "cotrimoxazole"
    assert result.unknown_medications == []


def test_combination_antibiotic_name_is_known():
    result = run_drug_agent(_request("amoxicillin/clavulanate 875 mg"))
    assert result.unknown_medications == []


def test_unknown_medication_is_flagged():
    result = run_drug_agent(_request("unobtainium"))
    assert result.unknown_medications == ["unobtainium"]
    assert result.status == "partial"
    assert result.overall_risk == "info"


def test_contraindicated_combination():
    result = run_drug_agent(_request("Viagra", "nitroglycerin"))
    assert result.overall_risk == "contraindicated"
    assert result.interactions[0].severity == "contraindicated"
