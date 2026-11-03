from __future__ import annotations

from app.agents.symptom.agent import run_symptom_agent
from app.agents.symptom.schemas import SymptomRequest


def test_extracts_symptoms_and_negation():
    result = run_symptom_agent(
        SymptomRequest(text="Dysuria and urinary frequency for 3 days. Denies fever.")
    )
    assert result.status == "ok"
    assert "E_DYSURIA" in result.evidence_codes
    assert "E_FREQUENCY" in result.evidence_codes
    assert "E_FEVER" not in result.evidence_codes
    assert "E_FEVER" in result.negated_codes
    assert "fever" in result.negated_symptoms
    assert result.backend == "lexicon"


def test_negation_does_not_leak_across_clauses():
    result = run_symptom_agent(
        SymptomRequest(text="Denies fever, chills, or night sweats but reports chest pain.")
    )
    assert "E_FEVER" in result.negated_codes
    assert "E_CHILLS" in result.negated_codes
    assert "E_SWEATING" in result.negated_codes
    assert "E_CHEST_PAIN" in result.evidence_codes
    assert result.red_flags == ["E_CHEST_PAIN"]


def test_red_flags_are_subset_of_positive_codes():
    result = run_symptom_agent(
        SymptomRequest(text="Sudden shortness of breath and hemoptysis.")
    )
    assert set(result.red_flags) <= set(result.evidence_codes)
    assert "E_SOB" in result.red_flags
    assert "E_HEMOPTYSIS" in result.red_flags


def test_input_hash_is_stable_and_entities_have_offsets():
    text = "Headache with photophobia and nausea."
    first = run_symptom_agent(SymptomRequest(text=text))
    second = run_symptom_agent(SymptomRequest(text=text))
    assert first.input_hash == second.input_hash
    assert first.input_hash.startswith("sha256:")
    for entity in first.entities:
        assert text[entity.start : entity.end].lower() == entity.text.lower()


def test_unknown_text_is_not_an_error():
    result = run_symptom_agent(SymptomRequest(text="Routine follow-up visit."))
    assert result.status == "ok"
    assert result.evidence_codes == []
    assert result.warnings
