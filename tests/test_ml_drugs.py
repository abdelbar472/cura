"""Tests for the hybrid drug-name resolver and its agent integration."""

from __future__ import annotations

import numpy as np

from app.agents.drugs.agent import run_drug_agent
from app.agents.drugs.normalize import KNOWN_INGREDIENTS
from app.agents.drugs.schemas import DrugSafetyRequest, Medication
from app.ml.drug_resolver import DrugResolver

_CATALOG = sorted(KNOWN_INGREDIENTS)


class CatalogEmbeddings:
    """Maps a paraphrase to one catalog ingredient via one-hot vectors."""

    name = "fake-embeddings"

    def __init__(self, target: str = "metformin") -> None:
        self._target = _CATALOG.index(target)

    @property
    def ready(self) -> bool:
        return True

    def encode(self, texts):
        texts = list(texts)
        if len(texts) > 1:
            return np.eye(len(texts), dtype=np.float32)
        vector = np.zeros((1, len(_CATALOG)), dtype=np.float32)
        vector[0, self._target] = 1.0
        return vector


def _request(*names: str) -> DrugSafetyRequest:
    return DrugSafetyRequest(medications=[Medication(name=name) for name in names])


def test_resolver_handles_exact_alias_salt_and_typos():
    resolver = DrugResolver(embeddings=None)

    assert resolver.resolve("ibuprofen").method == "exact"
    assert resolver.resolve("Advil").ingredient == "ibuprofen"
    assert resolver.resolve("Advil").method == "alias"
    assert resolver.resolve("metoprolol tartrate").ingredient == "metoprolol"
    assert resolver.resolve("metoprolol tartrate").method == "fuzzy"
    typo = resolver.resolve("ibuprofin")
    assert typo.ingredient == "ibuprofen"
    assert typo.method == "fuzzy"
    assert 0.0 < typo.confidence < 1.0


def test_resolver_never_guesses_an_unrecognised_name():
    match = DrugResolver(embeddings=None).resolve("unobtainium")
    assert match.ingredient is None
    assert match.method == "unresolved"
    assert match.confidence == 0.0


def test_resolver_falls_back_to_embeddings_for_paraphrases():
    resolver = DrugResolver(embeddings=CatalogEmbeddings("metformin"))
    match = resolver.resolve("sugar pill")
    assert match.ingredient == "metformin"
    assert match.method == "embedding"
    assert match.confidence >= 0.72


def test_disabled_resolver_reports_disabled_but_still_resolves_aliases():
    resolver = DrugResolver(embeddings=CatalogEmbeddings(), enabled=False)
    assert resolver.info.enabled is False
    assert resolver.resolve("Advil").ingredient == "ibuprofen"
    assert resolver.resolve("sugar pill").ingredient is None


def test_agent_warns_about_approximate_matches():
    resolver = DrugResolver(embeddings=None)
    result = run_drug_agent(_request("ibuprofin", "Coumadin"), resolver=resolver)

    by_input = {item.input_name: item for item in result.normalized_medications}
    assert by_input["ibuprofin"].ingredient == "ibuprofen"
    assert by_input["ibuprofin"].resolution == "fuzzy"
    assert by_input["Coumadin"].resolution == "alias"
    assert result.status == "partial"
    assert any("verify the name" in warning for warning in result.warnings)
    assert result.overall_risk == "major"
    assert result.ml is not None and result.ml.ready is True


def test_agent_exact_matches_stay_clean_and_ok():
    result = run_drug_agent(
        _request("ibuprofen", "warfarin"),
        resolver=DrugResolver(embeddings=None),
    )
    assert result.status == "ok"
    assert result.warnings == []
    assert {item.resolution for item in result.normalized_medications} == {"exact"}


def test_agent_falls_back_when_resolver_is_disabled():
    resolver = DrugResolver(embeddings=CatalogEmbeddings(), enabled=False)
    result = run_drug_agent(_request("unobtainium"), resolver=resolver)
    assert result.normalized_medications[0].resolution == "unresolved"
    assert result.ml is not None
    assert result.ml.ready is False
    assert result.ml.fallback is True
