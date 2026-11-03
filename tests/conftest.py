from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client() -> TestClient:
    settings = Settings(
        environment="test",
        kafka_enabled=False,
        log_level="WARNING",
        store_backend="memory",
        ml_enabled=False,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def uti_case() -> dict:
    return {
        "text": "Dysuria, urinary frequency and fever for two days. Denies chest pain.",
        "patient": {"age_years": 34, "sex": "female"},
        "labs": [
            {"name": "WBC", "value": 15.2, "unit": "10^9/L"},
            {"name": "CRP", "value": 48.0, "unit": "mg/L"},
        ],
        "medications": [{"name": "Ibuprofen 400 mg"}, {"name": "Coumadin 5 mg"}],
        "client_reference": "ED-1001",
    }
