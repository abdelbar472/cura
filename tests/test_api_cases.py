from __future__ import annotations

from fastapi.testclient import TestClient


def test_service_info_and_health(client: TestClient):
    info = client.get("/").json()
    assert info["name"] == "Cura"
    assert info["docs"] == "/docs"

    health = client.get("/healthz")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["bus"] == "memory"
    assert body["graph"] == "langgraph"
    assert body["agents"]["symptom"] == "ready"
    assert body["agents"]["imaging"] == "experimental"
    assert body["store"] == "memory"
    assert body["ml_enabled"] is False
    assert "symptom_semantic" in body["models"]


def test_agent_and_knowledge_catalogues(client: TestClient):
    agents = client.get("/v1/agents").json()
    assert {item["name"] for item in agents} >= {
        "symptom",
        "labs",
        "imaging",
        "differential",
        "drugs",
        "orchestrator",
    }

    conditions = client.get("/v1/knowledge/conditions").json()
    assert len(conditions) >= 10
    assert any(item["condition_id"] == "COND_UTI" for item in conditions)


def test_create_case_end_to_end(client: TestClient, uti_case: dict):
    response = client.post("/v1/cases", json=uti_case)
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["status"] == "ok"
    assert body["client_reference"] == "ED-1001"
    assert body["symptom"]["backend"] == "lexicon"
    assert "E_DYSURIA" in body["symptom"]["evidence_codes"]
    assert "E_CHEST_PAIN" in body["symptom"]["negated_codes"]

    assert body["labs"]["critical_flags"] == []
    assert body["imaging"] is None
    assert body["differential"]["considerations"][0]["condition_id"] == "COND_UTI"
    assert body["drug_safety"]["overall_risk"] == "major"
    assert body["triage"]["level"] == "urgent"
    assert "E_LEUKOCYTOSIS" in body["evidence_codes"]
    assert body["disclaimer"]


def test_case_store_listing_and_events(client: TestClient, uti_case: dict):
    case_id = client.post("/v1/cases", json=uti_case).json()["case_id"]

    fetched = client.get(f"/v1/cases/{case_id}")
    assert fetched.status_code == 200
    assert fetched.json()["case_id"] == case_id

    listing = client.get("/v1/cases").json()
    assert any(item["case_id"] == case_id for item in listing)
    assert listing[0]["triage_level"] == "urgent"

    events = client.get(f"/v1/cases/{case_id}/events").json()
    assert {event["agent"] for event in events} == {
        "symptom",
        "labs",
        "differential",
        "drugs",
    }
    assert all(event["case_id"] == case_id for event in events)
    assert all(event["topic"].endswith(".result") for event in events)


def test_imaging_request_makes_case_partial(client: TestClient, uti_case: dict):
    payload = dict(uti_case)
    payload["imaging"] = {
        "modality": "xray",
        "body_site": "chest",
        "clinical_question": "Infiltration?",
    }
    body = client.post("/v1/cases", json=payload).json()
    assert body["status"] == "partial"
    assert body["imaging"]["status"] == "unavailable"
    assert body["imaging"]["findings"] == []

    events = client.get(f"/v1/cases/{body['case_id']}/events").json()
    assert len(events) == 5


def test_validation_rejects_bad_input(client: TestClient):
    assert client.post("/v1/cases", json={"text": ""}).status_code == 422
    assert (
        client.post("/v1/cases", json={"text": "cough", "patient": {"age_years": 200}}).status_code
        == 422
    )
    assert client.post("/v1/cases", json={"text": "cough", "unexpected": True}).status_code == 422
    assert client.post("/v1/agents/imaging", json={"modality": "pet", "body_site": "chest"}).status_code == 422


def test_unknown_case_returns_404(client: TestClient):
    assert client.get("/v1/cases/case_missing").status_code == 404
    assert client.get("/v1/cases/case_missing/events").status_code == 404


def test_direct_agent_endpoints(client: TestClient):
    symptom = client.post(
        "/v1/agents/symptom",
        json={"text": "Crushing chest pain with sweating.", "backend": "lexicon"},
    )
    assert symptom.status_code == 200
    assert "E_CHEST_PAIN" in symptom.json()["evidence_codes"]

    labs = client.post(
        "/v1/agents/labs",
        json={"values": [{"name": "troponin", "value": 0.5, "unit": "ng/mL"}]},
    )
    assert labs.status_code == 200
    assert labs.json()["critical_flags"] == ["Troponin I"]

    differential = client.post(
        "/v1/agents/differential",
        json={"evidence_codes": ["E_DYSURIA", "E_FREQUENCY"], "patient": {"sex": "female"}},
    )
    assert differential.status_code == 200
    assert differential.json()["considerations"][0]["condition_id"] == "COND_UTI"

    drugs = client.post(
        "/v1/agents/drugs",
        json={"medications": [{"name": "warfarin"}, {"name": "aspirin"}]},
    )
    assert drugs.status_code == 200
    assert drugs.json()["overall_risk"] == "major"

    imaging = client.post(
        "/v1/agents/imaging",
        json={"modality": "xray", "body_site": "chest"},
    )
    assert imaging.status_code == 200
    assert imaging.json()["status"] == "unavailable"
