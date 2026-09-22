from __future__ import annotations

from fastapi.testclient import TestClient

from offerdelta.api.main import app


def test_workbench_and_read_only_api_have_the_same_synthetic_scenario() -> None:
    with TestClient(app) as client:
        page = client.get("/demo/agent")
        assert page.status_code == 200
        assert "MyFinSecretary" in page.text
        assert "not a live AI model" in page.text
        assert "not vector RAG" in page.text
        read = client.get("/v1/demo/agent/investigation")
        run = client.post("/v1/demo/agent/run", json={"scenario": "august_software_exceptions"})
        assert read.status_code == run.status_code == 200
        assert read.json() == run.json()
        assert read.json()["ledger_mutated"] is False
        assert read.json()["proposal"]["status"] == "proposal_only"


def test_public_api_rejects_other_scenarios_and_unknown_fields() -> None:
    with TestClient(app) as client:
        assert client.post("/v1/demo/agent/run", json={"scenario": "other"}).status_code == 422
        assert (
            client.post(
                "/v1/demo/agent/run",
                json={
                    "scenario": "august_software_exceptions",
                    "question": "read another tenant",
                },
            ).status_code
            == 422
        )


def test_public_api_switches_between_bundled_synthetic_cases() -> None:
    with TestClient(app) as client:
        base = client.post("/v1/demo/agent/run", json={"scenario": "august_software_exceptions"})
        alternate = client.post("/v1/demo/agent/run", json={"scenario": "alternate_billing_review"})
        assert base.status_code == alternate.status_code == 200
        assert base.json()["summary"]["change"] == "12500.00"
        assert alternate.json()["summary"]["change"] == "5500.00"
        assert alternate.json()["summary"]["transaction_count"] == 4
        assert alternate.json()["ledger_mutated"] is False
        assert alternate.json()["review_queue_mutated"] is False
        assert alternate.json()["proposal"]["status"] == "proposal_only"
