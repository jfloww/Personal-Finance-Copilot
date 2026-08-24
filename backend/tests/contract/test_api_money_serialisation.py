"""The money boundary.

Phase-1 gate item: no monetary value crosses the API as a JSON number.

This is enforced by walking the raw response body rather than the parsed model,
because the parsed model would hide exactly the defect being tested — a float
that survived serialisation still deserialises into something that looks fine.
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from offerdelta.api.main import app

#: The page that renders money. The root is the evaluation showcase, which
#: renders scores and has no amounts to get wrong.
_PAGE = "/demo/comparison"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _walk(value: object, path: str = "$") -> list[tuple[str, object]]:
    """Yield every (path, value) pair in a decoded JSON document."""
    found: list[tuple[str, object]] = [(path, value)]
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk(item, f"{path}[{index}]"))
    return found


def test_the_demo_derivation_is_served(client: TestClient) -> None:
    assert client.get("/v1/demo/derivation").status_code == 200


def test_no_amount_is_serialised_as_a_json_number(client: TestClient) -> None:
    body = json.loads(client.get("/v1/demo/derivation").text)
    offenders = [
        path
        for path, value in _walk(body)
        if path.endswith(".amount") and not isinstance(value, str)
    ]
    assert offenders == [], f"amounts must be strings, these were not: {offenders}"


def test_no_float_appears_anywhere_in_the_response(client: TestClient) -> None:
    # Broader than the amount check: nothing in a financial payload should be a
    # float, whatever its field name.
    body = json.loads(client.get("/v1/demo/derivation").text)
    floats = [(path, value) for path, value in _walk(body) if isinstance(value, float)]
    assert floats == [], f"floats found in response: {floats}"


def test_amounts_keep_their_exact_decimal_text(client: TestClient) -> None:
    body = client.get("/v1/demo/derivation").json()
    assert body["amount"] == "2335.00"


def test_costs_are_signed_negative_in_the_payload(client: TestClient) -> None:
    body = client.get("/v1/demo/derivation").json()
    housing = next(child for child in body["children"] if child["code"] == "housing")
    assert housing["amount"].startswith("-")


def test_every_node_reports_its_currency(client: TestClient) -> None:
    body = json.loads(client.get("/v1/demo/derivation").text)
    nodes = [value for path, value in _walk(body) if isinstance(value, dict) and "code" in value]
    assert nodes
    assert all(node["currency"] == "USD" for node in nodes)


def test_every_node_reports_its_provenance(client: TestClient) -> None:
    body = json.loads(client.get("/v1/demo/derivation").text)
    nodes = [value for path, value in _walk(body) if isinstance(value, dict) and "code" in value]
    allowed = {"SOURCED", "USER_CONFIRMED", "ASSUMED", "DERIVED"}
    assert all(node["evidence"] in allowed for node in nodes)


def test_liveness_and_readiness_are_served(client: TestClient) -> None:
    assert client.get("/v1/health/live").json() == {"status": "live"}
    # Readiness also reports the database, because "ready" alone would be
    # silent about the one thing that stops half the API working.
    ready = client.get("/v1/health/ready").json()
    assert ready["status"] == "ready"
    assert "database" in ready


def test_version_reports_the_engine(client: TestClient) -> None:
    body = client.get("/v1/version").json()
    assert body["engine"] == "0.1.0-skeleton"


def test_the_comparison_page_is_served(client: TestClient) -> None:
    """The root is the evaluation showcase; the money rules below govern the
    comparison page, which is the only one that renders amounts."""
    response = client.get(_PAGE)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_page_never_parses_money_loosely(client: TestClient) -> None:
    # parseFloat and parseInt have no legitimate use on this page: both stop at
    # the first character they cannot read, so a malformed amount becomes a
    # plausible wrong number instead of an error.
    for script in _scripts(client):
        assert "parseFloat" not in script
        assert "parseInt" not in script


def test_every_numeric_coercion_on_the_page_is_marked_as_geometry(
    client: TestClient,
) -> None:
    # A chart is 720 units wide and cannot render a cent of error, so pixel
    # positions legitimately become floats. Displayed amounts never may.
    #
    # Rather than ban coercion outright — which the chart made impossible — the
    # rule is that every coercion carries a GEOMETRY marker. An unannotated one
    # fails here, so the exception stays deliberate instead of spreading into a
    # figure someone reads.
    for script in _scripts(client):
        for number, line in enumerate(script.splitlines(), start=1):
            if "Number(" in line:
                assert "GEOMETRY" in line, (
                    f"line {number} coerces to a JavaScript number without a "
                    f"GEOMETRY marker: {line.strip()}"
                )


def test_displayed_amounts_come_from_the_string_formatter(client: TestClient) -> None:
    # The formatter is the only path to a rendered figure, and it works by
    # string surgery on the exact decimal the API sent.
    for script in _scripts(client):
        assert "function fmt(" in script


def _scripts(client: TestClient) -> list[str]:
    """The page's executable blocks, excluding prose that merely discusses them."""
    page = client.get(_PAGE).text
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, flags=re.DOTALL)
    assert scripts, "expected the page to contain a script block"
    return scripts


# ---------------------------------------------------------------- readiness


def test_readiness_reports_the_database_state(client: TestClient) -> None:
    """A probe that says "ready" while saying nothing about a missing database
    is asserting something untrue. It reports both."""
    body = client.get("/v1/health/ready").json()

    assert body["status"] == "ready"
    assert body["database"] in {"connected", "unconfigured", "unreachable"}


def test_readiness_stays_ready_without_a_database(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Deliberate: this path is the platform health check, and the demo
    endpoints are in-memory. Failing it would take a working demo offline to
    report a missing database."""
    monkeypatch.setattr("offerdelta.api.main._database_state", lambda: "unconfigured")

    response = client.get("/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["database"] == "unconfigured"


def test_the_demo_endpoints_do_not_need_a_database(client: TestClient) -> None:
    """They are in-memory, and the deployed service has no database at all."""
    assert client.get("/v1/demo/comparison").status_code == 200
    assert client.get("/v1/demo/derivation").status_code == 200
