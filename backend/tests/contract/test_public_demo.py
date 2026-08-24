"""What the public deployment is allowed to serve.

This deployment has no database, no API key, and no transactions. It does have
a summary of evaluation runs that read someone's actual bank statements, and
the whole point of publishing a summary rather than the runs is that the summary
cannot be walked back to a transaction.

So these are containment tests, not feature tests. They fail when something
personal appears on a public route, when the published decision drifts from the
prompt the code actually selects, or when an endpoint that needs a database
starts advertising itself on a deployment that has none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from offerdelta.api import main
from offerdelta.infrastructure.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPTS

#: `backend/tests/contract/x.py` -> repository root.
_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT = _ROOT / "docs" / "eval" / "public-results.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def published(client: TestClient) -> dict[str, object]:
    response = client.get("/demo/evaluation/latest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert isinstance(body, dict)
    return body


def _at(payload: dict[str, object], *keys: str) -> object:
    """Walk a nested key path, asserting the shape on the way down."""
    value: object = payload
    for key in keys:
        assert isinstance(value, dict), f"{key} is not under a mapping"
        value = value[key]
    return value


def _strings(value: object) -> list[str]:
    """Every string anywhere in the payload, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_strings(v)]]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_the_landing_page_is_the_evaluation_showcase(client: TestClient) -> None:
    body = client.get("/").text
    assert "Personal Finance Copilot" in body
    assert "/demo/evaluation/latest" in body


def test_the_landing_page_states_that_ingestion_is_disabled(client: TestClient) -> None:
    """A recruiter reading this should not have to wonder whether real
    statements are one URL away. It says so, on the page, unprompted."""
    body = client.get("/").text.lower()
    assert "public demo mode" in body
    assert "real financial ingestion is disabled" in body


def test_the_comparison_demo_is_still_reachable(client: TestClient) -> None:
    assert client.get("/demo/comparison").status_code == 200


def test_the_served_results_are_the_committed_artifact(client: TestClient) -> None:
    """Two copies exist - one committed under docs/, one shipped in the package.
    They are written by the same call, so a difference means someone edited the
    served one by hand."""
    served = client.get("/demo/evaluation/latest").text
    assert served == _ARTIFACT.read_text(encoding="utf-8")


def test_the_published_decision_matches_the_prompt_in_use(published: dict[str, object]) -> None:
    """The one that matters. The page says v1 was selected; if someone switches
    the default prompt without re-running the evaluation, the site would claim a
    decision the code no longer implements."""
    selected = _at(published, "prompt_experiment", "selected_prompt")
    assert selected == PROMPT_VERSION
    assert selected in SYSTEM_PROMPTS


def test_the_rejected_experiment_is_labelled_rejected(published: dict[str, object]) -> None:
    assert _at(published, "prompt_experiment", "decision") == "rejected"
    assert _at(published, "prompt_experiment", "status") == "rejected experiment"
    assert "categorise/v2" in SYSTEM_PROMPTS, "the rejected prompt is kept as evidence"


def test_no_money_amount_is_published(published: dict[str, object]) -> None:
    """Run costs are published; transaction amounts are not. The two are told
    apart by magnitude - a whole run costs cents, a rent payment does not - so
    this asserts on the fields rather than trying to classify loose numbers."""
    systems = published["systems"]
    assert isinstance(systems, list)
    for system in systems:
        assert isinstance(system, dict)
        cost = system["cost_usd"]
        assert cost is None or (isinstance(cost, float) and cost < 1), system["name"]
    assert "transactions" not in published
    assert "rows" not in published


def test_no_per_row_prediction_is_published(published: dict[str, object]) -> None:
    """Per-label aggregates are fine. A list of 135 anything is not."""

    def oversized(value: object, path: str = "$") -> list[str]:
        if isinstance(value, list) and len(value) > 60:
            return [f"{path} has {len(value)} entries"]
        if isinstance(value, dict):
            return [p for k, v in value.items() for p in oversized(v, f"{path}.{k}")]
        if isinstance(value, list):
            return [p for i, v in enumerate(value) for p in oversized(v, f"{path}[{i}]")]
        return []

    assert oversized(published) == []


def test_no_merchant_or_counterparty_name_is_published(published: dict[str, object]) -> None:
    """Label names are taxonomy constants. Anything else that reads like a
    merchant string got here from a transaction."""
    text = " ".join(_strings(published)).lower()
    forbidden = (
        "kroger",
        "tesla",
        "zelle",
        "starbucks",
        "amex",
        "american express",
        "chase",
        "capital one",
        "state farm",
        "person_",
    )
    found = [name for name in forbidden if name in text]
    assert found == [], f"merchant or counterparty names published: {found}"


def test_no_credential_is_published(published: dict[str, object]) -> None:
    text = " ".join(_strings(published))
    assert "sk-ant" not in text
    assert not re.search(r"postgres(ql)?://", text)


def test_no_transaction_date_is_published(published: dict[str, object]) -> None:
    """The generation timestamp is a date. A posting date would be a leak, and
    the two are distinguishable only by which field they sit in."""
    payload = dict(published)
    payload.pop("generated_at")
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", json.dumps(payload))
    assert dates == [], f"dates published outside generated_at: {dates}"


def test_transaction_endpoints_stay_out_of_the_schema_without_a_database() -> None:
    """A documented endpoint that can only return 503 is worse than an absent
    one: it invites a reader to post a transaction to a deployment that has
    nowhere to put it."""
    if main._DATABASE_CONFIGURED:
        pytest.skip("a database is configured here; the hidden case is the deployed one")
    advertised = {
        route.path
        for route in main.app.routes
        if getattr(route, "include_in_schema", False) and hasattr(route, "path")
    }
    assert "/v1/transactions" not in advertised


#: Every path the landing page reads out of the payload. The page renders from
#: the endpoint rather than baking figures in, which removes drift but adds a
#: silent failure mode: rename a field in the generator and the page shows a
#: blank where a number was, with nothing raising.
_PAGE_READS: tuple[tuple[str, ...], ...] = (
    ("generated_at",),
    ("dataset_flow", "imported_transactions"),
    ("dataset_flow", "accounts"),
    ("dataset_flow", "banks"),
    ("dataset_flow", "labelled_rows"),
    ("dataset_flow", "validation_benchmark_rows"),
    ("dataset_flow", "distinct_merchants_in_benchmark"),
    ("dataset_flow", "raw_annotator_agreement"),
    ("dataset_flow", "cohens_kappa"),
    ("method", "merchants_on_both_sides"),
    ("systems",),
    ("prompt_experiment", "hypothesis"),
    ("prompt_experiment", "controlled_change"),
    ("prompt_experiment", "reasoning"),
    ("prompt_experiment", "caveat"),
    ("prompt_experiment", "why_macro_f1_rose_anyway"),
    ("prompt_experiment", "selected_prompt"),
    ("prompt_experiment", "per_label_movement"),
    ("prompt_experiment", "outcome", "target_unmoved"),
    ("prompt_experiment", "outcome", "categories_regressed"),
    ("prompt_experiment", "outcome", "categories_improved"),
    ("prompt_experiment", "outcome", "v1_is_cheaper_by_pct"),
    ("prompt_experiment", "outcome", "v1_p95_latency_lower_by_ms"),
    ("backlog",),
)

_SYSTEM_FIELDS = (
    "name",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "coverage",
    "cost_per_row_usd",
    "p95_latency_ms",
)

_MOVEMENT_FIELDS = ("label", "support", "v1", "v2", "delta")


@pytest.mark.parametrize("path", _PAGE_READS, ids=[".".join(p) for p in _PAGE_READS])
def test_the_payload_carries_what_the_page_reads(
    published: dict[str, object], path: tuple[str, ...]
) -> None:
    assert _at(published, *path) is not None


def test_every_system_row_carries_what_the_table_renders(published: dict[str, object]) -> None:
    systems = published["systems"]
    assert isinstance(systems, list)
    assert systems
    for system in systems:
        assert isinstance(system, dict)
        assert set(_SYSTEM_FIELDS) <= set(system)


def test_every_movement_row_carries_what_the_table_renders(published: dict[str, object]) -> None:
    rows = _at(published, "prompt_experiment", "per_label_movement")
    assert isinstance(rows, list)
    assert rows
    for row in rows:
        assert isinstance(row, dict)
        assert set(_MOVEMENT_FIELDS) <= set(row)


def test_the_public_schema_advertises_the_evaluation_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/demo/evaluation/latest" in schema["paths"]
    assert "public demo mode" in schema["info"]["description"].lower()
