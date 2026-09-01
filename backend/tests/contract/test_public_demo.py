"""What the public deployment is allowed to serve.

The public deployment used to have no database at all - that was the entire
safety argument before this phase. It no longer holds: Phase 0 gives Render a
real connection and a `JWT_SECRET`, seeded with two synthetic demo tenants so
a visitor can log in and store one invented transaction against their own
tenant (`docs/superpowers/specs/2026-08-31-auth-and-tenant-isolation-design.md`
§6). It still has no LLM API key and no real bank data: the summary below is
generated from evaluation runs performed locally against real statements, and
the whole point of publishing a summary rather than the runs is that the
summary cannot be walked back to a transaction.

So these are containment tests, not feature tests. They fail when something
personal appears on a public route, when the published decision drifts from the
prompt the code actually selects, or when an endpoint that needs a database
starts advertising itself on a deployment that has none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import pytest
from fastapi.testclient import TestClient

from offerdelta.api import main
from offerdelta.infrastructure.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPTS

#: `backend/tests/contract/x.py` -> repository root.
_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT = _ROOT / "docs" / "eval" / "public-results.json"

#: Merchants and counterparties that appear in the imported statements. None of
#: them may reach a public route - not in the payload, and not in page copy
#: either. A real merchant used as a throwaway example in prose still says where
#: somebody shops, and it is not worth the sentence it improves.
FORBIDDEN: Final = (
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

#: Every page the deployment serves.
PAGES: Final = ("/", "/demo/comparison")


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
    """The one that matters. If someone switches the default prompt without
    re-running the evaluation, the site would claim a decision the code no
    longer implements."""
    selected = published["selected_prompt"]
    assert selected == PROMPT_VERSION
    assert selected in SYSTEM_PROMPTS


def test_the_adopted_iteration_is_the_selected_prompt(published: dict[str, object]) -> None:
    iterations = published["prompt_iterations"]
    assert isinstance(iterations, list)
    adopted = [i for i in iterations if isinstance(i, dict) and i["decision"] == "adopted"]
    assert len(adopted) == 1, "exactly one iteration can be the selected prompt"
    assert adopted[0]["version"] == published["selected_prompt"]


def test_every_rejected_iteration_is_labelled_rejected(published: dict[str, object]) -> None:
    """A rejected change is evidence and stays published as one. Quietly dropping
    it would leave a page showing only the changes that worked."""
    iterations = published["prompt_iterations"]
    assert isinstance(iterations, list)
    rejected = [i for i in iterations if isinstance(i, dict) and i["decision"] == "rejected"]
    assert rejected, "the rejected experiment is kept, not deleted"
    for iteration in rejected:
        assert iteration["status"] == "rejected experiment"
        assert iteration["version"] in SYSTEM_PROMPTS, "its prompt is still reproducible"


def test_a_macro_f1_move_is_published_with_its_denominator(
    published: dict[str, object],
) -> None:
    """Macro F1 can rise purely because a system stopped predicting a label it
    was never right about. Publishing the mean without the summed F1 beside it
    lets that read as an improvement."""
    iterations = published["prompt_iterations"]
    assert isinstance(iterations, list)
    for iteration in iterations:
        assert isinstance(iteration, dict)
        decomposition = iteration["macro_decomposition"]
        assert isinstance(decomposition, dict)
        assert set(decomposition) >= {
            "summed_f1_before",
            "summed_f1_after",
            "labels_in_average_before",
            "labels_in_average_after",
            "gain_from_better_answers",
            "gain_from_a_smaller_denominator",
        }


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
    found = [name for name in FORBIDDEN if name in text]
    assert found == [], f"merchant or counterparty names published: {found}"


@pytest.mark.parametrize("page", PAGES)
def test_no_merchant_name_appears_in_page_copy(client: TestClient, page: str) -> None:
    """The payload was scanned from the first draft; the prose around it was
    not, and a real merchant went out in an example sentence explaining the
    merchant-disjoint split. Both surfaces are public, so both are scanned."""
    text = client.get(page).text.lower()
    found = [name for name in FORBIDDEN if name in text]
    assert found == [], f"{page} names merchants or counterparties: {found}"


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
    ("method", "ambiguity_policy"),
    ("method", "authored_ambiguous_rows"),
    ("method", "reproduce"),
    ("method", "where_changes_are_chosen"),
    ("selected_prompt",),
    ("prompt_iterations",),
    ("development_did_not_predict_the_benchmark", "finding"),
    ("development_did_not_predict_the_benchmark", "development_macro_f1_gain"),
    ("development_did_not_predict_the_benchmark", "benchmark_macro_f1_gain"),
    ("development_did_not_predict_the_benchmark", "why"),
    ("failure_analysis", "failure_modes"),
    ("failure_analysis", "mean_confidence_when_right"),
    ("failure_analysis", "mean_confidence_when_wrong"),
    ("limitations",),
    ("backlog",),
)

_SYSTEM_FIELDS = (
    "name",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "coverage",
    "cost_per_row_usd",
    "mean_latency_ms",
    "p95_latency_ms",
)

_MOVEMENT_FIELDS = ("label", "support", "before", "after", "delta")


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
    iterations = published["prompt_iterations"]
    assert isinstance(iterations, list)
    adopted = next(i for i in iterations if isinstance(i, dict) and i["decision"] == "adopted")
    rows = adopted["per_label_movement"]
    assert isinstance(rows, list)
    assert rows
    for row in rows:
        assert isinstance(row, dict)
        assert set(_MOVEMENT_FIELDS) <= set(row)


def test_the_public_schema_advertises_the_evaluation_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/demo/evaluation/latest" in schema["paths"]
    assert "public demo mode" in schema["info"]["description"].lower()
    if main._AUTH_CONFIGURED:
        # The login route is part of the public surface now - nothing else in
        # this file pins its presence, and `_AUTH_CONFIGURED` is exactly the
        # constant deciding whether `/v1/auth/token` is advertised at all.
        assert "/v1/auth/token" in schema["paths"]
