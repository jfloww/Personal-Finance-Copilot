from __future__ import annotations

from _pytest.capture import CaptureFixture

import run_operations_agent
from offerdelta.agent.runtime import (
    OPERATIONS_PROVIDER_FAILURE_TEXT,
    OPERATIONS_TURN_LIMIT_TEXT,
)


def test_keyless_operations_cli_is_explicitly_scripted(capsys: CaptureFixture[str]) -> None:
    assert run_operations_agent.main([]) == 0
    output = capsys.readouterr().out
    assert "SCRIPTED DEMO" in output
    assert '"change": "12500.00"' in output
    assert '"ledger_mutated": false' in output


def test_operations_failure_copy_stays_in_the_investigation_domain() -> None:
    assert "transaction investigation" in OPERATIONS_PROVIDER_FAILURE_TEXT
    assert "transaction investigation" in OPERATIONS_TURN_LIMIT_TEXT
    assert "comparison" not in OPERATIONS_PROVIDER_FAILURE_TEXT
    assert "comparison" not in OPERATIONS_TURN_LIMIT_TEXT
