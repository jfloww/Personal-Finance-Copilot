from __future__ import annotations

from _pytest.capture import CaptureFixture

import run_operations_agent


def test_keyless_operations_cli_is_explicitly_scripted(capsys: CaptureFixture[str]) -> None:
    assert run_operations_agent.main([]) == 0
    output = capsys.readouterr().out
    assert "SCRIPTED DEMO" in output
    assert '"change": "12500.00"' in output
    assert '"ledger_mutated": false' in output
