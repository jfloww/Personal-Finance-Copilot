"""Run the synthetic operations investigation; live model calls are opt-in.

    PYTHONPATH=src uv run python run_operations_agent.py          # free fixed investigation
    PYTHONPATH=src uv run python run_operations_agent.py --live   # confirms before spending

No real tenant records, database access, policy documents, or writes are available
through this entry point. The live path is not a benchmark or production service.
"""

from __future__ import annotations

import argparse
import json

from offerdelta.agent.runtime import (
    OPERATIONS_PROVIDER_FAILURE_TEXT,
    OPERATIONS_SYSTEM_PROMPT,
    OPERATIONS_TURN_LIMIT_TEXT,
    AgentRuntime,
    InProcessToolSource,
)
from offerdelta.agent.tools.operations import build_operations_registry
from offerdelta.application.queries.operations_demo import QUESTION, investigate_demo
from offerdelta.infrastructure.llm.factory import build_agent_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Investigate synthetic transactions safely.")
    parser.add_argument("--live", action="store_true", help="allow a metered Anthropic agent run")
    parser.add_argument(
        "--yes", action="store_true", help="confirm a live model run non-interactively"
    )
    args = parser.parse_args(argv)
    if not args.live:
        report = investigate_demo()
        trace = report["trace"]
        if not isinstance(trace, list):
            raise RuntimeError("synthetic trace is malformed")
        print("SCRIPTED DEMO — not a live agent run or a benchmark score")
        print(
            json.dumps(
                {
                    "summary": report["summary"],
                    "duplicate": report["duplicate"],
                    "policy_citations": report["policy_citations"],
                    "proposal": report["proposal"],
                    "tool_order": [step["tool"] for step in trace if isinstance(step, dict)],
                },
                indent=2,
            )
        )
        return 0

    provider = build_agent_provider()
    if provider is None:
        print("--live requires ANTHROPIC_API_KEY; nothing was sent")
        return 2
    print(f"Model: {provider.model}; max turns: 8; synthetic data only; no mutations.")
    if not args.yes:
        try:
            confirmed = input("Send a metered model request? [y/N] ").strip().lower() == "y"
        except EOFError:
            confirmed = False
        if not confirmed:
            print("Nothing was sent")
            return 0
    result = AgentRuntime(
        provider,
        InProcessToolSource(build_operations_registry()),
        system_prompt=OPERATIONS_SYSTEM_PROMPT,
        provider_failure_text=OPERATIONS_PROVIDER_FAILURE_TEXT,
        turn_limit_text=OPERATIONS_TURN_LIMIT_TEXT,
    ).run(QUESTION)
    print(result.final_text)
    print(
        json.dumps(
            {
                "mode": "live_model_not_benchmarked",
                "model": provider.model,
                "audit": result.public_summary(),
                "tool_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": call.result.as_json(),
                    }
                    for call in result.tool_calls
                ],
            },
            indent=2,
        )
    )
    return 0 if result.stopped_reason == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
