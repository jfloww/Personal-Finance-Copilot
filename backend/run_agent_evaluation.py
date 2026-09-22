"""Run the 28-task agent tool-use evaluation, keyless by default.

    uv run python run_agent_evaluation.py          # scripted, free, proves the harness
    uv run python run_agent_evaluation.py --live   # real model, requires confirmation

The scripted path is an oracle, not a model score. It executes the computed
gold trace so the registry, runtime, fault injection, grounding, and aggregate
report can be tested in CI without a key. Only ``--live`` measures a model.
"""

from __future__ import annotations

import argparse
import json
import sys

from offerdelta.agent.runtime import (
    AgentRuntime,
    InProcessToolSource,
    ScriptedAgentProvider,
)
from offerdelta.agent.tools.definitions import build_tool_registry
from offerdelta.agent.transcript import ProviderResponse, TextBlock, ToolUseBlock
from offerdelta.evaluation.agent.report import evaluate_runs
from offerdelta.evaluation.agent.tasks import MaterializedTask, build_tasks, materialize
from offerdelta.infrastructure.llm.factory import build_agent_provider


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agent tool use and numeric grounding.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the configured Anthropic model; without this a scripted oracle is used",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the explicit confirmation before a live run",
    )
    return parser


def _scripted(materialized: MaterializedTask) -> ScriptedAgentProvider:
    calls = tuple(
        ToolUseBlock(
            id=f"gold-{index}",
            name=call.name,
            arguments=dict(call.arguments),
        )
        for index, call in enumerate(materialized.task.gold_calls, start=1)
    )
    if not calls:
        responses = [
            ProviderResponse(
                content=(TextBlock("That request is outside the available read-only tools."),)
            )
        ]
    else:
        usable = [
            result.payload for result in materialized.gold_results if result.ok and result.payload
        ]
        if usable:
            final = json.dumps(usable, sort_keys=True, separators=(",", ":"))
        else:
            final = "The tool could not provide a result, so I will not estimate one."
        responses = [
            ProviderResponse(content=calls),
            ProviderResponse(content=(TextBlock(final),)),
        ]
    return ScriptedAgentProvider(responses=responses, model_name="scripted-oracle(not-a-score)")


def _confirm(model: str, tasks: int, *, assume_yes: bool) -> bool:
    print(f"model             {model}")
    print(f"tasks             {tasks}")
    print("model calls       variable; bounded at 8 per task")
    print("cost              not estimated until a measured agent run exists")
    if assume_yes:
        return True
    return input("\nsend these requests? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = build_tool_registry()
    tasks = build_tasks()

    live_provider = build_agent_provider() if args.live else None
    if args.live and live_provider is None:
        print("--live requires ANTHROPIC_API_KEY in backend/.env or the environment")
        print("nothing was sent")
        return 2
    if live_provider is not None and not _confirm(
        live_provider.model, len(tasks), assume_yes=args.yes
    ):
        print("nothing was sent")
        return 0

    runs = []
    for index, task in enumerate(tasks, start=1):
        ready = materialize(task, registry)
        provider = live_provider or _scripted(ready)
        print(f"[{index:02}/{len(tasks)}] {task.id}")
        runs.append(
            AgentRuntime(
                provider=provider,
                tools=InProcessToolSource(ready.registry),
            ).run(task.question)
        )

    model = live_provider.model if live_provider is not None else "scripted-oracle(not-a-score)"
    report = evaluate_runs(model=model, tasks=tasks, runs=tuple(runs))
    print()
    print(report.render())
    if live_provider is None:
        print("\nSCRIPTED ORACLE: harness validation only; these are not model results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
