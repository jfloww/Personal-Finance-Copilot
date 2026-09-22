from __future__ import annotations

from offerdelta.agent.tools.definitions import build_tool_registry
from offerdelta.agent.tools.registry import ToolResult
from offerdelta.agent.transcript import AgentMessage, AgentRun, ToolCallRecord
from offerdelta.evaluation.agent.faults import FaultMode, inject_fault
from offerdelta.evaluation.agent.grounding import score_grounding
from offerdelta.evaluation.agent.tasks import TaskKind, build_tasks, materialize
from offerdelta.evaluation.agent.trace import GoldCall, score_trace


def _run(
    *,
    question: str,
    final: str,
    calls: tuple[ToolCallRecord, ...] = (),
) -> AgentRun:
    return AgentRun(
        question=question,
        final_text=final,
        messages=(AgentMessage(role="user", content=()),),
        tool_calls=calls,
        model_calls=1,
        input_tokens=0,
        output_tokens=0,
    )


def test_grounding_accepts_tool_values_question_values_and_rounding() -> None:
    call = ToolCallRecord(
        id="1",
        name="answer",
        arguments={},
        result=ToolResult.success({"amount": "4217.33", "month": 8}),
        latency_ms=0,
    )
    run = _run(
        question="Project 24 months.",
        final="Over 24 months it is about $4,217, with month 8 included.",
        calls=(call,),
    )

    report = score_grounding(run)

    assert report.total_numbers == 3
    assert report.grounded_numbers == 3
    assert report.fabrications == ()


def test_grounding_reports_fabricated_numbers() -> None:
    run = _run(question="What is it?", final="The answer is $4,300.")

    report = score_grounding(run)

    assert report.grounded_numbers == 0
    assert report.fabrications == ("$4,300",)


def test_grounding_handles_solver_precision_beyond_default_decimal_context() -> None:
    call = ToolCallRecord(
        id="1",
        name="equivalent_salary",
        arguments={},
        result=ToolResult.success({"salary": "88560.60028076171875"}),
        latency_ms=0,
    )

    report = score_grounding(_run(question="Salary?", final="$88,560.60", calls=(call,)))

    assert report.fabrications == ()


def test_trace_scores_selection_and_arguments_separately() -> None:
    calls = (
        ToolCallRecord(
            id="1",
            name="break_even",
            arguments={"current": "wrong"},
            result=ToolResult.failure("bad"),
            latency_ms=0,
        ),
        ToolCallRecord(
            id="2",
            name="list_profiles",
            arguments={},
            result=ToolResult.success({}),
            latency_ms=0,
        ),
    )
    gold = (
        GoldCall(name="break_even", arguments={"current": "right"}),
        GoldCall(name="equivalent_salary", arguments={}),
    )

    score = score_trace(_run(question="", final="", calls=calls), gold)

    assert str(score.selection_precision) == "0.5000"
    assert str(score.selection_recall) == "0.5000"
    assert str(score.argument_accuracy) == "0.0000"


def test_each_fault_mode_keeps_the_same_schema_and_changes_only_execution() -> None:
    registry = build_tool_registry()
    arguments = {"current": "auburn_current", "candidate": "new_jersey_candidate"}
    original = registry.get("break_even")
    assert original is not None

    for mode in FaultMode:
        faulted = inject_fault(registry, "break_even", mode)
        changed = faulted.get("break_even")
        assert changed is not None
        assert changed.input_schema == original.input_schema
        result = faulted.call("break_even", arguments)
        if mode is FaultMode.EMPTY:
            assert result.ok
            assert result.payload == {}
        else:
            assert not result.ok


def test_task_set_has_the_authored_strata_and_computed_gold() -> None:
    tasks = build_tasks()

    assert len(tasks) == 28
    assert {kind: sum(task.kind is kind for task in tasks) for kind in TaskKind} == {
        TaskKind.SINGLE_TOOL: 8,
        TaskKind.MULTI_TOOL: 6,
        TaskKind.DISTRACTOR: 6,
        TaskKind.OUT_OF_SCOPE: 4,
        TaskKind.FAULT_INJECTED: 4,
    }

    materialized = materialize(tasks[1], build_tool_registry())
    assert materialized.gold_results
    assert all(result.ok for result in materialized.gold_results)
