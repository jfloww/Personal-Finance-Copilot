"""A small authored task set whose gold results are always computed by tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from offerdelta.agent.tools.definitions import TARGET_METRIC
from offerdelta.agent.tools.registry import JsonValue, ToolRegistry, ToolResult
from offerdelta.application.queries.demo_profiles import PROFILE_KEYS
from offerdelta.evaluation.agent.faults import FaultMode, inject_fault
from offerdelta.evaluation.agent.trace import GoldCall

TASK_COUNT = 28


class TaskKind(StrEnum):
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    DISTRACTOR = "distractor"
    OUT_OF_SCOPE = "out_of_scope"
    FAULT_INJECTED = "fault_injected"


@dataclass(frozen=True)
class FaultSpec:
    tool_name: str
    mode: FaultMode


@dataclass(frozen=True)
class AgentTask:
    id: str
    kind: TaskKind
    question: str
    gold_calls: tuple[GoldCall, ...]
    fault: FaultSpec | None = None


@dataclass(frozen=True)
class MaterializedTask:
    task: AgentTask
    registry: ToolRegistry
    gold_results: tuple[ToolResult, ...]


def materialize(task: AgentTask, registry: ToolRegistry) -> MaterializedTask:
    active = (
        registry
        if task.fault is None
        else inject_fault(registry, task.fault.tool_name, task.fault.mode)
    )
    results = tuple(active.call(call.name, call.arguments) for call in task.gold_calls)
    return MaterializedTask(task=task, registry=active, gold_results=results)


def build_tasks() -> tuple[AgentTask, ...]:
    """Return 28 tasks: 8 single, 6 multi, 6 distractor, 4 scope, 4 fault."""
    current, candidate = PROFILE_KEYS
    pair: dict[str, JsonValue] = {"current": current, "candidate": candidate}

    def call(name: str, **extra: JsonValue) -> GoldCall:
        return GoldCall(name=name, arguments={**pair, **extra})

    compare_12 = call("compare_offers", horizon_months=12, move_date="2026-07-01")
    compare_36 = call("compare_offers", horizon_months=36, move_date="2026-07-01")
    break_even = call("break_even")
    equivalent = call("equivalent_salary")
    negotiation = call("negotiation_gap", target=TARGET_METRIC)
    housing = call("explain_component", component="housing")
    commute = call("explain_component", component="commute")
    health = call("explain_component", component="health")
    profiles = GoldCall(name="list_profiles", arguments={})

    tasks = (
        AgentTask(
            "single-profiles", TaskKind.SINGLE_TOOL, "Which demo profiles exist?", (profiles,)
        ),
        AgentTask(
            "single-comparison-12",
            TaskKind.SINGLE_TOOL,
            "Compare the Auburn and Jersey City offers over 12 months with a July 1, 2026 move.",
            (compare_12,),
        ),
        AgentTask(
            "single-comparison-36",
            TaskKind.SINGLE_TOOL,
            "Compare the two offers over 36 months with a July 1, 2026 move.",
            (compare_36,),
        ),
        AgentTask(
            "single-break-even",
            TaskKind.SINGLE_TOOL,
            "When does the move first break even, and when does it stay there?",
            (break_even,),
        ),
        AgentTask(
            "single-equivalent",
            TaskKind.SINGLE_TOOL,
            "What candidate salary matches current first-year disposable cash?",
            (equivalent,),
        ),
        AgentTask(
            "single-negotiation",
            TaskKind.SINGLE_TOOL,
            "What individual negotiation levers close the first-year cash gap?",
            (negotiation,),
        ),
        AgentTask(
            "single-housing", TaskKind.SINGLE_TOOL, "Explain the housing calculation.", (housing,)
        ),
        AgentTask(
            "single-commute", TaskKind.SINGLE_TOOL, "Explain the commute calculation.", (commute,)
        ),
        AgentTask(
            "multi-overview-break-even",
            TaskKind.MULTI_TOOL,
            "Give the 12-month cash comparison and say when the move breaks even.",
            (compare_12, break_even),
        ),
        AgentTask(
            "multi-overview-housing",
            TaskKind.MULTI_TOOL,
            "Compare the offers for 12 months and show what drives housing.",
            (compare_12, housing),
        ),
        AgentTask(
            "multi-salary-negotiation",
            TaskKind.MULTI_TOOL,
            "Give the equivalent salary and the available negotiation levers.",
            (equivalent, negotiation),
        ),
        AgentTask(
            "multi-break-even-commute",
            TaskKind.MULTI_TOOL,
            "When is break-even, and how is commute cash derived?",
            (break_even, commute),
        ),
        AgentTask(
            "multi-overview-health",
            TaskKind.MULTI_TOOL,
            "Compare 36 months and explain the health component.",
            (compare_36, health),
        ),
        AgentTask(
            "multi-discover-compare",
            TaskKind.MULTI_TOOL,
            "List the available profiles, then compare them over 12 months.",
            (profiles, compare_12),
        ),
        AgentTask(
            "distractor-salary-not-break-even",
            TaskKind.DISTRACTOR,
            "What base salary makes the candidate economically equivalent?",
            (equivalent,),
        ),
        AgentTask(
            "distractor-break-even-not-salary",
            TaskKind.DISTRACTOR,
            "In which month does cumulative cash stop being negative?",
            (break_even,),
        ),
        AgentTask(
            "distractor-health-not-overview",
            TaskKind.DISTRACTOR,
            "Show only the derivation of health costs, not the whole comparison.",
            (health,),
        ),
        AgentTask(
            "distractor-negotiation-not-equivalent",
            TaskKind.DISTRACTOR,
            "Which single offer terms could be negotiated to close the gap?",
            (negotiation,),
        ),
        AgentTask(
            "distractor-overview-not-components",
            TaskKind.DISTRACTOR,
            "What is the overall 12-month cash and wealth difference?",
            (compare_12,),
        ),
        AgentTask(
            "distractor-list-not-compare",
            TaskKind.DISTRACTOR,
            "Tell me the names of the available demo profiles only.",
            (profiles,),
        ),
        AgentTask(
            "scope-stock",
            TaskKind.OUT_OF_SCOPE,
            "Which stock should I buy tomorrow?",
            (),
        ),
        AgentTask(
            "scope-tax-filing",
            TaskKind.OUT_OF_SCOPE,
            "File my New Jersey tax return for me.",
            (),
        ),
        AgentTask(
            "scope-live-rent",
            TaskKind.OUT_OF_SCOPE,
            "Find today's cheapest live apartment listing in Manhattan.",
            (),
        ),
        AgentTask(
            "scope-private-account",
            TaskKind.OUT_OF_SCOPE,
            "Log in to my bank and transfer money to checking.",
            (),
        ),
        AgentTask(
            "fault-break-even-error",
            TaskKind.FAULT_INJECTED,
            "When does the move break even?",
            (break_even,),
            FaultSpec("break_even", FaultMode.ERROR),
        ),
        AgentTask(
            "fault-equivalent-empty",
            TaskKind.FAULT_INJECTED,
            "What is the equivalent salary?",
            (equivalent,),
            FaultSpec("equivalent_salary", FaultMode.EMPTY),
        ),
        AgentTask(
            "fault-comparison-refuse",
            TaskKind.FAULT_INJECTED,
            "Compare the offers over 12 months.",
            (compare_12,),
            FaultSpec("compare_offers", FaultMode.REFUSE),
        ),
        AgentTask(
            "fault-negotiation-error",
            TaskKind.FAULT_INJECTED,
            "What should I negotiate?",
            (negotiation,),
            FaultSpec("negotiation_gap", FaultMode.ERROR),
        ),
    )
    assert len(tasks) == TASK_COUNT
    return tasks
