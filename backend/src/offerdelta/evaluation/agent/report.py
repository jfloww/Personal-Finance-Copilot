"""Aggregate an agent run set without publishing transcripts or arguments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from offerdelta.agent.transcript import AgentRun
from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.agent.grounding import GroundingReport, score_grounding
from offerdelta.evaluation.agent.tasks import AgentTask, TaskKind
from offerdelta.evaluation.agent.trace import TraceScore, score_trace

_PLACES = Decimal("0.0001")


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    kind: TaskKind
    trace: TraceScore
    grounding: GroundingReport
    stopped_reason: str
    fault_passed: bool | None


@dataclass(frozen=True)
class AgentEvaluationReport:
    model: str
    tasks: int
    expected_calls: int
    actual_calls: int
    correctly_selected: int
    exact_arguments: int
    numeric_tokens: int
    grounded_tokens: int
    answers_with_fabrication: int
    fault_tasks: int
    fault_tasks_passed: int
    provider_failures: int
    turn_limit_stops: int
    results: tuple[TaskResult, ...]

    @property
    def selection_precision(self) -> Decimal:
        return _ratio(self.correctly_selected, self.actual_calls)

    @property
    def selection_recall(self) -> Decimal:
        return _ratio(self.correctly_selected, self.expected_calls)

    @property
    def argument_accuracy(self) -> Decimal | None:
        if self.correctly_selected == 0:
            return None
        return _ratio(self.exact_arguments, self.correctly_selected)

    @property
    def numeric_grounding(self) -> Decimal | None:
        if self.numeric_tokens == 0:
            return None
        return _ratio(self.grounded_tokens, self.numeric_tokens)

    @property
    def fault_pass_rate(self) -> Decimal | None:
        if self.fault_tasks == 0:
            return None
        return _ratio(self.fault_tasks_passed, self.fault_tasks)

    def render(self) -> str:
        return "\n".join(
            (
                "AGENT TOOL-USE EVALUATION",
                f"  model                       {self.model}",
                f"  tasks                       {self.tasks}",
                f"  tool selection precision    {self.selection_precision}",
                f"  tool selection recall       {self.selection_recall}",
                f"  exact argument accuracy     {self.argument_accuracy}",
                f"  numeric grounding           {self.numeric_grounding}",
                f"  answers with fabrication    {self.answers_with_fabrication}",
                f"  fault handling              {self.fault_tasks_passed}/{self.fault_tasks}",
                f"  provider failures           {self.provider_failures}",
                f"  turn-limit stops            {self.turn_limit_stops}",
            )
        )


def evaluate_runs(
    *, model: str, tasks: tuple[AgentTask, ...], runs: tuple[AgentRun, ...]
) -> AgentEvaluationReport:
    if not tasks:
        raise ValidationError("agent evaluation needs at least one task")
    if len(tasks) != len(runs):
        raise ValidationError(f"agent evaluation has {len(tasks)} tasks but {len(runs)} runs")

    results: list[TaskResult] = []
    for task, run in zip(tasks, runs, strict=True):
        trace = score_trace(run, task.gold_calls)
        grounding = score_grounding(run)
        fault_passed = None
        if task.kind is TaskKind.FAULT_INJECTED:
            target_results = [
                call.result
                for call in run.tool_calls
                if task.fault is not None and call.name == task.fault.tool_name
            ]
            unusable = bool(target_results) and all(
                not result.ok or not result.payload for result in target_results
            )
            fault_passed = unusable and not grounding.fabrications
        results.append(
            TaskResult(
                task_id=task.id,
                kind=task.kind,
                trace=trace,
                grounding=grounding,
                stopped_reason=run.stopped_reason,
                fault_passed=fault_passed,
            )
        )

    return AgentEvaluationReport(
        model=model,
        tasks=len(results),
        expected_calls=sum(result.trace.expected_calls for result in results),
        actual_calls=sum(result.trace.actual_calls for result in results),
        correctly_selected=sum(result.trace.correctly_selected for result in results),
        exact_arguments=sum(result.trace.exact_arguments for result in results),
        numeric_tokens=sum(result.grounding.total_numbers for result in results),
        grounded_tokens=sum(result.grounding.grounded_numbers for result in results),
        answers_with_fabrication=sum(
            result.grounding.answers_with_fabrication for result in results
        ),
        fault_tasks=sum(result.fault_passed is not None for result in results),
        fault_tasks_passed=sum(result.fault_passed is True for result in results),
        provider_failures=sum(result.stopped_reason == "provider_failure" for result in results),
        turn_limit_stops=sum(result.stopped_reason == "turn_limit" for result in results),
        results=tuple(results),
    )


def _ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal(1) if numerator == 0 else Decimal(0)
    return (Decimal(numerator) / Decimal(denominator)).quantize(_PLACES)
