# Agent, MCP, and tool-use evaluation — design

**Date:** 2026-08-24
**Status:** implemented; live model benchmark and public aggregate are pending
**Scope:** V3. Builds on the V2 categorisation benchmark; changes nothing in it.

---

## 1. What this is

A single agent that answers offer-comparison questions by calling the
deterministic finance engine, exposed as tools; the same tools published over
MCP; and an evaluation that measures how well the agent uses them.

The point is not the agent. The point is the **evaluation of tool use**, which
is where the project's existing evidence — a labelled benchmark, a rejected
experiment, a measured iteration — extends into agent behaviour.

### Why offer comparison, and not spending analysis

The engine is deterministic, so **gold answers are computed rather than
authored**. For any task, the correct final number is whatever
`ComparisonEngine` produces, and the correct tool call is the one that produces
it. Nothing has to be hand-graded, which is what makes a tool-use eval tractable
at this size.

It also touches no personal data: demo profiles only, no database, no
transactions. The V2 constraint that real financial data never reaches an
unauthenticated path is satisfied by construction rather than by care.

### The claim being tested

The architecture asserts that **the model stays outside the calculation
boundary** — it proposes, the engine computes. Until now that has been a
structural claim backed by import contracts. An agent that writes prose
containing numbers can violate it invisibly: it can state a figure it computed
itself rather than one a tool returned.

Answer grounding (§6.1) turns that claim into a number that can move.

---

## 2. Non-goals

- **No multi-agent anything.** One agent, one loop.
- **No second provider.** Claude only, as in V2.
- **No public agent endpoint.** Render stays database-free and key-free; nothing
  deployed reaches a model. See §8.
- **No change to the categoriser.** V2's published figures depend on its request
  construction; touching it would invalidate them. Consolidating the two Claude
  clients is later, separately-measured work.
- **No trajectory-efficiency score.** Cost and latency are recorded because
  `Usage` already provides them; they are not a graded dimension.

---

## 3. Package layout and boundaries

```
offerdelta/
  agent/
    tools/
      registry.py     Tool, ToolResult, ToolRegistry
      definitions.py  the six tools, pure over application + domain
    runtime.py        the agent loop
    transcript.py     one recorded run
  infrastructure/mcp/
    server.py         stdio MCP server over the registry
  evaluation/agent/
    tasks.py          the task set
    grounding.py      answer-grounding scorer
    trace.py          tool-selection and argument scorer
    faults.py         fault injection
    report.py         the agent-eval report
```

`agent/` is a sibling of `evaluation/` and `ingest/`, outside the
`api → application → domain` layer contract, exactly as they are.

### New import-linter contract

```toml
[[tool.importlinter.contracts]]
name = "Agent tools depend only on application and domain"
type = "forbidden"
source_modules = ["offerdelta.agent.tools"]
forbidden_modules = [
    "anthropic",
    "mcp",
    "offerdelta.infrastructure",
    "offerdelta.api",
]
```

The tool surface must stay free of transport and SDK concerns, because two
different consumers use it and neither may leak into it. Demonstrated
red-green: written first against a deliberate violating import, then made to
pass.

---

## 4. The tool surface

Six tools, all read-only, all over demo profiles.

| Tool | Input | Returns |
|---|---|---|
| `list_profiles` | — | profile keys and labels |
| `compare_offers` | `current`, `candidate`, `horizon_months`, `move_date` | component deltas, cumulative series, both derivations |
| `explain_component` | `current`, `candidate`, `component` | that component's derivation tree |
| `break_even` | `current`, `candidate` | months until the move pays for itself, or that it never does |
| `equivalent_salary` | `current`, `candidate` | salary in the candidate location matching the current outcome |
| `negotiation_gap` | `current`, `candidate`, `target` | the raise needed to close the gap |

### Registry shape

```python
@dataclass(frozen=True)
class ToolResult:
    """What a tool returned. `payload` carries decimal strings, never floats."""
    ok: bool
    payload: Mapping[str, object]
    error: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, object]
    call: Callable[[Mapping[str, object]], ToolResult]
```

**Schemas are written explicitly, not generated from signatures.** Three
reasons: MCP requires an `inputSchema`; `strict: true` requires
`additionalProperties: false` and an explicit `required`, which guarantees
`tool_use.input` validates before any domain code sees it; and a generated
schema would make the conformance test in §5 vacuous — it would be comparing a
thing to itself.

**Every monetary value crosses as a decimal string.** The same rule the HTTP
boundary already enforces, with the same reasoning: a float here is an
approximation that the agent will then quote to a person.

---

## 5. MCP adapter

`infrastructure/mcp/server.py` serves the registry over stdio, launchable from
Claude Desktop or Claude Code.

The agent runtime takes a **tool source**, either:

- **in-process** — registry entries called directly. The default for evaluation:
  fast, no subprocess, no protocol round-trip.
- **MCP** — `mcp_tool(...)` from `anthropic.lib.tools.mcp`, over a
  `stdio_client` session.

### The conformance test

Runs a fixed subset of tasks through both sources and asserts:

1. the MCP server advertises exactly the registry's tool names;
2. each advertised `inputSchema` is byte-identical to the registry's;
3. tool results are identical for identical inputs.

This is what makes "the agent uses MCP" a checked claim rather than a hope. If
the adapter drifts from what the agent uses in evaluation, the build fails.

---

## 6. The evaluation

A **task** is a question, a gold trace, and a gold answer. Gold answers come
from calling the engine directly — the same code path the tools wrap — so they
cannot drift from what a correct agent would find.

### 6.1 Answer grounding — the headline

Every numeric token in the agent's final text must be accounted for. A number is
**grounded** if it is:

- present in some tool result payload for this run; or
- present in the question itself (a horizon of 24 months, a stated salary); or
- a rounding of a grounded number, where the grounded value rounded to the
  answer's own precision equals the answer's value.

Anything else is a **fabrication**: the model produced a figure instead of
calling for one.

Reported as: grounded numbers / total numbers, fabrications per answer, and the
count of answers with at least one fabrication. That last one is the number that
matters — one invented figure spoils an answer regardless of how many correct
ones surround it.

The rounding rule is deliberately generous. `$4,217` against a tool's
`4217.33` is grounded; `$4,300` is not. Being generous here means a reported
fabrication is a real one.

### 6.2 Tool selection and arguments

Against the gold trace:

- **Selection** — precision and recall over tool names called.
- **Arguments** — of the correctly-selected calls, the share whose arguments
  match the gold constraints exactly.

Scored through the existing `evaluation.metrics` module rather than a second
scorer, so an agent's precision means the same thing as the categoriser's.

### 6.3 Failure handling

`faults.py` wraps a tool so that it:

- **errors** — raises, returning `ToolResult(ok=False)`;
- **returns empty** — succeeds with nothing useful;
- **refuses** — reports the input is out of its supported range.

Scored on whether the agent **abstains rather than inventing an answer**. A run
passes a fault task when both hold:

1. **zero fabrications** — no number in the answer is ungrounded by §6.1; and
2. **no final figure is asserted** for the quantity the question asked about.

Both are mechanical. Whether the agent *explains* the failure well is recorded
as a note on the run and is not scored, because grading explanation quality
means grading prose, which this harness deliberately does not do.

### 6.4 Task set

28 tasks across five kinds — 8 single-tool, 6 multi-tool, 6 distractor,
4 out-of-scope, 4 fault-injected:

| Kind | What it probes |
|---|---|
| Single-tool | the obvious call is made |
| Multi-tool | several calls, right order not required, all required present |
| Distractor | a nearby wrong tool is tempting (`break_even` when `equivalent_salary` is meant) |
| Out of scope | no tool can answer; the agent should decline |
| Fault-injected | §6.3 |

Small enough to hand-check, large enough that selection precision means
something. Every task's gold answer is computed, never typed.

---

## 7. Runtime configuration

- **Model:** `claude-opus-5`. Tool selection is the thing being measured;
  cheapening the model to reduce eval cost would measure a different system. The
  task set is tens of tasks, so this costs less than V2's 135-row run.
- **Thinking:** adaptive. Effort left at its default, tunable per run and
  recorded with the result.
- **Loop:** `client.beta.messages.tool_runner`. The SDK owns the loop; this
  project owns the tools. No server tools are used, so `pause_turn` cannot occur
  and no restart handling is needed.
- **Strict tools:** `strict: true` on every definition.
- **Tool inputs** are parsed with `json.loads`, never string-matched.
- **History mirroring:** the runner keeps its own message history and does not
  expose it, so `transcript.py` mirrors messages while iterating. That mirror is
  the eval's raw material.

### Determinism, and its limit

V2 pinned `temperature=0`. **Sampling parameters are rejected on Claude Opus 5**,
so the agent evaluation cannot be made deterministic the same way. This is a
real limitation and is reported as one: results are single runs, and run-to-run
variance is not separated from any change being measured. A repeated-run
subset may be added later to quantify it; it is not in this scope.

---

## 8. What ships, and what does not

Same shape as V2, which is proven:

- Runs happen **locally**, with a key, against demo profiles.
- Transcripts stay local. They contain no personal data — demo profiles only —
  but they are per-task detail, and the artifact publishes aggregates.
- The public deployment gains **aggregate agent-eval results** in the existing
  `/demo/evaluation/latest` payload and a section on the landing page.
- No agent endpoint is deployed. Nothing public reaches a model.

The existing containment tests extend to the new payload keys: no merchant
names, no dates outside the generation timestamp, no credentials, no list long
enough to be per-row.

---

## 9. Dependencies

Add `anthropic[mcp]` (requires Python ≥ 3.10; this project is 3.12).

This introduces a **second Claude client**: the SDK for the agent, the existing
hand-rolled `urllib` client for the categoriser. That is deliberate. Migrating
the categoriser would change how its requests are constructed, which could shift
V2's published numbers — a change that must be measured, not assumed, and not
bundled into unrelated work.

The tool runner is **beta** (`client.beta.messages.tool_runner`). Recorded as a
known risk: a beta surface can change. The manual loop remains available as a
fallback and is a contained change, since the loop is one file.

---

## 10. Risks

**The tool-runner binding for registry-defined schemas is unverified.** The
documented path is the `@beta_tool` decorator, which generates schemas from
function signatures. Feeding explicit schemas from a registry may require a
different construction. **The first implementation task is a spike that pins
this against the installed SDK**, with the decorated-function path as the
fallback — in which case the registry supplies the explicit schema to MCP, and a
test asserts the generated and explicit schemas agree.

**Grounding by numeric extraction is approximate.** A number written as words
("about four thousand") escapes it, and one that coincidentally matches a tool
value is scored as grounded. Both directions are documented in the report. The
check is a floor on fabrication, not a proof of its absence.

**The task set is small and authored by one person.** It probes what its author
thought to probe. Stated as a limitation next to the results, as with V2's
benchmark.

---

## 11. Testing

- Unit: registry construction, schema validity, every tool's happy path and
  error path, grounding scorer (grounded / rounded / fabricated / from-question),
  trace scorer, fault wrappers.
- Contract: MCP conformance (§5); the agent-eval payload stays sanitized;
  published agent figures come from a recorded run.
- Integration: one end-to-end agent run against a scripted provider, no key
  required — mirroring how V2's harness runs keyless.
- The existing suite must stay green, and the keyless path must keep working:
  without a key the agent reports that it cannot run, as the categoriser does.

---

## 12. Definition of done

1. Six tools callable in-process, each with an explicit strict-compatible schema.
2. The MCP server serves them, and the conformance test proves it matches.
3. An agent answers offer-comparison questions through the tool runner.
4. The task set exists, with computed gold answers.
5. Grounding, selection/argument, and failure-handling scores are recorded from a
   real run.
6. Aggregate results published; transcripts stay local.
7. One documented command reproduces the run.
8. All five checks pass, including the new import contract.
9. Limitations stated: single runs, no sampling control on Opus 5, small
   authored task set, approximate grounding.
