# The auditable monthly report — design

**Date:** 2026-09-01
**Status:** approved for planning
**Scope:** Phase 1. Builds on Phase 0's authentication and tenant isolation.
Changes no evaluation artifact and no published figure.

---

## 1. What this is

A monthly spending report over imported transactions, where **every figure
expands to the transactions that produced it**, and where the report says
plainly how much of the month it could account for.

### The starting position, stated accurately

An earlier reading of this phase called it "assembly and exposure, not
construction", on the grounds that `total_income`, `total_spending`,
`net_cash_flow`, and `detect_recurring` already exist, are tested, and have no
caller. That much is true. It is also not the whole picture, and the missing
half is the work.

Those functions consume the **domain** `Transaction`, which carries a
`TransactionKind` and, for spending, a `CostCategory`. Nothing anywhere in
`src/` constructs one — only `LabelledTransaction`, from the evaluation CSV,
and `StoredTransaction`, a raw database row. The `transactions` table has no
label column of any kind. The categorisers live entirely inside
`offerdelta.evaluation` and have no caller in `application/` or
`infrastructure/`.

```
CSV -> import -> transactions table (raw rows, unclassified)
                        |
                     (nothing)

hand-labelled CSV -> categorisers -> metrics -> artifacts
```

So the domain functions are not idle waiting to be wired up; they have nothing
to eat. **The missing link is classification, and building it is most of this
phase.** The report is what becomes possible afterwards.

### The claim being made

**Every imported row in a reported month is accounted for exactly once, and
the report distinguishes what a person confirmed from what a model guessed.**

Note what that claim does *not* say. It says nothing about whether any row is
labelled *correctly* — see §7.

---

## 2. Non-goals

- **No accuracy claim from the report.** Label accuracy is what the frozen
  validation benchmark measures. The report measures coverage. Conflating them
  would let a confident-looking screen stand in for a measured one.
- **No balance reconciliation.** No bank export in this project carries a
  running balance, so agreeing with the bank's own figure is not available at
  any price. The report knows movement, not balance, and says so.
- **No re-labelling of the evaluation dataset.** The hand-labelled 400-row CSV
  is a frozen artifact; this phase neither reads nor writes it.
- **No agent, no natural-language questions.** That is the next phase, and it
  is easier to ground against a report that already exists.
- **No new unauthenticated route.**
- **No model calls from the deployed service.** Classification is a local
  operation; the deployment reads its results.

---

## 3. Classification storage

```
transactions
  + suggested_label       text NULL      -- machine
  + suggested_source      text NULL      -- 'rules' | 'llm'
  + suggested_confidence  numeric NULL
  + suggested_by          text NULL      -- 'rules/v1', 'claude-haiku-4-5:categorise/v3'
  + suggested_at          timestamptz NULL
  + confirmed_label       text NULL      -- human; re-classification never touches it
  + confirmed_at          timestamptz NULL
```

Every column is nullable and added in one migration. There is no backfill and
nothing destructive: existing rows stay unclassified, which is what they are.

### One label, not a kind and a category

`evaluation/labels.py` composes the label space as `CostCategory` plus
`INCOME`, `TRANSFER`, `REFUND`, plus `UNKNOWN` for abstention — and derives the
spending half from the enum rather than restating it, "so the taxonomy has
exactly one definition."

One label therefore determines both facts: a `CostCategory` value means the row
is spending and names its category; `INCOME` / `TRANSFER` / `REFUND` name the
kind and carry no category. Storing a separate `kind` column would be a second
copy of a derivable fact, and two copies can disagree — the same reasoning that
put tenancy on one table in Phase 0.

### The effective label

```
effective = confirmed_label ?? suggested_label
```

Re-classification writes only the `suggested_*` columns. Erasing a person's
decision is not something a caller must remember to avoid; it is not
expressible.

### NULL and `'UNKNOWN'` are different

`NULL` means *never looked at*. `'UNKNOWN'` means *looked at, and the
categoriser declined to answer* — the abstention the label space exists to make
measurable. A report that merged them would hide unfinished work behind a model
that knew it was unsure.

### No database check constraint

The label space has exactly one definition, derived from `CostCategory` in
Python. A `CHECK` listing the labels would be a second copy that drifts the
first time a category is added. Validation happens at the repository boundary
against `LABEL_SPACE`, and a test asserts an invalid label is refused.

---

## 4. The classification path

A new root script, `categorise.py` — not a subcommand of `transactions.py`.

```
categorise.py --user <email> [--limit N] [--reclassify] [--yes]
```

It is separate because it needs the LLM client and an API key, and the import
tool deliberately does not. Keeping that dependency out of the import path is
why classification does not happen at import time; the same line is worth
holding one level down. It joins the four file lists and `known-first-party`
with everything else.

**Rules first, then the model.** The same order the evaluation uses, for the
same reason: a free deterministic baseline answers what it can — about 20% of
rows on the benchmark — and only the remainder costs anything.

**Cost is stated before it is spent.** The command prints the number of
unclassified rows and the estimated cost, and writes nothing without `--yes`.
742 rows is roughly $1.41; the amount is not the point. This repository already
refuses to write an import without confirmation.

**Idempotent by default.** Only rows with `suggested_label IS NULL` are
processed unless `--reclassify` is passed. Either way `confirmed_label` is
untouched.

**Provenance travels with every write.** `suggested_by` records which ruleset
or which model-and-prompt produced the label, so a report can answer what
produced its numbers.

### The review threshold is measured, not invented

The queue's confidence threshold comes from a sweep over the saved predictions:
chosen on the development split, reported once on the frozen validation
benchmark. That is the discipline that selected prompt v3, applied to a routing
decision. It needs no new inference and costs nothing.

The failure analysis already supplies the ground: mean confidence 0.8411 when
right and 0.6078 when wrong, with 20 wrong benchmark rows already answered
below 0.60.

**The threshold is applied when the queue is read, not when a row is written.**
Classification stores the confidence it measured; the queue is defined as rows
whose effective label is unconfirmed and whose `suggested_confidence` is below
the threshold, or whose label is `UNKNOWN` or `NULL`. Storing a queued flag
instead would freeze one threshold into the data and require re-classifying —
at cost — to change a number that is a reading decision, not a measurement.

---

## 5. The report tree

`DerivationNode` moves from `domain/comparisons/derivation.py` to
`domain/common/derivation.py`. It already imports only from `domain/common`,
and a monthly report is not a comparison; leaving it where it is would make the
package boundary say something untrue. The new module is
`domain/reports/monthly.py`, pure domain, ignorant of sessions and
infrastructure.

### The root is the sum of every row, not net cash flow

```
2026-03   (sum of every imported row in the window)
├─ income
├─ spending
│   ├─ LIVING_DINING
│   ├─ LIVING_GROCERY
│   └─ …
├─ refunds
├─ transfers            <- a branch, not an exclusion
└─ unclassified
    ├─ not yet examined     (NULL)
    └─ examined, no answer  (UNKNOWN)
```

`DerivationNode` refuses to construct a node whose children do not sum to it.
Rooting the tree at *every row* therefore makes completeness a structural
property rather than a check somebody remembers to run: drop a row or count one
twice and **the report cannot be built**.

Transfers stay in the tree for that reason. Excluding them would mean the root
is no longer the sum of everything, and a transfer misclassified as spending
would leave a tree that still balances — losing exactly the error the structure
exists to catch.

**The headline figures are derived views of the tree**: net cash flow is
income − spending + refunds, savings rate follows from it. They lead the
screen; the balancing tree is what makes them answerable.

Period is `PeriodKind.MONTHLY`.

### Evidence already carries the review state

`Evidence` distinguishes `USER_CONFIRMED` from `ASSUMED`, and its own comment
says an assumption "must be visually distinct wherever it is displayed". A leaf
built from a confirmed label is `USER_CONFIRMED`; one built from a machine
suggestion is `ASSUMED` — the amount is sourced, the categorisation is a guess.

Branch evidence is already the weakest of its children, so a category subtotal
containing one unreviewed row is marked as containing a guess, and that
propagates to the root without any new machinery. **Emptying the review queue
is visible as the tree turning from assumed to confirmed.**

---

## 6. API and exposed surface

```
GET  /v1/reports/months               available months, each complete or partial
GET  /v1/reports/monthly/{YYYY-MM}    the tree, with coverage
GET  /v1/review-queue                 rows awaiting review, most recent first
POST /v1/transactions/{id}/label      record a confirmed label
```

The queue is ordered by `posted_on` descending — most recent transactions
first, because a person reviewing their spending recognises last week's
merchants and has forgotten March's. A `?month=YYYY-MM` filter narrows it to
the month a report is complaining about, which is how the report and the queue
connect.

`POST .../label` accepts one label, validated against `LABEL_SPACE` (422 if
not), and answers **404 for a transaction id belonging to another tenant** —
the same rule as everywhere else, for the same reason.

All four depend on Phase 0's `_scope`. No new unauthenticated route exists, and
the containment test that pinned `/v1/auth/token` is extended to pin these as
authenticated.

`DerivationNodeSchema.of()` already serialises a tree for `/v1/demo/derivation`,
so the report reuses it — and inherits the contract test that fails if any
amount is emitted as a JSON number.

### Partial months are read from declared data

A snapshot import records `window_start` and `window_end` on its batch as a
claim that the window is complete. A month is complete when the account's
declared windows cover it, and partial otherwise. This is the importer's
declaration, not an inference from row density.

August 2026 runs to the 21st and is therefore partial. **Month-over-month
comparison runs between complete months only**; seven complete months
(January–July) are available.

### The deployment finally shows the product

Phase 0 seeded two synthetic tenants. Giving them transactions and confirmed
labels lets a reviewer with the demo account see a real monthly report — the
first time the deployed service shows the product rather than a static
evaluation page. Synthetic data is honestly `USER_CONFIRMED`. Real bank data
stays on the machine that imported it.

---

## 7. What proves it

| Scenario | Expected |
|---|---|
| Report over a known set of rows | root equals their sum |
| A row dropped or double-counted by the builder | `ValidationError`, no report |
| Every row confirmed | root `USER_CONFIRMED` |
| One row machine-suggested | root `ASSUMED` |
| That row then confirmed | root turns `USER_CONFIRMED` |
| `NULL` and `UNKNOWN` both present | two distinct leaves |
| `--reclassify` re-run | `confirmed_label` unchanged |
| A label outside `LABEL_SPACE` | refused at the repository |
| A's report | contains no row of B's |
| A's review queue | contains no row of B's |
| A labels a transaction id belonging to B | 404 |
| Declared windows do not cover the month | marked partial, excluded from comparison |

The last two isolation rows are new work, not covered by Phase 0. That phase
proved the repositories; the report and the queue are new query paths and have
to prove their own scoping.

### The limit, stated plainly

**The tree proves that every row is accounted for. It does not prove that any
row is labelled correctly.** A transfer misclassified as spending leaves a tree
that balances perfectly. Label accuracy is what the frozen validation benchmark
measures; coverage is what the report measures. The header reads "N classified,
M awaiting review" because that is a coverage statement, and it must not be
read as an accuracy one.

Everything above runs in CI, which since Phase 0 supplies both a database and a
signing key and fails rather than skips when either is missing.

---

## 8. Deferred, with reasons

**A trained classifier.** 400 labelled rows across 33 labels is about twelve
per label. It will not beat the model it would replace, and the value such a
model would add — a calibrated probability to route on — is partly available
already from the measured confidence signal. Revisit when the labelled set is
much larger.

**Drift monitoring, active learning, shadow deployment.** Each needs a stream
of production traffic. There is one user and one machine.

**A second vendor on the benchmark.** Worth doing after the agent phase, so one
adapter scores two vendors across both the categoriser and the agent.

**Balance reconciliation.** Not deferred — unavailable. See §2.

---

## 9. What this unblocks

The next phase's agent answers questions by calling tools over a deterministic
engine. A monthly report whose every figure already carries its own derivation
path makes answer grounding checkable rather than asserted: the number in a
sentence either traces to a node in the tree or it does not.
