"""Give stored transactions a label.

Separate from `transactions.py` on purpose. This tool needs the LLM client
and an API key; the import tool deliberately does not, and keeping that
dependency off the import path is why classification is not done at import
time. The same line is worth holding one level down.

Rules run first and the model runs on what is left. That ordering is the same
one the evaluation uses, for the same reason: a deterministic answer costs
nothing, so it is always worth trying before anything that does - regardless
of how much it actually answers.

How much it answers is a separate claim, and this tool does not get to make
the evaluation benchmark's version of it. The published "a free deterministic
baseline answers about a fifth of rows" describes a `RuleBaseline` *fitted* on
real transaction history; this tool builds one unfitted (see `_rule_baseline`
for why), so on a real run it answers only what its two built-in heuristics
catch - far below that figure. The published per-row cost and accuracy figures
therefore describe the benchmark's run, not this tool's; `_run` prints a caveat
to that effect whenever the rule tier is unfitted, which today is every time.

The `account_type` this sends is *not* one of those differences, though an
earlier version of this docstring claimed it was. The frozen benchmark's CSV
carries no `account_type` column at all - `build_eval_subset.COLUMNS` does not
write one - so `evaluation.csv_loader` gave all 400 of its rows the same
`"unknown"` this tool sends. See `_UNKNOWN_ACCOUNT_TYPE`.

Writing is chunked rather than done in one transaction at the end: the model is
asked about `_CHUNK_ROWS` rows, those answers are committed, and only then does
the next chunk start. See `_CHUNK_ROWS` for why a run being non-atomic is the
point rather than a compromise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.llm_categoriser import LLMCategoriser
from offerdelta.evaluation.rule_baseline import RuleBaseline
from offerdelta.infrastructure.llm.factory import build_provider
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import (
    StoredTransaction,
    TransactionRepository,
    UserRepository,
)

#: Measured on the frozen evaluation benchmark for the selected prompt,
#: `categorise/v3`, and published in README.md and
#: `docs/eval/public-results.json` ("Claude, categorise/v3 (selected) ...
#: $0.001913"). Not re-derived here - the benchmark earned this number, and
#: `estimate_cost` only quotes it.
_COST_PER_ROW_USD: Final = Decimal("0.001913")

#: `StoredTransaction` carries no account type: `accounts` has no such column
#: (see `AccountRow` in `infrastructure.postgres.models`). Every categoriser
#: needs *some* value for `TransactionView.account_type`, so every row gets the
#: same placeholder `evaluation.csv_loader` already uses for a missing
#: account_type field, rather than inventing a second convention for the same
#: absence.
#:
#: That placeholder is also exactly what the frozen benchmark scored: its CSV
#: has no `account_type` column either (`build_eval_subset.COLUMNS` writes
#: none), so every one of its 400 rows reached the model as `"unknown"` too.
#: On this axis the deployed input matches what was measured. Giving
#: `accounts` a real account type would move this tool *away* from the scored
#: distribution, and the benchmark holds no account types to re-measure
#: against - so that is a measured decision with a cost, not the small
#: correction it looks like.
_UNKNOWN_ACCOUNT_TYPE: Final = "unknown"

#: Strictly above the highest confidence `Prediction` allows (`0 <= confidence
#: <= 1`), so `awaiting_review`'s `suggested_confidence < threshold` leg is
#: true for every stored value. `TransactionRepository` has no "every
#: transaction" query; `awaiting_review` at this threshold is the closest
#: thing to one it exposes - every row this tenant owns that a person has not
#: confirmed, whatever a categoriser previously said about it. That also
#: means `--reclassify` never re-spends on a row a person already confirmed:
#: `confirmed_label` already settles what that row reports, so touching
#: `suggested_label` again could not change anything downstream.
_RECLASSIFY_THRESHOLD: Final = Decimal("1.001")


#: How many rows the model is asked about between commits.
#:
#: A run over real history spends minutes inside `predict_many`. Holding one
#: transaction open across all of it keeps a connection checked out for the
#: whole run, which is precisely the case `pool_pre_ping` cannot cover: it
#: validates a connection at *checkout*, so a pooler dropping the idle
#: connection mid-run surfaces as a failure at the next statement rather than
#: being replaced. Committing between chunks hands the connection back to the
#: pool before each model call, so every chunk's first write checks a fresh one
#: out.
#:
#: The price is that a run is no longer one transaction. That is deliberate
#: here and would be wrong elsewhere: classification rows are independent, and
#: `unclassified()` skips whatever already carries a suggestion, so re-running
#: the command *is* resume, and the answers already paid for survive a failure.
#: `seed_demo.py` keeps its single commit for the opposite reason - a
#: half-seeded demo is a broken one.
_CHUNK_ROWS: Final = 50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="categorise.py", description=__doc__)
    parser.add_argument(
        "--user", required=True, help="email of the user whose transactions this labels"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="label at most N rows",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help=(
            "also re-run rows that already carry a suggestion (a confirmed label is never touched)"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm spending on the rows the rules could not answer",
    )
    return parser


def _open_scope(email: str) -> TenantScope:
    """Resolve the user and open the session the whole run works through.

    A module-level function precisely so a test can replace the database
    entirely, the same reason `transactions.py`'s `_scope_for` exists - except
    that one is handed an already-open session from inside a `with` block, and
    this one has to own the session itself: it is the only thing standing
    between `main` and a database.
    """
    session = Session(get_engine())
    stored = UserRepository(session).by_email(email)
    if stored is None:
        raise ValidationError(f"no user {email!r}; create it first with users.py create")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _rule_baseline() -> RuleBaseline:
    """The free first pass every row gets before anything can cost money.

    Deliberately not fitted from `data/eval/transactions.csv`: that file is
    private, gitignored research data (see `.gitignore`'s "financial data"
    section) that neither CI nor a real deployment ever has, so a tool that
    needed it to run would not run there. An empty merchant table still runs
    `RuleBaseline`'s two built-in heuristics - the transfer-keyword scan and
    the positive-amount-is-income guess - so this starts at the bottom of
    what the class can do rather than at nothing.

    Real rule coverage today is therefore far below the ~20% the evaluation
    benchmark measures for a merchant table fitted on real history; see this
    tool's task report for that as a named concern.
    """
    return RuleBaseline(merchant_labels={})


@dataclass
class _ProvenancedLLM:
    """Wraps `LLMCategoriser` so this tool's provenance string lives in one place.

    `LLMCategoriser.name` is `llm:<model>`, which is what the evaluation
    report's column headers need. `record_suggestion`'s `by` needs
    `<model>:<prompt_version>` instead, so a run made against an older
    recorded prompt is distinguishable in the stored data itself, not only in
    whichever log happened to be kept - Task 3's own example value,
    `"claude-haiku-4-5:categorise/v3"`, is exactly this shape.
    """

    categoriser: LLMCategoriser
    by: str

    @property
    def name(self) -> str:
        return self.by

    def predict(self, view: TransactionView) -> Prediction:
        return self.categoriser.predict(view)

    def predict_many(self, views: Sequence[TransactionView]) -> list[Prediction]:
        return self.categoriser.predict_many(views)


def _llm_categoriser() -> _ProvenancedLLM:
    """Build the real model client - the one seam a test cannot cross for free.

    A database can be stood in for with a plain Python object; a live model
    call cannot be, since it needs a real key and spends real money. Naming
    this as its own module-level function is what lets a test replace the
    whole model step with a stand-in, the same reason `_open_scope` exists
    for the database.
    """
    provider = build_provider(get_settings())
    if provider is None:
        raise ValidationError(
            "ANTHROPIC_API_KEY is not set; add it to backend/.env or the "
            "environment before running categorise.py with --yes"
        )
    return _ProvenancedLLM(
        categoriser=LLMCategoriser(provider),
        by=f"{provider.model}:{provider.prompt_version}",
    )


def _view(row: StoredTransaction) -> TransactionView:
    return TransactionView(
        normalised_merchant=row.normalised_merchant,
        raw_description=row.description,
        amount=row.amount,
        account_type=_UNKNOWN_ACCOUNT_TYPE,
    )


#: One categoriser's verdict on one row, held in memory rather than written
#: immediately. Splitting "predict" from "write" is what lets `_run` know how
#: many rows need the model - and print that - before committing to writing
#: anything at all: nothing may reach the database while a person is still
#: being asked whether to spend on the next stage, and the rule tier's free
#: answers are no exception, however tempting it is to bank them early.
_Answered = tuple[StoredTransaction, Prediction]


def _split(
    rows: Sequence[StoredTransaction], predictions: Sequence[Prediction]
) -> tuple[list[_Answered], list[_Answered]]:
    """(row, prediction) pairs, split by whether the categoriser abstained.

    Both halves carry the prediction, not just the row: an abstention is
    still a verdict - `Prediction.label` is already `"UNKNOWN"` - and a
    caller that means to escalate an abstained row to the next stage reads
    the row back out of the pair, while a caller that means to record it
    (there is no further stage after the model) can hand the pair straight
    to `_record` like any other answer. Keeping both in the same shape is
    what let `_run`'s final abstentions become recordable by reusing
    `_record` rather than needing a second write path for "examined, no
    answer" - the property Fix 2 exists for.
    """
    abstained: list[_Answered] = []
    answered: list[_Answered] = []
    for row, prediction in zip(rows, predictions, strict=True):
        (abstained if prediction.abstained else answered).append((row, prediction))
    return abstained, answered


def _record(
    repo: TransactionRepository, answered: Sequence[_Answered], *, source: str, by: str
) -> None:
    """Write every (row, prediction) pair produced by one categoriser's pass.

    The only place `record_suggestion` is called from `_run` - so the point
    at which this is first called *is* the point at which the run starts
    writing, and `_run` only calls it once the `--yes` gate has been cleared.
    """
    for row, prediction in answered:
        repo.record_suggestion(
            row.id,
            label=prediction.label,
            source=source,
            confidence=prediction.confidence,
            by=by,
        )


def _chunks(rows: Sequence[StoredTransaction], size: int) -> Iterator[Sequence[StoredTransaction]]:
    """Successive slices of `rows`, the last one short."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _rows_to_classify(
    repo: TransactionRepository, args: argparse.Namespace
) -> list[StoredTransaction]:
    """Idempotent by default: only rows nobody has suggested a label for yet.

    `--reclassify` widens that to every row a person has not confirmed - see
    `_RECLASSIFY_THRESHOLD` for how that is expressed through
    `awaiting_review`, the closest thing `TransactionRepository` exposes to
    "every transaction" without a new repository method. `--limit` has no
    equivalent on `awaiting_review`, so it is applied here instead.
    """
    if args.reclassify:
        rows = repo.awaiting_review(_RECLASSIFY_THRESHOLD)
        return rows[: args.limit] if args.limit is not None else rows
    return repo.unclassified(limit=args.limit)


def estimate_cost(rows: int) -> Decimal:
    """Upper-bound cost of running the model over `rows` rows.

    Uses the published per-row figure rather than a live token count: this
    has to be printable before any request is sent, which is the entire
    point of showing it before `--yes` is checked.
    """
    return _COST_PER_ROW_USD * Decimal(rows)


def _report(error: ValidationError | RuntimeError | SQLAlchemyError) -> int:
    """Say what went wrong without echoing anything we have not vetted.

    `config.py` promises the connection string is never logged, echoed, or put
    into an error message. A SQLAlchemy exception breaks that promise if
    printed: it can carry the DSN and the SQL that was running, real amounts
    and descriptions included. So only the redacted host is named, and nothing
    from the exception itself.
    """
    if isinstance(error, SQLAlchemyError):
        host = get_settings().redacted_dsn
        print(f"database error while reaching {host}; the operation did not complete")
    elif isinstance(error, RuntimeError) and get_settings().database_available:
        # The RuntimeError we expect is get_engine() finding CONNECTION_STRING
        # unset, whose message names no host and no SQL. Any other RuntimeError
        # came from somewhere we have not reasoned about, so it is not echoed.
        print("the operation did not complete; an unexpected internal error occurred")
    else:
        # A ValidationError's message is always ours, as is the unset-DSN one.
        print(error)
    return 1


def _run(args: argparse.Namespace) -> int:
    scope = _open_scope(args.user)
    try:
        repo = TransactionRepository(scope)
        rows = _rows_to_classify(repo, args)

        # Predict, but do not write yet - see `_split`'s docstring. Everything
        # from here down through the `--yes` check must stay read-only.
        rules = _rule_baseline()
        if rules.merchant_count == 0:
            print(
                "note: the rule tier is unfitted (no merchant table), so nearly "
                "every row will reach the model; the published per-row cost and "
                "accuracy figures describe a fitted tier, not this run"
            )
        rule_abstained, rule_answers = _split(rows, rules.predict_many([_view(r) for r in rows]))
        remaining = [row for row, _prediction in rule_abstained]

        n = len(remaining)
        print(f"{n} rows need the model; estimated ${estimate_cost(n)}")
        if not args.yes:
            return 2

        # Past the gate: now it is safe to write, starting with the rule
        # tier's own free answers, which were held back until this point too.
        # Committing them here rather than at the end is what leaves no
        # transaction open across the first model call.
        _record(repo, rule_answers, source="rules", by=rules.name)
        scope.session.commit()

        answered = 0
        abstained = 0
        if remaining:
            llm = _llm_categoriser()
            for chunk in _chunks(remaining, _CHUNK_ROWS):
                chunk_abstained, chunk_answers = _split(
                    chunk, llm.predict_many([_view(r) for r in chunk])
                )
                _record(repo, chunk_answers, source="llm", by=llm.name)
                # The model is the last tier: an abstention here is recorded
                # too, not merely printed, so `unclassified()` stops handing
                # this row back out every run and `awaiting_review()` can
                # surface it as "examined, no answer" instead of "nobody has
                # looked yet" - the property Fix 2 exists for.
                # `Prediction.abstain`'s label is already `"UNKNOWN"`, so
                # `_record` needs no special case for it.
                _record(repo, chunk_abstained, source="llm", by=llm.name)
                scope.session.commit()

                answered += len(chunk_answers)
                abstained += len(chunk_abstained)
                # Printed after the commit, so the last line a failed run
                # leaves behind names what is actually on disk - which is all
                # a resume needs, since re-running skips exactly those rows.
                print(f"  committed {answered + abstained} / {len(remaining)}")

        print(
            f"rules answered {len(rule_answers)}, model answered {answered}, abstained {abstained}"
        )
        return 0
    finally:
        scope.session.close()


def main(argv: list[str]) -> int:
    """Parse, run, and turn any expected failure into an exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValidationError, RuntimeError, SQLAlchemyError) as error:
        return _report(error)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
