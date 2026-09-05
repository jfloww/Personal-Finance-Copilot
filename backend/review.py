"""Work the review queue: turn a categoriser's uncertainty into a decision.

The queue is where a person's judgement enters the data, and this is the only
command that writes `confirmed_label`. Everything else - the API route
included - either proposes a label or reads one back.

Two decisions shape the interaction, and both exist because the taxonomy holds
33 labels while a real queue holds hundreds of rows. A tool that asks for a
full label name once per row is a tool nobody finishes using, and a queue
nobody finishes is worth about as much as no queue at all.

**Enter accepts what the model already said.** A low-confidence row is not an
unanswered one: the model did answer, it just answered below
`REVIEW_THRESHOLD`. Agreeing is the common case, so it must cost one keystroke
while disagreeing costs a word. A row nobody examined, and a row the model
examined and declined, have no answer to agree with - there Enter stands for
nothing and the label has to be typed.

**A merchant is offered as a group, never forced into one.** Rows sharing a
normalised merchant usually share a label - the same regularity a fitted rule
tier would exploit - so the queue is walked merchant by merchant and one
answer can settle several rows. Offered, not applied: `confirmed_label` means
a person decided, so every row is printed before the question is asked, and
`s` splits the group back into individual rows for the merchants where one
label genuinely will not do.

Writing is committed per merchant. The connection would otherwise sit checked
out while a person thinks, which is the failure `categorise.py`'s `_CHUNK_ROWS`
documents; and stopping halfway has to keep what was already decided, because
a review session that loses an hour of answers is one nobody starts again.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.application.reports.review import REVIEW_THRESHOLD, confirm, queue
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import StoredTransaction, UserRepository

#: Sorted once so `?` and the ambiguity message list labels in a stable order.
_LABELS: Final = tuple(sorted(LABEL_SPACE))

_USER_HELP: Final = "email of the user whose review queue this works"

#: Months are 1-12 and a year is four digits. Named rather than inline so the
#: bounds read as the calendar rather than as arithmetic.
_FIRST_MONTH: Final = 1
_LAST_MONTH: Final = 12
_YEAR_DIGITS: Final = 4

#: How many labels `?` prints per line, and how wide each column is. The
#: longest label is `HOUSING_PARKING_RESIDENTIAL` at 27 characters.
_LABEL_COLUMNS: Final = 3
_LABEL_WIDTH: Final = 30

#: Descriptions are printed to a fixed width so the amount column lines up.
#: A bank description longer than this is truncated for display only; the
#: stored row is untouched.
_DESCRIPTION_WIDTH: Final = 40


def _month(raw: str) -> tuple[int, int]:
    """Parse `YYYY-MM` as argparse's `type=`, so a bad value is a usage error."""
    year, _, month = raw.partition("-")
    if len(year) != _YEAR_DIGITS or not year.isdigit() or not month.isdigit():
        raise argparse.ArgumentTypeError(f"{raw!r} is not a month; use YYYY-MM")
    if not _FIRST_MONTH <= int(month) <= _LAST_MONTH:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a month; use YYYY-MM")
    return int(year), int(month)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show the queue grouped by merchant; writes nothing")
    listing.add_argument("--user", required=True, help=_USER_HELP)
    listing.add_argument("--month", type=_month, default=None, metavar="YYYY-MM")

    work = sub.add_parser("work", help="confirm labels, one merchant at a time")
    work.add_argument("--user", required=True, help=_USER_HELP)
    work.add_argument("--month", type=_month, default=None, metavar="YYYY-MM")

    return parser


def _open_scope(email: str) -> TenantScope:
    """Resolve the user and open the session the whole session works through.

    A module-level function precisely so a test can replace the database
    entirely, the same reason `categorise.py`'s `_open_scope` exists - and it
    owns the session for the same reason too: it is the only thing standing
    between `main` and a database.
    """
    session = Session(get_engine())
    stored = UserRepository(session).by_email(email)
    if stored is None:
        raise ValidationError(f"no user {email!r}; create it first with users.py create")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _prompt(message: str) -> str:
    """Read one answer. Its own function so a test can supply the answers."""
    return input(message)


def _is_a_terminal() -> bool:
    """Whether there is somebody there to answer. Also a test seam."""
    return sys.stdin.isatty()


def resolve_label(entry: str) -> tuple[str | None, tuple[str, ...]]:
    """One label, or the candidates that left what was typed ambiguous.

    Matching is case-insensitive and by substring rather than by prefix,
    because every label in this taxonomy is `AREA_THING`: prefix matching
    would make `din` mean nothing and force `living_din` for `LIVING_DINING`.
    An exact match always wins, so a label that is a substring of another can
    still be reached by typing it in full.

    Returning the candidates rather than picking one is the point. An entry
    that could mean four things is not an answer, and this refuses to invent
    one - see `_ask_label`, which prints them and asks again.
    """
    typed = entry.strip().upper()
    if not typed:
        return None, ()
    if typed in LABEL_SPACE:
        return typed, ()
    matches = tuple(label for label in _LABELS if typed in label)
    if len(matches) == 1:
        return matches[0], ()
    return None, matches


def suggested_default(rows: Sequence[StoredTransaction]) -> str | None:
    """The label Enter accepts for a group, or `None` when there is nothing to accept.

    `None` in two cases, both meaning "the model has said nothing you can
    agree with": no row carries a real suggestion, or the rows carry
    *different* suggestions. The second matters - taking the commonest would
    hide the model's own disagreement behind a single keystroke, which is
    exactly the kind of unexamined confirmation `confirmed_label` must not
    collect.

    `ABSTAIN` is not a suggestion for this purpose. The model examined the row
    and declined to answer, so there is no proposal to accept; a person may
    still type `UNKNOWN` deliberately, which means something quite different.
    """
    proposed: set[str] = set()
    for row in rows:
        label = row.suggested_label
        if label is not None and label != ABSTAIN:
            proposed.add(label)
    return proposed.pop() if len(proposed) == 1 else None


def _state(row: StoredTransaction) -> str:
    """Why this row is in the queue, in the queue's own three-way vocabulary."""
    if row.suggested_label is None:
        return "never examined"
    if row.suggested_label == ABSTAIN:
        return "examined, no answer"
    return f"{row.suggested_source} {row.suggested_label} {row.suggested_confidence}"


def group_by_merchant(
    rows: Sequence[StoredTransaction],
) -> dict[str, list[StoredTransaction]]:
    """Rows by normalised merchant, keeping the queue's own ordering.

    `awaiting_review` returns the most recent uncertain row first and a plain
    dict preserves insertion order, so the merchant carrying the most recent
    uncertain spending is the first one asked about.
    """
    groups: dict[str, list[StoredTransaction]] = {}
    for row in rows:
        groups.setdefault(row.normalised_merchant, []).append(row)
    return groups


def _total(rows: Sequence[StoredTransaction]) -> Decimal:
    return sum((row.amount.amount for row in rows), Decimal(0))


def _print_labels() -> None:
    for start in range(0, len(_LABELS), _LABEL_COLUMNS):
        row = _LABELS[start : start + _LABEL_COLUMNS]
        print("  " + "".join(f"{label:<{_LABEL_WIDTH}}" for label in row).rstrip())


def _print_row(row: StoredTransaction) -> None:
    description = row.description[:_DESCRIPTION_WIDTH]
    print(
        f"  {row.posted_on}  {description:<{_DESCRIPTION_WIDTH}}"
        f"  {row.amount.amount:>10}  {_state(row)}"
    )


def _render_group(
    position: int, total: int, merchant: str, rows: Sequence[StoredTransaction]
) -> None:
    print(f"\n[{position} / {total}]  {merchant}   {len(rows)} rows   {_total(rows)}")
    for row in rows:
        _print_row(row)


@dataclass(frozen=True)
class _Answer:
    """One prompt's outcome: a label, or a decision to stop the session."""

    label: str | None = None
    stop: bool = False


@dataclass(frozen=True)
class _GroupOutcome:
    """How many rows one merchant settled, and whether the session continues."""

    confirmed: int
    stop: bool


def _ask_label(message: str, default: str | None) -> _Answer:
    """Ask until the answer names exactly one label, or the person stops.

    Every path that is not a single label re-asks and writes nothing: an
    ambiguous entry prints what it could have meant, an unmatched one says so,
    and an empty one with no suggestion behind it explains why Enter did not
    work here. EOF counts as stopping rather than raising - a review session
    reading from a closed pipe must end, not crash.
    """
    while True:
        try:
            entry = _prompt(message).strip()
        except EOFError:
            return _Answer(stop=True)

        if entry.lower() == "q":
            return _Answer(stop=True)
        if entry == "?":
            _print_labels()
            continue
        if not entry:
            if default is None:
                print("  nothing to accept here; type a label, ? to list them, or q to stop")
                continue
            return _Answer(label=default)

        label, candidates = resolve_label(entry)
        if label is not None:
            return _Answer(label=label)
        if candidates:
            print(f"  {entry!r} could be: {', '.join(candidates)}")
        else:
            print(f"  no label matches {entry!r}; ? lists them all")


def _ask_scope(count: int) -> str:
    """What to do with one label across a group: `y`, `s`, or `q`."""
    while True:
        try:
            entry = _prompt(f"apply to all {count}?  [y] yes  [s] split row-by-row  [q] quit: ")
        except EOFError:
            return "q"
        choice = entry.strip().lower()
        if choice in {"y", "s", "q"}:
            return choice
        print("  answer y, s, or q")


def _split_group(
    scope: TenantScope, rows: Sequence[StoredTransaction], group_label: str
) -> _GroupOutcome:
    """Ask about each row of a group separately.

    The group's own answer is not thrown away: it becomes the default for any
    row the model had nothing to say about, so splitting AMAZON into groceries
    and clothing still costs one keystroke on the rows that agree.
    """
    confirmed = 0
    for row in rows:
        _print_row(row)
        default = row.suggested_label
        if default is None or default == ABSTAIN:
            default = group_label
        answer = _ask_label("  label> ", default)
        if answer.stop or answer.label is None:
            return _GroupOutcome(confirmed=confirmed, stop=True)
        confirm(scope, row.id, answer.label)
        confirmed += 1
    return _GroupOutcome(confirmed=confirmed, stop=False)


def _work_group(scope: TenantScope, rows: Sequence[StoredTransaction]) -> _GroupOutcome:
    """Settle one merchant. The caller renders it and commits afterwards."""
    answer = _ask_label("label> ", suggested_default(rows))
    if answer.stop or answer.label is None:
        return _GroupOutcome(confirmed=0, stop=True)

    if len(rows) == 1:
        confirm(scope, rows[0].id, answer.label)
        return _GroupOutcome(confirmed=1, stop=False)

    choice = _ask_scope(len(rows))
    if choice == "q":
        return _GroupOutcome(confirmed=0, stop=True)
    if choice == "s":
        return _split_group(scope, rows, answer.label)

    for row in rows:
        confirm(scope, row.id, answer.label)
    return _GroupOutcome(confirmed=len(rows), stop=False)


def _list(args: argparse.Namespace) -> int:
    """Show what is waiting, and write nothing at all."""
    scope = _open_scope(args.user)
    try:
        groups = group_by_merchant(queue(scope, REVIEW_THRESHOLD, month=args.month))
        if not groups:
            print("the review queue is empty")
            return 0
        for merchant, rows in groups.items():
            print(f"{len(rows):>4}  {_total(rows):>12}  {merchant}")
        rows_total = sum(len(rows) for rows in groups.values())
        print(f"\n{rows_total} rows across {len(groups)} merchants")
        return 0
    finally:
        scope.session.close()


def _work(args: argparse.Namespace) -> int:
    """Walk the queue merchant by merchant, committing each one as it is settled."""
    if not _is_a_terminal():
        # Fails closed, as `transactions.py`'s `_confirm` does: this command
        # writes a person's decisions, and there is nobody here to make them.
        print("review work asks a question per merchant; run it from a terminal")
        return 1

    scope = _open_scope(args.user)
    try:
        groups = group_by_merchant(queue(scope, REVIEW_THRESHOLD, month=args.month))
        if not groups:
            print("the review queue is empty")
            return 0

        confirmed = 0
        for position, (merchant, rows) in enumerate(groups.items(), start=1):
            _render_group(position, len(groups), merchant, rows)
            outcome = _work_group(scope, rows)
            confirmed += outcome.confirmed
            # Committed before the loop can end, so stopping - or an EOF -
            # keeps every answer already given.
            scope.session.commit()
            if outcome.stop:
                break

        print(f"\nconfirmed {confirmed} rows")
        return 0
    finally:
        scope.session.close()


def _report(error: ValidationError | RuntimeError | SQLAlchemyError) -> int:
    """Say what went wrong without echoing anything we have not vetted.

    The same three branches as `categorise.py`'s `_report`, for the same
    reason: `config.py` promises the connection string never reaches an error
    message, and a SQLAlchemy exception can carry the DSN and the SQL that was
    running - real descriptions and amounts included.
    """
    if isinstance(error, SQLAlchemyError):
        host = get_settings().redacted_dsn
        print(f"database error while reaching {host}; the operation did not complete")
    elif isinstance(error, RuntimeError) and get_settings().database_available:
        print("the operation did not complete; an unexpected internal error occurred")
    else:
        print(error)
    return 1


def main(argv: list[str]) -> int:
    """Parse, dispatch, and turn any expected failure into an exit code."""
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "list": _list,
        "work": _work,
    }

    try:
        return handlers[args.command](args)
    except (ValidationError, RuntimeError, SQLAlchemyError) as error:
        return _report(error)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
