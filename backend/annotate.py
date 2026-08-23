"""Label a bank export into the annotation dataset, one row at a time.

    uv run python annotate.py statement.csv --annotator=a
    uv run python annotate.py --resume --annotator=b
    uv run python annotate.py --resume --annotator=final

Three passes. A and B label independently; `final` visits only the rows where
they disagree and records the adjudicated answer. Skipping the third pass is
not neutral - `validate_dataset.py` refuses a file with unadjudicated
disagreements, because leaving `final_label` blank makes the dataset silently
adopt A and the disagreement disappears.

Four hundred rows is a lot of typing, so three things carry the weight.

**Codes, not spelling.** Thirty labels is too many to type. Each has a
two-character code grouped by prefix - `l2` is `LIVING_DINING` - and any unique
prefix of the label itself also works, so `groc` picks `LIVING_GROCERY`.

**Merchant memory.** Bank exports repeat: one coffee shop can be forty rows.
Once a merchant is labelled, later rows with the same normalised merchant
pre-fill that label and Enter accepts it. It is a suggestion, never automatic -
your own earlier decision offered back, which keeps labels consistent rather
than biasing them.

**Independence is structural.** `data/eval/README.md` requires annotator B to
label without seeing A. That cannot rest on willpower, so the B pass never
loads, displays, or remembers anything from the A pass - not the labels, and
not A's merchant memory.

Ambiguity is authored, never inferred: `?` records `acceptable_labels` and
demands the note explaining why the row has no single right answer. Two
annotators disagreeing does not make a row ambiguous - that usually means one
of them was wrong, and treating it otherwise inflates every score.

Progress is saved as you go, so stopping at row 47 costs nothing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.costs.categories import CostCategory
from offerdelta.domain.transactions.parsing import normalise_description
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE, NON_SPENDING_LABELS
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv

DEFAULT_DATASET: Final = Path("data/eval/transactions.csv")

#: Written every N decisions. Small enough that a crash costs seconds of work,
#: large enough that a four-hundred-row session is not four hundred writes.
SAVE_EVERY: Final = 10

#: Column order from `evaluation.csv_loader.REQUIRED_COLUMNS`, plus the one
#: optional column worth keeping: a date makes a disputed row findable in the
#: original statement.
COLUMNS: Final = (
    "transaction_id",
    "description",
    "amount",
    "annotator_a_label",
    "annotator_b_label",
    "final_label",
    "acceptable_labels",
    "ambiguity_note",
    "posted_on",
)

_GROUP_KEYS: Final[dict[str, str]] = {
    "HOUSING": "h",
    "HEALTH": "e",
    "COMMUTE": "c",
    "LIVING": "l",
    "RELOCATION": "r",
}


def _build_codes() -> dict[str, str]:
    """Two-character code per label, grouped so the codes are learnable.

    Derived from `CostCategory` rather than written out, so a new category
    cannot be added to the taxonomy and forgotten here.
    """
    codes: dict[str, str] = {}
    counters: dict[str, int] = dict.fromkeys(_GROUP_KEYS.values(), 0)

    for category in CostCategory:
        group = category.value.split("_", 1)[0]
        key = _GROUP_KEYS[group]
        counters[key] += 1
        codes[f"{key}{counters[key]}"] = category.value

    # The non-spending labels matter more than their count suggests: a real
    # statement is full of transfers, and a categoriser that cannot say
    # TRANSFER will read a savings move as spending.
    for index, label in enumerate(sorted(NON_SPENDING_LABELS), start=1):
        codes[f"n{index}"] = label

    codes["u"] = ABSTAIN
    return codes


CODES: Final[dict[str, str]] = _build_codes()


@dataclass
class Row:
    """One transaction awaiting, or carrying, a label."""

    transaction_id: str
    description: str
    amount: str
    posted_on: str
    normalised_merchant: str
    annotator_a_label: str = ""
    annotator_b_label: str = ""
    final_label: str = ""
    acceptable_labels: str = ""
    ambiguity_note: str = ""

    def label_for(self, annotator: str) -> str:
        if annotator == "a":
            return self.annotator_a_label
        return self.annotator_b_label if annotator == "b" else self.final_label

    def set_label(self, annotator: str, label: str) -> None:
        if annotator == "a":
            self.annotator_a_label = label
        elif annotator == "b":
            self.annotator_b_label = label
        else:
            self.final_label = label

    @property
    def disputed(self) -> bool:
        """Both annotators labelled it, and they disagree.

        Only these rows need adjudicating. Leaving one unadjudicated makes the
        dataset silently adopt A, and the disagreement - which is a signal
        about the taxonomy, not noise - disappears.
        """
        return bool(
            self.annotator_a_label
            and self.annotator_b_label
            and self.annotator_a_label != self.annotator_b_label
        )

    def as_record(self) -> dict[str, str]:
        return {column: getattr(self, column) for column in COLUMNS}


@dataclass
class Session:
    """Everything one annotation pass needs to resume and finish."""

    rows: list[Row]
    annotator: str
    dataset: Path
    memory: dict[str, str] = field(default_factory=dict)

    @property
    def pending(self) -> list[Row]:
        if self.annotator == "final":
            return [row for row in self.rows if row.disputed and not row.final_label]
        return [row for row in self.rows if not row.label_for(self.annotator)]

    @property
    def done_count(self) -> int:
        if self.annotator == "final":
            return sum(1 for row in self.rows if row.disputed and row.final_label)
        return sum(1 for row in self.rows if row.label_for(self.annotator))


def resolve(entry: str) -> str | None:
    """Turn a keystroke or a typed prefix into a label.

    Returns None when nothing matches or a prefix is ambiguous - the caller
    reprompts rather than guessing, because a wrong label recorded silently is
    worse than a second keystroke.
    """
    text = entry.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in CODES:
        return CODES[lowered]

    upper = text.upper().replace("-", "_").replace(" ", "_")
    if upper in LABEL_SPACE:
        return upper

    matches = sorted(label for label in LABEL_SPACE if upper in label)
    return matches[0] if len(matches) == 1 else None


def render_menu() -> str:
    """The code grid, grouped, two columns wide enough for the longest label."""
    lines: list[str] = []
    ordered = sorted(CODES.items(), key=lambda item: (item[0][0], int(item[0][1:] or 0)))
    pairs = [f"{code:>3} {label}" for code, label in ordered]
    for index in range(0, len(pairs), 2):
        lines.append("  " + "".join(f"{cell:<34}" for cell in pairs[index : index + 2]).rstrip())
    return "\n".join(lines)


def load_bank_csv(
    path: Path, *, mapping: ColumnMapping | None, date_order: DateOrder | None
) -> list[Row]:
    """Parse a statement with the importer, so there is only one CSV reader.

    Reusing `preview_csv` means ragged rows, embedded newlines, and date-order
    detection behave exactly as they do on the import path - and a statement
    that annotates cleanly is one that will import cleanly.
    """
    preview = preview_csv(path, mapping=mapping, date_order=date_order)
    if preview.mapping is None:
        raise ValidationError(
            f"could not work out which columns {path.name} uses; pass --map explicitly"
        )
    if preview.errors:
        first = preview.errors[0]
        raise ValidationError(
            f"{len(preview.errors)} row(s) in {path.name} could not be parsed, "
            f"starting at line {first.line}: {first.reason}. Fix the file and re-run - "
            f"annotating a file the importer will later refuse wastes the annotation."
        )

    stem = path.stem
    return [
        Row(
            transaction_id=f"{stem}-{row.line:04d}",
            description=row.description,
            amount=f"{row.amount.amount:.2f}",
            posted_on=row.posted_on.isoformat(),
            normalised_merchant=row.normalised_merchant,
        )
        for row in preview.rows
    ]


def load_dataset(path: Path) -> list[Row]:
    """Read an existing annotation file so a pass can be resumed."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))

    rows: list[Row] = []
    for record in records:
        description = (record.get("description") or "").strip()
        rows.append(
            Row(
                transaction_id=(record.get("transaction_id") or "").strip(),
                description=description,
                amount=(record.get("amount") or "").strip(),
                posted_on=(record.get("posted_on") or "").strip(),
                normalised_merchant=normalise_description(description),
                annotator_a_label=(record.get("annotator_a_label") or "").strip(),
                annotator_b_label=(record.get("annotator_b_label") or "").strip(),
                final_label=(record.get("final_label") or "").strip(),
                acceptable_labels=(record.get("acceptable_labels") or "").strip(),
                ambiguity_note=(record.get("ambiguity_note") or "").strip(),
            )
        )
    return rows


def merge(existing: list[Row], incoming: list[Row]) -> list[Row]:
    """Add rows from a new statement without disturbing labelled ones."""
    seen = {row.transaction_id for row in existing}
    return existing + [row for row in incoming if row.transaction_id not in seen]


def save(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_record())


def build_memory(rows: Sequence[Row], annotator: str) -> dict[str, str]:
    """What this annotator has already decided, keyed by normalised merchant.

    Built only from this annotator's own column. Seeding B's suggestions from
    A's labels would defeat the independence the dataset depends on.
    """
    memory: dict[str, str] = {}
    for row in rows:
        label = row.label_for(annotator)
        if label and row.normalised_merchant:
            memory[row.normalised_merchant] = label
    return memory


def _prompt_ambiguity(row: Row, ask: Callable[[str], str]) -> bool:
    """Record why a row has no single right answer. Both fields or neither."""
    raw = ask("  acceptable labels (space separated, blank to cancel) > ")
    entries = [resolve(part) for part in raw.split()]
    if not entries or any(entry is None for entry in entries):
        print("  cancelled: every label must resolve")
        return False

    labels = sorted({entry for entry in entries if entry is not None})
    if len(labels) < 2:  # noqa: PLR2004 - one acceptable label is not ambiguity
        print("  cancelled: ambiguity needs at least two acceptable labels")
        return False

    note = ask("  why is this ambiguous? > ").strip()
    if not note:
        print("  cancelled: an ambiguous row needs a note saying why")
        return False

    row.acceptable_labels = "|".join(labels)
    row.ambiguity_note = note
    return True


def annotate(session: Session, ask: Callable[[str], str]) -> str:
    """Run one pass. Returns why it ended: "finished" or "quit"."""
    total = (
        sum(1 for row in session.rows if row.disputed)
        if session.annotator == "final"
        else len(session.rows)
    )
    history: list[Row] = []
    since_save = 0

    print(render_menu())
    print("\n  [enter] accept suggestion   [?] ambiguous   [s] skip   [u] undo   [q] save+quit\n")

    index = 0
    while index < len(session.rows):
        row = session.rows[index]
        settled = (
            row.final_label or not row.disputed
            if session.annotator == "final"
            else row.label_for(session.annotator)
        )
        if settled:
            index += 1
            continue

        suggestion = session.memory.get(row.normalised_merchant, "")
        hint = f"  [{suggestion}]" if suggestion else ""
        print(f"\n  [{session.done_count + 1}/{total}]  {row.amount:>10}  {row.description}")
        entry = ask(f"  label{hint} > ").strip()

        if entry.lower() == "q":
            return "quit"

        if entry.lower() == "s":
            index += 1
            continue

        if entry.lower() == "u":
            if not history:
                print("  nothing to undo")
                continue
            previous = history.pop()
            previous.set_label(session.annotator, "")
            previous.acceptable_labels = ""
            previous.ambiguity_note = ""
            index = session.rows.index(previous)
            continue

        if entry == "?":
            if not _prompt_ambiguity(row, ask):
                continue
            entry = ask("  primary label > ").strip()

        label = suggestion if not entry and suggestion else resolve(entry)
        if label is None:
            print(f"  no label matches {entry!r} - try a code, or a longer prefix")
            continue

        row.set_label(session.annotator, label)
        if row.normalised_merchant:
            session.memory[row.normalised_merchant] = label
        history.append(row)
        index += 1

        since_save += 1
        if since_save >= SAVE_EVERY:
            save(session.rows, session.dataset)
            since_save = 0

    return "finished"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotate.py", description=__doc__)
    parser.add_argument("file", type=Path, nargs="?", help="a bank export to add")
    parser.add_argument(
        "--annotator",
        required=True,
        choices=["a", "b", "final"],
        help="a and b label independently; final adjudicates where they disagree",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--map", dest="mapping", default=None)
    parser.add_argument("--dates", dest="dates", choices=[o.value for o in DateOrder], default=None)
    parser.add_argument(
        "--resume", action="store_true", help="continue the dataset without adding a file"
    )
    return parser


def _mapping(raw: str | None) -> ColumnMapping | None:
    if not raw:
        return None
    fields: dict[str, str] = {}
    for pair in raw.split(","):
        name, _, column = pair.partition(":")
        if not name or not column:
            raise ValidationError(f"--map entries look like field:Column, got {pair!r}")
        fields[name.strip()] = column.strip()
    return ColumnMapping(**fields)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if not args.file and not args.resume:
        print("give a bank export to annotate, or --resume to continue the dataset")
        return 2

    try:
        rows = load_dataset(args.dataset)
        if args.file:
            incoming = load_bank_csv(
                args.file,
                mapping=_mapping(args.mapping),
                date_order=DateOrder(args.dates) if args.dates else None,
            )
            before = len(rows)
            rows = merge(rows, incoming)
            print(f"{args.file.name}: {len(incoming)} rows, {len(rows) - before} new")
    except ValidationError as error:
        print(error)
        return 1

    if not rows:
        print(f"nothing to annotate in {args.dataset}")
        return 1

    session = Session(
        rows=rows,
        annotator=args.annotator,
        dataset=args.dataset,
        memory=build_memory(rows, args.annotator),
    )

    remaining = len(session.pending)
    if not remaining:
        save(session.rows, session.dataset)
        if args.annotator == "final":
            print("nothing to adjudicate: the annotators agree on every labelled row")
        else:
            print(f"annotator {args.annotator}: all {len(rows)} rows already labelled")
        return 0

    if not sys.stdin.isatty():
        print("annotating needs a terminal; stdin is not one")
        return 2

    print(
        f"\nannotator {args.annotator}: {remaining} of {len(rows)} rows to label"
        f"{' (B never sees A)' if args.annotator == 'b' else ''}\n"
    )

    try:
        outcome = annotate(session, input)
    except (EOFError, KeyboardInterrupt):
        outcome = "quit"
        print()

    save(session.rows, session.dataset)
    print(f"\nsaved {session.dataset}: {session.done_count} of {len(rows)} labelled ({outcome})")
    print(f"check it with: uv run python validate_dataset.py {session.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
