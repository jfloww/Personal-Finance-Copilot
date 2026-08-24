"""Draw the annotation subset from imported transactions, reproducibly.

    uv run python build_eval_subset.py
    uv run python build_eval_subset.py --cap 5 --target 400 --seed 20260824

Four hundred rows is what the benchmark needs; 742 are imported. Which four
hundred is a decision, so it is recorded rather than left to whatever the
database returned that day.

**Merchants are capped, not sampled.** Every merchant with more than one row
contributes up to `--cap` of them. A cap of five keeps 98 merchants present
without letting the largest — 51 Tesla charging rows — become 13% of the
benchmark and drag the headline number around on its own. The remainder is
filled with one-off merchants, which is where categorisation is actually hard.

**Transfers stay in.** They are a fifth of real activity, and `TRANSFER` is the
label whose misclassification is most expensive: calling a savings move
"spending" is the double count the taxonomy exists to prevent. A benchmark
without them cannot measure the error that matters most. Macro F1 weights
classes equally, so their being easy does not inflate the headline.

**Names are masked, and the mapping is never written down.** Zelle descriptions
carry a real person's name. Each distinct person becomes a stable
`PERSON_NNN`, so a repeated counterparty stays recognisable and TO/FROM
direction survives — those are what a categoriser needs. The alias is derived
by hashing, so it is reproducible from the source without a mapping file
existing anywhere to leak.

The manifest records the parameters and the selected rows by account,
fingerprint, and occurrence — one-way values that identify a row without
describing it — so the same four hundred can be drawn again and a reported
score can name the set it came from.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import sqlalchemy as sa

from offerdelta.infrastructure.postgres.engine import get_engine

DEFAULT_OUT: Final = Path("data/eval/transactions.csv")
DEFAULT_MANIFEST: Final = Path("../docs/eval/subset-manifest.json")

#: Column order `evaluation.csv_loader` requires, plus the date.
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

#: A Zelle description is "<prefix> FROM|TO <name> [reference]". The direction
#: is meaningful to a categoriser; the name is not.
_ZELLE: Final = re.compile(r"^(?P<head>.*?\b(?P<dir>from|to)\b)\s*(?P<tail>.*)$", re.IGNORECASE)

#: Trailing bank reference on a Zelle line. Stripped before aliasing so the
#: same person is one alias rather than one per transaction.
_REFERENCE: Final = re.compile(r"\s+([A-Z0-9]{6,})$")

#: How many bands to stratify the singleton draw across, by absolute amount.
_BANDS: Final = 4


@dataclass(frozen=True)
class Row:
    account: str
    fingerprint: str
    occurrence: int
    posted_on: str
    description: str
    amount: Decimal
    merchant: str

    @property
    def transaction_id(self) -> str:
        """Stable across runs: derived from the row, not from its position."""
        return f"{self.account}-{self.fingerprint[:8]}-{self.occurrence}"


def load_rows() -> list[Row]:
    """Every imported transaction, ordered so a run is deterministic."""
    with get_engine().connect() as connection:
        records = connection.execute(
            sa.text("""
                SELECT a.key AS account, t.fingerprint, t.occurrence, t.posted_on,
                       t.description, t.amount, t.normalised_merchant
                FROM transactions t JOIN accounts a ON a.id = t.account_id
                ORDER BY a.key, t.fingerprint, t.occurrence
            """)
        ).all()
    return [
        Row(
            account=r.account,
            fingerprint=r.fingerprint,
            occurrence=r.occurrence,
            posted_on=r.posted_on.isoformat(),
            description=r.description,
            amount=r.amount,
            merchant=r.normalised_merchant,
        )
        for r in records
    ]


def _person_key(tail: str) -> str:
    """The counterparty, without the per-transaction bank reference."""
    return _REFERENCE.sub("", tail).strip().upper()


def build_aliases(rows: list[Row]) -> dict[str, str]:
    """A stable `PERSON_NNN` per distinct counterparty.

    Ordered by hash rather than alphabetically, so the number carries no
    information about the name it stands for. Held in memory only: the caller
    writes masked descriptions, never this mapping.
    """
    people: set[str] = set()
    for row in rows:
        match = _ZELLE.match(row.description)
        if match and "zelle" in row.description.lower():
            key = _person_key(match.group("tail"))
            if key:
                people.add(key)

    ordered = sorted(people, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    return {name: f"PERSON_{index:03d}" for index, name in enumerate(ordered, start=1)}


def mask(description: str, aliases: dict[str, str]) -> str:
    """Replace a counterparty with its alias, keeping the direction."""
    if "zelle" not in description.lower():
        return description
    match = _ZELLE.match(description)
    if not match:
        return description
    key = _person_key(match.group("tail"))
    alias = aliases.get(key)
    if alias is None:
        return description
    return f"{match.group('head').strip()} {alias}"


def choose(
    rows: list[Row], *, cap: int, target: int, seed: int
) -> tuple[list[Row], dict[str, int]]:
    """Cap the repeated merchants, then fill with singletons.

    Singletons are drawn across amount bands so the filler is not accidentally
    all micro-charges from one account, which would make the long tail look
    easier than it is.
    """
    by_merchant: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_merchant[row.merchant].append(row)

    repeated: list[Row] = []
    singletons: list[Row] = []
    for merchant in sorted(by_merchant):
        group = by_merchant[merchant]
        if len(group) == 1:
            singletons.append(group[0])
        else:
            repeated.extend(group[:cap])

    rng = random.Random(seed)
    wanted = max(target - len(repeated), 0)

    banded: dict[int, list[Row]] = defaultdict(list)
    ordered = sorted(singletons, key=lambda r: abs(r.amount))
    for position, row in enumerate(ordered):
        banded[min(position * _BANDS // max(len(ordered), 1), _BANDS - 1)].append(row)

    picked: list[Row] = []
    per_band = wanted // _BANDS
    for band in range(_BANDS):
        pool = sorted(banded[band], key=lambda r: r.transaction_id)
        take = per_band if band < _BANDS - 1 else wanted - len(picked)
        picked.extend(rng.sample(pool, min(take, len(pool))))

    chosen = repeated + picked
    return chosen, {
        "repeated_merchant_rows": len(repeated),
        "singleton_rows": len(picked),
        "merchants_capped": sum(1 for g in by_merchant.values() if len(g) > cap),
        "distinct_merchants_available": len(by_merchant),
    }


def write_dataset(rows: list[Row], aliases: dict[str, str], path: Path) -> None:
    """The annotation file, with every counterparty already masked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.posted_on, r.transaction_id)):
            writer.writerow(
                {
                    "transaction_id": row.transaction_id,
                    "description": mask(row.description, aliases),
                    "amount": f"{row.amount:.2f}",
                    "annotator_a_label": "",
                    "annotator_b_label": "",
                    "final_label": "",
                    "acceptable_labels": "",
                    "ambiguity_note": "",
                    "posted_on": row.posted_on,
                }
            )


def write_manifest(
    rows: list[Row], path: Path, *, cap: int, target: int, seed: int, counts: dict[str, int]
) -> str:
    """Enough to draw the same rows again, and nothing that describes them.

    Rows are named by account, fingerprint, and occurrence. A fingerprint is a
    one-way digest, so the manifest identifies a transaction without revealing
    its merchant, and can be committed where the dataset cannot.
    """
    selection = sorted(f"{r.account}:{r.fingerprint}:{r.occurrence}" for r in rows)
    digest = hashlib.sha256("\n".join(selection).encode()).hexdigest()

    per_account: dict[str, int] = defaultdict(int)
    for row in rows:
        per_account[row.account] += 1

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parameters": {"cap": cap, "target": target, "seed": seed, "amount_bands": _BANDS},
        "counts": {
            "selected": len(rows),
            **counts,
            "per_account": dict(sorted(per_account.items())),
        },
        "selection_sha256": digest,
        "selection": selection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return digest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="build_eval_subset.py", description=__doc__)
    parser.add_argument("--cap", type=int, default=5, help="max rows per repeated merchant")
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing dataset, losing any labels in it",
    )
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        print(
            f"{args.out} already exists. Overwriting would discard any labels in it; "
            f"pass --force if that is what you want."
        )
        return 1

    rows = load_rows()
    if not rows:
        print("no imported transactions to draw from")
        return 1

    aliases = build_aliases(rows)
    chosen, counts = choose(rows, cap=args.cap, target=args.target, seed=args.seed)
    write_dataset(chosen, aliases, args.out)
    digest = write_manifest(
        chosen, args.manifest, cap=args.cap, target=args.target, seed=args.seed, counts=counts
    )

    masked = sum(1 for r in chosen if mask(r.description, aliases) != r.description)
    print(f"{args.out}: {len(chosen)} rows")
    print(f"  {counts['repeated_merchant_rows']} from repeated merchants (cap {args.cap})")
    print(f"  {counts['singleton_rows']} singletons, seed {args.seed}")
    print(f"  {masked} descriptions masked across {len(aliases)} distinct counterparties")
    print(f"{args.manifest}: selection {digest[:16]}...")
    print("\nlabel it with: uv run python annotate.py --resume --annotator=a")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
