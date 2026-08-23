# Annotation dataset

The hand-labelled benchmark. `transactions.csv` is the real file;
`transactions.example.csv` shows the conventions on ten rows.

Label it with the annotator rather than a spreadsheet:

```bash
uv run python annotate.py statement.csv --annotator=a   # first pass
uv run python annotate.py --resume --annotator=b        # independent second pass
uv run python annotate.py --resume --annotator=final    # adjudicate disagreements
```

It parses the statement with the same importer the real import path uses, so a
file that annotates cleanly is one that will import cleanly. Labels are chosen
by two-character code or by unique prefix, a repeated merchant pre-fills its
previous label for you to accept or override, and the B pass never loads or
displays anything from the A pass - the independence this dataset depends on is
the tool's job, not your willpower.

Check it as you go, not at the end:

```bash
uv run python validate_dataset.py
```

A misspelled label caught at row 40 costs a moment. The same label caught after
four hundred rows costs a re-read of all of them.

## Columns

| Column | Required | Meaning |
|---|---|---|
| `transaction_id` | yes | Unique. Any stable string. |
| `description` | yes | The raw bank description, exactly as exported. |
| `amount` | yes | Signed: negative is money out. `$1,234.56` and `(45.00)` both parse. |
| `annotator_a_label` | yes | First annotator's independent label. |
| `annotator_b_label` | blank unless double-annotated | Second annotator's **independent** label. Assign it without seeing A. |
| `final_label` | blank unless adjudicated | The adjudicated answer. Blank means A's label stands. |
| `acceptable_labels` | blank unless ambiguous | Pipe-separated, e.g. `LIVING_GROCERY\|LIVING_OTHER`. |
| `ambiguity_note` | required if `acceptable_labels` is set | Why this row has no single right answer. |

Optional if you have them: `posted_on` (ISO date), `account_type`, `source`,
`bank_format`. Absent columns are defaulted, not rejected.

`normalised_merchant` is **not** a column. It is derived from `description`
using the same normaliser the categorisers use, so the split key can never drift
from the key the systems actually see.

## The two rules that keep scores honest

**Ambiguity is authored, never inferred.** A row is ambiguous exactly when you
fill in `acceptable_labels`. Annotator disagreement does *not* make a row
ambiguous — disagreement usually means one annotator was wrong, and treating
every disagreement as legitimate ambiguity would quietly inflate every system's
score.

**Every disagreement must be adjudicated.** If A and B differ and `final_label`
is blank, the validator refuses the file. Left alone the dataset would silently
adopt A's label as gold and the disagreement would vanish with nothing
downstream able to detect it.

So adjudication has two outcomes:

- one annotator was wrong → put the correct label in `final_label`, leave
  `acceptable_labels` blank;
- the transaction genuinely has no single answer → put your chosen gold in
  `final_label`, list the defensible labels in `acceptable_labels` (the gold
  must be among them), and explain why in `ambiguity_note`.

## Targets

- 400 transactions
- 100 independently double-annotated, spread across the label space rather than
  drawn from one corner — the validator warns if the subset covers too few
  labels, because kappa would then describe that corner rather than the dataset
- both annotators label independently before any adjudication

## Labels

The label space is every `CostCategory` value plus `INCOME`, `TRANSFER`,
`REFUND`, and `UNKNOWN`. `UNKNOWN` is for systems to abstain with — annotators
should not use it; if a row is unclear, that is what `acceptable_labels` and
`ambiguity_note` are for.

`TRANSFER` matters more than its frequency suggests. A movement between your own
accounts read as spending reports money spent that is still there, and the
matching leg read as income reports money earned that was already there. Both
totals wrong, in opposite directions, both plausible.

## Privacy

This directory is for **redacted** data. No account numbers, no full card
numbers, no names of counterparties you would not publish. The deployed demo
uses synthetic data; this file exists to produce a headline F1 and stays local
unless you deliberately decide otherwise.
