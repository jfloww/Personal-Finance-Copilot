"""Score the rule baseline, the LLM, and the hybrid on one frozen holdout.

    uv run python run_evaluation.py                      # stand-in, free
    uv run python run_evaluation.py --live --limit 25    # 25 rows against a real model
    uv run python run_evaluation.py --live               # the whole holdout

**The stand-in is the default, and `--live` is the only way to spend money.**
Without `--live` the LLM slot is filled by a provider that always abstains, so
the pipeline, the split, and the report can all be exercised before any model
exists. The report names the provider that ran, so a stand-in result can never
be mistaken for a real one.

`--limit` exists for the first live run. Costs scale linearly with rows, and
discovering a prompt problem on row 400 costs four hundred times what
discovering it on row 25 does. The estimate printed before a live run is
deliberately shown *before* anything is sent, so the run can be abandoned.

Rules are fitted on the development split and every system is scored on the
holdout, which no merchant crosses. The dataset checksum appears in the header
and is verified after the run: if any system mutated the labels it was scored
against, no report is produced.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.csv_loader import load_labelled_csv
from offerdelta.evaluation.dataset import LabelledDataset
from offerdelta.evaluation.llm_categoriser import HybridCategoriser, LLMCategoriser
from offerdelta.evaluation.providers import LLMProvider, LLMResponse, ScriptedProvider
from offerdelta.evaluation.report import evaluate
from offerdelta.evaluation.rule_baseline import fit_rules
from offerdelta.evaluation.splitting import merchant_disjoint_split
from offerdelta.evaluation.validation import validate_labelled_csv
from offerdelta.infrastructure.llm.factory import build_provider
from offerdelta.infrastructure.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPTS

DEFAULT_PATH = Path("data/eval/transactions.csv")
HOLDOUT_FRACTION = 0.3

#: Published per-million token prices, per model. Keyed by the exact model id
#: the provider reports, so a report can never price one model at another's
#: rate - the previous single pair of constants silently did, and a cheaper
#: model would have been reported costing three times what it did.
#:
#: A model absent from this table is reported as "not priced" rather than
#: guessed. A wrong cost is worse than no cost: it reads as measured.
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "claude-haiku-4-5": (Decimal(1), Decimal(5)),
    "claude-sonnet-5": (Decimal(3), Decimal(15)),
    "claude-opus-5": (Decimal(5), Decimal(25)),
}

#: Fallback for the pre-run estimate when the model is not in the table: the
#: most expensive one known, so an unknown model over-states rather than
#: under-states. A known model is priced at its own rate - an estimate five
#: times the real cost cannot do its job, which is to let someone say no.
_UNKNOWN_MODEL_PRICES = max(PRICES.values(), key=lambda pair: pair[1])


def prices_for(model: str) -> tuple[Decimal | None, Decimal | None]:
    """Published rates for a model, or (None, None) if we do not know them."""
    found = PRICES.get(model)
    return found if found is not None else (None, None)


#: Per-transaction token cost, for the estimate shown before a live run.
#:
#: Measured, not guessed: a 25-row live run against claude-haiku-4-5 used
#: 32,521 input and 2,320 output tokens, i.e. ~1,301 in and ~93 out per call.
#: The earlier figures were derived from an offline smoke test and were nearly
#: three times too low, so the estimate under-stated the cost - the one
#: direction it must never err in, since its whole job is to let someone say no.
#:
#: Rounded up from the measurement to keep it an upper bound. The prompt carries
#: the full taxonomy on every call, so input dominates and grows with the
#: category list.
ESTIMATED_INPUT_TOKENS = 1400
ESTIMATED_OUTPUT_TOKENS = 110


def _stand_in() -> ScriptedProvider:
    """A provider that always abstains.

    Deliberately useless. It proves the pipeline runs without pretending to a
    result, and its abstentions show up as coverage rather than as accuracy.
    """
    return ScriptedProvider(
        default=LLMResponse(
            label="UNKNOWN",
            confidence=Decimal(0),
            reason="no model configured; scripted stand-in",
            latency_ms=0,
        ),
        model_name="stand-in(no-key)",
    )


def _estimate_cost(rows: int, model: str) -> Decimal:
    """What a live run would cost, before it is made.

    The hybrid calls the model only where the rules abstain, so the true cost is
    below this. Over-estimating is the right direction for a number whose job is
    to let someone say no.
    """
    input_price, output_price = prices_for(model)
    if input_price is None or output_price is None:
        input_price, output_price = _UNKNOWN_MODEL_PRICES
    input_cost = Decimal(rows * ESTIMATED_INPUT_TOKENS) / Decimal(1_000_000) * input_price
    output_cost = Decimal(rows * ESTIMATED_OUTPUT_TOKENS) / Decimal(1_000_000) * output_price
    # Twice: the LLM system and the hybrid are scored separately, and each makes
    # its own calls.
    return (input_cost + output_cost) * 2


def _confirm_live_run(rows: int, model: str, prompt: str, *, assume_yes: bool) -> bool:
    """Show what a live run will cost, and get a yes before spending it.

    The estimate is printed whether or not it is going to be asked about, so a
    `--yes` run still leaves a record of what it expected to spend next to what
    it actually did.
    """
    rate_in, rate_out = prices_for(model)
    basis = (
        f"at {rate_in}/{rate_out} per Mtok"
        if rate_in is not None
        else "priced at the dearest known model; this one is not in the table"
    )
    print(f"model            {model}")
    print(f"prompt           {prompt}")
    print(f"rows to score    {rows}")
    print(f"estimated cost   ${_estimate_cost(rows, model):.4f} upper bound, {basis}")

    if not assume_yes and input("\nsend these requests? [y/N] ").strip().lower() != "y":
        print("nothing was sent")
        return False

    print()
    return True


def _resolve_provider(live: bool, prompt_version: str) -> tuple[LLMProvider, bool]:
    """Pick the provider, and say plainly which one came back.

    Returns the provider and whether it is live. Never prints, logs, or returns
    the key itself - only whether one was found.
    """
    if not live:
        return _stand_in(), False

    provider = build_provider(get_settings(), prompt_version=prompt_version)
    if provider is None:
        print("--live was requested but ANTHROPIC_API_KEY is not set.")
        print("Add it to backend/.env or the environment. Nothing was sent.")
        raise SystemExit(2)

    return provider, True


def _holdout_for_run(holdout: LabelledDataset, limit: int | None) -> LabelledDataset:
    """Optionally shrink the holdout for a cheap first live run.

    The rows are taken in order rather than sampled, so two runs at the same
    limit score the same transactions and can be compared.
    """
    if limit is None or limit >= len(holdout.records):
        return holdout
    return LabelledDataset(
        dataset_version=f"{holdout.dataset_version}[:{limit}]",
        records=holdout.records[:limit],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score rules, LLM, and hybrid on one holdout.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the real model (costs money); without this a stand-in is used",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="score only the first N holdout rows - use this for the first live run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the cost confirmation prompt on a live run",
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT_VERSION,
        choices=sorted(SYSTEM_PROMPTS),
        metavar="VERSION",
        help=(
            "which recorded prompt to send (default: the selected one). Naming "
            "an older version reproduces the score archived under it."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path: Path = args.path

    report = validate_labelled_csv(path)
    if not report.usable:
        print(report.render())
        print("\nrefusing to evaluate an unusable dataset")
        return 1
    if report.warnings:
        print(report.render())
        print()

    dataset = load_labelled_csv(path, dataset_version=path.stem)

    try:
        split = merchant_disjoint_split(dataset, holdout_fraction=HOLDOUT_FRACTION)
    except ValidationError as error:
        print(f"cannot split: {error}")
        return 1

    print(
        f"split by merchant: {split.development_merchants} development / "
        f"{split.holdout_merchants} holdout, "
        f"{len(split.development)} / {len(split.holdout)} rows"
    )
    overlap = split.development.merchants & split.holdout.merchants
    print(f"merchants on both sides: {len(overlap)}\n")

    holdout = _holdout_for_run(split.holdout, args.limit)
    if len(holdout) != len(split.holdout):
        print(f"limited to the first {len(holdout)} holdout rows of {len(split.holdout)}")
        print("a partial holdout is a smoke test, not a benchmark result\n")

    provider, is_live = _resolve_provider(args.live, args.prompt)

    if is_live and not _confirm_live_run(
        len(holdout), provider.model, args.prompt, assume_yes=args.yes
    ):
        return 0

    rules = fit_rules(split.development)
    llm = LLMCategoriser(provider)
    hybrid = HybridCategoriser(fit_rules(split.development), LLMCategoriser(provider))

    # Priced at the rate for the model that actually ran, not a constant:
    # the report has to be able to say what this run cost, not what some other
    # model would have cost.
    input_price, output_price = prices_for(provider.model)
    if input_price is None:
        print(
            f"note: no published prices recorded for {provider.model!r}; "
            f"tokens will be reported without a cost"
        )

    result = evaluate(
        holdout,
        [rules, llm, hybrid],
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )
    print(result.render())

    if is_live:
        usage = llm.usage()
        print()
        print(f"model calls      {usage.calls}")
        print(f"failures         {usage.failures}")
        print(f"rejected outputs {usage.rejected_outputs}  (outside the taxonomy, discarded)")
        if args.limit is not None:
            print("\npartial holdout: a smoke test, not a benchmark result")

    return 0


if __name__ == "__main__":
    sys.exit(main())
