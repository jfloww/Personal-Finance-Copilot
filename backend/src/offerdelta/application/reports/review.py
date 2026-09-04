"""The review queue and the label-confirmation write, thin over the repository.

`offerdelta.api` may not import `offerdelta.infrastructure.postgres.repositories`
directly - see the "API does not reach repositories directly" import-linter
contract. `queue` and `confirm` are the use case that stands in for it on the
review routes, the same role `offerdelta.application.reports.monthly` plays
for the report routes: a route that could construct a repository itself could
construct one without a scope, and the scope is the only thing keeping one
tenant's rows out of another's response.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Final

from offerdelta.application.scope import TenantScope
from offerdelta.infrastructure.postgres.repositories import (
    StoredTransaction,
    TransactionRepository,
)

#: The confidence bar below which a suggested label counts as "awaiting
#: review" rather than settled. Chosen on the *development* split of the
#: frozen benchmark by Youden's J statistic - the threshold that best
#: separates rows the categoriser got right from rows it got wrong there -
#: then measured exactly once, after that choice, on the held-out split. See
#: `docs/eval/review-threshold.json` for the swept curve this number was read
#: off and the coverage/accuracy the held-out split measured at it.
#:
#: That curve was swept over LLM predictions made with a real `account_type`
#: and no rule tier ahead of the model - `categorise.py`'s deployed pipeline
#: has neither: the rule tier runs unfitted and every row reaches the model
#: with `account_type="unknown"` (see that module's docstring). So this
#: number was not chosen against the input distribution the running queue
#: actually produces. It is kept at 0.80 anyway - re-picking it now, on a
#: distribution nobody has swept a curve over, would not be a measured choice
#: either.
#:
#: That file is not read here. It records an analysis, not configuration: a
#: deployed service reading a docs artifact at request time would acquire a
#: dependency on the repository's layout that has nothing to do with serving
#: a request, for a number that does not change between requests. The number
#: the analysis produced is configuration, so it is committed as one, here.
REVIEW_THRESHOLD: Final = Decimal("0.80")


def queue(
    scope: TenantScope, threshold: Decimal, *, month: tuple[int, int] | None = None
) -> list[StoredTransaction]:
    """Rows nobody has confirmed and no categoriser confidently resolved.

    A thin pass-through to `TransactionRepository.awaiting_review` - see that
    method's docstring for what "awaiting review" means and why the three
    underlying conditions stay an `OR`. Kept here rather than called directly
    from a route so the API layer never needs to construct a repository
    itself; see this module's docstring.
    """
    return TransactionRepository(scope).awaiting_review(threshold, month=month)


def confirm(scope: TenantScope, transaction_id: uuid.UUID, label: str) -> None:
    """Record a person's decision on one transaction.

    A thin pass-through to `TransactionRepository.confirm_label`, which
    raises `ValidationError` both for a label outside the taxonomy and for a
    transaction id naming another tenant's row (or no row at all) - with the
    same message for the latter two, so a caller cannot use it to tell
    "not yours" from "does not exist". The route calling this validates the
    label against the taxonomy at the wire boundary before it ever reaches
    here (see `LabelConfirmationSchema`), so by the time this can raise, only
    the tenancy failure is left for it to mean.
    """
    TransactionRepository(scope).confirm_label(transaction_id, label)
