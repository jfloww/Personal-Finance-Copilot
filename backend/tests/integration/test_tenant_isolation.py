"""What this whole phase is for.

Two users, each with a valid identity. Nothing either does may be visible to
the other, and the same account key and the same transaction fingerprint must
be able to exist on both sides at once.

The failure mode guarded against is the quiet one: not a crash, but one
person's coffee showing up in another person's statement because a query
forgot its `WHERE user_id`. Every assertion below is written from the second
tenant's point of view, because a repository that filters correctly and one
that does not both look identical from the first tenant's.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    TransactionRepository,
    UserRepository,
)
from offerdelta.records.transactions import TransactionRecord
from tests.integration.conftest import requires_database

pytestmark = requires_database

KEY = "chase-checking-5718"


def _scope(session: Session, email: str) -> TenantScope:
    """A real tenant, created the way the rest of the system creates one.

    Emails are made unique per call: this suite runs against a live shared
    database, and a fixed address would collide with a row a previous,
    committed run left behind.
    """
    address = f"{uuid.uuid4().hex[:12]}-{email}"
    stored = UserRepository(session).create(address, f"Owner of {address}")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _entry(amount: str = "-12.34") -> ManualEntry:
    return ManualEntry(
        account_key=KEY,
        posted_on=date(2026, 3, 1),
        description="BLUE BOTTLE COFFEE",
        amount=Money.parse(amount),
        repeat=False,
    )


def test_one_tenant_cannot_read_another_tenants_account(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")

    assert AccountRepository(a).by_key(KEY) is not None
    assert AccountRepository(b).by_key(KEY) is None


def test_listing_accounts_shows_only_your_own(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Amex Gold 1006")

    assert [x.key for x in AccountRepository(a).all()] == [KEY]
    assert [x.key for x in AccountRepository(b).all()] == ["amex-gold-1006"]


def test_both_tenants_may_register_the_same_account_key(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Chase Checking 5718")

    # Bound to locals rather than chained: `by_key` returns an Optional, and
    # mypy runs over tests too.
    held_by_a = AccountRepository(a).by_key(KEY)
    held_by_b = AccountRepository(b).by_key(KEY)
    assert held_by_a is not None
    assert held_by_b is not None
    assert held_by_a.id != held_by_b.id


def test_entering_a_transaction_into_another_tenants_account_is_refused(
    session: Session,
) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")

    # Matched, not bare: `enter_transaction` raises ValidationError for a
    # blank description too, and a bare `raises` would keep this green if the
    # tenancy check were removed and some unrelated refusal took its place.
    with pytest.raises(ValidationError, match="no account"):
        enter_transaction(b, _entry())


def test_the_refusal_does_not_name_another_tenants_accounts(session: Session) -> None:
    """The error lists the accounts you could have meant. Only yours.

    Naming every registered account is genuinely useful when the key is a
    typo, and it is an enumeration of other people's banks the moment that
    list is not scoped.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Amex Gold 1006")

    with pytest.raises(ValidationError) as raised:
        enter_transaction(b, _entry())

    assert KEY not in str(raised.value).removeprefix(f"no account {KEY!r}.")
    assert "amex-gold-1006" in str(raised.value)


def test_opening_a_batch_against_another_tenants_account_is_refused(
    session: Session,
) -> None:
    """The import path, guarded at the repository rather than by call site."""
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")

    with pytest.raises(ValidationError, match="no account"):
        ImportBatchRepository(b).open(
            account_of_a.id,
            source_file="statement.csv",
            source_sha256="0" * 64,
            mode="snapshot",
            window_start=date(2026, 3, 1),
            window_end=date(2026, 3, 31),
            row_count=1,
        )


def test_inserting_a_batch_row_directly_is_refused_too(session: Session) -> None:
    """The private insert checks for itself; it does not inherit `open`'s check.

    `_insert` is the one method here that writes a row against an account id
    it was handed, and the foreign key alone is satisfied by any account - so
    a call that skipped `open` would file a batch, and through `batch_id` the
    transactions hanging off it, inside somebody else's account. Reached
    directly here for the same reason `test_a_losing_racer_gets_the_existing_
    batch_not_a_crash` does: "open is its only caller" is a fact about today's
    call sites, and that is precisely the kind of fact this phase refuses to
    rest on.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")

    with pytest.raises(ValidationError, match="no account"):
        ImportBatchRepository(b)._insert(
            account_of_a.id,
            source_file="statement.csv",
            source_sha256="1" * 64,
            mode="snapshot",
            window_start=date(2026, 3, 1),
            window_end=date(2026, 3, 31),
            row_count=1,
            now=None,
        )


def test_writing_transactions_into_another_tenants_account_is_refused(
    session: Session,
) -> None:
    """`add_many` takes an account id from its caller, so it verifies it.

    Chain of custody - "the id came from a scoped AccountRepository" - is an
    argument about today's call sites, not a property of this method.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")

    record = TransactionRecord(
        account_id=account_of_a.id,
        posted_on=date(2026, 3, 1),
        description="BLUE BOTTLE COFFEE",
        normalised_merchant="BLUE BOTTLE COFFEE",
        amount=Money.parse("-12.34"),
        external_id=None,
        occurrence=1,
        provenance=None,
    )

    with pytest.raises(ValidationError, match="no account"):
        TransactionRepository(b).add_many([record])


def test_writing_transactions_with_another_tenants_batch_id_is_refused(
    session: Session,
) -> None:
    """`add_many` also takes a `batch_id` from its caller, and the foreign key
    alone accepts any batch - so a caller who owns the account but not the
    batch must still be refused. Owning the account is what makes this case
    different from the one above: without its own check, `batch_id` would
    let a caller tag rows it is otherwise allowed to write onto somebody
    else's import history.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")
    account_of_b = AccountRepository(b).register("Amex Gold 1006")

    batch_of_a, _created = ImportBatchRepository(a).open(
        account_of_a.id,
        source_file="statement.csv",
        source_sha256="2" * 64,
        mode="snapshot",
        window_start=date(2026, 3, 1),
        window_end=date(2026, 3, 31),
        row_count=1,
    )

    record = TransactionRecord(
        account_id=account_of_b.id,
        posted_on=date(2026, 3, 1),
        description="BLUE BOTTLE COFFEE",
        normalised_merchant="BLUE BOTTLE COFFEE",
        amount=Money.parse("-12.34"),
        external_id=None,
        occurrence=1,
        provenance=None,
    )

    with pytest.raises(ValidationError, match="no batch"):
        TransactionRepository(b).add_many([record], batch_id=batch_of_a.id)


def test_the_same_transaction_may_exist_in_both_tenants(session: Session) -> None:
    """Deduplication is per tenant, not global.

    The same account key, the same day, the same merchant, the same amount.
    Neither tenant's row may be reported to the other as already stored, and
    each must see exactly one.

    The two fingerprints differ, and deliberately so: `compute_fingerprint`
    takes `account_id`, and the two accounts are two rows. That is why this
    asserts on what a person can observe - both written, each tenant holding
    one - rather than on the hash.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")
    account_of_b = AccountRepository(b).register("Chase Checking 5718")

    first = enter_transaction(a, _entry())
    second = enter_transaction(b, _entry())

    assert first.stored is True
    assert second.stored is True, "deduplication must be per tenant, not global"
    assert second.occurrence == 1, "b's first charge is not b's second"
    assert first.transaction_id != second.transaction_id
    assert TransactionRepository(a).count(account_id=account_of_a.id) == 1
    assert TransactionRepository(b).count(account_id=account_of_b.id) == 1


def test_counting_transactions_never_reaches_across_tenants(session: Session) -> None:
    """`count()` with no account is "mine", not "everyone's".

    An unfiltered count is the easiest query to leave untenanted, and the
    least likely to be noticed: it returns a plausible number either way.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Chase Checking 5718")

    enter_transaction(a, _entry())

    assert TransactionRepository(a).count() == 1
    assert TransactionRepository(b).count() == 0


def test_a_transaction_cannot_be_fetched_by_id_from_another_tenant(
    session: Session,
) -> None:
    """A primary key is not an authorisation.

    `get` used to be `session.get(TransactionRow, id)`, which answers for any
    row in the table. An id that leaked into a URL would have been enough.
    """
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")

    stored = enter_transaction(a, _entry())
    assert stored.transaction_id is not None

    assert TransactionRepository(a).get(stored.transaction_id) is not None
    assert TransactionRepository(b).get(stored.transaction_id) is None
