"""Imported transactions against real PostgreSQL.

The unit suite exercises the repository quickly against a fake; these tests
verify the claims that belong to PostgreSQL itself: exact NUMERIC round trips,
JSON lineage, and the multiplicity-aware unique constraint created by Alembic.

The shared integration fixture wraps every test in an outer transaction and
rolls it back, so verification leaves no bank rows, and no accounts, behind.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy.orm import Session

from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.records import Provenance, TransactionRecord
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database


def _account_name(prefix: str) -> str:
    # Unique per test run so a live shared database can never collide with a
    # name a previous run happened to register.
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _account(session: Session, prefix: str = "integration-checking") -> uuid.UUID:
    return AccountRepository(session).register(_account_name(prefix)).id


def _record(
    account_id: uuid.UUID,
    *,
    posted_on: date = date(2026, 8, 17),
    merchant: str = "BLUE BOTTLE",
    amount: str = "-4.50",
    occurrence: int = 1,
    provenance: Provenance | None = None,
) -> TransactionRecord:
    return TransactionRecord(
        account_id=account_id,
        posted_on=posted_on,
        description=merchant,
        normalised_merchant=merchant,
        amount=Money.parse(amount),
        external_id=None,
        occurrence=occurrence,
        provenance=provenance,
    )


def test_a_record_round_trips_exactly_with_its_provenance(session: Session) -> None:
    account_id = _account(session)
    provenance = Provenance(
        source_file="statement.csv", source_line=2, raw_cells={"Amount": "-4.50"}
    )
    record = _record(account_id, provenance=provenance)

    result = TransactionRepository(session).add_many([record])
    stored = TransactionRepository(session).get(result.imported_ids[0])

    assert stored is not None
    assert stored.amount == Money.parse("-4.50")
    assert stored.account_id == account_id
    assert stored.source_file == "statement.csv"
    assert stored.source_line == 2
    assert stored.raw_cells == {"Amount": "-4.50"}


def test_a_second_identical_record_is_reported_already_stored_not_written(
    session: Session,
) -> None:
    account_id = _account(session)
    record = _record(account_id)
    repo = TransactionRepository(session)

    first = repo.add_many([record])
    second = repo.add_many([record])

    assert first.imported_count == 1
    assert second.imported_count == 0
    assert second.already_stored_count == 1
    assert repo.count(account_id=account_id) == 1


def test_two_accounts_do_not_share_identities(session: Session) -> None:
    repo = TransactionRepository(session)
    first_account = _account(session, "account-one")
    second_account = _account(session, "account-two")

    # Same date, merchant, and amount in both accounts: without account_id in
    # the fingerprint this would collide and the second import would be
    # reported as already stored.
    first = repo.add_many([_record(first_account)])
    second = repo.add_many([_record(second_account)])

    assert first.imported_count == 1
    assert second.imported_count == 1
    assert second.already_stored_count == 0
    assert repo.count(account_id=first_account) == 1
    assert repo.count(account_id=second_account) == 1


def test_count_is_correct(session: Session) -> None:
    repo = TransactionRepository(session)
    account_id = _account(session)
    other_account = _account(session, "other-account")
    before_total = repo.count()

    repo.add_many(
        [
            _record(account_id, occurrence=1),
            _record(account_id, merchant="NETFLIX", occurrence=1),
        ]
    )
    repo.add_many([_record(other_account)])

    assert repo.count(account_id=account_id) == 2
    assert repo.count(account_id=other_account) == 1
    assert repo.count() == before_total + 3
