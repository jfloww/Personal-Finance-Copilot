"""Storing what a categoriser said, and what a person said instead."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
    UserRepository,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database

KEY = "chase-checking-5718"


def _scope(session: Session, email: str) -> TenantScope:
    stored = UserRepository(session).create(email, f"Owner {email}")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _one_transaction(scope: TenantScope, description: str = "BLUE BOTTLE") -> uuid.UUID:
    AccountRepository(scope).register("Chase Checking 5718")
    outcome = enter_transaction(
        scope,
        ManualEntry(
            account_key=KEY,
            posted_on=date(2026, 3, 1),
            description=description,
            amount=Money.parse("-12.34"),
            repeat=False,
        ),
    )
    assert outcome.transaction_id is not None
    return outcome.transaction_id


def test_a_new_transaction_is_unclassified(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    _one_transaction(scope)

    pending = TransactionRepository(scope).unclassified()
    assert len(pending) == 1
    assert pending[0].suggested_label is None
    assert pending[0].effective_label is None


def test_a_suggestion_is_recorded_with_its_provenance(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id,
        label="LIVING_DINING",
        source="llm",
        confidence=Decimal("0.910"),
        by="claude-haiku-4-5:categorise/v3",
    )

    stored = repo.for_month(2026, 3)[0]
    assert stored.suggested_label == "LIVING_DINING"
    assert stored.suggested_source == "llm"
    assert stored.suggested_confidence == Decimal("0.910")
    assert stored.suggested_by == "claude-haiku-4-5:categorise/v3"
    assert stored.effective_label == "LIVING_DINING"


def test_a_confirmation_outranks_a_suggestion(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="LIVING_OTHER", source="llm", confidence=Decimal("0.400"), by="x"
    )
    repo.confirm_label(txn_id, "LIVING_DINING")

    stored = repo.for_month(2026, 3)[0]
    assert stored.confirmed_label == "LIVING_DINING"
    assert stored.suggested_label == "LIVING_OTHER"
    assert stored.effective_label == "LIVING_DINING"


def test_a_later_suggestion_does_not_erase_a_confirmation(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.confirm_label(txn_id, "LIVING_DINING")
    repo.record_suggestion(
        txn_id, label="TRANSFER", source="llm", confidence=Decimal("0.990"), by="x"
    )

    stored = repo.for_month(2026, 3)[0]
    assert stored.effective_label == "LIVING_DINING"


def test_a_label_outside_the_space_is_refused(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)

    with pytest.raises(ValidationError, match="not a label"):
        TransactionRepository(scope).confirm_label(txn_id, "COFFEE")


def test_unknown_is_a_valid_label_and_is_not_the_same_as_never_examined(
    session: Session,
) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="UNKNOWN", source="llm", confidence=Decimal("0.000"), by="x"
    )

    assert repo.unclassified() == []
    assert repo.for_month(2026, 3)[0].suggested_label == "UNKNOWN"


def test_awaiting_review_takes_low_confidence_and_unknown_and_never_examined(
    session: Session,
) -> None:
    scope = _scope(session, "a@example.test")
    repo = TransactionRepository(scope)
    AccountRepository(scope).register("Chase Checking 5718")

    ids = []
    for day, description in (
        (1, "CONFIDENT"),
        (2, "UNSURE"),
        (3, "ABSTAINED"),
        (4, "UNSEEN"),
        (4, "UNSEEN TOO"),
    ):
        outcome = enter_transaction(
            scope,
            ManualEntry(
                account_key=KEY,
                posted_on=date(2026, 3, day),
                description=description,
                amount=Money.parse("-1.00"),
                repeat=False,
            ),
        )
        assert outcome.transaction_id is not None
        ids.append(outcome.transaction_id)

    repo.record_suggestion(
        ids[0], label="LIVING_DINING", source="llm", confidence=Decimal("0.950"), by="x"
    )
    repo.record_suggestion(
        ids[1], label="LIVING_DINING", source="llm", confidence=Decimal("0.400"), by="x"
    )
    repo.record_suggestion(
        ids[2], label="UNKNOWN", source="llm", confidence=Decimal("0.000"), by="x"
    )
    # ids[3] and ids[4] (day 4) are left untouched: two rows sharing a date,
    # both never examined, to pin the tie-break below.

    queued = repo.awaiting_review(Decimal("0.600"))
    assert {t.description for t in queued} == {"UNSURE", "ABSTAINED", "UNSEEN", "UNSEEN TOO"}
    assert [t.posted_on.day for t in queued] == [4, 4, 3, 2], "most recent first"
    same_day_ids = [t.id for t in queued if t.posted_on.day == 4]
    assert same_day_ids == sorted(same_day_ids), "rows sharing a date break ties by id"


def test_confirming_takes_a_row_out_of_the_queue(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="LIVING_OTHER", source="llm", confidence=Decimal("0.100"), by="x"
    )
    assert len(repo.awaiting_review(Decimal("0.600"))) == 1

    repo.confirm_label(txn_id, "LIVING_DINING")
    assert repo.awaiting_review(Decimal("0.600")) == []


def test_one_tenant_cannot_classify_another_tenants_transaction(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    txn_id = _one_transaction(a)

    with pytest.raises(ValidationError, match="no transaction"):
        TransactionRepository(b).confirm_label(txn_id, "LIVING_DINING")


def test_one_tenant_never_sees_another_tenants_month(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    _one_transaction(a)

    assert TransactionRepository(a).for_month(2026, 3) != []
    assert TransactionRepository(b).for_month(2026, 3) == []
