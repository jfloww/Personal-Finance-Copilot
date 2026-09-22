"""Load section-aware chunks from the bundled, explicitly synthetic corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib.resources import files
from typing import cast

from offerdelta.domain.common.errors import ValidationError

type JsonObject = dict[str, object]
MAX_THRESHOLD_DECIMAL_PLACES = 2


@dataclass(frozen=True)
class PolicyChunk:
    """One independently citable policy section."""

    chunk_id: str
    corpus_version: str
    policy_id: str
    version: str
    title: str
    effective_date: str
    section: str
    heading: str
    text: str
    tags: tuple[str, ...]
    threshold: str | None = None

    @property
    def source(self) -> str:
        return f"synthetic://policies/{self.policy_id}@{self.version}#{self.section}"

    @property
    def searchable_text(self) -> str:
        return " ".join((self.title, self.heading, self.text, *self.tags))

    def citation(self) -> dict[str, str | list[str] | None]:
        return {
            "chunk_id": self.chunk_id,
            "policy_id": self.policy_id,
            "id": self.policy_id,
            "version": self.version,
            "title": self.title,
            "effective_date": self.effective_date,
            "section": self.section,
            "heading": self.heading,
            "text": self.text,
            "tags": list(self.tags),
            "threshold": self.threshold,
            "source": self.source,
        }


def _string(item: JsonObject, key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"policy corpus field {key!r} must be a non-empty string")
    return value


def _objects(item: JsonObject, key: str) -> list[JsonObject]:
    value = item.get(key)
    if not isinstance(value, list) or not value:
        raise ValidationError(f"policy corpus field {key!r} must be a non-empty list")
    if not all(isinstance(entry, dict) for entry in value):
        raise ValidationError(f"policy corpus field {key!r} must contain objects")
    return [cast(JsonObject, entry) for entry in value]


def _tags(item: JsonObject) -> tuple[str, ...]:
    value = item.get("tags")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(tag, str) and tag.strip() for tag in value)
    ):
        raise ValidationError("policy section tags must be a non-empty string list")
    tags = tuple(cast(list[str], value))
    if len(set(tags)) != len(tags):
        raise ValidationError("policy section tags must be unique")
    return tags


def _iso_date(item: JsonObject, key: str) -> str:
    value = _string(item, key)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValidationError(f"policy corpus field {key!r} must be an ISO date") from error
    return value


def _threshold(item: JsonObject) -> str | None:
    value = item.get("threshold")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("policy threshold must be a decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValidationError("policy threshold must be a decimal string") from error
    exponent = amount.as_tuple().exponent
    if (
        not amount.is_finite()
        or amount <= 0
        or not isinstance(exponent, int)
        or exponent < -MAX_THRESHOLD_DECIMAL_PLACES
    ):
        raise ValidationError("policy threshold must be positive with at most two decimal places")
    return value


@lru_cache(maxsize=1)
def load_policy_corpus() -> tuple[PolicyChunk, ...]:
    """Parse and validate package data once; malformed content fails closed."""
    resource = files("offerdelta.policy.data").joinpath("synthetic_policies.json")
    raw = cast(object, json.loads(resource.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValidationError("policy corpus root must be an object")
    root = cast(JsonObject, raw)
    corpus_version = _string(root, "corpus_version")
    chunks: list[PolicyChunk] = []
    seen: set[str] = set()
    for document in _objects(root, "documents"):
        policy_id = _string(document, "policy_id")
        version = _string(document, "version")
        title = _string(document, "title")
        effective_date = _iso_date(document, "effective_date")
        for section_data in _objects(document, "sections"):
            section = _string(section_data, "section")
            chunk_id = f"{policy_id}@{version}#{section}"
            if chunk_id in seen:
                raise ValidationError(f"duplicate policy chunk {chunk_id!r}")
            seen.add(chunk_id)
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    corpus_version=corpus_version,
                    policy_id=policy_id,
                    version=version,
                    title=title,
                    effective_date=effective_date,
                    section=section,
                    heading=_string(section_data, "heading"),
                    text=_string(section_data, "text"),
                    tags=_tags(section_data),
                    threshold=_threshold(section_data),
                )
            )
    if not chunks:
        raise ValidationError("policy corpus cannot be empty")
    return tuple(chunks)
