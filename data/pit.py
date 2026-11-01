"""Strict clocks for point-in-time research; synthetic D#### clocks stay separate."""
from __future__ import annotations

from datetime import datetime, timezone
import re


def parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp with seconds and an explicit offset, returning UTC.

    Date-only and naive clocks are rejected, since assigning a timezone silently
    can let an event enter a decision before it was observable.
    """
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value
    ):
        raise ValueError(f"Explicit timezone and full ISO timestamp required: {value!r}")
    if value[-1] != "Z" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
        raise ValueError(f"Invalid ISO timezone offset: {value!r}")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {value!r}") from exc
    return timestamp.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Serialize an aware datetime to an unambiguous UTC Z timestamp."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An aware datetime is required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
