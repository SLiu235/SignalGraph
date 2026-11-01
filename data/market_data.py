"""Validated historical market inputs for event research.

Prices must explicitly be split-and-dividend adjusted. Only ratios of adjusted
closes are suitable for the current return calculations: present-day restated
price levels must not become historical features. The schema cannot certify a
vendor's adjustment methodology or survivorship coverage; record both in source
metadata. Exchange open/close timestamps are supplied by a provider or calendar,
never inferred from weekdays.

Fama-French factors are decimal daily returns. Fit stock excess returns (stock
return minus rf) on mkt_rf, smb and hml with an intercept. Add rf back to obtain
the expected total stock return. Factors downloaded today may be revised and
are for ex-post risk adjustment, not point-in-time predictive features.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import zipfile

from data.pit import iso_utc, parse_utc


FAMA_FRENCH_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FACTOR_DOCUMENTATION_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html"
)
ALPHA_VANTAGE_DOCUMENTATION_URL = "https://www.alphavantage.co/documentation/#dailyadj"
RAW_RECEIPT_VERSION = 1
MAX_RAW_OBJECT_BYTES = 20_000_000
FAMA_FRENCH_DAILY_ADAPTER = "fama_french_daily"
FAMA_FRENCH_DAILY_PARSER_VERSION = "ff-daily-4-v1"
FAMA_FRENCH_DAILY_PROVIDER = "Kenneth R. French Data Library"
FAMA_FRENCH_DAILY_IDENTITY = {
    "columns": "Mkt-RF,SMB,HML,RF",
    "dataset": "F-F_Research_Data_Factors_daily",
    "frequency": "daily",
}
FAMA_FRENCH_FACTOR_LINEAGE_ID = "ken_french:F-F_Research_Data_Factors_daily"
FAMA_FRENCH_BENCHMARK_DEFINITION = "Fama-French market total return = mkt_rf + rf"
FAMA_FRENCH_USAGE = "Ex-post risk adjustment; not historical predictive availability"
ALPHA_VANTAGE_ADJUSTED_ADAPTER = "alpha_vantage_daily_adjusted"
ALPHA_VANTAGE_PARSER_VERSION = "av-adjusted-v1"
ALPHA_VANTAGE_ADJUSTED_SOURCE = "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED"
ALPHA_VANTAGE_ADJUSTED_USAGE = "Adjusted-price ratios only; levels are restated as of retrieval"


def _utc(value, field: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field} must be an ISO UTC timestamp with an explicit offset")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO UTC timestamp") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must explicitly use UTC (Z or +00:00)")
    return result


def _session(value, field="session") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid calendar date") from exc
    return value


def _text(value, field) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _number(value, field, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return float(value)


def _rows(bundle, name, *, optional=False):
    value = bundle.get(name, [] if optional else None)
    if not isinstance(value, list) or (not optional and not value):
        raise ValueError(f"{name} must be a {'possibly empty ' if optional else 'nonempty '}list")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} entries must be objects")
    return value


def _validate_bars(bars):
    seen, previous = set(), {}
    for row in bars:
        ticker = _text(row.get("ticker"), "bars.ticker")
        session = _session(row.get("session"), "bars.session")
        if (ticker, session) in seen:
            raise ValueError(f"duplicate bar: {ticker} {session}")
        seen.add((ticker, session))
        opened = _utc(row.get("open_at"), "bars.open_at")
        closed = _utc(row.get("close_at"), "bars.close_at")
        if opened >= closed:
            raise ValueError("bar open_at must precede close_at")
        if row.get("available_at") is not None:
            available = _utc(row["available_at"], "bars.available_at")
            if available < closed:
                raise ValueError("bars.available_at cannot precede bars.close_at")
        if ticker in previous:
            last_session, last_close = previous[ticker]
            if session <= last_session or opened <= last_close:
                raise ValueError(f"bars must be chronological and nonoverlapping for {ticker}")
        previous[ticker] = (session, closed)
        _number(row.get("adjusted_close"), "bars.adjusted_close", positive=True)
        if "volume" in row:
            _number(row["volume"], "bars.volume", nonnegative=True)


def _validate_session_schedule(value, *, required=False) -> dict | None:
    """Validate a versioned, source-identified timestamped exchange-session schedule.

    Daily benchmark/factor rows carry dates but not a market close instant.  A
    separate schedule is therefore the authority for execution timing and can
    include future announced sessions even when return observations do not yet
    exist.  It is intentionally not inferred from the price bars.
    """
    if value is None:
        if required:
            raise ValueError("market bundle requires a timestamped session_schedule for execution research")
        return None
    if not isinstance(value, dict):
        raise ValueError("session_schedule must be an object")
    if value.get("version") != 1:
        raise ValueError("session_schedule.version must be 1")
    _text(value.get("calendar_id"), "session_schedule.calendar_id")
    _text(value.get("source"), "session_schedule.source")
    retrieved = _utc(value.get("retrieved_at"), "session_schedule.retrieved_at")
    rows = _rows(value, "sessions")
    seen, previous_session, previous_close, normalized = set(), "", None, []
    for row in rows:
        session = _session(row.get("session"), "session_schedule.sessions.session")
        if session in seen or (previous_session and session <= previous_session):
            raise ValueError("session_schedule sessions must be unique and chronological")
        opened = _utc(row.get("open_at"), "session_schedule.sessions.open_at")
        closed = _utc(row.get("close_at"), "session_schedule.sessions.close_at")
        if opened >= closed:
            raise ValueError("session_schedule session open_at must precede close_at")
        if previous_close is not None and opened <= previous_close:
            raise ValueError("session_schedule sessions must be nonoverlapping")
        seen.add(session)
        previous_session, previous_close = session, closed
        normalized.append({"session": session, "open_at": iso_utc(opened), "close_at": iso_utc(closed)})
    return {"version": 1, "calendar_id": value["calendar_id"], "source": value["source"],
            "retrieved_at": iso_utc(retrieved), "sessions": normalized}


def parse_session_schedule(bundle: dict) -> dict:
    """Return one normalized execution schedule; reject date-only bundles.

    This helper is shared by walk-forward evaluation and prospective paper
    marking so the same availability-to-next-close rule cannot drift between
    those two research paths.
    """
    schedule = _validate_session_schedule(bundle.get("session_schedule"), required=True)
    rows = schedule["sessions"]
    return {**schedule, "rows": rows, "by_session": {row["session"]: row for row in rows},
            "index": {row["session"]: index for index, row in enumerate(rows)}}


def plan_next_close(schedule: dict, decision_at: str | datetime, latency_seconds: float,
                    horizons: list[int]) -> dict:
    """Freeze the first scheduled close after a decision and all planned exits.

    ``horizon=1`` means the first *session after* the entry session.  The
    result contains only schedule identities/timestamps, never a price.  This
    lets a prospective ledger record a plan before future bars and factor rows
    exist, while later marking can fail closed against that exact plan.
    """
    if isinstance(decision_at, datetime):
        if decision_at.tzinfo is None:
            raise ValueError("decision_at must include a timezone")
        decision = decision_at.astimezone(timezone.utc)
    else:
        decision = parse_utc(decision_at)
    if isinstance(latency_seconds, bool) or not isinstance(latency_seconds, (int, float)) or not math.isfinite(latency_seconds) or latency_seconds < 0:
        raise ValueError("latency_seconds must be a finite nonnegative number")
    if not isinstance(horizons, list) or not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("horizons must be a nonempty list of positive session counts")
    execution_after = decision + timedelta(seconds=float(latency_seconds))
    rows = schedule.get("rows") or schedule.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("session schedule has no sessions")
    normalized = []
    for row in rows:
        session = _session(row.get("session"), "session schedule session")
        opened = _utc(row.get("open_at"), "session schedule open_at")
        closed = _utc(row.get("close_at"), "session schedule close_at")
        normalized.append({"session": session, "open_at": iso_utc(opened), "close_at": iso_utc(closed)})
    entry_index = next((index for index, row in enumerate(normalized)
                        if parse_utc(row["close_at"]) > execution_after), None)
    if entry_index is None:
        raise ValueError("session schedule has no close strictly after the decision boundary")
    unique_horizons = sorted(set(horizons))
    max_exit = entry_index + unique_horizons[-1]
    if max_exit >= len(normalized):
        raise ValueError("session schedule does not cover the planned paper horizon")
    exits = {str(horizon): normalized[entry_index + horizon] for horizon in unique_horizons}
    window = normalized[entry_index:max_exit + 1]
    frozen = {"calendar_id": schedule.get("calendar_id"), "source": schedule.get("source"),
              "retrieved_at": schedule.get("retrieved_at"), "decision_at": iso_utc(decision),
              "executable_after": iso_utc(execution_after), "entry": normalized[entry_index],
              "window": window, "exits": exits}
    frozen["window_sha256"] = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return frozen


def validate_market_bundle(bundle: dict) -> dict:
    """Validate a complete bundle; return a deep copy without mutating its input.

    Required keys: metadata, bars, benchmark, factors. Optional sector_returns
    and memberships contain historical sector controls and universe membership.
    Coverage across these series is assessed by the research evaluator: missing
    return/factor sessions are not silently forward-filled by this importer.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get("metadata"), dict):
        raise ValueError("market bundle requires metadata object")
    meta = bundle["metadata"]
    if meta.get("data_kind") not in ("historical", "fixture"):
        raise ValueError("metadata.data_kind must explicitly be historical or fixture")
    if meta.get("adjustment") != "split_and_dividend":
        raise ValueError("metadata.adjustment must explicitly be split_and_dividend")
    _text(meta.get("source"), "metadata.source")
    _text(meta.get("universe_policy"), "metadata.universe_policy")
    if "price_lineage" in meta:
        lineages = meta["price_lineage"]
        if not isinstance(lineages, list) or not lineages:
            raise ValueError("metadata.price_lineage must be a nonempty list when supplied")
        seen_lineages = set()
        for lineage in lineages:
            if not isinstance(lineage, dict):
                raise ValueError("metadata.price_lineage entries must be objects")
            identifier = _text(lineage.get("id"), "metadata.price_lineage.id")
            source = _text(lineage.get("source"), "metadata.price_lineage.source")
            if identifier in seen_lineages:
                raise ValueError("metadata.price_lineage IDs must be unique")
            seen_lineages.add(identifier)
            if not isinstance(lineage.get("identity"), dict) or not lineage["identity"]:
                raise ValueError("metadata.price_lineage.identity must be a nonempty object")
            if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                   for key, value in lineage["identity"].items()):
                raise ValueError("metadata.price_lineage.identity values must be nonempty strings")
    if "factor_lineage" in meta:
        lineage = meta["factor_lineage"]
        if not isinstance(lineage, dict):
            raise ValueError("metadata.factor_lineage must be an object when supplied")
        _text(lineage.get("id"), "metadata.factor_lineage.id")
        _text(lineage.get("source"), "metadata.factor_lineage.source")
        if not isinstance(lineage.get("identity"), dict) or not lineage["identity"]:
            raise ValueError("metadata.factor_lineage.identity must be a nonempty object")
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
               for key, value in lineage["identity"].items()):
            raise ValueError("metadata.factor_lineage.identity values must be nonempty strings")
    retrieved = _utc(meta.get("retrieved_at"), "metadata.retrieved_at")
    schedule = _validate_session_schedule(bundle.get("session_schedule"))
    if schedule is not None and _utc(schedule["retrieved_at"], "session_schedule.retrieved_at") > retrieved:
        raise ValueError("session_schedule.retrieved_at cannot follow metadata.retrieved_at")
    schedule_by_session = {row["session"]: row for row in schedule["sessions"]} if schedule else {}
    bars = _rows(bundle, "bars")
    _validate_bars(bars)
    if any(_utc(row["close_at"], "bars.close_at") > retrieved for row in bars):
        raise ValueError("bars cannot close after metadata.retrieved_at")
    if schedule is not None:
        for row in bars:
            clock = schedule_by_session.get(row["session"])
            if clock is None:
                raise ValueError("bar session is absent from the timestamped session_schedule")
            if (_utc(row["open_at"], "bars.open_at") != _utc(clock["open_at"], "session_schedule.sessions.open_at")
                    or _utc(row["close_at"], "bars.close_at") != _utc(clock["close_at"], "session_schedule.sessions.close_at")):
                raise ValueError("bar clocks must exactly match the timestamped session_schedule")
    for name, fields in (("benchmark", ("return",)),
                         ("factors", ("mkt_rf", "smb", "hml", "rf")),
                         ("sector_returns", ("return",))):
        seen, previous = set(), {}
        for row in _rows(bundle, name, optional=name == "sector_returns"):
            session = _session(row.get("session"), f"{name}.session")
            if schedule is not None and session not in schedule_by_session:
                raise ValueError(f"{name} session is absent from the timestamped session_schedule")
            group = _text(row.get("sector"), "sector_returns.sector") if name == "sector_returns" else "all"
            key = (group, session)
            if key in seen:
                raise ValueError(f"duplicate {name} session: {key}")
            seen.add(key)
            if group in previous and session <= previous[group]:
                raise ValueError(f"{name} must be chronological")
            previous[group] = session
            for field in fields:
                value = _number(row.get(field), f"{name}.{field}")
                if name != "factors" and value < -1:
                    raise ValueError(f"{name}.return cannot be below -1")
            if row.get("available_at") is not None:
                available = _utc(row["available_at"], f"{name}.available_at")
                if schedule is not None and available < _utc(schedule_by_session[session]["close_at"],
                                                            "session_schedule.sessions.close_at"):
                    raise ValueError(f"{name}.available_at cannot precede the scheduled close")
    effective = {}
    for row in _rows(bundle, "memberships", optional=True):
        ticker = _text(row.get("ticker"), "memberships.ticker")
        _text(row.get("sector"), "memberships.sector")
        start = _utc(row.get("valid_from"), "memberships.valid_from")
        end = _utc(row["valid_to"], "memberships.valid_to") if row.get("valid_to") is not None else None
        _utc(row.get("available_at"), "memberships.available_at")
        if end is not None and end <= start:
            raise ValueError("memberships.valid_to must follow valid_from")
        for old_start, old_end in effective.setdefault(ticker, []):
            if (end is None or old_start < end) and (old_end is None or start < old_end):
                raise ValueError(f"overlapping membership intervals for {ticker}")
        effective[ticker].append((start, end))
    return copy.deepcopy(bundle)


def _json_bytes(raw: bytes):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _cache_object(cache: Path, digest: str) -> Path:
    return cache / "objects" / digest[:2] / digest


def _read_cache_object(path: str | Path, label: str) -> bytes:
    """Read one bounded, non-symlink cache object exactly once for replay."""
    object_path = Path(path)
    if not object_path.is_file() or object_path.is_symlink():
        raise ValueError(f"{label} is unavailable")
    with object_path.open("rb") as handle:
        raw = handle.read(MAX_RAW_OBJECT_BYTES + 1)
    if len(raw) > MAX_RAW_OBJECT_BYTES:
        raise ValueError(f"{label} exceeds {MAX_RAW_OBJECT_BYTES} bytes")
    return raw


def _cache_root_from_object(path: Path, label: str) -> Path:
    """Return the cache root only for the canonical objects/<prefix>/<hash> layout."""
    if path.name == "" or path.parent.name != path.name[:2] or path.parent.parent.name != "objects":
        raise ValueError(f"{label} is not content addressed")
    return path.parent.parent.parent


def _response_stays_at_approved_endpoint(response, endpoint: str, label: str) -> None:
    """Reject cross-origin, downgrade, or path-changing redirects before caching bytes.

    The local receipt records a stable credential-free endpoint.  A redirect to
    another origin would make that record misleading, so it is rejected rather
    than silently attributing bytes to the requested endpoint.  Some small test
    doubles do not implement ``geturl``; they are treated as the requested URL.
    """
    geturl = getattr(response, "geturl", None)
    actual = geturl() if callable(geturl) else endpoint
    if not isinstance(actual, str):
        actual = endpoint
    expected_parts, actual_parts = urlsplit(endpoint), urlsplit(actual)
    if (actual_parts.scheme != "https" or actual_parts.scheme != expected_parts.scheme
            or actual_parts.netloc != expected_parts.netloc or actual_parts.path != expected_parts.path):
        raise ValueError(f"{label} response redirected away from the approved HTTPS endpoint")


def _write_content_addressed(cache: Path, raw: bytes) -> tuple[str, Path]:
    if len(raw) > MAX_RAW_OBJECT_BYTES:
        raise ValueError(f"raw market object exceeds {MAX_RAW_OBJECT_BYTES} bytes")
    digest = hashlib.sha256(raw).hexdigest()
    object_path = _cache_object(cache, digest)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with object_path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if hashlib.sha256(_read_cache_object(object_path, "raw market cache object")).hexdigest() != digest:
            raise ValueError(f"raw market cache integrity failure: {object_path.name}") from None
    return digest, object_path


def _write_receipt_sidecar(cache: Path, endpoint: str, receipt: dict) -> str:
    """Persist one canonical receipt beside the content-addressed raw object."""
    encoded = _canonical_json_bytes(receipt)
    endpoint_key = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
    receipt_path = cache / "receipts" / endpoint_key / f"{hashlib.sha256(encoded).hexdigest()}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if receipt_path.read_bytes() != encoded:
            raise ValueError(f"raw market receipt integrity failure: {receipt_path.name}") from None
    return str(receipt_path.resolve())


def _normalize_price_sessions(sessions: list[dict] | dict) -> list[dict]:
    if isinstance(sessions, dict):
        sessions = _validate_session_schedule(sessions, required=True)["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("an explicit exchange session table is required")
    result, seen, previous = [], set(), ""
    for clock in sessions:
        if not isinstance(clock, dict):
            raise ValueError("price session table entries must be objects")
        day = _session(clock.get("session"), "price session")
        opened = _utc(clock.get("open_at"), "price session open_at")
        closed = _utc(clock.get("close_at"), "price session close_at")
        if opened >= closed:
            raise ValueError("price session open_at must precede close_at")
        if day in seen or (previous and day <= previous):
            raise ValueError("price session table must be unique and chronological")
        seen.add(day)
        previous = day
        result.append({"session": day, "open_at": iso_utc(opened), "close_at": iso_utc(closed)})
    return result


def _persist_raw_snapshot(cache_dir, raw: bytes, *, provider: str, endpoint: str,
                          identity: dict, sessions: list[dict], adapter: str,
                          parser_version: str, retrieved_at: datetime,
                          content_type: str = "application/json") -> tuple[dict, str]:
    """Write an immutable-content-addressed provider response and receipt.

    The receipt deliberately contains only public request identity.  In
    particular, an API key must never appear in a response path, receipt, or
    derived market fragment.  Local files are still a local trust boundary; the
    exact sidecar/object relationship makes later accidental alteration visible.
    """
    if not isinstance(cache_dir, (str, Path)) or not str(cache_dir):
        raise ValueError("a raw market cache directory is required for provider snapshots")
    _text(provider, "raw receipt provider")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        raise ValueError("raw receipt endpoint must be an HTTPS URL")
    if not isinstance(identity, dict) or any(not isinstance(key, str) or not key
                                             or not isinstance(value, str) or not value
                                             for key, value in identity.items()):
        raise ValueError("raw receipt identity must contain nonempty public strings")
    _text(content_type, "raw receipt content_type")
    stamp = iso_utc(retrieved_at)
    cache = Path(cache_dir)
    raw_digest, object_path = _write_content_addressed(cache, raw)
    normalized_sessions = _normalize_price_sessions(sessions)
    session_bytes = _canonical_json_bytes(normalized_sessions)
    sessions_digest, sessions_path = _write_content_addressed(cache, session_bytes)
    _text(adapter, "raw receipt adapter")
    _text(parser_version, "raw receipt parser_version")
    receipt = {"version": RAW_RECEIPT_VERSION, "adapter": adapter,
               "parser_version": parser_version, "provider": provider, "endpoint": endpoint,
               "identity": dict(sorted(identity.items())), "sha256": raw_digest,
               "sessions_sha256": sessions_digest, "sessions_path": str(sessions_path.resolve()),
               "retrieved_at": stamp, "content_type": content_type,
               "raw_path": str(object_path.resolve())}
    return receipt, _write_receipt_sidecar(cache, endpoint, receipt)


def _persist_factor_snapshot(cache_dir, raw: bytes, *, retrieved_at: datetime) -> tuple[dict, str]:
    """Store one raw Fama--French response with a parser-versioned receipt.

    Factor data has no selected exchange-session payload, so it deliberately uses
    a separate receipt schema from adjusted prices.  It is still content
    addressed and replayable.  The receipt time is a local vintage clock only;
    it must not be used as a historical publication-time claim.
    """
    if not isinstance(cache_dir, (str, Path)) or not str(cache_dir):
        raise ValueError("a raw factor cache directory is required for provider snapshots")
    cache = Path(cache_dir)
    digest, object_path = _write_content_addressed(cache, raw)
    content_type = "application/zip" if raw.startswith(b"PK") else "text/csv"
    receipt = {
        "version": RAW_RECEIPT_VERSION,
        "adapter": FAMA_FRENCH_DAILY_ADAPTER,
        "parser_version": FAMA_FRENCH_DAILY_PARSER_VERSION,
        "provider": FAMA_FRENCH_DAILY_PROVIDER,
        "endpoint": FAMA_FRENCH_DAILY_URL,
        "identity": dict(FAMA_FRENCH_DAILY_IDENTITY),
        "sha256": digest,
        "raw_path": str(object_path.resolve()),
        "retrieved_at": iso_utc(retrieved_at),
        "content_type": content_type,
    }
    return receipt, _write_receipt_sidecar(cache, FAMA_FRENCH_DAILY_URL, receipt)


def verify_raw_receipt(metadata: dict, role: str = "prices") -> tuple[dict, bytes, list[dict]]:
    """Verify a historical provider fragment's raw bytes, sessions, and receipt."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{role} metadata must be an object")
    receipt = metadata.get("raw_receipt")
    expected = {"version", "adapter", "parser_version", "provider", "endpoint", "identity",
                "sha256", "sessions_sha256", "sessions_path", "retrieved_at", "content_type", "raw_path"}
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise ValueError(f"{role} historical fragment requires a complete raw_receipt")
    if receipt.get("version") != RAW_RECEIPT_VERSION:
        raise ValueError(f"{role} raw receipt has an unsupported version")
    _text(receipt.get("adapter"), f"{role} raw receipt adapter")
    _text(receipt.get("parser_version"), f"{role} raw receipt parser_version")
    _text(receipt.get("provider"), f"{role} raw receipt provider")
    if not isinstance(receipt.get("endpoint"), str) or not receipt["endpoint"].startswith("https://"):
        raise ValueError(f"{role} raw receipt requires an HTTPS endpoint")
    if (not isinstance(receipt.get("identity"), dict) or
            any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in receipt["identity"].items())):
        raise ValueError(f"{role} raw receipt identity is invalid")
    digest = receipt.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{role} raw receipt requires a SHA256")
    _utc(receipt.get("retrieved_at"), f"{role} raw receipt retrieved_at")
    _text(receipt.get("content_type"), f"{role} raw receipt content_type")
    raw_path = receipt.get("raw_path")
    if not isinstance(raw_path, str):
        raise ValueError(f"{role} raw receipt object is unavailable")
    raw_object_path = Path(raw_path)
    raw = _read_cache_object(raw_object_path, f"{role} raw receipt object")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{role} raw receipt object hash mismatch")
    if raw_object_path.name != digest or raw_object_path.parent.name != digest[:2]:
        raise ValueError(f"{role} raw receipt object is not content addressed")
    cache_root = _cache_root_from_object(raw_object_path, f"{role} raw receipt object")
    sessions_digest = receipt.get("sessions_sha256")
    sessions_path = receipt.get("sessions_path")
    if (not isinstance(sessions_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", sessions_digest)
            or not isinstance(sessions_path, str)):
        raise ValueError(f"{role} raw receipt session selection is unavailable")
    sessions_object_path = Path(sessions_path)
    expected_sessions_path = _cache_object(cache_root, sessions_digest)
    if sessions_object_path.resolve() != expected_sessions_path.resolve():
        raise ValueError(f"{role} raw receipt sessions are not in the same content-addressed cache")
    session_bytes = _read_cache_object(sessions_object_path, f"{role} raw receipt session selection")
    if hashlib.sha256(session_bytes).hexdigest() != sessions_digest:
        raise ValueError(f"{role} raw receipt session selection hash mismatch")
    if sessions_object_path.name != sessions_digest or sessions_object_path.parent.name != sessions_digest[:2]:
        raise ValueError(f"{role} raw receipt sessions are not content addressed")
    try:
        sessions = _normalize_price_sessions(_json_bytes(session_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{role} raw receipt session selection is invalid") from exc
    if _canonical_json_bytes(sessions) != session_bytes:
        raise ValueError(f"{role} raw receipt sessions are not canonical")
    sidecar_path = metadata.get("raw_receipt_path")
    if not isinstance(sidecar_path, str):
        raise ValueError(f"{role} raw receipt sidecar is unavailable")
    canonical = _canonical_json_bytes(receipt)
    receipt_hash = hashlib.sha256(canonical).hexdigest()
    expected_sidecar_path = (cache_root / "receipts" /
                             hashlib.sha256(receipt["endpoint"].encode("utf-8")).hexdigest() /
                             f"{receipt_hash}.json")
    sidecar = Path(sidecar_path)
    if sidecar.resolve() != expected_sidecar_path.resolve():
        raise ValueError(f"{role} raw receipt sidecar is not content addressed")
    if _read_cache_object(sidecar, f"{role} raw receipt sidecar") != canonical:
        raise ValueError(f"{role} raw receipt sidecar does not match its declared receipt")
    return receipt, raw, sessions


def load_market_bundle(path) -> dict:
    """Load validated JSON, recording a hash of the exact input bytes."""
    path = Path(path)
    raw = path.read_bytes()
    result = validate_market_bundle(_json_bytes(raw))
    inherited = result.get("provenance", {})
    if not isinstance(inherited, dict):
        raise ValueError("provenance must be an object")
    result["provenance"] = {**inherited,
        "input_path": str(path.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def parse_fama_french_daily(raw: bytes) -> list[dict]:
    """Read official FF daily CSV or ZIP bytes; convert percentages to decimals.

    Reject missing sentinels, duplicate/out-of-order dates and malformed daily
    rows. Text headers/footers are ignored. ZIP members are read in memory.
    """
    if raw.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
            if len(members) != 1 or members[0].file_size > 20_000_000:
                raise ValueError("factor ZIP must contain one CSV of at most 20 MB")
            raw = archive.read(members[0])
    rows, previous, header_found = [], "", False
    for fields in csv.reader(io.StringIO(raw.decode("utf-8-sig"))):
        fields = [field.strip() for field in fields]
        if len(fields) >= 5 and fields[1:5] == ["Mkt-RF", "SMB", "HML", "RF"]:
            header_found = True
            continue
        if not fields or not re.fullmatch(r"\d{8}", fields[0]):
            continue
        if not header_found or len(fields) != 5:
            raise ValueError("factor daily row does not match expected Mkt-RF,SMB,HML,RF columns")
        session = _session(f"{fields[0][:4]}-{fields[0][4:6]}-{fields[0][6:]}")
        if session <= previous:
            raise ValueError("factor dates must be unique and chronological")
        previous = session
        values = [float(field) for field in fields[1:]]
        if any(not math.isfinite(value) or value in (-99.99, -999.0) for value in values):
            raise ValueError("factor CSV contains missing or non-finite values")
        rows.append({"session": session, **dict(zip(("mkt_rf", "smb", "hml", "rf"),
                                                   (value / 100.0 for value in values)))})
    if not rows:
        raise ValueError("no daily factor observations found")
    return rows


def _read_factor_receipt(metadata: dict) -> tuple[dict, bytes]:
    """Verify a Fama--French factor receipt and return its immutable raw bytes."""
    if not isinstance(metadata, dict):
        raise ValueError("factors metadata must be an object")
    receipt = metadata.get("raw_receipt")
    expected = {"version", "adapter", "parser_version", "provider", "endpoint", "identity",
                "sha256", "raw_path", "retrieved_at", "content_type"}
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise ValueError("historical factor fragment requires a complete raw_receipt")
    if receipt.get("version") != RAW_RECEIPT_VERSION:
        raise ValueError("factor raw receipt has an unsupported version")
    if (receipt.get("adapter") != FAMA_FRENCH_DAILY_ADAPTER
            or receipt.get("parser_version") != FAMA_FRENCH_DAILY_PARSER_VERSION
            or receipt.get("provider") != FAMA_FRENCH_DAILY_PROVIDER
            or receipt.get("endpoint") != FAMA_FRENCH_DAILY_URL
            or receipt.get("identity") != FAMA_FRENCH_DAILY_IDENTITY):
        raise ValueError("unsupported Fama-French factor adapter or provider identity")
    digest = receipt.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("factor raw receipt requires a SHA256")
    _utc(receipt.get("retrieved_at"), "factor raw receipt retrieved_at")
    if receipt.get("content_type") not in ("application/zip", "text/csv"):
        raise ValueError("factor raw receipt content_type is invalid")
    raw_path = receipt.get("raw_path")
    if not isinstance(raw_path, str):
        raise ValueError("factor raw receipt object is unavailable")
    raw_object_path = Path(raw_path)
    raw = _read_cache_object(raw_object_path, "factor raw receipt object")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("factor raw receipt object hash mismatch")
    if raw_object_path.name != digest or raw_object_path.parent.name != digest[:2]:
        raise ValueError("factor raw receipt object is not content addressed")
    cache_root = _cache_root_from_object(raw_object_path, "factor raw receipt object")
    expected_type = "application/zip" if raw.startswith(b"PK") else "text/csv"
    if receipt["content_type"] != expected_type:
        raise ValueError("factor raw receipt content_type does not match its raw object")
    receipt_hash = metadata.get("raw_receipt_sha256")
    canonical = _canonical_json_bytes(receipt)
    if (not isinstance(receipt_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_hash)
            or receipt_hash != hashlib.sha256(canonical).hexdigest()):
        raise ValueError("factor raw receipt hash is missing or inconsistent")
    sidecar_path = metadata.get("raw_receipt_path")
    if not isinstance(sidecar_path, str):
        raise ValueError("factor raw receipt sidecar is unavailable")
    expected_sidecar_path = (cache_root / "receipts" /
                             hashlib.sha256(FAMA_FRENCH_DAILY_URL.encode("utf-8")).hexdigest() /
                             f"{receipt_hash}.json")
    sidecar = Path(sidecar_path)
    if sidecar.resolve() != expected_sidecar_path.resolve():
        raise ValueError("factor raw receipt sidecar is not content addressed")
    if _read_cache_object(sidecar, "factor raw receipt sidecar") != canonical:
        raise ValueError("factor raw receipt sidecar does not match its declared receipt")
    return receipt, raw


def _replay_fama_french_factor_fragment(metadata: dict) -> tuple[list[dict], list[dict], dict]:
    """Reconstruct factor and benchmark rows from an immutable provider response."""
    receipt, raw = _read_factor_receipt(metadata)
    if (metadata.get("data_kind") != "historical"
            or metadata.get("source") != FAMA_FRENCH_DAILY_URL
            or metadata.get("retrieved_at") != receipt["retrieved_at"]
            or metadata.get("input_sha256") != receipt["sha256"]
            or metadata.get("factor_lineage_id") != FAMA_FRENCH_FACTOR_LINEAGE_ID
            or metadata.get("return_unit") != "decimal"
            or metadata.get("factor_vintage") != "retrieved_snapshot"
            or metadata.get("benchmark_definition") != FAMA_FRENCH_BENCHMARK_DEFINITION
            or metadata.get("usage") != FAMA_FRENCH_USAGE):
        raise ValueError("factor fragment metadata differs from its immutable provider receipt")
    factors = parse_fama_french_daily(raw)
    benchmark = [{"session": row["session"], "return": row["mkt_rf"] + row["rf"]}
                 for row in factors]
    return factors, benchmark, receipt


def verify_historical_factor_fragment(fragment: dict) -> tuple[list[dict], list[dict], dict]:
    """Replay a Fama--French fragment and compare factor and benchmark rows exactly."""
    if not isinstance(fragment, dict) or not isinstance(fragment.get("metadata"), dict):
        raise ValueError("factors fragment requires metadata")
    factors, benchmark, receipt = _replay_fama_french_factor_fragment(fragment["metadata"])
    if _rows(fragment, "factors") != factors:
        raise ValueError("factors differ from the immutable provider replay")
    if _rows(fragment, "benchmark") != benchmark:
        raise ValueError("benchmark differs from the immutable factor replay")
    return factors, benchmark, receipt


def fetch_fama_french_daily(cache_dir) -> dict:
    """Fetch official daily factors and preserve an immutable, replayable snapshot.

    The derived benchmark is the Fama--French market total return, ``mkt_rf + rf``.
    Its local receipt clock captures this download's vintage; it is not a claim
    that the factor rows were known at a historical trading decision.
    """
    request = Request(FAMA_FRENCH_DAILY_URL, headers={"User-Agent": "SignalGraph research/1.0"})
    with urlopen(request, timeout=30) as response:
        _response_stays_at_approved_endpoint(response, FAMA_FRENCH_DAILY_URL, "Fama-French")
        raw = response.read(20_000_001)
    if len(raw) > 20_000_000:
        raise ValueError("factor download exceeds 20 MB")
    factors = parse_fama_french_daily(raw)
    retrieved = datetime.now(timezone.utc)
    receipt, receipt_path = _persist_factor_snapshot(cache_dir, raw, retrieved_at=retrieved)
    return {
        "metadata": {"data_kind": "historical", "source": FAMA_FRENCH_DAILY_URL,
                     "retrieved_at": receipt["retrieved_at"], "input_sha256": receipt["sha256"],
                     "raw_receipt": receipt, "raw_receipt_path": receipt_path,
                     "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest(),
                     "factor_lineage_id": FAMA_FRENCH_FACTOR_LINEAGE_ID,
                     "return_unit": "decimal", "factor_vintage": "retrieved_snapshot",
                     "benchmark_definition": FAMA_FRENCH_BENCHMARK_DEFINITION,
                     "usage": FAMA_FRENCH_USAGE},
        "factors": factors,
        "benchmark": [{"session": row["session"], "return": row["mkt_rf"] + row["rf"]}
                      for row in factors],
    }


def parse_alpha_vantage_adjusted(payload: dict, ticker: str, sessions: list[dict] | dict,
                                 *, available_at: str | None = None) -> list[dict]:
    """Convert adjusted daily responses using an explicit exchange session table.

    ``sessions`` selects dates to import and contains session/open_at/close_at.
    Every requested session must have a vendor bar; no gaps are filled. The
    endpoint is premium; HTTP-200 quota/entitlement messages fail explicitly.
    """
    ticker = _text(ticker, "ticker").upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", ticker):
        raise ValueError("ticker must be a supported uppercase provider symbol")
    series = payload.get("Time Series (Daily)") if isinstance(payload, dict) else None
    if not isinstance(series, dict):
        raise ValueError("Alpha Vantage adjusted daily data unavailable; check entitlement, key and quota")
    provider_symbol = payload.get("Meta Data", {}).get("2. Symbol")
    if not isinstance(provider_symbol, str) or not provider_symbol.strip():
        raise ValueError("Alpha Vantage response must identify its provider symbol")
    if provider_symbol.upper() != ticker.upper():
        raise ValueError("Alpha Vantage response symbol does not match requested ticker")
    sessions = _normalize_price_sessions(sessions)
    if available_at is not None:
        _utc(available_at, "price provider available_at")
    bars = []
    for clock in sessions:
        day = _session(clock.get("session"))
        vendor = series.get(day)
        if not isinstance(vendor, dict) or "5. adjusted close" not in vendor:
            raise ValueError(f"missing adjusted vendor bar for {ticker} {day}")
        bars.append({"ticker": ticker, "session": day, "open_at": clock.get("open_at"),
                     "close_at": clock.get("close_at"),
                     "adjusted_close": float(vendor["5. adjusted close"]),
                     "volume": float(vendor["6. volume"]),
                     **({"available_at": available_at} if available_at is not None else {})})
    _validate_bars(bars)
    return bars


def fetch_alpha_vantage_adjusted(ticker: str, api_key: str, sessions: list[dict] | dict,
                                 *, cache_dir) -> dict:
    """Download one symbol with an explicit key; never persist/log its request URL.

    Return a price fragment with source hash and retrieval provenance. The API
    can require a paid entitlement; callers decide whether to use it. No retry
    storm, alternative source or synthetic fallback is performed.
    """
    ticker = _text(ticker, "ticker").upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", ticker):
        raise ValueError("ticker must be a supported uppercase provider symbol")
    _text(api_key, "api_key")
    query = urlencode({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker,
                       "outputsize": "full", "apikey": api_key})
    try:
        with urlopen(Request(f"https://www.alphavantage.co/query?{query}"), timeout=30) as response:
            _response_stays_at_approved_endpoint(
                response, "https://www.alphavantage.co/query", "Alpha Vantage")
            raw = response.read(20_000_001)
    except (HTTPError, URLError, TimeoutError):
        # urllib exceptions may include the URL, so do not propagate key-bearing
        # exception strings or exception chains to CLI output.
        raise RuntimeError("Alpha Vantage request failed; check connectivity and entitlement") from None
    if len(raw) > 20_000_000:
        raise ValueError("Alpha Vantage response exceeds 20 MB")
    retrieved = datetime.now(timezone.utc)
    retrieved_at = iso_utc(retrieved)
    selected_sessions = _normalize_price_sessions(sessions)
    receipt, receipt_path = _persist_raw_snapshot(
        cache_dir, raw, provider="Alpha Vantage", endpoint="https://www.alphavantage.co/query",
        identity={"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker},
        sessions=selected_sessions, adapter=ALPHA_VANTAGE_ADJUSTED_ADAPTER,
        parser_version=ALPHA_VANTAGE_PARSER_VERSION, retrieved_at=retrieved,
    )
    bars = parse_alpha_vantage_adjusted(_json_bytes(raw), ticker, selected_sessions, available_at=retrieved_at)
    return {"metadata": {"data_kind": "historical", "source": ALPHA_VANTAGE_ADJUSTED_SOURCE,
                         "price_lineage_id": f"alpha_vantage:TIME_SERIES_DAILY_ADJUSTED:{ticker}",
                         "adjustment": "split_and_dividend", "retrieved_at": retrieved_at,
                         "input_sha256": receipt["sha256"], "raw_receipt": receipt,
                         "raw_receipt_path": receipt_path,
                         "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest(),
                         "usage": ALPHA_VANTAGE_ADJUSTED_USAGE},
            "bars": bars}


def _replay_alpha_price_fragment(metadata: dict) -> tuple[list[dict], dict]:
    """Reconstruct one Alpha Vantage fragment solely from immutable cache bytes."""
    receipt, raw, sessions = verify_raw_receipt(metadata, "prices")
    receipt_hash = metadata.get("raw_receipt_sha256")
    if (not isinstance(receipt_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_hash)
            or receipt_hash != hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()):
        raise ValueError("prices raw receipt hash is missing or inconsistent")
    if receipt["adapter"] != ALPHA_VANTAGE_ADJUSTED_ADAPTER or receipt["parser_version"] != ALPHA_VANTAGE_PARSER_VERSION:
        raise ValueError("unsupported historical price adapter or parser version")
    if (receipt["provider"] != "Alpha Vantage"
            or receipt["endpoint"] != "https://www.alphavantage.co/query"):
        raise ValueError("Alpha Vantage raw receipt provider endpoint is invalid")
    identity = receipt["identity"]
    if set(identity) != {"function", "symbol"} or identity["function"] != "TIME_SERIES_DAILY_ADJUSTED":
        raise ValueError("Alpha Vantage raw receipt request identity is invalid")
    ticker = identity["symbol"].upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", ticker):
        raise ValueError("Alpha Vantage raw receipt ticker is invalid")
    if metadata.get("retrieved_at") != receipt["retrieved_at"]:
        raise ValueError("prices raw receipt retrieved_at must match its fragment metadata")
    if (metadata.get("data_kind") != "historical"
            or metadata.get("source") != ALPHA_VANTAGE_ADJUSTED_SOURCE
            or metadata.get("input_sha256") != receipt["sha256"]
            or metadata.get("usage") != ALPHA_VANTAGE_ADJUSTED_USAGE):
        raise ValueError("prices fragment metadata differs from its immutable provider receipt")
    lineage_id = f"alpha_vantage:TIME_SERIES_DAILY_ADJUSTED:{ticker}"
    if metadata.get("price_lineage_id") != lineage_id:
        raise ValueError("prices price_lineage_id does not match its immutable provider identity")
    expected = parse_alpha_vantage_adjusted(_json_bytes(raw), ticker, sessions,
                                            available_at=receipt["retrieved_at"])
    return expected, receipt


def verify_historical_price_fragment(fragment: dict) -> tuple[list[dict], dict]:
    """Replay a supported historical price fragment and compare every bar exactly."""
    if not isinstance(fragment, dict) or not isinstance(fragment.get("metadata"), dict):
        raise ValueError("prices fragment requires metadata")
    metadata = fragment["metadata"]
    if metadata.get("data_kind") != "historical":
        raise ValueError("price replay requires a historical fragment")
    if metadata.get("adjustment") != "split_and_dividend":
        raise ValueError("price replay requires split_and_dividend adjustment")
    expected, receipt = _replay_alpha_price_fragment(metadata)
    actual = _rows(fragment, "bars")
    if actual != expected:
        raise ValueError("prices bars differ from the immutable provider replay")
    return expected, receipt


def assemble_market_bundle(price_paths: list, factor_path, universe_policy: str,
                           session_schedule_path, memberships_path=None, sector_returns_path=None) -> dict:
    """Compose separately sourced fragments into one validated market bundle.

    Each input's exact bytes are hashed. Original component metadata is retained,
    including vendor hashes and factor vintage. A composition containing any
    fixture-labelled price or factor fragment remains a fixture. Session schedules,
    memberships, and sector controls are currently schema-validated declarations;
    this function does not infer historical constituents or authenticate those
    optional upstream inputs.
    """
    _text(universe_policy, "universe_policy")
    if not price_paths:
        raise ValueError("at least one price fragment is required")
    provenance, components, kinds, retrieval_times, sources, price_lineage = [], [], [], [], [], []
    factor_lineage = None

    def read_component(path, role, *, fragment=True):
        path = Path(path)
        raw = path.read_bytes()
        data = _json_bytes(raw)
        provenance.append({"role": role, "input_path": str(path.resolve()),
                           "input_sha256": hashlib.sha256(raw).hexdigest()})
        if fragment:
            if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
                raise ValueError(f"{role} fragment requires metadata")
            meta = data["metadata"]
            if meta.get("data_kind") not in ("historical", "fixture"):
                raise ValueError(f"{role} fragment must declare historical or fixture data_kind")
            kinds.append(meta["data_kind"])
            sources.append(_text(meta.get("source"), f"{role}.source"))
            retrieval_times.append(_utc(meta.get("retrieved_at"), f"{role}.retrieved_at"))
            components.append({"role": role, "input_path": str(path.resolve()), "metadata": copy.deepcopy(meta)})
        return data

    schedule_path = Path(session_schedule_path)
    schedule_raw = schedule_path.read_bytes()
    schedule_data = _json_bytes(schedule_raw)
    schedule = _validate_session_schedule(schedule_data, required=True)
    provenance.append({"role": "session_schedule", "input_path": str(schedule_path.resolve()),
                       "input_sha256": hashlib.sha256(schedule_raw).hexdigest()})
    components.append({"role": "session_schedule", "input_path": str(schedule_path.resolve()),
                       "calendar_id": schedule["calendar_id"], "source": schedule["source"],
                       "retrieved_at": schedule["retrieved_at"]})
    retrieval_times.append(_utc(schedule["retrieved_at"], "session_schedule.retrieved_at"))
    sources.append(schedule["source"])

    bars = []
    for path in price_paths:
        prices = read_component(path, "prices")
        if prices["metadata"].get("adjustment") != "split_and_dividend":
            raise ValueError("price fragments must explicitly declare split_and_dividend adjustment")
        if prices["metadata"].get("data_kind") == "historical":
            _, receipt = verify_historical_price_fragment(prices)
            lineage_id = _text(prices["metadata"].get("price_lineage_id"), "prices.price_lineage_id")
            price_lineage.append({"id": lineage_id, "source": prices["metadata"]["source"],
                                  "identity": receipt["identity"]})
        else:
            # Fixtures remain explicitly labelled and have a deterministic,
            # non-provider lineage solely for integration-test traceability.
            price_lineage.append({"id": f"fixture:{prices['metadata']['source']}",
                                  "source": prices["metadata"]["source"],
                                  "identity": {"fixture_source": prices["metadata"]["source"]}})
        rows = _rows(prices, "bars")
        _validate_bars(rows)
        received = _utc(prices["metadata"]["retrieved_at"], "prices.retrieved_at")
        if any(_utc(row["close_at"], "bars.close_at") > received for row in rows):
            raise ValueError("price fragment bars cannot close after its retrieved_at")
        bars.extend(rows)
    factors = read_component(factor_path, "factors")
    if factors["metadata"].get("return_unit") != "decimal":
        raise ValueError("factor fragment must explicitly declare return_unit=decimal")
    if factors["metadata"].get("data_kind") == "historical":
        replayed_factors, replayed_benchmark, receipt = verify_historical_factor_fragment(factors)
        factor_lineage = {"id": factors["metadata"]["factor_lineage_id"],
                          "source": factors["metadata"]["source"],
                          "identity": receipt["identity"]}
    else:
        replayed_factors, replayed_benchmark = _rows(factors, "factors"), _rows(factors, "benchmark")
        factor_lineage = {"id": f"fixture:{factors['metadata']['source']}",
                          "source": factors["metadata"]["source"],
                          "identity": {"fixture_source": factors["metadata"]["source"]}}
    schedule_sessions = {row["session"] for row in schedule["sessions"]}
    benchmark = [row for row in replayed_benchmark if row.get("session") in schedule_sessions]
    factor_rows = [row for row in replayed_factors if row.get("session") in schedule_sessions]
    result = {
        "metadata": {"data_kind": "historical" if all(kind == "historical" for kind in kinds) else "fixture",
                     "source": "; ".join(dict.fromkeys(sources)), "adjustment": "split_and_dividend",
                     "universe_policy": universe_policy, "retrieved_at": max(retrieval_times).isoformat(),
                     "price_lineage": sorted(price_lineage, key=lambda row: row["id"]),
                     "factor_lineage": factor_lineage,
                     "return_unit": "decimal", "component_metadata": components,
                     "usage": "Adjusted-price ratios for returns; downloaded factor vintages for ex-post risk adjustment."},
        "bars": sorted(bars, key=lambda row: (row["ticker"], row["session"])),
        "session_schedule": schedule,
        "benchmark": benchmark, "factors": factor_rows,
        "provenance": {"components": provenance},
    }
    for field, path in (("memberships", memberships_path), ("sector_returns", sector_returns_path)):
        if path is not None:
            result[field] = read_component(path, field, fragment=False)
    return validate_market_bundle(result)


def verify_historical_market_provenance(market: dict) -> dict:
    """Replay supported price/factor components and disclose unreplayed inputs.

    ``validate_market_bundle`` checks public shape and clocks.  Historical
    admission additionally reconstructs every adjusted-price bar and every
    factor/benchmark row from retained provider bytes.  Schedules, memberships,
    and sector controls remain schema-validated declared inputs pending their
    own provider adapters.  Imported ``verified`` markers are ignored.
    """
    market = validate_market_bundle(market)
    if market["metadata"]["data_kind"] != "historical":
        return {"status": "not_applicable_fixture", "price_receipt_hashes": [],
                "factor_receipt_hashes": []}
    metadata = market["metadata"]
    components = metadata.get("component_metadata")
    if not isinstance(components, list):
        raise ValueError("historical market bundle requires retained component_metadata")
    schedule = _validate_session_schedule(market.get("session_schedule"), required=True)
    schedule_sessions = {row["session"] for row in schedule["sessions"]}
    actual_bars = {(row["ticker"], row["session"]): row for row in market["bars"]}
    expected_bars, expected_price_lineage, price_receipt_hashes = {}, [], []
    expected_factors = expected_benchmark = expected_factor_lineage = None
    factor_receipt_hashes = []
    bundle_retrieved = _utc(metadata["retrieved_at"], "metadata.retrieved_at")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("historical market bundle has an invalid component")
        role = component.get("role")
        if role not in ("prices", "factors"):
            continue
        fragment_metadata = component.get("metadata")
        if not isinstance(fragment_metadata, dict) or fragment_metadata.get("data_kind") != "historical":
            raise ValueError(f"historical market bundle has an invalid {role} component")
        if role == "prices":
            bars, receipt = _replay_alpha_price_fragment(fragment_metadata)
            if _utc(receipt["retrieved_at"], "prices raw receipt retrieved_at") > bundle_retrieved:
                raise ValueError("historical market bundle predates a retained price receipt")
            receipt_hash = fragment_metadata.get("raw_receipt_sha256")
            expected_price_lineage.append({"id": fragment_metadata["price_lineage_id"],
                                           "source": fragment_metadata["source"],
                                           "identity": receipt["identity"]})
            price_receipt_hashes.append(receipt_hash)
            for row in bars:
                key = (row["ticker"], row["session"])
                if key in expected_bars:
                    raise ValueError("historical price replay has duplicate bar ownership")
                expected_bars[key] = row
        else:
            if expected_factors is not None:
                raise ValueError("historical market bundle requires exactly one replayable factor component")
            factors, benchmark, receipt = _replay_fama_french_factor_fragment(fragment_metadata)
            if _utc(receipt["retrieved_at"], "factor raw receipt retrieved_at") > bundle_retrieved:
                raise ValueError("historical market bundle predates a retained factor receipt")
            expected_factors = [row for row in factors if row["session"] in schedule_sessions]
            expected_benchmark = [row for row in benchmark if row["session"] in schedule_sessions]
            expected_factor_lineage = {"id": fragment_metadata["factor_lineage_id"],
                                       "source": fragment_metadata["source"],
                                       "identity": receipt["identity"]}
            factor_receipt_hashes.append(fragment_metadata.get("raw_receipt_sha256"))
    if not expected_bars:
        raise ValueError("historical market bundle requires at least one replayable price component")
    if expected_factors is None:
        raise ValueError("historical market bundle requires exactly one replayable factor component")
    if set(actual_bars) != set(expected_bars):
        raise ValueError("assembled bars do not exactly match the historical price replay coverage")
    for key, row in expected_bars.items():
        if actual_bars[key] != row:
            raise ValueError("assembled bar differs from the immutable historical price replay")
    if metadata.get("price_lineage") != sorted(expected_price_lineage, key=lambda row: row["id"]):
        raise ValueError("historical market price_lineage differs from replayed provider identity")
    if market["factors"] != expected_factors:
        raise ValueError("assembled factors differ from the immutable provider replay")
    if market["benchmark"] != expected_benchmark:
        raise ValueError("assembled benchmark differs from the immutable factor replay")
    if metadata.get("factor_lineage") != expected_factor_lineage:
        raise ValueError("historical market factor_lineage differs from replayed provider identity")
    unreplayed_components = ["session_schedule"]
    if market.get("memberships"):
        unreplayed_components.append("memberships")
    if market.get("sector_returns"):
        unreplayed_components.append("sector_returns")
    return {"status": "price_factor_replayed_only",
            "replayed_components": ["prices", "factors"],
            "unreplayed_components": unreplayed_components,
            "price_receipt_hashes": sorted(price_receipt_hashes),
            "factor_receipt_hashes": sorted(factor_receipt_hashes)}


def _write_json_exclusive(path, value):
    """Persist a new artifact; an existing research input is never overwritten."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate a complete market bundle and replay supported historical inputs")
    validate.add_argument("path")
    factors = sub.add_parser("fetch-factors", help="Download a public FF factor/benchmark fragment")
    factors.add_argument("--cache-dir", required=True)
    factors.add_argument("--output", required=True)
    prices = sub.add_parser("fetch-prices", help="Download an adjusted price fragment; paid API may be required")
    prices.add_argument("--ticker", required=True)
    prices.add_argument("--sessions", required=True, help="JSON list with session, open_at, close_at")
    prices.add_argument("--api-key-env", default="ALPHAVANTAGE_API_KEY")
    prices.add_argument("--cache-dir", required=True,
                        help="content-addressed raw-response cache; receipts never include the API key")
    prices.add_argument("--output", required=True)
    assemble = sub.add_parser("assemble", help="Compose adjusted price and factor fragments into a validated market bundle")
    assemble.add_argument("--prices", action="append", required=True, help="Price fragment JSON; repeat for multiple issuers")
    assemble.add_argument("--factors", required=True, help="Factor/benchmark fragment JSON")
    assemble.add_argument("--session-schedule", required=True,
                          help="JSON session_schedule object with calendar_id/source/retrieved_at/sessions")
    assemble.add_argument("--universe-policy", required=True, help="Explicit universe selection and survivorship declaration")
    assemble.add_argument("--memberships", help="JSON list of historical membership intervals")
    assemble.add_argument("--sector-returns", help="JSON list of decimal sector returns by session")
    assemble.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = load_market_bundle(args.path)
        integrity = (verify_historical_market_provenance(result)
                     if result["metadata"]["data_kind"] == "historical"
                     else {"status": "not_applicable_fixture"})
        print(json.dumps({"valid": True, "data_kind": result["metadata"]["data_kind"],
                          "bars": len(result["bars"]), "provenance": result["provenance"],
                          "integrity": integrity}, indent=2))
        return
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists; choose a new path: {args.output}")
    if args.command == "fetch-factors":
        result = fetch_fama_french_daily(args.cache_dir)
    elif args.command == "assemble":
        result = assemble_market_bundle(args.prices, args.factors, args.universe_policy,
                                        args.session_schedule, args.memberships, args.sector_returns)
    else:
        result = fetch_alpha_vantage_adjusted(args.ticker, os.environ.get(args.api_key_env, ""),
                                             _json_bytes(Path(args.sessions).read_bytes()),
                                             cache_dir=args.cache_dir)
    _write_json_exclusive(args.output, result)
    print(f"Wrote {args.command} output to {args.output}")


if __name__ == "__main__":
    main()
