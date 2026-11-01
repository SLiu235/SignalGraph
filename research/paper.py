"""Prospective decision ledger and hypothetical close-price marks; no broker orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from common import load_config, read_json, write_json
from data.market_data import (load_market_bundle, parse_session_schedule,
                              plan_next_close, validate_market_bundle,
                              verify_historical_market_provenance)
from data.pit import iso_utc, parse_utc
from evaluation.walk_forward import score_event
from research.features import validate_events
from research.pipeline import DEFAULT_RESEARCH_CONFIG, load_inputs


# v4 also freezes the filing-acceptance freshness policy for each decision.
# Older local ledgers are rejected rather than being interpreted under weaker
# marking rules.
LEDGER_VERSION = 4
DEFAULT_MAX_FILING_AGE_SECONDS = 72 * 60 * 60


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _market_reference(market: dict, schedule: dict) -> dict:
    """Keep a compact, deterministic reference to the input used at record/mark."""
    provenance = market.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    # Unit callers may pass an in-memory validated bundle rather than a loaded
    # file. Hash that canonical object in that case, rather than omitting lineage.
    input_hash = provenance.get("input_sha256")
    if not isinstance(input_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", input_hash):
        input_hash = _json_hash(market)
    price_lineage = market["metadata"].get("price_lineage")
    if not isinstance(price_lineage, list) or not price_lineage:
        raise ValueError("paper market reference requires replayed price_lineage")
    return {"input_sha256": input_hash, "source": market["metadata"]["source"],
            "adjustment": market["metadata"]["adjustment"],
            "universe_policy": market["metadata"]["universe_policy"],
            "price_lineage": price_lineage,
            "retrieved_at": market["metadata"]["retrieved_at"],
            "calendar_id": schedule["calendar_id"], "calendar_source": schedule["source"],
            "calendar_retrieved_at": schedule["retrieved_at"]}


def _verify_execution_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValueError("paper ledger execution plan is missing")
    saved_hash = plan.get("window_sha256")
    payload = {key: value for key, value in plan.items() if key != "window_sha256"}
    if not isinstance(saved_hash, str) or _json_hash(payload) != saved_hash:
        raise ValueError("paper ledger execution-plan hash is invalid")
    for field in ("calendar_id", "source", "retrieved_at", "decision_at", "executable_after"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise ValueError("paper ledger execution plan lacks calendar identity")
    parse_utc(plan["retrieved_at"])
    decision, executable = parse_utc(plan["decision_at"]), parse_utc(plan["executable_after"])
    if executable < decision:
        raise ValueError("paper ledger execution plan precedes its decision")
    window = plan.get("window")
    if not isinstance(window, list) or not window:
        raise ValueError("paper ledger execution plan has no session window")
    prior_close = None
    for row in window:
        if not isinstance(row, dict) or not isinstance(row.get("session"), str):
            raise ValueError("paper ledger execution plan has an invalid session")
        opened, closed = parse_utc(row.get("open_at", "")), parse_utc(row.get("close_at", ""))
        if opened >= closed or (prior_close is not None and opened <= prior_close):
            raise ValueError("paper ledger execution plan session clocks are invalid")
        prior_close = closed
    if plan.get("entry") != window[0]:
        raise ValueError("paper ledger entry is not the first frozen session")
    exits = plan.get("exits")
    if not isinstance(exits, dict) or not exits:
        raise ValueError("paper ledger execution plan lacks planned exits")
    for key, row in exits.items():
        if not isinstance(key, str) or not key.isdigit() or int(key) < 1:
            raise ValueError("paper ledger execution plan has an invalid horizon")
        if int(key) >= len(window) or row != window[int(key)]:
            raise ValueError("paper ledger execution plan exit does not match its frozen window")


def record_decisions(events: list[dict], ledger_path: Path, *, settings: dict,
                     data_kind: str, market: dict, now: datetime | None = None) -> dict:
    """Serialize local writers with a POSIX advisory lock and atomic replacement."""
    import fcntl
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.with_suffix(ledger_path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _record_decisions(events, ledger_path, settings=settings, data_kind=data_kind,
                                 market=market, now=now)


def _record_decisions(events: list[dict], ledger_path: Path, *, settings: dict,
                      data_kind: str, market: dict, now: datetime | None = None) -> dict:
    """Append local evidence decisions at the actual clock time, with no backdating.

    Explicit `now` is a test seam; the CLI always uses the current UTC clock. The
    ledger hash chain detects accidental corruption, not a malicious local rewrite.
    """
    if data_kind != "historical":
        raise ValueError("paper recording requires real inputs, not a fixture")
    market = validate_market_bundle(market)
    if market["metadata"]["data_kind"] != "historical":
        raise ValueError("paper recording requires a historical market bundle")
    schedule = parse_session_schedule(market)
    at = now or datetime.now(timezone.utc)
    stamp = iso_utc(at)
    ledger_path = Path(ledger_path)
    existing = read_json(ledger_path) if ledger_path.exists() else {"version": LEDGER_VERSION, "entries": []}
    verify_ledger(existing)
    seen = {e["event_id"] for e in existing["entries"]}
    validate_events(events)
    max_filing_age = settings.get("max_filing_age_seconds", DEFAULT_MAX_FILING_AGE_SECONDS)
    if (isinstance(max_filing_age, bool) or not isinstance(max_filing_age, (int, float))
            or not math.isfinite(max_filing_age) or max_filing_age <= 0):
        raise ValueError("max_filing_age_seconds must be finite and positive")
    for event in events:
        if event["event_id"] in seen:
            continue
        mode = event.get("availability_mode", event.get("provenance", {}).get("availability_mode"))
        if mode != "live":
            raise ValueError("prospective paper decisions require live availability, not historical reconstruction")
        for field in ("available_at", "features_available_at", "retrieved_at"):
            if parse_utc(event[field]) > at:
                raise ValueError(f"future {field} cannot enter a paper decision")
        # A late first fetch moves live available_at to the fetch time. It must
        # not turn an old accession into a fresh prospective signal.
        age = (at - parse_utc(event["event_time"])).total_seconds()
        if age < 0 or age > max_filing_age:
            raise ValueError("filing acceptance is too old for a prospective paper decision")
    verify_historical_market_provenance(market)
    if parse_utc(market["metadata"]["retrieved_at"]) > at:
        raise ValueError("paper recording cannot use a market bundle retrieved after the decision clock")
    if parse_utc(schedule["retrieved_at"]) > at:
        raise ValueError("paper recording cannot use a session schedule retrieved after the decision clock")
    market_reference = _market_reference(market, schedule)
    for event in events:
        if event["event_id"] in seen:
            continue
        cost = float(settings.get("cost_bps_per_side", 5))
        slippage = float(settings.get("slippage_bps_per_side", 5))
        latency = float(settings.get("latency_seconds", 60))
        horizons = settings.get("horizons", [1, 5, 20])
        if not isinstance(horizons, list) or not horizons or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError("paper horizons must be planned positive session counts")
        if any(not math.isfinite(x) or x < 0 for x in (cost, slippage, latency)):
            raise ValueError("paper costs must be finite and nonnegative")
        plan = plan_next_close(schedule, at, latency, sorted(set(horizons)))
        entry = {"event_id": event["event_id"], "ticker": event["ticker"], "decision_at": stamp,
                 "event_time": event["event_time"], "available_at": event["available_at"],
                 "max_filing_age_seconds": max_filing_age,
                 "document_sha256": event["document_sha256"],
                 "score": score_event(event), "features": event["features"],
                 "feature_version": event.get("feature_version"), "cost_bps_per_side": cost,
                 "slippage_bps_per_side": slippage, "latency_seconds": latency,
                 "planned_horizons": sorted(set(horizons)),
                 "execution_plan": plan, "market_reference": market_reference,
                 "previous_hash": existing["entries"][-1]["record_hash"] if existing["entries"] else "",
                 "status": "recorded; awaiting hypothetical frozen next-close entry"}
        entry["record_hash"] = _digest(entry)
        existing["entries"].append(entry)
        seen.add(event["event_id"])
    # Never overwrite a corrupt ledger; verification above occurs before any write.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=ledger_path.parent, delete=False) as stream:
        json.dump(existing, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, ledger_path)
    return existing


def _digest(entry: dict) -> str:
    return _json_hash(entry)


def verify_ledger(ledger: dict) -> None:
    if ledger.get("version") != LEDGER_VERSION or not isinstance(ledger.get("entries"), list):
        raise ValueError("paper ledger requires schema version 4 with frozen execution plans, price lineage and filing freshness")
    previous, seen = "", set()
    for entry in ledger["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("paper ledger entry must be an object")
        payload = {k: v for k, v in entry.items() if k != "record_hash"}
        if entry.get("previous_hash") != previous or _digest(payload) != entry.get("record_hash"):
            raise ValueError("paper ledger hash chain invalid")
        if not isinstance(entry.get("event_id"), str) or not entry["event_id"]:
            raise ValueError("paper ledger entry lacks an event_id")
        if entry["event_id"] in seen:
            raise ValueError("duplicate paper event")
        decision = parse_utc(entry.get("decision_at", ""))
        event_time = parse_utc(entry.get("event_time", ""))
        max_age = entry.get("max_filing_age_seconds")
        if (isinstance(max_age, bool) or not isinstance(max_age, (int, float))
                or not math.isfinite(max_age) or max_age <= 0
                or not 0 <= (decision - event_time).total_seconds() <= max_age):
            raise ValueError("paper ledger filing freshness policy is invalid")
        plan = entry.get("execution_plan")
        _verify_execution_plan(plan)
        if parse_utc(plan["decision_at"]) != decision:
            raise ValueError("paper ledger plan decision does not match the entry decision")
        latency = entry.get("latency_seconds")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("paper ledger latency is invalid")
        if parse_utc(plan["executable_after"]) != decision + timedelta(seconds=float(latency)):
            raise ValueError("paper ledger plan executable boundary does not match the entry latency")
        horizons = entry.get("planned_horizons")
        if not isinstance(horizons, list) or sorted(set(horizons)) != horizons or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError("paper ledger planned horizons are invalid")
        if set(plan["exits"]) != {str(horizon) for horizon in horizons}:
            raise ValueError("paper ledger plan exits do not match the planned horizons")
        reference = entry.get("market_reference")
        if (not isinstance(reference, dict) or not isinstance(reference.get("input_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", reference["input_sha256"])
                or not isinstance(reference.get("source"), str) or not reference["source"]
                or reference.get("adjustment") != "split_and_dividend"
                or not isinstance(reference.get("universe_policy"), str) or not reference["universe_policy"]
                or not isinstance(reference.get("price_lineage"), list) or not reference["price_lineage"]
                or reference.get("calendar_id") != plan["calendar_id"]
                or reference.get("calendar_source") != plan["source"]):
            raise ValueError("paper ledger market reference is missing")
        parse_utc(reference.get("retrieved_at", ""))
        parse_utc(reference.get("calendar_retrieved_at", ""))
        seen.add(entry["event_id"])
        previous = entry["record_hash"]


def mark_positions(ledger: dict, market: dict, as_of: str, horizon: int = 5) -> dict:
    verify_ledger(ledger)
    market = validate_market_bundle(market)
    if market["metadata"]["data_kind"] != "historical":
        raise ValueError("paper marks require real market data")
    verify_historical_market_provenance(market)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive session count")
    cutoff = parse_utc(as_of)
    schedule = parse_session_schedule(market)
    if parse_utc(market["metadata"]["retrieved_at"]) > cutoff:
        raise ValueError("paper marking cannot use a market bundle retrieved after the mark cutoff")
    if parse_utc(schedule["retrieved_at"]) > cutoff:
        raise ValueError("paper marking cannot use a session schedule retrieved after the mark cutoff")
    market_reference = _market_reference(market, schedule)
    by_ticker = {}
    for bar in market["bars"]:
        by_ticker.setdefault(bar["ticker"], {})[bar["session"]] = bar
    rows = []
    for entry in ledger["entries"]:
        decision = parse_utc(entry["decision_at"])
        if decision > cutoff:
            continue
        if horizon not in entry["planned_horizons"]:
            raise ValueError("mark horizon was not planned in the prospective decision")
        plan = entry["execution_plan"]
        frozen_reference = entry["market_reference"]
        if (market_reference["source"] != frozen_reference["source"]
                or market_reference["adjustment"] != frozen_reference["adjustment"]
                or market_reference["universe_policy"] != frozen_reference["universe_policy"]
                or market_reference["price_lineage"] != frozen_reference["price_lineage"]):
            raise ValueError("later market bundle does not match the frozen price lineage")
        if (schedule["calendar_id"] != plan["calendar_id"] or schedule["source"] != plan["source"]):
            raise ValueError("later market bundle does not match the frozen calendar identity")
        planned_window = plan["window"]
        for planned in planned_window:
            observed_clock = schedule["by_session"].get(planned["session"])
            if observed_clock != planned:
                raise ValueError("later market bundle changed or omitted a frozen execution session")
        exit_plan = plan["exits"].get(str(horizon))
        if exit_plan is None:
            raise ValueError("mark horizon does not have a frozen execution exit")
        exit_index = planned_window.index(exit_plan)
        path = planned_window[:exit_index + 1]
        bars_by_session = by_ticker.get(entry["ticker"], {})
        entry_clock = parse_utc(path[0]["close_at"])
        row_base = {"event_id": entry["event_id"], "ticker": entry["ticker"],
                    "planned_entry_session": path[0]["session"], "planned_entry_at": path[0]["close_at"],
                    "planned_exit_session": path[-1]["session"], "planned_exit_at": path[-1]["close_at"],
                    "execution_plan_sha256": plan["window_sha256"]}
        if entry_clock > cutoff:
            rows.append({**row_base, "status": "pending", "net_return": None,
                         "reason": "scheduled entry close has not occurred at the mark cutoff"})
            continue
        observed_bars = []
        for planned in path:
            close = parse_utc(planned["close_at"])
            if close > cutoff:
                break
            bar = bars_by_session.get(planned["session"])
            if bar is None:
                raise ValueError("missing planned market bar after its scheduled close")
            observed_bars.append(bar)
        if not observed_bars:
            raise AssertionError("scheduled entry should have been observed or rejected")
        entry_bar = observed_bars[0]
        mark_bar = observed_bars[-1]
        closed = len(observed_bars) == len(path)
        gross = entry["score"] * (mark_bar["adjusted_close"] / entry_bar["adjusted_close"] - 1)
        sides = 2 if closed else 1
        costs = abs(entry["score"]) * sides * (entry["cost_bps_per_side"] + entry["slippage_bps_per_side"]) / 10000
        rows.append({**row_base, "status": "closed" if closed else "open",
                     "entry_at": entry_bar["close_at"], "marked_at": mark_bar["close_at"],
                     "gross_return": gross, "cost": costs, "net_return": gross - costs})
    output = {"as_of": as_of, "horizon_sessions": horizon, "rows": rows,
            "ledger_head": ledger["entries"][-1]["record_hash"] if ledger["entries"] else None,
            "market_reference": market_reference,
            "interpretation": "Hypothetical total-return close marks on frozen exposure; no executed fills, borrow fees, liquidity model or broker verification.",
            "capital_deployment_allowed": False}
    output["report_sha256"] = _json_hash(output)
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("record", "mark"))
    ap.add_argument("--config", default=str(DEFAULT_RESEARCH_CONFIG))
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--market", help="newer validated market bundle for marking")
    ap.add_argument("--output", help="new mark report path (required for mark)")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.action == "record":
        events, market, _ = load_inputs(cfg)
        settings = {**cfg["research"]["evaluation"], **cfg["research"].get("paper", {})}
        ledger = record_decisions(events, Path(args.ledger), settings=settings,
                                  data_kind=market["metadata"]["data_kind"], market=market)
        print(f"Paper ledger has {len(ledger['entries'])} prospective decisions: {args.ledger}")
    else:
        if not args.market or not args.output:
            ap.error("mark requires --market and --output")
        if Path(args.output).exists():
            raise FileExistsError("paper mark report exists; choose a new output path")
        marks = mark_positions(read_json(args.ledger), load_market_bundle(args.market),
                               iso_utc(datetime.now(timezone.utc)), args.horizon)
        write_json(args.output, marks)
        print(f"Paper marks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
