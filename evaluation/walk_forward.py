"""Point-in-time event research with expanding, purged chronological evaluation.

This module neither trains a return predictor nor submits orders. Scores are frozen
rules. Per-event factor regressions fit only pre-entry session observations; their
Fama--French inputs are ex-post risk-adjustment artifacts unless an independent
availability adapter supplies a historically valid availability clock. All reported returns
are decimal, adjusted-close-to-adjusted-close total returns.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import math
import random
from typing import Any

import numpy as np

from data.market_data import (parse_session_schedule, plan_next_close, validate_market_bundle,
                              verify_historical_market_provenance)
from research.features import prepare_events, validate_events


FEATURE_WEIGHTS = {
    "credibility": 0.40, "access_cost": 0.20,
    "novelty": 0.25, "graph_propagation": 0.15,
}
DEFAULTS = {
    "horizons": [1, 5, 20], "min_train_events": 20, "test_events": 10,
    "embargo_sessions": 5, "estimation_sessions": 120,
    "min_estimation_sessions": 30, "cost_bps_per_side": 5.0,
    "slippage_bps_per_side": 5.0, "latency_seconds": 60,
    "seed": 7, "momentum_sessions": 20, "bootstrap_samples": 500,
    "min_bootstrap_dates": 5, "min_oos_events": 100, "min_oos_folds": 3,
    "portfolio_capital_units": 100.0,
}


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamps must be timezone-aware ISO strings")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if result.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value!r}")
    return result.astimezone(timezone.utc)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def score_event(event: dict, ablation: str | None = None) -> float:
    """Frozen signed research exposure; an ablated feature contributes zero.

    Access cost is transformed as cost/(1+cost); other feature values must be in
    [0,1]. We do not redistribute ablated weights. Imported scores are never
    accepted: every evaluation exposure is regenerated from the frozen features.
    Ablations always refer to the explicit fixed feature rule.
    """
    if ablation is not None and ablation not in FEATURE_WEIGHTS:
        raise ValueError(f"unknown ablation: {ablation}")
    features = event.get("features", {})
    direction = _number(features.get("direction"), "direction")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    values = {}
    for name in FEATURE_WEIGHTS:
        value = _number(features.get(name), name)
        if name == "access_cost":
            if value < 0:
                raise ValueError("access_cost must be nonnegative")
            value = value / (1.0 + value)
        elif not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
        values[name] = value
    return direction * sum(weight * values[name] for name, weight in
                           FEATURE_WEIGHTS.items() if name != ablation)


def _settings(settings: dict) -> dict:
    result = {**DEFAULTS, **settings}
    for key in ("min_train_events", "test_events", "estimation_sessions",
                "min_estimation_sessions", "momentum_sessions", "min_bootstrap_dates",
                "min_oos_events", "min_oos_folds"):
        value = _number(result[key], key)
        if value < 1 or int(value) != value:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(value)
    for key in ("embargo_sessions", "bootstrap_samples"):
        value = _number(result[key], key)
        if value < 0 or int(value) != value:
            raise ValueError(f"{key} must be a nonnegative integer")
        result[key] = int(value)
    for key in ("cost_bps_per_side", "slippage_bps_per_side", "latency_seconds"):
        result[key] = _number(result[key], key)
        if result[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    result["portfolio_capital_units"] = _number(result["portfolio_capital_units"],
                                                "portfolio_capital_units")
    if result["portfolio_capital_units"] <= 0:
        raise ValueError("portfolio_capital_units must be positive")
    if result["min_estimation_sessions"] < 5:
        raise ValueError("min_estimation_sessions must be at least five for four OLS coefficients")
    if result["min_estimation_sessions"] > result["estimation_sessions"]:
        raise ValueError("min_estimation_sessions exceeds estimation_sessions")
    horizons = result["horizons"]
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("horizons must be a nonempty list")
    if any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    result["horizons"] = sorted(set(horizons))
    return result


def _indexed(rows: list[dict], key: str, fields: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"duplicate {key}: {identity}")
        result[identity] = {**row, **{f: _number(row.get(f), f) for f in fields}}
    return result


def _market(market: dict) -> dict:
    """Normalize one fully validated bundle for historical event evaluation."""
    market = validate_market_bundle(market)
    integrity = None
    if market["metadata"]["data_kind"] == "historical":
        integrity = verify_historical_market_provenance(market)
    schedule = parse_session_schedule(market)
    metadata = market["metadata"]
    benchmark = _indexed(market["benchmark"], "session", ("return",))
    factors = _indexed(market["factors"], "session", ("mkt_rf", "smb", "hml", "rf"))
    if not benchmark or not factors:
        raise ValueError("benchmark and factor observations are required")
    if any(row["return"] <= -1 for row in benchmark.values()):
        raise ValueError("benchmark returns must exceed -1")
    calendar = sorted(benchmark)
    for session in calendar:
        try:
            if datetime.strptime(session, "%Y-%m-%d").strftime("%Y-%m-%d") != session:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise ValueError("session must be an ISO YYYY-MM-DD date") from exc
    by_ticker = defaultdict(dict)
    for bar in market["bars"]:
        ticker, session = bar["ticker"], bar["session"]
        if session in by_ticker[ticker]:
            raise ValueError(f"duplicate bar: {ticker}/{session}")
        price = _number(bar.get("adjusted_close"), "adjusted_close")
        if price <= 0:
            raise ValueError("adjusted_close must be positive")
        opened, closed = _time(bar["open_at"]), _time(bar["close_at"])
        if opened >= closed:
            raise ValueError("open_at must precede close_at")
        observed_at = bar.get("available_at")
        if observed_at is not None:
            observed_at = _time(observed_at)
            if observed_at < closed:
                raise ValueError("bar available_at cannot precede its exchange close")
        if session not in benchmark:
            raise ValueError(f"missing benchmark session for bar: {session}")
        by_ticker[ticker][session] = {
            **bar, "adjusted_close": price, "_open": opened, "_close": closed,
            "_available_at": observed_at,
        }
    for ticker, bars in by_ticker.items():
        ordered = [bars[s] for s in sorted(bars)]
        if any(a["_close"] >= b["_open"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError(f"overlapping or unordered sessions for {ticker}")
    sectors = defaultdict(dict)
    sector_rows = market.get("sector_returns", [])
    if isinstance(sector_rows, dict):
        sector_rows = [{"sector": sector, **row} for sector, rows in sector_rows.items()
                       for row in rows]
    for row in sector_rows:
        sector, session = row["sector"], row["session"]
        if session in sectors[sector]:
            raise ValueError(f"duplicate sector return: {sector}/{session}")
        value = _number(row.get("return"), "sector return")
        if value <= -1:
            raise ValueError("sector return must exceed -1")
        sectors[sector][session] = value
    return {"metadata": metadata, "integrity": integrity,
            "schedule": schedule, "bars": dict(by_ticker), "benchmark": benchmark,
            "factors": factors, "calendar": calendar, "sectors": dict(sectors),
            "index": {session: i for i, session in enumerate(calendar)}}


def _compound(values) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def _event_row(event: dict, horizon: int, market: dict, cfg: dict) -> dict:
    available = _time(event["available_at"])
    feature_time = _time(event["features_available_at"])
    if feature_time > available:
        raise ValueError("features_available_at is after event availability")
    score = score_event(event)
    bars = market["bars"].get(event["ticker"], {})
    plan = plan_next_close(market["schedule"], available, cfg["latency_seconds"], [horizon])
    entry_session = plan["entry"]["session"]
    path_sessions = [row["session"] for row in plan["window"]]
    if any(session not in market["benchmark"] for session in path_sessions):
        raise ValueError("incomplete forward benchmark coverage for the scheduled execution window")
    if entry_session not in bars:
        raise ValueError("missing candidate entry stock bar at the scheduled execution close")
    entry = bars[entry_session]
    entry_index = market["index"][entry_session]
    exit_index = market["index"][path_sessions[-1]]
    calendar = market["calendar"]
    if any(s not in bars for s in path_sessions):
        raise ValueError("missing stock bar inside forward window")
    outcomes = path_sessions[1:]
    if any(s not in market["factors"] for s in outcomes):
        raise ValueError("missing factor observation inside forward window")
    # Returns whose close equals entry are excluded: fit ends strictly before it.
    training, missing_training_bars = [], []
    for index in range(max(1, entry_index - cfg["estimation_sessions"]), entry_index):
        session, previous = calendar[index], calendar[index - 1]
        if session not in market["factors"]:
            raise ValueError("missing factor observation inside pre-entry estimation window")
        if session not in bars or previous not in bars:
            missing_training_bars.append(session)
            continue
        factors = market["factors"][session]
        ret = bars[session]["adjusted_close"] / bars[previous]["adjusted_close"] - 1
        training.append((session, [1, factors["mkt_rf"], factors["smb"], factors["hml"]],
                         ret - factors["rf"]))
    if len(training) < cfg["min_estimation_sessions"]:
        raise ValueError("insufficient pre-entry factor estimation observations")
    x = np.array([row[1] for row in training], dtype=float)
    y = np.array([row[2] for row in training], dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 4:
        raise ValueError("rank-deficient pre-entry factor design")
    expected = []
    for session in outcomes:
        f = market["factors"][session]
        expected.append(f["rf"] + float(np.dot(coefficients,
                        [1, f["mkt_rf"], f["smb"], f["hml"]])))
    if any(value <= -1 for value in expected):
        raise ValueError("factor model predicts inadmissible daily simple return")
    gross = bars[path_sessions[-1]]["adjusted_close"] / entry["adjusted_close"] - 1
    benchmark = _compound(market["benchmark"][s]["return"] for s in outcomes)
    sector_data = market["sectors"].get(event.get("sector", "unknown"), {})
    sector_return = (_compound(sector_data[s] for s in outcomes)
                     if all(s in sector_data for s in outcomes) else None)
    # Price-based controls require a provider observation clock, not merely an
    # exchange date.  Historical adjusted bars without it remain valid outcome
    # data but cannot be treated as a predictive momentum input.
    prior = [s for s in calendar if s in bars and bars[s]["_available_at"] is not None
             and bars[s]["_available_at"] <= available]
    momentum = None
    if len(prior) > cfg["momentum_sessions"]:
        final_index = market["index"][prior[-1]]
        start_index = final_index - cfg["momentum_sessions"]
        momentum_path = calendar[max(0, start_index):final_index + 1]
        if (start_index >= 0 and all(
                s in bars and bars[s]["_available_at"] is not None
                and bars[s]["_available_at"] <= available
                for s in momentum_path)):
            momentum = bars[prior[-1]]["adjusted_close"] / bars[momentum_path[0]]["adjusted_close"] - 1
    regime = "unknown"
    if len(prior) >= cfg["momentum_sessions"]:
        end = market["index"][prior[-1]] + 1
        begin = end - cfg["momentum_sessions"]
        if begin >= 0:
            regime_sessions = calendar[begin:end]
            # Fama-French rows are normally ex-post/revisable. They cannot create
            # a point-in-time market-regime feature unless the input explicitly
            # carries an availability clock no later than the event.
            if all(market["benchmark"][s].get("available_at") is not None
                   and _time(market["benchmark"][s]["available_at"]) <= available
                   for s in regime_sessions):
                trailing_market = _compound(market["benchmark"][s]["return"] for s in regime_sessions)
                regime = "up" if trailing_market > 0 else "down" if trailing_market < 0 else "flat"
            else:
                regime = "unavailable_without_point_in_time_benchmark"
    is_earnings = event.get("features", {}).get("is_earnings")
    if is_earnings is not None and not isinstance(is_earnings, bool):
        raise ValueError("is_earnings must be an explicit boolean when supplied")
    return {
        "event_id": event["event_id"], "ticker": event["ticker"],
        "available_at": available.isoformat(), "features_available_at": feature_time.isoformat(),
        "source": event.get("source", "unknown"), "sector": event.get("sector", "unknown"),
        "form": event.get("form", "unknown"), "market_regime": regime,
        "horizon": horizon, "entry_session": entry["session"],
        "entry_at": entry["close_at"], "exit_session": path_sessions[-1],
        "exit_at": bars[path_sessions[-1]]["close_at"],
        "gross_return": gross, "benchmark_return": benchmark,
        "sector_return": sector_return, "factor_expected_return": _compound(expected),
        "factor_residual_return": gross - _compound(expected), "score": score,
        "momentum": momentum, "is_earnings": is_earnings,
        "factor_fit": {"n": len(training), "first_session": training[0][0],
                       "last_session": training[-1][0], "intercept_daily": float(coefficients[0]),
                       "excluded_missing_stock_return_sessions": missing_training_bars,
                       "betas": dict(zip(("mkt_rf", "smb", "hml"), map(float, coefficients[1:])))},
        "_event": event, "_entry_index": entry_index, "_exit_index": exit_index,
        "_execution_plan": plan,
        "_path": [{"session": s, "relative_price": bars[s]["adjusted_close"] / entry["adjusted_close"]}
                  for s in path_sessions],
    }


def _walk_forward_splits(rows: list[dict], cfg: dict) -> list[dict]:
    """Group simultaneous releases; purge labels at/after the embargo boundary."""
    groups = []
    for row in sorted(rows, key=lambda r: (r["available_at"], r["event_id"])):
        if not groups or groups[-1][0]["available_at"] != row["available_at"]:
            groups.append([])
        groups[-1].append(row)
    splits, prior, group_index = [], [], 0
    while group_index < len(groups) and len(prior) < cfg["min_train_events"]:
        prior.extend(groups[group_index])
        group_index += 1
    while group_index < len(groups):
        test = []
        while group_index < len(groups) and len(test) < cfg["test_events"]:
            test.extend(groups[group_index])
            group_index += 1
        boundary = min(r["_entry_index"] for r in test) - cfg["embargo_sessions"]
        # Also require each train label to end before the first test information
        # time, which matters for pre-open releases and long execution latency.
        first_test_at = min(_time(r["available_at"]) for r in test)
        train = [r for r in prior if r["_exit_index"] < boundary
                 and _time(r["exit_at"]) < first_test_at]
        if len(train) >= cfg["min_train_events"]:
            splits.append({"fold": len(splits), "train": train, "test": test,
                           "purged_train_events": len(prior) - len(train),
                           "embargo_boundary_index": boundary})
        prior.extend(test)
    return splits


def _ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + end - 1) / 2
        start = end
    return ranks


def _rank_ic(scores: list[float], outcomes: list[float]) -> float | None:
    if len(scores) < 3 or len(set(scores)) < 2 or len(set(outcomes)) < 2:
        return None
    return float(np.corrcoef(_ranks(scores), _ranks(outcomes))[0, 1])


def _ci(rows: list[dict], values: list[float], cfg: dict) -> list[float] | None:
    clusters = defaultdict(list)
    for row, value in zip(rows, values):
        clusters[row["entry_session"]].append(value)
    if len(clusters) < cfg["min_bootstrap_dates"] or cfg["bootstrap_samples"] < 100:
        return None
    rng = random.Random(cfg["seed"])
    dates = sorted(clusters)
    draws = []
    for _ in range(cfg["bootstrap_samples"]):
        sample = [value for day in rng.choices(dates, k=len(dates)) for value in clusters[day]]
        draws.append(sum(sample) / len(sample))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _sleeves(rows: list[dict], scores: list[float], cfg: dict) -> dict:
    """One initial unit per event; no reinvestment or imposed capital constraint."""
    pnl, entries, exits = defaultdict(float), defaultdict(float), defaultdict(float)
    fee = (cfg["cost_bps_per_side"] + cfg["slippage_bps_per_side"]) / 10000
    for row, score in zip(rows, scores):
        size = abs(score)
        path = row["_path"]
        entries[path[0]["session"]] += size
        exits[path[-1]["session"]] += size
        pnl[path[0]["session"]] -= size * fee
        pnl[path[-1]["session"]] -= size * fee
        for previous, current in zip(path, path[1:]):
            pnl[current["session"]] += score * (current["relative_price"] - previous["relative_price"])
    capital = cfg["portfolio_capital_units"]
    equity = peak = capital
    drawdown = gross_open = max_open = 0.0
    series = []
    for session in sorted(set(pnl) | set(entries) | set(exits)):
        equity += pnl[session]
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
        # Both new and exiting sleeves are counted conservatively at this close.
        gross_open += entries[session]
        max_open = max(max_open, gross_open)
        gross_open -= exits[session]
        series.append({"session": session, "pnl_units": pnl[session], "equity_units": equity})
    return {"scope": "Fixed initial event notional; overlapping sleeves; no capital or borrow constraint. Not investable NAV.",
            "initial_capital_units": capital, "net_pnl_units": equity - capital,
            "return_on_configured_capital": equity / capital - 1,
            "max_drawdown_on_configured_capital": drawdown,
            "two_way_turnover_on_configured_capital": 2 * sum(map(abs, scores)) / capital,
            "max_gross_initial_notional_on_capital": max_open / capital,
            "daily_pnl": series}


def _metrics(rows: list[dict], scores: list[float], cfg: dict,
             outcome_key: str = "gross_return", include_sleeves: bool = False) -> dict:
    fee = 2 * (cfg["cost_bps_per_side"] + cfg["slippage_bps_per_side"]) / 10000
    net = [score * row[outcome_key] - abs(score) * fee for row, score in zip(rows, scores)]
    residual = [score * row["factor_residual_return"] - abs(score) * fee
                for row, score in zip(rows, scores)]
    active = [(row, score) for row, score in zip(rows, scores) if score != 0]
    n = len(rows)
    result = {
        "n_events": n, "n_active": len(active),
        "mean_net_event_return": sum(net) / n if n else None,
        "mean_net_factor_residual": sum(residual) / n if n and outcome_key == "gross_return" else None,
        "mean_net_event_return_ci95": _ci(rows, net, cfg) if n else None,
        "mean_net_factor_residual_ci95": _ci(rows, residual, cfg) if n and outcome_key == "gross_return" else None,
        "rank_ic": _rank_ic(scores, [r[outcome_key] for r in rows]),
        "direction_hit_rate": (sum(s * r[outcome_key] > 0 for r, s in active) / len(active)) if active else None,
        "net_positive_rate": sum(value > 0 for value in net) / n if n else None,
        "mean_round_trip_cost": fee * sum(map(abs, scores)) / n if n else None,
        "two_way_event_notional_turnover": 2 * sum(map(abs, scores)),
        "distinct_entry_dates": len({r["entry_session"] for r in rows}),
    }
    if include_sleeves:
        result["event_sleeves"] = _sleeves(rows, scores, cfg)
    return result


def _random_score(event_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{event_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / ((1 << 64) - 1) * 2 - 1


def _paired_comparison(rows: list[dict], baseline_scores: list[float], cfg: dict,
                       outcome_key: str = "gross_return") -> dict:
    fee = 2 * (cfg["cost_bps_per_side"] + cfg["slippage_bps_per_side"]) / 10000
    differences = [r["score"] * r["gross_return"] - abs(r["score"]) * fee
                   - score * r[outcome_key] + abs(score) * fee
                   for r, score in zip(rows, baseline_scores)]
    return {
        "n_events": len(rows),
        "model": _metrics(rows, [r["score"] for r in rows], cfg),
        "baseline": _metrics(rows, baseline_scores, cfg, outcome_key),
        "mean_net_advantage": sum(differences) / len(rows) if rows else None,
        "mean_net_advantage_ci95": _ci(rows, differences, cfg) if rows else None,
        "absolute_exposure_matched": all(abs(r["score"]) == abs(s) for r, s in zip(rows, baseline_scores)),
    }


def _baselines(rows: list[dict], cfg: dict) -> dict:
    scores = [r["score"] for r in rows]
    sector_rows = [r for r in rows if r["sector_return"] is not None]
    momentum_rows = [r for r in rows if r["momentum"] is not None]
    earnings_rows = [r for r in rows if r["is_earnings"] is not None]
    positive_earnings = [r for r in earnings_rows if r["is_earnings"]]
    matched = {
        "market": _paired_comparison(rows, scores, cfg, "benchmark_return"),
        "sector": _paired_comparison(sector_rows, [r["score"] for r in sector_rows], cfg, "sector_return"),
        "momentum": _paired_comparison(momentum_rows, [abs(r["score"]) *
                     (1 if r["momentum"] >= 0 else -1) for r in momentum_rows], cfg),
        "random_direction": _paired_comparison(rows, [abs(r["score"]) *
                            (1 if _random_score(r["event_id"], cfg["seed"]) >= 0 else -1) for r in rows], cfg),
        "earnings_long": _paired_comparison(positive_earnings, [abs(r["score"]) for r in positive_earnings], cfg),
    }
    for name, missing in (("market", 0), ("sector", len(rows) - len(sector_rows)),
                          ("momentum", len(rows) - len(momentum_rows)), ("random_direction", 0),
                          ("earnings_long", len(rows) - len(earnings_rows))):
        matched[name]["missing_events"] = missing
    matched["earnings_long"]["population"] = "Explicit earnings events only; non-earnings events are outside this paired comparison."
    matched["momentum"]["definition"] = "Trailing-return sign with abs(model score) exposure; exact zero trailing return breaks ties long."
    return {
        "exposure_matched_comparisons": matched,
        "descriptive_controls_note": "Legacy unit-momentum and uniform-random controls differ in exposure; use exposure_matched_comparisons for paired advantage.",
        "market_benchmark_same_exposure": _metrics(rows, scores, cfg, "benchmark_return"),
        "sector_benchmark_same_exposure": {
            **_metrics(sector_rows, [r["score"] for r in sector_rows], cfg, "sector_return"),
            "missing_events": len(rows) - len(sector_rows),
            "model_same_subset": _metrics(sector_rows, [r["score"] for r in sector_rows], cfg),
        },
        "trailing_momentum": {
            **_metrics(momentum_rows, [float(np.sign(r["momentum"])) for r in momentum_rows], cfg),
            "missing_events": len(rows) - len(momentum_rows),
            "model_same_subset": _metrics(momentum_rows, [r["score"] for r in momentum_rows], cfg),
        },
        "explicit_earnings_filter": {
            **_metrics(earnings_rows, [r["score"] if r["is_earnings"] else 0.0 for r in earnings_rows], cfg),
            "missing_events": len(rows) - len(earnings_rows),
            "definition": "Filter the frozen score to explicitly identified earnings events; 8-K alone is insufficient.",
            "model_same_subset": _metrics(earnings_rows, [r["score"] for r in earnings_rows], cfg),
        },
        "deterministic_random_ranking": _metrics(rows, [_random_score(r["event_id"], cfg["seed"]) for r in rows], cfg),
    }


def _feature_coverage(rows: list[dict]) -> dict:
    result = {}
    for feature in FEATURE_WEIGHTS:
        values = [float(row["_event"]["features"][feature]) for row in rows]
        distinct = len(set(values))
        by_entry = defaultdict(set)
        entry_counts = defaultdict(int)
        for row, value in zip(rows, values):
            by_entry[row["entry_session"]].add(value)
            entry_counts[row["entry_session"]] += 1
        varied_entry_sessions = sum(len(group) > 1 for group in by_entry.values())
        result[feature] = {
            "n_observations": len(values), "distinct_values": distinct,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "variation_status": "varied" if distinct > 1 else "constant" if distinct == 1 else "no_data",
            "sample_variation_present": distinct > 1,
            "entry_sessions_with_multiple_events": sum(count > 1 for count in entry_counts.values()),
            "entry_sessions_with_feature_variation": varied_entry_sessions,
            "cross_sectional_variation_present": varied_entry_sessions > 0,
            "interpretation": "Across-sample variation may be temporal. Cross-sectional variation requires different values among events entering the same session; variation alone does not identify a feature effect.",
        }
    return result


def _promotion_checks(events: list[dict], market: dict, horizons: dict,
                      exclusions: list[dict], cfg: dict, *,
                      historical_source_verified: bool) -> dict:
    def check(status, detail):
        return {"status": status, "detail": detail}

    temporal = [e for e in exclusions if any(term in e["reason"] for term in
                ("timestamp", "timezone", "features_available_at", "available_at", "candidate entry", "calendar coverage"))]
    checks = {
        "point_in_time": check("fail" if temporal else "pass",
            f"{len(temporal)} temporal validation exclusions; operational UTC/availability/entry checks only, not certification of vendor history."),
        "historical_source": check("pass" if historical_source_verified else "fail",
            "Current historical route requires sec_edgar events replayed from retained SEC receipts during this evaluation; another source requires an explicit verified adapter."),
        "market_replay_scope": check(
            "not_verified" if market["metadata"]["data_kind"] == "historical" else "not_applicable",
            "Supported price and factor artifacts replay from raw bytes. Session schedules, memberships, and sector controls remain schema-validated declarations pending dedicated upstream replay adapters."),
        "causal_mechanism": check("not_verified", "A reviewer must document an economically plausible mechanism and falsifiers."),
        "independent_paper_evidence": check("not_verified", "This evaluator has not verified an independent prospective paper-trading record."),
        "drift_monitoring": check("not_verified", "A live drift-monitoring and response process has not been verified."),
        "horizons": {},
    }
    for horizon, report in horizons.items():
        metrics = report["metrics"]
        interval = metrics["mean_net_factor_residual_ci95"]
        baseline_checks = {}
        for name, comparison in report["baselines"]["exposure_matched_comparisons"].items():
            ci = comparison["mean_net_advantage_ci95"]
            if comparison["missing_events"] or not comparison["n_events"] or ci is None:
                status = "inconclusive"
            elif ci[0] > 0:
                status = "pass"
            else:
                status = "fail"
            baseline_checks[name] = check(status,
                f"Matched-population, matched-absolute-exposure paired interval: {ci}; missing events: {comparison['missing_events']}.")
        statuses = [entry["status"] for entry in baseline_checks.values()]
        checks["horizons"][horizon] = {
            "sufficient_events": check("pass" if metrics["n_events"] >= cfg["min_oos_events"] else "fail",
                f"{metrics['n_events']} OOS events; required {cfg['min_oos_events']}."),
            "sufficient_folds": check("pass" if report["n_folds"] >= cfg["min_oos_folds"] else "fail",
                f"{report['n_folds']} folds; required {cfg['min_oos_folds']}."),
            "positive_residual_interval": check("inconclusive" if interval is None else "pass" if interval[0] > 0 else "fail",
                f"Preliminary entry-date-cluster interval: {interval}; serial dependence and multiple testing remain uncorrected."),
            "beats_baselines": check("pass" if all(s == "pass" for s in statuses) else
                "inconclusive" if "inconclusive" in statuses else "fail",
                "All required controls must have positive paired net-advantage intervals on their defined common populations."),
            "baseline_details": baseline_checks,
        }
    return checks


def evaluate_research(events: list[dict], market: dict, settings: dict, *,
                      historical_source_config: dict | None = None) -> dict:
    """Evaluate a historical/fixture bundle; invalid events are explicitly excluded.

    Structural market errors raise ValueError. Individual missing windows, factors,
    timestamps, or features create exclusion records. No missing returns are filled.
    The deployment gate always requires independent paper/causal/monitoring review.
    """
    cfg = _settings(settings)
    historical_source_verified = False
    data_kind = market.get("metadata", {}).get("data_kind") if isinstance(market, dict) else None
    if data_kind == "historical":
        # This public API must not rely on a mutable in-memory marker applied by
        # ``load_inputs``.  A caller that wants historical evaluation supplies
        # the full research config so the SEC cache chain is replayed here too.
        if historical_source_config is None:
            raise ValueError("historical evaluation requires historical_source_config for SEC receipt replay; use research.pipeline.run")
        from research.pipeline import verify_sec_snapshots
        verify_sec_snapshots(events, historical_source_config)
        historical_source_verified = True
    parsed = _market(market)
    validate_events(events)
    if any("score" in event for event in events):
        raise ValueError("evaluation accepts frozen feature events, not externally supplied scores")
    if data_kind == "historical":
        # SEC replay binds source text and clocks, but the public evaluator must
        # also bind the feature values used by score_event to that source. The
        # pipeline normally prepares them; direct callers cannot substitute a
        # favorable direction, novelty, or sector while retaining SEC receipts.
        derived = {row["event_id"]: row for row in
                   prepare_events(events, market, historical_source_config)}
        for event in events:
            expected = derived[event["event_id"]]
            for field in ("features", "features_available_at", "feature_version",
                          "feature_provenance", "sector", "sender"):
                if event.get(field) != expected[field]:
                    raise ValueError(f"historical derived {field} differs from the frozen feature rule")
    identifiers = [event.get("event_id") for event in events]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("every event requires a nonempty event_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate event_id")
    exclusions, horizons, all_splits = [], {}, []
    for horizon in cfg["horizons"]:
        eligible = []
        for event in events:
            try:
                eligible.append(_event_row(event, horizon, parsed, cfg))
            except (ValueError, KeyError) as exc:
                exclusions.append({"event_id": event["event_id"], "horizon": horizon, "reason": str(exc)})
        splits = _walk_forward_splits(eligible, cfg)
        oos = []
        for split in splits:
            oos.extend([{**row, "fold": split["fold"]} for row in split["test"]])
            all_splits.append({"horizon": horizon, "fold": split["fold"],
                               "train_event_ids": [r["event_id"] for r in split["train"]],
                               "test_event_ids": [r["event_id"] for r in split["test"]],
                               "purged_train_events": split["purged_train_events"],
                               "embargo_boundary_index": split["embargo_boundary_index"]})
        scores = [row["score"] for row in oos]
        metrics = _metrics(oos, scores, cfg, include_sleeves=True)
        subgroups = {}
        for field in ("sector", "source", "market_regime"):
            subgroups[field] = {}
            for name in sorted({str(r[field]) for r in oos}):
                subset = [r for r in oos if str(r[field]) == name]
                subgroups[field][name] = _metrics(subset, [r["score"] for r in subset], cfg)
        fixed_scores = [score_event({k: v for k, v in r["_event"].items() if k != "score"}) for r in oos]
        ablations = {
            "reference": "Fixed feature rule on exactly the same OOS events; external score overrides do not apply.",
            "full_feature_reference": _metrics(oos, fixed_scores, cfg),
            "drop": {name: _metrics(oos, [score_event(r["_event"], name) for r in oos], cfg)
                     for name in FEATURE_WEIGHTS},
        }
        fee = 2 * (cfg["cost_bps_per_side"] + cfg["slippage_bps_per_side"]) / 10000
        public_rows = []
        for row in oos:
            public_rows.append({**{k: v for k, v in row.items() if not k.startswith("_")},
                                "net_event_return": row["score"] * row["gross_return"] - abs(row["score"]) * fee,
                                "net_factor_residual": row["score"] * row["factor_residual_return"] - abs(row["score"]) * fee})
        horizons[str(horizon)] = {"eligible_events": len(eligible), "n_folds": len(splits),
                                  "not_oos_events": len(eligible) - len(oos), "rows": public_rows,
                                  "metrics": metrics, "baselines": _baselines(oos, cfg),
                                  "ablations": ablations, "feature_coverage": _feature_coverage(oos),
                                  "subgroups": subgroups}
    reasons = ["Independent paper-trading evidence, causal review and drift monitoring have not been verified."]
    if parsed["metadata"]["data_kind"] != "historical":
        reasons.append("Fixture data cannot establish market alpha.")
    elif parsed["integrity"]["unreplayed_components"]:
        names = ", ".join(parsed["integrity"]["unreplayed_components"])
        reasons.append(f"Upstream replay/authentication is not yet available for: {names}.")
    for horizon, report in horizons.items():
        metrics = report["metrics"]
        if metrics["n_events"] < cfg["min_oos_events"]:
            reasons.append(f"Horizon {horizon}: fewer than min_oos_events.")
        if report["n_folds"] < cfg["min_oos_folds"]:
            reasons.append(f"Horizon {horizon}: fewer than min_oos_folds.")
        interval = metrics["mean_net_factor_residual_ci95"]
        if interval is None or interval[0] <= 0:
            reasons.append(f"Horizon {horizon}: positive net factor residual is not established by the date-cluster interval.")
    return {
        "methodology": {
            "price_basis": "split/dividend adjusted close-to-close total return proxy; strict next close after availability + latency",
            "score": "Frozen feature rule; external scores are rejected; signed unit exposure, no predictor training.",
            "feature_weights": FEATURE_WEIGHTS,
            "factor_model": "OLS(stock daily return - rf) on intercept,mkt_rf,smb,hml; training closes strictly before entry; rf added back daily.",
            "factor_residual": "Realized compounded stock return minus compounded factor prediction (including historical intercept); not a regression-alpha estimate.",
            "split": "Expanding chronological groups; purge overlapping labels and embargo before each test block; same-time releases stay together.",
            "uncertainty": "95% percentile bootstrap resamples whole entry dates; does not correct cross-date serial correlation, overlapping horizons or multiple testing.",
            "costs": "Two sides of fees plus slippage charged against initial absolute event notional; no borrow, financing, impact or capacity model.",
            "earnings": "Requires explicit is_earnings; never inferred from 8-K form alone.",
            "limitations": ["Historical adjusted prices are a return basis, not executable quote prices.",
                            "Eligibility uses complete outcome windows, so missing/delisted histories can induce sample selection; inspect exclusions per horizon.",
                            "Calendar coverage and UTC session labels must describe the same exchange; missing candidate entry bars are excluded conservatively.",
                            "No portfolio optimizer, broker, execution simulator or capital allocation constraint.",
                            "Factor archives may be revised; historical provenance and vintages must be independently audited.",
                            "Score and horizon choices must be preregistered; comparison tables are not multiple-testing-adjusted evidence."],
        },
        "settings": cfg, "input_summary": {"events": len(events), "data_kind": parsed["metadata"]["data_kind"],
                                            "tickers": len(parsed["bars"]), "sessions": len(parsed["calendar"])},
        "exclusions": exclusions, "splits": all_splits, "horizons": horizons,
        "promotion_gate": {"deployable": False, "reasons": reasons,
                           "checks": _promotion_checks(
                               events, parsed, horizons, exclusions, cfg,
                               historical_source_verified=historical_source_verified)},
    }
