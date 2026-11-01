from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import load_config
from evaluation.walk_forward import (
    _event_row, _feature_coverage, _market, _settings, _walk_forward_splits,
    evaluate_research, score_event,
)
from research.features import prepare_events
from research.pipeline import DEFAULT_RESEARCH_CONFIG
from tests.test_research_pipeline import sec_snapshot_fixture


def research_fixture():
    dates = []
    day = datetime(2024, 1, 1, tzinfo=timezone.utc)
    while len(dates) < 100:
        if day.weekday() < 5:
            dates.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    rng = np.random.default_rng(7)
    price, bars, benchmark, factors = 100.0, [], [], []
    for i, session in enumerate(dates):
        mkt, smb, hml = rng.uniform(-0.01, 0.01, 3)
        rf = 0.0002 + (i % 5) * 0.00001
        stock_return = rf + 0.0003 + 1.2 * mkt + 0.4 * smb - 0.2 * hml
        if i:
            price *= 1 + stock_return
        bars.append({"ticker": "TEST", "session": session,
                     "open_at": session + "T14:30:00Z", "close_at": session + "T21:00:00Z",
                     "available_at": session + "T21:00:00Z", "adjusted_close": price})
        benchmark.append({"session": session, "return": float(mkt + rf)})
        factors.append({"session": session, "mkt_rf": float(mkt), "smb": float(smb),
                        "hml": float(hml), "rf": float(rf)})
    events = []
    for i in range(12, 80, 2):
        available = dates[i] + "T12:00:00Z"
        text = f"Fixture filing {i}"
        events.append({"event_id": f"event-{i}", "ticker": "TEST", "event_time": available,
                       "retrieved_at": "2025-01-01T00:00:00Z", "document_text": text,
                       "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "available_at": available, "features_available_at": available,
                       "source": "fixture_edgar", "sector": "technology",
                       "form": "8-K", "features": {"direction": 1 if i % 4 else -1,
                       "credibility": 0.8, "access_cost": 1.0, "novelty": 0.9,
                       "graph_propagation": 0.6, "is_earnings": i % 4 == 0}})
    sessions = [{"session": session, "open_at": session + "T14:30:00Z",
                 "close_at": session + "T21:00:00Z"} for session in dates]
    market = {"metadata": {"data_kind": "fixture", "source": "walk-forward test fixture",
                            "adjustment": "split_and_dividend", "universe_policy": "one fictional issuer",
                            "retrieved_at": "2025-01-01T00:00:00Z"},
              "session_schedule": {"version": 1, "calendar_id": "fixture-xnys-v1",
                                   "source": "test fixed UTC calendar", "retrieved_at": "2023-12-01T00:00:00Z",
                                   "sessions": sessions},
              "bars": bars, "benchmark": benchmark, "factors": factors,
              "sector_returns": [{"sector": "technology", **row} for row in benchmark]}
    settings = {"horizons": [2], "min_train_events": 2, "test_events": 3,
                "embargo_sessions": 1, "estimation_sessions": 20,
                "min_estimation_sessions": 6, "momentum_sessions": 5,
                "bootstrap_samples": 100, "min_bootstrap_dates": 5,
                "latency_seconds": 0, "cost_bps_per_side": 5,
                "slippage_bps_per_side": 5}
    return events, market, settings


class PointInTimeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.events, self.market, self.settings = research_fixture()

    def row(self, event=None, market=None):
        return _event_row(event or self.events[0], 2, _market(market or self.market),
                          _settings(self.settings))

    def test_release_at_close_executes_strictly_next_close(self):
        event = copy.deepcopy(self.events[0])
        event["available_at"] = event["available_at"].replace("12:00", "21:00")
        event["features_available_at"] = event["available_at"]
        row = self.row(event)
        self.assertEqual(row["entry_session"], self.market["bars"][13]["session"])
        self.assertEqual(row["exit_session"], self.market["bars"][15]["session"])

    def test_latency_equal_to_remaining_session_defers_execution(self):
        self.settings["latency_seconds"] = 9 * 60 * 60
        row = self.row()
        self.assertEqual(row["entry_session"], self.market["bars"][13]["session"])

    def test_late_features_are_excluded_and_naive_timestamps_fail_closed(self):
        self.events[0]["features_available_at"] = self.events[0]["available_at"].replace("12:00", "13:00")
        self.events[1]["available_at"] = self.events[1]["available_at"][:-1]
        with self.assertRaisesRegex(ValueError, "timezone"):
            evaluate_research(self.events, self.market, self.settings)
        # Once structurally invalid input is removed, event-specific feature
        # timing remains an explicit exclusion rather than a silent repair.
        self.events.pop(1)
        report = evaluate_research(self.events, self.market, self.settings)
        reasons = {row["event_id"]: row["reason"] for row in report["exclusions"]}
        self.assertIn("after event availability", reasons[self.events[0]["event_id"]])

    def test_missing_forward_factor_or_stock_bar_is_not_zero_filled(self):
        market = copy.deepcopy(self.market)
        market["factors"].pop(13)
        with self.assertRaisesRegex(ValueError, "missing factor"):
            self.row(market=market)
        market = copy.deepcopy(self.market)
        market["bars"].pop(13)
        with self.assertRaisesRegex(ValueError, "missing stock bar"):
            self.row(market=market)

    def test_missing_entry_bar_cannot_silently_defer_until_next_price(self):
        market = copy.deepcopy(self.market)
        market["bars"].pop(12)
        with self.assertRaisesRegex(ValueError, "candidate entry"):
            self.row(market=market)

    def test_missing_training_factor_is_explicitly_excluded(self):
        market = copy.deepcopy(self.market)
        market["factors"].pop(4)
        with self.assertRaisesRegex(ValueError, "pre-entry estimation"):
            self.row(market=market)

    def test_regime_and_momentum_ignore_post_decision_prices(self):
        original = self.row()
        market = copy.deepcopy(self.market)
        for bar in market["bars"][12:]:
            bar["adjusted_close"] *= 1.5
        for benchmark in market["benchmark"][12:]:
            benchmark["return"] = 0.5
        changed = self.row(market=market)
        self.assertEqual(original["market_regime"], changed["market_regime"])
        self.assertEqual(original["momentum"], changed["momentum"])

    def test_momentum_requires_availability_for_its_entire_price_window(self):
        market = copy.deepcopy(self.market)
        # The final pre-event close remains available, but the start of its
        # five-session lookback arrives after the event.  It must not be used
        # merely because every calendar date has a bar.
        market["bars"][6]["available_at"] = self.events[0]["available_at"].replace(
            "12:00:00", "13:00:00")
        self.assertIsNone(self.row(market=market)["momentum"])

    def test_regime_requires_every_benchmark_row_to_have_a_timely_clock(self):
        market = copy.deepcopy(self.market)
        # The five sessions ending at the last usable price close are known at
        # the event clock. This permits a descriptive regime label.
        for row in market["benchmark"][7:12]:
            row["available_at"] = self.events[0]["available_at"]
        self.assertIn(self.row(market=market)["market_regime"], {"up", "down", "flat"})

        # A single later benchmark observation makes the whole regime feature
        # unavailable rather than silently mixing point-in-time and ex-post data.
        market["benchmark"][9]["available_at"] = self.events[0]["available_at"].replace(
            "12:00:00", "13:00:00")
        self.assertEqual(self.row(market=market)["market_regime"],
                         "unavailable_without_point_in_time_benchmark")

    def test_missing_benchmark_and_unadjusted_prices_fail_closed(self):
        market = copy.deepcopy(self.market)
        market["benchmark"].pop(13)
        with self.assertRaisesRegex(ValueError, "missing benchmark"):
            _market(market)
        market = copy.deepcopy(self.market)
        market["metadata"]["adjustment"] = "split_only"
        with self.assertRaisesRegex(ValueError, "split_and_dividend"):
            _market(market)


class FactorModelTests(unittest.TestCase):
    def test_factors_fit_pre_entry_and_rf_is_added_back(self):
        events, market, settings = research_fixture()
        cfg = _settings(settings)
        row = _event_row(events[0], 2, _market(market), cfg)
        self.assertLess(row["factor_fit"]["last_session"], row["entry_session"])
        self.assertAlmostEqual(row["factor_fit"]["intercept_daily"], 0.0003, places=11)
        self.assertAlmostEqual(row["factor_fit"]["betas"]["mkt_rf"], 1.2, places=10)
        self.assertAlmostEqual(row["factor_fit"]["betas"]["smb"], 0.4, places=10)
        self.assertAlmostEqual(row["factor_residual_return"], 0.0, places=10)
        changed = copy.deepcopy(market)
        for factor in changed["factors"][12:]:
            factor["mkt_rf"] += 0.1
        altered = _event_row(events[0], 2, _market(changed), cfg)
        self.assertEqual(row["factor_fit"], altered["factor_fit"])
        self.assertNotAlmostEqual(row["factor_expected_return"], altered["factor_expected_return"])

    def test_rank_deficient_factor_design_is_excluded(self):
        events, market, settings = research_fixture()
        for factor in market["factors"]:
            factor["smb"] = factor["mkt_rf"]
        report = evaluate_research(events, market, settings)
        self.assertEqual(report["horizons"]["2"]["metrics"]["n_events"], 0)
        self.assertTrue(all("rank-deficient" in row["reason"] for row in report["exclusions"]))


class ChronologicalSplitTests(unittest.TestCase):
    def test_purge_removes_overlapping_labels_and_keeps_equal_times_together(self):
        events, market, settings = research_fixture()
        cfg = _settings(settings)
        parsed = _market(market)
        rows = [_event_row(event, 2, parsed, cfg) for event in events]
        duplicate = copy.deepcopy(rows[4])
        duplicate["event_id"] += "-second-issuer"
        rows.append(duplicate)
        splits = _walk_forward_splits(rows, cfg)
        self.assertTrue(splits)
        for split in splits:
            test_times = {r["available_at"] for r in split["test"]}
            self.assertFalse(test_times & {r["available_at"] for r in split["train"]})
            self.assertTrue(all(r["_exit_index"] < split["embargo_boundary_index"] for r in split["train"]))
            self.assertTrue(all(r["exit_at"] < min(t["available_at"] for t in split["test"]) for r in split["train"]))
        matching = [split for split in splits if rows[4]["event_id"] in {r["event_id"] for r in split["test"]}]
        if matching:
            self.assertIn(duplicate["event_id"], [r["event_id"] for r in matching[0]["test"]])
        self.assertTrue(any(split["purged_train_events"] > 0 for split in splits))


class ResearchMetricsTests(unittest.TestCase):
    def test_costs_scale_exposure_and_closed_gate_survives_fixture_success(self):
        events, market, settings = research_fixture()
        report = evaluate_research(events, market, settings)
        horizon = report["horizons"]["2"]
        self.assertGreater(horizon["metrics"]["n_events"], 0)
        for row in horizon["rows"]:
            self.assertAlmostEqual(row["net_event_return"],
                                   row["score"] * row["gross_return"] - abs(row["score"]) * 0.002)
        sleeves = horizon["metrics"]["event_sleeves"]
        self.assertAlmostEqual(sleeves["net_pnl_units"], sum(r["net_event_return"] for r in horizon["rows"]))
        self.assertFalse(report["promotion_gate"]["deployable"])
        self.assertTrue(any("Fixture" in reason for reason in report["promotion_gate"]["reasons"]))
        json.dumps(report, allow_nan=False)

    def test_baselines_use_explicit_earnings_and_common_ablation_population(self):
        events, market, settings = research_fixture()
        for event in events:
            event["features"].pop("is_earnings")
        report = evaluate_research(events, market, settings)
        horizon = report["horizons"]["2"]
        self.assertEqual(horizon["baselines"]["explicit_earnings_filter"]["n_events"], 0)
        self.assertEqual(horizon["baselines"]["explicit_earnings_filter"]["missing_events"], len(horizon["rows"]))
        for metric in horizon["ablations"]["drop"].values():
            self.assertEqual(metric["n_events"], len(horizon["rows"]))
        reverse = evaluate_research(list(reversed(events)), market, settings)
        self.assertEqual(horizon["baselines"]["deterministic_random_ranking"],
                         reverse["horizons"]["2"]["baselines"]["deterministic_random_ranking"])

    def test_feature_ablation_and_external_score_override_is_rejected(self):
        events, _, _ = research_fixture()
        event = events[0]
        full = score_event(event)
        reduced = score_event(event, "credibility")
        self.assertAlmostEqual(abs(full) - abs(reduced), 0.4 * 0.8)
        event["score"] = 0.123
        self.assertEqual(score_event(event), full)
        self.assertEqual(score_event(event, "credibility"), reduced)
        _, market, settings = research_fixture()
        with self.assertRaisesRegex(ValueError, "externally supplied scores"):
            evaluate_research([event], market, settings)

    def test_inadequate_cluster_count_has_no_confidence_interval(self):
        events, market, settings = research_fixture()
        settings["min_bootstrap_dates"] = 100
        report = evaluate_research(events, market, settings)
        self.assertIsNone(report["horizons"]["2"]["metrics"]["mean_net_event_return_ci95"])

    def test_promotion_controls_match_exposure_and_missing_sector_is_inconclusive(self):
        events, market, settings = research_fixture()
        market["sector_returns"] = []
        report = evaluate_research(events, market, settings)
        horizon = report["horizons"]["2"]
        paired = horizon["baselines"]["exposure_matched_comparisons"]
        for name in ("momentum", "random_direction", "earnings_long"):
            self.assertTrue(paired[name]["absolute_exposure_matched"])
            self.assertEqual(paired[name]["model"]["mean_round_trip_cost"],
                             paired[name]["baseline"]["mean_round_trip_cost"])
        checks = report["promotion_gate"]["checks"]
        self.assertEqual(checks["historical_source"]["status"], "fail")
        self.assertEqual(checks["independent_paper_evidence"]["status"], "not_verified")
        self.assertEqual(checks["horizons"]["2"]["baseline_details"]["sector"]["status"], "inconclusive")
        self.assertEqual(horizon["feature_coverage"]["credibility"]["variation_status"], "constant")
        self.assertFalse(horizon["feature_coverage"]["credibility"]["cross_sectional_variation_present"])

    def test_sample_variation_is_not_mislabeled_cross_sectional(self):
        events, _, _ = research_fixture()
        events[0]["features"]["novelty"] = 0.2
        events[1]["features"]["novelty"] = 0.8
        rows = [{"_event": events[0], "entry_session": "2024-01-02"},
                {"_event": events[1], "entry_session": "2024-01-03"}]
        coverage = _feature_coverage(rows)["novelty"]
        self.assertTrue(coverage["sample_variation_present"])
        self.assertFalse(coverage["cross_sectional_variation_present"])
        self.assertEqual(coverage["entry_sessions_with_feature_variation"], 0)
        rows[1]["entry_session"] = rows[0]["entry_session"]
        coverage = _feature_coverage(rows)["novelty"]
        self.assertTrue(coverage["cross_sectional_variation_present"])
        self.assertEqual(coverage["entry_sessions_with_feature_variation"], 1)


class HistoricalDerivedFeatureTests(unittest.TestCase):
    def test_direct_evaluator_recomputes_features_from_sec_text_and_config(self):
        _, market, settings = research_fixture()
        market["metadata"]["data_kind"] = "historical"
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config(DEFAULT_RESEARCH_CONFIG)
            cfg["research"]["edgar"]["cache_dir"] = folder
            event = sec_snapshot_fixture(Path(folder))
            prepared = prepare_events([event], market, cfg)
            # This test isolates SEC-derived feature integrity. The existing
            # market tests cover raw price/factor replay independently.
            with patch("evaluation.walk_forward.verify_historical_market_provenance",
                       return_value={"unreplayed_components": ["session_schedule"]}):
                evaluate_research(prepared, market, settings,
                                  historical_source_config=cfg)
                for field, value in (
                    ("features", {**prepared[0]["features"], "direction": 1}),
                    ("features_available_at", "2024-01-02T21:00:00Z"),
                    ("feature_version", "forged-v2"),
                    ("feature_provenance", {}),
                    ("sector", "forged-sector"),
                    ("sender", "issuer:forged"),
                ):
                    with self.subTest(field=field):
                        forged = copy.deepcopy(prepared)
                        forged[0][field] = value
                        with self.assertRaisesRegex(ValueError, f"historical derived {field}"):
                            evaluate_research(forged, market, settings,
                                              historical_source_config=cfg)


if __name__ == "__main__":
    unittest.main()
