from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

from agents import types as agent_types
from agents.simulation import realize_action
from agents.strategies import BayesStrategy
from common import DEFAULT_CONFIG_PATH, load_config
from data import _edgar, ingest, seed_graph
from evaluation.decision_bench import directional_confidence, impact_direction
from extraction import retrieve
from graph import ontology as O
from graph.build_graph import assemble_graph
from signaling.signal_game import characterize_equilibrium
from signaling.types import SignalTypeSpec


def config() -> dict:
    """A fresh mutable copy of the checked-in offline configuration."""
    return copy.deepcopy(load_config(DEFAULT_CONFIG_PATH))


class SeedConfigurationTests(unittest.TestCase):
    def test_requested_player_count_is_honored(self):
        cfg = config()
        cfg["data"]["num_companies"] = 1
        cfg["data"]["num_players_per_company"] = 3
        companies, players, _ = seed_graph.build_seed_records(cfg)

        self.assertEqual(len(companies), 1)
        self.assertEqual(len(players), 3)
        self.assertEqual([p.role for p in players], O.ROLES[:3])

    def test_invalid_synthetic_universe_sizes_fail_fast(self):
        cfg = config()
        cfg["data"]["num_companies"] = len(seed_graph.SEED_COMPANIES) + 1
        with self.assertRaisesRegex(ValueError, "num_companies"):
            seed_graph.build_seed_records(cfg)

        cfg = config()
        cfg["data"]["num_players_per_company"] = 0
        with self.assertRaisesRegex(ValueError, "num_players_per_company"):
            seed_graph.build_seed_records(cfg)

    def test_incremental_feed_stays_within_configured_universe(self):
        cfg = config()
        cfg["data"]["num_companies"] = 1
        signals, events = ingest._feed_poll(cfg, {"sources": {}}, "feed")
        self.assertTrue(signals)
        self.assertTrue(events)
        self.assertEqual({s.company for s in signals}, {"AAPL"})
        self.assertEqual({e.company for e in events}, {"AAPL"})

    def test_live_initial_build_fails_instead_of_silently_using_synthetic_data(self):
        cfg = config()
        cfg["data"]["mode"] = "edgar"
        with self.assertRaisesRegex(ValueError, "initial graph build"):
            assemble_graph(cfg)


class RetrievalSafetyTests(unittest.TestCase):
    def test_events_do_not_get_misread_as_signals(self):
        cfg = config()
        G = assemble_graph(cfg)
        O.add_event(G, O.Event(
            event_id="test-event", event_type="price_move", timestamp="D0010",
            company="AAPL", payload={"move_pct": 0.03},
        ))

        subgraph = retrieve.retrieve_subgraph(G, "AAPL", 40)
        self.assertTrue(subgraph["signals"])
        self.assertNotIn("test-event", {s["signal_id"] for s in subgraph["signals"]})

    def test_non_synthetic_timestamp_cannot_silently_bypass_cutoff(self):
        with self.assertRaises(ValueError):
            retrieve.day_of("2026-01-01")


class ModelSemanticsTests(unittest.TestCase):
    def test_equilibrium_characterization_is_config_explicit(self):
        cfg = config()
        costly = SignalTypeSpec("certified_filing", "costly", "filings", 2.0)
        eq = characterize_equilibrium(costly, 2.0, cfg)
        self.assertEqual(eq["regime"], "separating")

        cfg["signaling"]["cost"]["informed_cost_mult"] = 1.0
        eq = characterize_equilibrium(costly, 2.0, cfg)
        self.assertEqual(eq["regime"], "pooling")

    def test_directional_confidence_is_symmetric_for_bearish_calls(self):
        self.assertEqual(impact_direction(0.01, 0.02), 0)
        self.assertEqual(impact_direction(-0.03, 0.02), -1)
        self.assertEqual(directional_confidence(0.8, 1), 0.8)
        self.assertAlmostEqual(directional_confidence(0.8, -1), 0.2)
        self.assertEqual(directional_confidence(0.8, 0), 0.5)


class EdgarParserTests(unittest.TestCase):
    def test_sec_index_and_nested_submission_records_are_parsed(self):
        cfg = config()
        cfg["data"]["edgar"]["user_agent"] = "SignalGraph test contact@example.com"

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Requests:
            urls = []

            @classmethod
            def get(cls, url, headers):
                cls.urls.append(url)
                if url == cfg["data"]["edgar"]["submissions_url"]:
                    return Response({"0": {"ticker": "AAPL", "cik_str": 320193}})
                return Response({"filings": {"recent": {
                    "form": ["10-K", "S-1", "8-K"],
                    "filingDate": ["2026-01-02", "2026-01-03", "2026-01-04"],
                    "accessionNumber": ["a", "ignored", "b"],
                }}})

        with patch.object(_edgar, "_require_requests", return_value=Requests), \
             patch.object(_edgar.time, "sleep"):
            rows = _edgar._fetch_raw(cfg, ["AAPL"])

        self.assertEqual([row["accn"] for row in rows], ["a", "b"])
        self.assertIn("CIK0000320193.json", Requests.urls[-1])


class AgentLearningTests(unittest.TestCase):
    def test_bayes_strategy_consumes_each_neutral_observation_once(self):
        cfg = config()
        spec = agent_types.AgentSpec(
            player_id="agent", role="analyst", company_ref="AAPL",
            strategy="bayes", info_access="full", response_time="fast",
        )
        strategy = BayesStrategy(spec, cfg)
        neutral = agent_types.ObservedSignal(
            round=0, signal_id="neutral", sender="other", role="analyst",
            signal_type="commentary", kind="cheap_talk", content="neutral",
            direction=0, credibility=0.9,
        )
        strategy.learn([neutral])
        strategy.learn([neutral])

        self.assertEqual(len(strategy.observations), 1)
        self.assertEqual(strategy.market_posterior, 0.5)

    def test_payoff_policy_generates_a_traceable_action(self):
        cfg = config()
        spec = agent_types.AgentSpec(
            player_id="agent", role="analyst", company_ref="AAPL",
            strategy="bayes", info_access="full", response_time="fast",
            true_theta={"truth": 2.0, "contrarian": 0.0, "stay": 0.0},
        )
        intent = agent_types.Reaction(
            round=1, agent="agent", role="analyst", company_ref="AAPL",
            action_type="react", direction=1, conviction=1.0,
            signal_type="commentary", content="intent", credibility=0.5,
        )
        realized = realize_action(intent, spec, cfg, np.random.default_rng(7), "D0043")

        self.assertEqual(realized.belief_context, 1)
        self.assertEqual(realized.timestamp, "D0043")
        self.assertIn(realized.direction, (-1, 0, 1))
        self.assertIn(realized.action_type, ("react", "stay"))


class RunIdentityTests(unittest.TestCase):
    def test_environment_run_id_is_honored_when_config_is_blank(self):
        with patch.dict(os.environ, {"SIGNAL_GRAPH_RUN_ID": "test-run"}, clear=False):
            cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg["run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
