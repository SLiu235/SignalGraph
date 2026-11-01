from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from agents.counterfactuals import apply_intervention, run_counterfactuals, scenario_definitions
from agents.simulation import load_seed_signals, run_simulation
from common import DEFAULT_CONFIG_PATH, load_config
from graph.build_graph import assemble_graph
from signaling.signal_game import score_signals


def config():
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["verbose"] = False
    cfg["data"]["num_companies"] = 1
    cfg["data"]["num_players_per_company"] = 2
    cfg["agents"]["n_rounds"] = 2
    return cfg


class CounterfactualTests(unittest.TestCase):
    def test_injected_credibility_avoids_artifacts_and_requires_coverage(self):
        cfg = config()
        graph = assemble_graph(cfg)
        records = score_signals(graph, cfg)
        with patch("agents.simulation.read_json", side_effect=AssertionError("artifact read")):
            sim = run_simulation(cfg, graph, credibility_records=records)
            self.assertEqual(len(sim["companies"]["AAPL"]["rounds"]), 2)
            with self.assertRaisesRegex(ValueError, "missing credibility"):
                load_seed_signals(cfg, graph, "AAPL", credibility_records=[])

    def test_cost_intervention_recomputes_scores_without_mutating_inputs(self):
        cfg = config()
        graph = assemble_graph(cfg)
        initial = score_signals(graph, cfg)
        scenario = {"name": "cost_double", "cost_multiplier": 2.0}
        changed, modified, records, _, controls = apply_intervention(cfg, graph, scenario)
        self.assertGreater(controls["scaled_transmit_edges"], 0)
        by_id = {row["signal_id"]: row for row in records}
        for row in initial:
            self.assertAlmostEqual(by_id[row["signal_id"]]["access_transmit_cost"], 2 * row["access_transmit_cost"])
        self.assertEqual(score_signals(graph, cfg), initial)
        self.assertEqual(changed["agents"]["payoff"], cfg["agents"]["payoff"])
        self.assertNotEqual(records, initial)

    def test_sender_ablation_suppresses_seed_and_reaction_communication_only(self):
        cfg = config()
        graph = assemble_graph(cfg)
        records = score_signals(graph, cfg)
        sim = run_simulation(cfg, graph, records, {"insider"})
        run = sim["companies"]["AAPL"]
        insider_ids = {pid for pid, state in run["states"].items() if state["role"] == "insider"}
        for rnd in run["rounds"]:
            self.assertEqual(len(rnd["reactions"]), 2)
            for visible in rnd["observations"].values():
                self.assertTrue(all(row["sender"] not in insider_ids for row in visible))

    def test_paired_design_retains_coordinates_and_zero_base_disagreement(self):
        cfg = config()
        with patch("agents.simulation.read_json", side_effect=AssertionError("artifact read")):
            result = run_counterfactuals(cfg, [11, 12, 13])
        self.assertEqual(len(result["scenarios"]), 7)
        for run in result["runs"]:
            base = run["scenarios"]["base"]
            self.assertEqual(base["metrics"]["action_disagreement"], 0)
            for scenario in run["scenarios"].values():
                self.assertEqual(scenario["metrics"]["actions"], base["metrics"]["actions"])
            self.assertGreater(run["scenarios"]["silence_insider"]["controls"]["seed_signals_suppressed"], 0)
        self.assertEqual(result["paired_distributions"]["base"]["reaction_rate"]["paired_delta"]["mean"], 0)
        self.assertEqual(cfg["seed"], load_config(DEFAULT_CONFIG_PATH)["seed"])

    def test_full_fast_and_slow_controls_are_applied(self):
        cfg = config()
        graph = assemble_graph(cfg)
        for scenario in scenario_definitions([])[1:3]:
            changed, _, _, _, _ = apply_intervention(cfg, graph, scenario)
            for defaults in changed["agents"]["role_defaults"].values():
                self.assertEqual(defaults["response_time"], scenario["response_time"])
                if "info_access" in scenario:
                    self.assertEqual(defaults["info_access"], "full")

    def test_live_mode_and_insufficient_paired_seeds_are_rejected(self):
        cfg = config()
        with self.assertRaisesRegex(ValueError, "three distinct"):
            run_counterfactuals(cfg, [1, 1, 2])
        cfg["data"]["mode"] = "edgar"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            run_counterfactuals(cfg)


if __name__ == "__main__":
    unittest.main()
