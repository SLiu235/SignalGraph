"""Paired synthetic-agent counterfactuals; no price or profitability mechanism.

All scenarios for a seed share its generated graph, latent agent payoffs and
random action draws. Only the declared communication/cost intervention changes.
The original graph and signaling artifacts are neither read nor overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import statistics

from common import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config, log, write_json
from graph import ontology as O
from graph.build_graph import assemble_graph
from signaling.signal_game import score_signals
from agents.simulation import run_simulation


def scenario_definitions(roles: list[str]) -> list[dict]:
    return [
        {"name": "base", "description": "Configured access, response times and sender costs."},
        {"name": "everyone_full_fast", "info_access": "full", "response_time": "fast",
         "description": "Every agent receives all information without a response-round lag."},
        {"name": "everyone_slow", "response_time": "slow",
         "description": "Every agent has a one-round response lag; existing slow agents remain slow."},
        {"name": "cost_half", "cost_multiplier": 0.5,
         "description": "Multiply sender access costs by 0.5 and recompute equilibrium credibility."},
        {"name": "cost_double", "cost_multiplier": 2.0,
         "description": "Multiply sender access costs by 2 and recompute equilibrium credibility."},
    ] + [{"name": f"silence_{role}", "excluded_sender_roles": [role],
          "description": f"Suppress {role} seed/reaction communication; retain its agents and action draws."}
         for role in sorted(roles)]


def apply_intervention(cfg: dict, graph, scenario: dict):
    """Copy inputs and alter exactly the declared intervention, without IO."""
    changed_cfg, changed_graph = copy.deepcopy(cfg), graph.copy()
    for role in O.ROLES:
        defaults = changed_cfg["agents"]["role_defaults"].setdefault(role, {})
        for field in ("info_access", "response_time"):
            if field in scenario:
                defaults[field] = scenario[field]
    multiplier = float(scenario.get("cost_multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("cost_multiplier must be positive")
    changed_edges = 0
    if multiplier != 1:
        for _, _, _, attrs in changed_graph.edges(keys=True, data=True):
            if attrs.get("relation") == O.SIGNAL_ACCESS and attrs.get("can_transmit"):
                attrs["transmit_cost"] *= multiplier
                changed_edges += 1
    records = score_signals(changed_graph, changed_cfg)
    excluded = set(scenario.get("excluded_sender_roles", []))
    visible = [record for record in records if record["role"] not in excluded]
    controls = {
        "cost_multiplier": multiplier, "scaled_transmit_edges": changed_edges,
        "seed_signals_total": len(records), "seed_signals_visible": len(visible),
        "seed_signals_suppressed": len(records) - len(visible),
        "equilibrium_counts": dict(Counter(record["equilibrium"] for record in visible)),
        "mean_seed_credibility": statistics.mean(record["credibility"] for record in visible) if visible else None,
        "excluded_sender_roles": sorted(excluded),
    }
    return changed_cfg, changed_graph, records, excluded, controls


def summarize_simulation(sim: dict, baseline: dict | None = None) -> dict:
    decisions, consensus, theta_errors, observations = [], [], [], 0
    for ticker, company in sim["companies"].items():
        for rnd in company["rounds"]:
            actions = rnd["reactions"]
            if actions:
                counts = Counter(action["direction"] for action in actions)
                consensus.append(max(counts.values()) / len(actions))
            decisions.extend({"company": ticker, "round": rnd["round"],
                              "agent": action["agent"], "role": action["role"],
                              "direction": action["direction"],
                              "belief_context": action["belief_context"]}
                             for action in actions)
        for state in company["states"].values():
            observations += state["n_observations"]
            theta_errors.extend(model["theta_error"] for model in state["analysis"].values()
                                if model.get("theta_error") is not None)
    indexed = {(row["company"], row["round"], row["agent"]): row["direction"] for row in decisions}
    if baseline is None:
        disagreement = 0.0
    else:
        base = {(row["company"], row["round"], row["agent"]): row["direction"]
                for row in baseline["decisions"]}
        if indexed.keys() != base.keys():
            raise ValueError("paired scenarios must preserve agent-round decision coordinates")
        disagreement = sum(value != base[key] for key, value in indexed.items()) / len(indexed) if indexed else 0.0
    return {"metrics": {
        "actions": len(decisions),
        "reaction_rate": sum(row["direction"] != 0 for row in decisions) / len(decisions) if decisions else 0.0,
        "action_disagreement": disagreement,
        "consensus": statistics.mean(consensus) if consensus else None,
        "theta_error": statistics.mean(theta_errors) if theta_errors else None,
        "theta_comparisons": len(theta_errors), "unique_agent_observations": observations,
    }, "decisions": decisions}


def _distribution(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def run_counterfactuals(cfg: dict, seeds: list[int] | None = None) -> dict:
    """Run paired experiments entirely in memory and return JSON-ready results."""
    if cfg["data"].get("mode", "synthetic") != "synthetic":
        raise ValueError("agent counterfactuals currently require synthetic data.mode")
    if int(cfg["agents"].get("n_rounds", 0)) < 1:
        raise ValueError("counterfactuals require at least one agent round")
    configured = cfg["agents"].get("counterfactuals", {}).get("seeds")
    seeds = seeds if seeds is not None else configured
    seeds = list(seeds) if seeds is not None else [int(cfg["seed"]) + offset for offset in range(3)]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("counterfactuals require at least three distinct nonnegative integer seeds")
    runs, definitions = [], None
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"], seed_cfg["verbose"] = seed, False
        graph = assemble_graph(seed_cfg)
        roles = sorted({attrs["role"] for _, attrs in graph.nodes(data=True) if attrs.get("kind") == O.PLAYER})
        definitions = scenario_definitions(roles)
        baseline = None
        scenarios = {}
        for scenario in definitions:
            changed_cfg, changed_graph, records, excluded, controls = apply_intervention(seed_cfg, graph, scenario)
            sim = run_simulation(changed_cfg, changed_graph, credibility_records=records,
                                 excluded_sender_roles=excluded)
            result = summarize_simulation(sim, baseline)
            result["controls"] = controls
            scenarios[scenario["name"]] = result
            if scenario["name"] == "base":
                baseline = result
        runs.append({"seed": seed, "scenarios": scenarios})
    metrics = ("reaction_rate", "action_disagreement", "consensus", "theta_error", "unique_agent_observations")
    distributions = {}
    for scenario in definitions:
        name = scenario["name"]
        distributions[name] = {}
        for metric in metrics:
            values, deltas = [], []
            for run in runs:
                value = run["scenarios"][name]["metrics"][metric]
                base = run["scenarios"]["base"]["metrics"][metric]
                if value is not None:
                    values.append(value)
                if value is not None and base is not None:
                    deltas.append(value - base)
            distributions[name][metric] = {"level": _distribution(values), "paired_delta": _distribution(deltas)}
    return {"run_id": cfg["run_id"], "data_kind": "synthetic_simulation", "seeds": seeds,
            "scope": "Behavioral mechanism sensitivity only; no price formation, returns, or profitability model.",
            "pairing": "Same per-seed graph, agent ordering, payoff draws and action random stream for every scenario.",
            "cost_mechanism": "Access costs change equilibrium credibility and observed agent information; action utility has no direct transmission-cost term.",
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "metric_definitions": {
                "reaction_rate": "Nonzero actions / all agent-round actions.",
                "action_disagreement": "Fraction of agent-round decisions differing from the paired base.",
                "consensus": "Mean largest action-category share per company-round, including stay.",
                "theta_error": "Mean recorded IRL payoff-parameter error; null when no estimates exist.",
                "unique_agent_observations": "Sum of each agent's deduplicated observation count; shared evidence counts once per recipient.",
                "sample_std": "Sample standard deviation across seeds; not a significance test or confidence interval.",
            }, "scenarios": definitions, "paired_distributions": distributions, "runs": runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seeds", nargs="+", type=int, help="At least three distinct seeds; default master seed + 0,1,2")
    parser.add_argument("--output", default=str(RESULTS_DIR / "agents" / "counterfactuals.json"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = run_counterfactuals(cfg, args.seeds)
    write_json(args.output, result)
    log(cfg, f"counterfactuals: {len(result['seeds'])} paired seeds x {len(result['scenarios'])} scenarios -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
