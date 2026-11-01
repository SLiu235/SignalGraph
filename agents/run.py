"""
agents.run — stage: run the multi-agent signaling-game simulation and write results.

Inputs : config/config.yaml (agents.*), the knowledge graph, and results/signaling/
         (credibility.json for seed-signal credibility).
Outputs: results/agents/
           agents.json              — roster: roles, strategies, info access, response time, true payoff
           simulation.json          — per-company round-by-round observations + reactions
           learning.json            — per-agent strategy state (market posterior, opponent models)
           strategy_analysis.json   — the "analyze each other's strategies" deliverable
           comparison.json          — learning-quality metrics across strategies / tiers
"""
from __future__ import annotations

import argparse
import sys

from common import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config, log, write_json
from graph import storage
from . import simulation as S


def main(argv=None):
    ap = argparse.ArgumentParser(description="multi-agent signaling-game simulation")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--stage", default="agents")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.get("agents", {}).get("enabled", True):
        log(cfg, "agents stage disabled (agents.enabled=false); nothing to do")
        return 0

    G = storage.open_graph(cfg)
    sim = S.run_simulation(cfg, G)

    out = RESULTS_DIR / "agents"
    rng = S.get_rng(cfg, salt="agents")

    roster = S.agent_roster(cfg, G, rng)
    write_json(out / "agents.json", {"run_id": cfg["run_id"], "agents": roster})

    write_json(out / "simulation.json", sim)

    learning = S.learning_state(sim)
    write_json(out / "learning.json", {"run_id": cfg["run_id"], "agents": learning})

    analysis = S.strategy_analysis(sim)
    write_json(out / "strategy_analysis.json", {"run_id": cfg["run_id"], "agents": analysis})

    comp = S.comparison(sim, cfg)
    write_json(out / "comparison.json", {"run_id": cfg["run_id"], **comp})

    n_agents = len(roster)
    actions = [a for run in sim["companies"].values() for rnd in run["rounds"]
               for a in rnd["reactions"]]
    n_actions = len(actions)
    n_react = sum(a["direction"] != 0 for a in actions)
    by_strat = {k: f"err={v['theta_error']}" for k, v in comp["by_strategy"].items()}
    log(cfg, f"ran {n_agents} agents over {len(sim['companies'])} companies, "
             f"{cfg['agents']['n_rounds']} rounds; {n_react} non-stay reactions "
             f"out of {n_actions} actions; "
             f"by-strategy IRL theta_error: {by_strat}")
    log(cfg, f"wrote -> {out}/agents.json, simulation.json, learning.json, "
             f"strategy_analysis.json, comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
