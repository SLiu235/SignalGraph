"""
agents.simulation — the multi-agent signaling-game round loop.

For each company, the graph's players become agents. The transmitted-signal corpus
(from signaling, with game-derived credibility) seeds round 0. Each round:

  1. Each agent sees the signals its info-access tier grants (full / partial / limited),
     and, for slow agents, only prior-round signals (one-round response lag).
  2. Each agent updates its opponent model via its learning strategy.
  3. Each agent chooses a reaction action (confirm / counter / stay) from its model
     + belief; reactions become observable signals for the next round.

Each agent's FINAL strategy state (opponent models + the "analyze each other" reads)
is recorded in `states` inside the per-company trace, so result shaping reads real
learned state without replaying the loop.

Deterministic: all randomness comes from get_rng(cfg, salt="agents").
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from common import RESULTS_DIR, get_rng, read_json
from graph import ontology as O
from . import types as T
from .strategies import (
    INDEX_ACTION,
    aggregate_belief,
    choice_utility,
    logit_probs,
    make_strategy,
)


def build_agent_specs(cfg: dict, G, ticker: str, rng) -> list[T.AgentSpec]:
    """Derive AgentSpec for every player linked to a company, from graph player nodes."""
    defaults = cfg["agents"]["role_defaults"]
    pay = cfg["agents"]["payoff"]
    specs = []
    # players of this company (linked_to edges)
    players = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k != O.LINKED_TO:
            continue
        pnode = u if G.nodes[u].get("kind") == O.PLAYER else v
        if G.nodes[pnode].get("company_ref") == ticker:
            players.append(pnode)
    linked_ids = [G.nodes[p]["player_id"] for p in players]

    for pnode in players:
        pattr = dict(G.nodes[pnode])
        role = pattr["role"]
        rd = defaults.get(role, {})
        # observe channels: signal_access edges where can_observe
        obs = []
        for u2, v2, k2, a2 in G.edges(keys=True, data=True):
            if k2 != O.SIGNAL_ACCESS or u2 != pnode:
                continue
            if a2.get("can_observe") and G.nodes[v2].get("kind") == O.SIGNAL_TYPE:
                obs.append(v2.replace("stype:", ""))
        theta = {
            "truth": float(rng.uniform(*pay["theta_truth_range"])),
            "contrarian": float(rng.uniform(*pay["theta_contrarian_range"])),
            "stay": float(rng.uniform(*pay["theta_stay_range"])),
        }
        specs.append(T.AgentSpec(
            player_id=pattr["player_id"], role=role, company_ref=ticker,
            strategy=str(rd.get("strategy", T.STRATEGY_IRL)),
            info_access=str(rd.get("info_access", T.ACCESS_PARTIAL)),
            response_time=str(rd.get("response_time", T.RESPONSE_FAST)),
            credibility_prior=float(pattr["credibility_prior"]),
            informed_prior=float(pattr["informed_prior"]),
            aligned_prior=float(pattr["aligned_prior"]),
            observe_channels=obs, linked_agents=linked_ids, true_theta=theta))
    return specs


def _direction(content: str) -> int:
    """Reuse the deterministic sentiment heuristic from signaling.belief_update."""
    from signaling.belief_update import _direction as _d
    return _d(content)


def load_seed_signals(cfg: dict, G, ticker: str,
                      credibility_records: list[dict] | None = None,
                      excluded_sender_roles: set[str] | None = None) -> list[T.ObservedSignal]:
    """Transmitted signals with game credibility; injection avoids artifact IO.

    Counterfactual callers supply recomputed records and may silence specified
    sender roles while preserving their agents and paired random action draws.
    """
    if credibility_records is None:
        cred_path = RESULTS_DIR / "signaling" / "credibility.json"
        try:
            credibility_records = read_json(cred_path)["signals"]
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Agent simulation requires signaling credibility output. Run `make signaling` "
                "or invoke `make agents` so its prerequisites run first."
            ) from exc
    cred_by_id = {s["signal_id"]: s["credibility"] for s in credibility_records}
    excluded_sender_roles = excluded_sender_roles or set()
    out = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k != O.TRANSMITS:
            continue
        s = G.nodes[v]
        if s.get("company") != ticker:
            continue
        if G.nodes[u].get("role") in excluded_sender_roles:
            continue
        if s["signal_id"] not in cred_by_id:
            raise ValueError(f"missing credibility for transmitted signal {s['signal_id']}")
        out.append(T.ObservedSignal(
            round=0, signal_id=s["signal_id"], sender=s["sender"],
            role=G.nodes[u].get("role"), signal_type=s["signal_type"],
            kind=s["kind_of_signal"], content=s["content"],
            direction=_direction(s["content"]),
            credibility=cred_by_id[s["signal_id"]],
            is_reaction=False, timestamp=s.get("timestamp", "")))
    return out


def _filter_access(spec: T.AgentSpec, signals: list[T.ObservedSignal]) -> list[T.ObservedSignal]:
    """Apply the agent's info-access tier to a view of signals/reactions."""
    if spec.info_access == T.ACCESS_FULL:
        return signals
    if spec.info_access == T.ACCESS_LIMITED:
        return [s for s in signals
                if not s.is_reaction and s.signal_type in spec.observe_channels]
    # partial: own observe channels + reactions of same-company agents
    out = []
    for s in signals:
        if s.signal_type in spec.observe_channels:
            out.append(s)
        elif s.is_reaction and s.sender in spec.linked_agents:
            out.append(s)
    return out


def realize_action(intent: T.Reaction, spec: T.AgentSpec, cfg: dict, rng,
                   timestamp: str) -> T.Reaction:
    """Convert a strategy's intended belief direction into a realized action.

    The latent payoff parameters now genuinely generate observed behaviour under the
    documented logit model. This makes IRL's parameter-recovery error interpretable:
    it is limited by finite observations, partial information, and the configured grid,
    rather than comparing an inference target to an unrelated random tag.
    """
    temperature = float(cfg["agents"].get("payoff", {}).get(
        "temperature", cfg["agents"].get("temperature", 0.5)
    ))
    belief = intent.direction
    probs = logit_probs(choice_utility(spec.true_theta, belief), temperature)
    action = INDEX_ACTION[int(rng.choice(len(probs), p=probs))]
    word = "confirms" if action > 0 else "counters" if action < 0 else "stays"
    context = "bullish" if belief > 0 else "bearish" if belief < 0 else "neutral"
    content = (f"{spec.role}@{spec.company_ref} reaction (round {intent.round}): "
               f"public belief {context}; {word} the signal; "
               f"choice probability {probs[[1, -1, 0].index(action)] * 100:.0f}%.")
    return replace(
        intent,
        action_type="react" if action else "stay",
        direction=action,
        conviction=float(probs[[1, -1, 0].index(action)]),
        content=content,
        belief_context=belief,
        timestamp=timestamp,
    )


def run_company(cfg: dict, G, ticker: str, rng,
                credibility_records: list[dict] | None = None,
                excluded_sender_roles: set[str] | None = None) -> dict:
    """Run the round loop for one company's agent population."""
    n_rounds = int(cfg["agents"]["n_rounds"])
    specs = build_agent_specs(cfg, G, ticker, rng)
    agents = []
    for spec in specs:
        strat = make_strategy(spec.strategy, spec, cfg)
        strat._specs = specs   # for IRL true_theta_of lookups
        agents.append(strat)

    excluded_sender_roles = excluded_sender_roles or set()
    all_signals = load_seed_signals(cfg, G, ticker, credibility_records,
                                    excluded_sender_roles)   # round 0
    trace_rounds = []
    for r in range(1, n_rounds + 1):
        reaction_day = int(cfg["evaluation"]["brief_day"]) + r * int(cfg["agents"]["round_step_days"])
        reaction_timestamp = f"D{reaction_day:04d}"
        available = [s for s in all_signals if s.round <= r - 1]
        round_obs = {}
        round_reactions = []
        for strat in agents:
            view = available
            if strat.spec.response_time == T.RESPONSE_SLOW:
                view = [s for s in available if s.round <= r - 2]
            visible = _filter_access(strat.spec, view)
            strat.learn(visible)
            belief_sign, conv = aggregate_belief(visible)
            intent = strat.react(belief_sign, conv, visible, r)
            reac = realize_action(intent, strat.spec, cfg, rng, reaction_timestamp)
            round_reactions.append(reac)
            round_obs[strat.spec.player_id] = [
                {"round": s.round, "signal_id": s.signal_id, "sender": s.sender,
                 "signal_type": s.signal_type, "kind": s.kind, "direction": s.direction,
                 "credibility": s.credibility, "is_reaction": s.is_reaction,
                 "belief_context": s.belief_context, "timestamp": s.timestamp}
                for s in visible]
        # record reactions as observable signals for the next round
        for reac in round_reactions:
            if reac.role not in excluded_sender_roles:
                all_signals.append(reac.as_observed())
        trace_rounds.append({
            "round": r, "timestamp": reaction_timestamp,
            "observations": round_obs,
            "reactions": [{"agent": a.agent, "role": a.role, "action_type": a.action_type,
                           "direction": a.direction, "belief_context": a.belief_context,
                           "timestamp": a.timestamp,
                           "conviction": round(a.conviction, 4),
                           "signal_type": a.signal_type, "credibility": round(a.credibility, 4)}
                          for a in round_reactions],
        })

    # record each agent's final strategy state (opponent models + analysis reads)
    states = {}
    for strat in agents:
        states[strat.spec.player_id] = {
            "role": strat.spec.role, "strategy": strat.spec.strategy,
            "info_access": strat.spec.info_access, "response_time": strat.spec.response_time,
            "market_posterior": getattr(strat, "market_posterior", None),
            "n_observations": len(strat.observations),
            "opponent_models": {pid: _model_summary(strat, pid)
                                for pid in strat.opponent_models},
            "analysis": strat.analyze_opponents(),
        }
    return {"ticker": ticker, "agents": [a.spec.player_id for a in agents],
            "rounds": trace_rounds, "states": states}


def run_simulation(cfg: dict, G, credibility_records: list[dict] | None = None,
                   excluded_sender_roles: set[str] | None = None) -> dict:
    """Run the multi-agent simulation over every company."""
    rng = get_rng(cfg, salt="agents")
    companies = sorted({G.nodes[n]["ticker"] for n in G.nodes if G.nodes[n].get("kind") == "company"})
    per_company = {}
    for ticker in companies:
        per_company[ticker] = run_company(cfg, G, ticker, rng, credibility_records,
                                          excluded_sender_roles)
    return {"run_id": cfg["run_id"], "n_rounds": cfg["agents"]["n_rounds"],
            "companies": per_company}


# ---- result shaping (reads the recorded states, no replay needed) ----
def agent_roster(cfg, G, rng) -> list[dict]:
    tickers = sorted({G.nodes[n]["ticker"] for n in G.nodes if G.nodes[n].get("kind") == "company"})
    roster = []
    for t in tickers:
        for spec in build_agent_specs(cfg, G, t, rng):
            roster.append({
                "player_id": spec.player_id, "role": spec.role, "company": spec.company_ref,
                "strategy": spec.strategy, "info_access": spec.info_access,
                "response_time": spec.response_time,
                "priors": {"credibility": spec.credibility_prior,
                           "informed": spec.informed_prior, "aligned": spec.aligned_prior},
                "observe_channels": spec.observe_channels,
                "true_theta": {k: round(v, 3) for k, v in spec.true_theta.items()},
            })
    return roster


def _model_summary(strat, pid: str) -> dict:
    m = strat.opponent_models[pid]
    if strat.name == T.STRATEGY_IRL:
        return {"theta": m.get("theta"), "n": m.get("n", 0)}
    if strat.name == T.STRATEGY_BAYES:
        return {"type_posterior": m.get("type_posterior"), "n": m.get("n", 0)}
    if strat.name == T.STRATEGY_FP:
        return {"dist": m.get("dist"), "n": m.get("n", 0)}
    return {"n": m.get("n", 0)}


def learning_state(sim: dict) -> dict:
    """Per-agent strategy state: market posterior, opponent models, reaction tally."""
    out = {}
    for t, run in sim["companies"].items():
        for pid, st in run["states"].items():
            out[pid] = {
                "company": t, "role": st["role"], "strategy": st["strategy"],
                "market_posterior": st.get("market_posterior"),
                "n_observations": st["n_observations"],
                "n_opponents_modeled": len(st["opponent_models"]),
                "opponent_models": st["opponent_models"],
            }
    return out


def strategy_analysis(sim: dict) -> dict:
    """The 'analyze each other's strategies' deliverable: per agent, per opponent read."""
    out = {}
    for t, run in sim["companies"].items():
        for pid, st in run["states"].items():
            out[pid] = {
                "company": t, "role": st["role"], "strategy": st["strategy"],
                "info_access": st["info_access"], "response_time": st["response_time"],
                "opponents": st["analysis"],
            }
    return out


def comparison(sim: dict, cfg: dict) -> dict:
    """Aggregate learning-quality metrics across strategies (IRL fidelity, reaction
    rates by role / access / response tier)."""
    by_strategy = defaultdict(lambda: {
        "theta_error": 0.0, "n_irl": 0, "n_agents": 0,
        "actions": 0, "reactions": 0,
    })
    by_tier = defaultdict(lambda: {"actions": 0, "reactions": 0, "agents": 0})
    by_response = defaultdict(lambda: {"actions": 0, "reactions": 0, "agents": 0})
    n_agents = 0
    for t, run in sim["companies"].items():
        for rnd in run["rounds"]:
            for a in rnd["reactions"]:
                st = run["states"][a["agent"]]
                for bucket in (by_tier[st["info_access"]], by_response[st["response_time"]],
                               by_strategy[st["strategy"]]):
                    bucket["actions"] += 1
                    if a["direction"] != 0:
                        bucket["reactions"] += 1
        for pid, st in run["states"].items():
            n_agents += 1
            s = by_strategy[st["strategy"]]
            s["n_agents"] += 1
            by_tier[st["info_access"]]["agents"] += 1
            by_response[st["response_time"]]["agents"] += 1
            if st["strategy"] == T.STRATEGY_IRL:
                for opp in st["analysis"].values():
                    if opp.get("theta_error") is not None:
                        s["theta_error"] += opp["theta_error"]
                        s["n_irl"] += 1
    for k, v in by_strategy.items():
        if v["n_irl"]:
            v["theta_error"] = round(v["theta_error"] / v["n_irl"], 3)
        v["reaction_rate"] = round(v["reactions"] / (v["n_agents"] * sim["n_rounds"]), 3) if v["n_agents"] else 0.0
    for buckets in (by_tier, by_response):
        for v in buckets.values():
            v["reaction_rate"] = round(v["reactions"] / (v["agents"] * sim["n_rounds"]), 3) if v["agents"] else 0.0
    return {
        "n_agents": n_agents,
        "by_strategy": dict(by_strategy),
        "by_info_access": {k: dict(v) for k, v in by_tier.items()},
        "by_response_time": {k: dict(v) for k, v in by_response.items()},
    }
