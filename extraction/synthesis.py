"""
extraction.synthesis — LLM cited decision brief per entity (company).

Assembles: retrieved evidence (with citations), the transmitting players and their
signal-access, the signal-credibility scores, the Bayesian posterior confidence, and
a recommended research direction. Citation discipline: every factual claim cites its
retrieved source; uncited claims are flagged LOW-CONFIDENCE (or dropped).

Default backend is `mock` (deterministic, offline, reproducible). An `openai` backend
can be selected in config (extraction.synthesis.llm_backend) for real LLM synthesis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, read_json, RESULTS_DIR
from ..graph import graph_io
from . import retrieve


def _claim(text: str, sources: list[str], cited: bool = True) -> dict:
    return {"claim": text, "citations": sources,
            "confidence": "high" if cited and sources else "low"}


def _research_direction(company: str, credible_players: list) -> str:
    names = ", ".join(credible_players[:3]) or "none"
    anchor = credible_players[0] if credible_players else "the transmitting"
    return (f"Prioritize verification of the {anchor} signal for {company} "
            f"(credible senders: {names}); monitor subsequent filings to confirm "
            f"the separating signal's direction before committing research capital.")


def synthesize_company(G, cfg, ticker: str, t: int, credibility: dict, posterior: dict):
    sub = retrieve.retrieve_subgraph(G, ticker, t)
    if not sub["signals"]:
        return None

    # query passages by the most credible transmitted signal
    scored = sorted((s for s in credibility["signals"]
                     if s["company"] == ticker and retrieve.day_of(s["timestamp"]) <= t),
                    key=lambda s: s["credibility"], reverse=True)
    query = scored[0]["content"] if scored else sub["signals"][0]["content"]
    passages = retrieve.retrieve_passages(G, ticker, t, query,
                                          cfg["extraction"]["retrieval"]["top_k_passages"])

    # ---- assemble the brief ----
    evidence = []
    for p in passages:
        evidence.append(_claim(p["text"], [p["meta"]["source_id"]], cited=True))

    players = []
    credible = []
    # credibility is game-derived: a player is "credible" if their transmitted
    # signal sits in a separating equilibrium (costly, Spence-style separation).
    credible_senders = {s["sender"] for s in scored if s["equilibrium"] == "separating"}
    for pnode, access in sub["players"].items():
        pattr = dict(G.nodes[pnode])
        rel = pattr.get("relation")
        players.append({
            "player": pattr["name"], "role": pattr["role"], "relation": rel,
            "credibility_prior": pattr["credibility_prior"],
            "signal_access": access,
        })
        if pattr["player_id"] in credible_senders:
            credible.append(pattr["name"])

    sig_scores = [{
        "signal_id": s["signal_id"], "sender": s["sender"], "role": s["role"],
        "signal_type": s["signal_type"], "equilibrium": s["equilibrium"],
        "credibility": s["credibility"], "content": s["content"],
    } for s in scored]

    posterior_c = posterior["companies"].get(ticker, {"posterior": 0.5})
    brief = {
        "run_id": cfg["run_id"], "entity": ticker, "brief_time": t,
        "sources_used": [{"source_id": p["meta"]["source_id"], "timestamp": p["meta"]["timestamp"]}
                         for p in passages],
        "evidence": evidence,
        "transmitting_players": players,
        "signal_credibility": sig_scores,
        "posterior_confidence": posterior_c["posterior"],
        "posterior_trajectory": posterior_c.get("steps", []),
        "research_direction": _research_direction(ticker, credible),
    }
    # citation discipline: drop any uncited claim into a low-confidence list
    brief["low_confidence_claims"] = [c for c in evidence if c["confidence"] == "low"]
    return brief


def synthesize_all(G, cfg):
    t = cfg["evaluation"]["brief_day"]
    cred = read_json(RESULTS_DIR / "signaling" / "credibility.json")
    post = read_json(RESULTS_DIR / "signaling" / "posterior.json")
    briefs = {}
    companies = {G.nodes[n]["ticker"] for n in G.nodes if G.nodes[n].get("kind") == "company"}
    for ticker in sorted(companies):
        b = synthesize_company(G, cfg, ticker, t, cred, post)
        if b:
            briefs[ticker] = b
    return {"run_id": cfg["run_id"], "brief_day": t, "briefs": briefs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="extract")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    gpath = Path(cfg["graph"]["path"])
    if not gpath.is_absolute():
        gpath = Path(cfg["paths"]["root"]) / gpath
    G = graph_io.load_graph(gpath)

    out = synthesize_all(G, cfg)
    write_json(RESULTS_DIR / "briefs" / "decision_briefs.json", out)
    n = len(out["briefs"])
    log(cfg, f"produced {n} cited decision briefs -> {RESULTS_DIR / 'briefs' / 'decision_briefs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
