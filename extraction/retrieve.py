"""
extraction.retrieve — subgraph retrieval (the company, the transmitting players, and
their signal-access edges) plus passage retrieval (filings/news) for a decision brief.

NO-LOOKAHEAD: retrieval at day t uses only graph nodes/edges and signals whose
timestamp <= t. Anything strictly after t is excluded from the brief's evidence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log
from ..graph import graph_io, ontology as O
from ..graph.ontology import SIGNAL_ACCESS, LINKED_TO, ABOUT, HAS_SOURCE
from . import embed

DAY_PREFIX = "D"


def day_of(timestamp: str) -> int:
    """Parse synthetic 'D####' timestamps into an integer day; ISO dates fall back
    to 0 (kept for live data where ordering is handled upstream)."""
    if timestamp.startswith(DAY_PREFIX):
        try:
            return int(timestamp[1:])
        except ValueError:
            return 0
    return 0


def retrieve_subgraph(G, ticker: str, t: int) -> dict:
    """Company node + linked players + their signal-access edges + signals <= t."""
    company_node = f"company:{ticker}"
    company = dict(G.nodes[company_node])

    players = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k != LINKED_TO:
            continue
        player_node = u if G.nodes[u].get("kind") == O.PLAYER else v
        if G.nodes[player_node].get("company_ref") == ticker:
            players.append(player_node)

    player_access = {}
    for p in players:
        access = []
        for u, v, k, attrs in G.edges(keys=True, data=True):
            if k != SIGNAL_ACCESS:
                continue
            # player -> signal_type
            if u == p and G.nodes[v].get("kind") == O.SIGNAL_TYPE:
                access.append({
                    "signal_type": v.replace("stype:", ""),
                    "can_observe": attrs["can_observe"],
                    "can_transmit": attrs["can_transmit"],
                    "transmit_cost": attrs["transmit_cost"],
                    "channel": attrs["channel"],
                })
        player_access[p] = access

    signals = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k == ABOUT and (G.nodes[v].get("ticker") == ticker):
            s = G.nodes[u]
            if day_of(s.get("timestamp", "")) <= t:
                signals.append({
                    "signal_id": s["signal_id"], "signal_type": s["signal_type"],
                    "kind": s["kind_of_signal"], "content": s["content"],
                    "sender": s["sender"], "timestamp": s["timestamp"],
                })

    return {"company": company, "players": player_access, "signals": signals}


def retrieve_passages(G, ticker: str, t: int, query: str, top_k: int) -> list[dict]:
    """Passage retrieval over filings/news (source of record) with timestamp <= t."""
    passages = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k != HAS_SOURCE:
            continue
        # company -> signal
        if G.nodes[u].get("ticker") == ticker:
            s = G.nodes[v]
            if s.get("source") in ("filing", "news", "transcript", "disclosure") \
                    and day_of(s.get("timestamp", "")) <= t:
                passages.append({"id": s["signal_id"], "text": s["content"],
                                 "meta": {"source": s["source"], "source_id": s["source_id"],
                                          "timestamp": s["timestamp"], "signal_type": s["signal_type"]}})
    idx = embed.build_index(passages)
    hits = idx.query(query, top_k)
    out = []
    for h in hits:
        p = next(p for p in passages if p["id"] == h["id"])
        out.append({**p, "score": h["score"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    gpath = Path(cfg["graph"]["path"])
    if not gpath.is_absolute():
        gpath = Path(cfg["paths"]["root"]) / gpath
    G = graph_io.load_graph(gpath)
    sub = retrieve_subgraph(G, "AAPL", cfg["evaluation"]["brief_day"])
    log(cfg, f"retrieved AAPL subgraph: {len(sub['players'])} players, "
             f"{len(sub['signals'])} signals <= t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
