"""
extraction.retrieve — subgraph retrieval (the company, the transmitting players, and
their signal-access edges) plus passage retrieval (filings/news) for a decision brief.

NO-LOOKAHEAD: retrieval at day t uses only graph nodes/edges and signals whose
timestamp <= t. Anything strictly after t is excluded from the brief's evidence.
"""
from __future__ import annotations

import argparse
import sys

from common import DEFAULT_CONFIG_PATH, day_of, load_config, log
from graph import storage
from graph import ontology as O
from graph.ontology import ABOUT, HAS_SOURCE, LINKED_TO, SIGNAL_ACCESS
from . import embed

def evidence_visible(signal: dict, cutoff: int | str) -> bool:
    """Keep synthetic and UTC timelines explicit, using actual availability in UTC."""
    if isinstance(cutoff, int):
        return day_of(signal.get("timestamp", "")) <= cutoff
    from data.pit import parse_utc
    return parse_utc(signal.get("available_at") or signal.get("timestamp", "")) <= parse_utc(cutoff)


def retrieve_subgraph(G, ticker: str, t: int | str) -> dict:
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

    if isinstance(t, str):
        # A full historical archive contains future issuers/relationships. Only
        # expose a player after an observed transmission exists at the cutoff.
        visible_senders = {s.get("sender") for _, s in G.nodes(data=True)
                           if s.get("kind") == O.SIGNAL and s.get("company") == ticker
                           and evidence_visible(s, t)}
        players = [p for p in players if G.nodes[p].get("player_id") in visible_senders]

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
            # Events also use ABOUT edges. Only signal nodes belong in the signal
            # retrieval payload; indexing an event as a signal previously raised a
            # KeyError after incremental ingestion.
            if s.get("kind") != O.SIGNAL:
                continue
            if evidence_visible(s, t):
                signals.append({
                    "signal_id": s["signal_id"], "signal_type": s["signal_type"],
                    "kind": s["kind_of_signal"], "content": s["content"],
                    "sender": s["sender"], "timestamp": s["timestamp"],
                    "available_at": s.get("available_at", s["timestamp"]),
                    "document_sha256": s.get("document_sha256"),
                })

    return {"company": company, "players": player_access, "signals": signals}


def retrieve_passages(G, ticker: str, t: int | str, query: str, top_k: int) -> list[dict]:
    """Passage retrieval over filings/news (source of record) with timestamp <= t."""
    passages = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        if k != HAS_SOURCE:
            continue
        # company -> signal
        if G.nodes[u].get("ticker") == ticker:
            s = G.nodes[v]
            if s.get("source") in ("filing", "news", "transcript", "disclosure") \
                    and evidence_visible(s, t):
                passages.append({"id": s["signal_id"], "text": s["content"],
                                 "meta": {"source": s["source"], "source_id": s["source_id"],
                                          "timestamp": s["timestamp"], "signal_type": s["signal_type"],
                                          "available_at": s.get("available_at", s["timestamp"]),
                                          "document_sha256": s.get("document_sha256")}})
    idx = embed.build_index(passages)
    hits = idx.query(query, top_k)
    out = []
    for h in hits:
        p = next(p for p in passages if p["id"] == h["id"])
        out.append({**p, "score": h["score"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    G = storage.open_graph(cfg)
    sub = retrieve_subgraph(G, "AAPL", cfg["evaluation"]["brief_day"])
    log(cfg, f"retrieved AAPL subgraph: {len(sub['players'])} players, "
             f"{len(sub['signals'])} signals <= t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
