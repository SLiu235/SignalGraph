"""
graph.update — incremental updates to the knowledge graph as new filings,
disclosures, and events arrive. Loads the existing graph, applies a batch of
new signals/events, validates, and re-serializes. Designed so the live graph and
the offline seed graph share the same ontology.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

from ..common import load_config, log, write_json, RESULTS_DIR
from ..graph import ontology as O
from ..graph import graph_io


class Updater:
    def __init__(self, G: nx.MultiGraph):
        self.G = G

    def add_signal(self, s: O.Signal) -> None:
        O.add_signal(self.G, s)

    def add_event(self, e: O.Event) -> None:
        O.add_event(self.G, e)

    def add_player(self, p: O.Player, access_profile: dict) -> None:
        """Add a new player with their full signal-access profile."""
        O.add_player(self.G, p)
        O.add_linked_to(self.G, p.player_id, p.company_ref, p.relation)
        for stype, spec in access_profile.items():
            O.add_signal_access(self.G, p.player_id, stype, spec["can_observe"],
                                spec["can_transmit"], spec["transmit_cost"],
                                O.SIGNAL_TYPES[stype]["channel"])

    def apply_batch(self, signals: list, events: list) -> list:
        for s in signals:
            self.add_signal(s)
        for e in events:
            self.add_event(e)
        return O.validate(self.G)


def main(argv=None):
    ap = argparse.ArgumentParser(description="incremental graph update")
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="update")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(cfg["graph"]["path"])
    if not out.is_absolute():
        out = Path(cfg["paths"]["root"]) / out

    G = graph_io.load_graph(out)
    upd = Updater(G)
    issues = upd.apply_batch(signals=[], events=[])
    log(cfg, f"update applied (no new events): nodes={G.number_of_nodes()} "
             f"edges={G.number_of_edges()} issues={len(issues)}")
    graph_io.save_graph(G, out)
    log(cfg, f"graph re-serialized -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
