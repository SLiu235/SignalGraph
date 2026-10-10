"""
graph.update — incremental updates to the knowledge graph as new filings,
disclosures, and events arrive.

Real incremental event ingestion:
  * Loads the existing graph from the configured backend (graph.storage),
  * polls the ingest sources (data/ingest.py: edgar live, feed/news synthetic)
    and pulls ONLY events new since the cursor,
  * applies the batch through the ontology (validation included),
  * re-persists to the configured backend(s).

Designed so the live graph and the offline seed graph share the same ontology.
"""
from __future__ import annotations

import argparse
import sys

import networkx as nx

from common import DEFAULT_CONFIG_PATH, load_config, log
from graph import ontology as O, storage
from data import ingest as _ingest


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
    ap = argparse.ArgumentParser(description="incremental graph update + event ingestion")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--stage", default="update")
    ap.add_argument("--sources", default=None,
                    help="comma-separated ingest sources to poll (default: all enabled)")
    ap.add_argument("--dry-run", action="store_true", help="validate but do not persist")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    G = storage.open_graph(cfg)
    upd = Updater(G)

    sources = args.sources.split(",") if args.sources else None
    signals, events = _ingest.ingest_new(cfg, sources_filter=sources, commit=not args.dry_run)
    issues = upd.apply_batch(signals, events)

    log(cfg, f"update applied: +{len(signals)} signals +{len(events)} events "
             f"-> nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
             f"validation issues={len(issues)}")
    if issues:
        log(cfg, "validation WARNINGS:\n  " + "\n  ".join(issues))

    if not args.dry_run:
        storage.save_graph(G, cfg)
        log(cfg, f"graph re-persisted -> {storage.resolve_graph_path(cfg)} "
                 f"(storage={storage.storage_backend(cfg)})")
    else:
        log(cfg, "dry-run: graph not persisted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
