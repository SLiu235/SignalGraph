"""
graph.build_graph — stage 1: assemble the knowledge graph from seed + data sources.

Steps: (1) generate deterministic seed records, (2) instantiate ontology entities and
relations (including every player's signal-access edges), (3) validate, (4) serialize
to results/graph/graph.json and write a reproducibility manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, RESULTS_DIR
from ..graph import ontology as O
from ..graph import graph_io
from ..data import seed_graph


def assemble_graph(cfg: dict):
    companies, players, signals = seed_graph.build_seed_records(cfg)
    transmitted = seed_graph.build_transmitted_signals(cfg, players)

    G = O.new_graph()
    # signal types
    for name, spec in O.SIGNAL_TYPES.items():
        O.add_signal_type(G, name, spec)

    # companies + player nodes + linked_to relations + signal-access edges
    for c in companies:
        O.add_company(G, c)
    for p in players:
        O.add_player(G, p)
        O.add_linked_to(G, p.player_id, p.company_ref, p.relation)
        for stype, spec in p.access_profile.items():
            O.add_signal_access(G, p.player_id, stype, spec["can_observe"],
                                spec["can_transmit"], spec["transmit_cost"],
                                O.SIGNAL_TYPES[stype]["channel"])

    # signals: source-of-record (filings/news) + player-transmitted
    for s in signals + transmitted:
        O.add_signal(G, s)

    issues = O.validate(G)
    log(cfg, f"graph nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
             f"validation issues={len(issues)}")
    if issues:
        log(cfg, "validation WARNINGS:\n  " + "\n  ".join(issues))
    return G


def main(argv=None):
    ap = argparse.ArgumentParser(description="build knowledge graph")
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="graph")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    G = assemble_graph(cfg)

    out = Path(cfg["graph"]["path"])
    if not out.is_absolute():
        out = Path(cfg["paths"]["root"]) / out
    graph_io.save_graph(G, out)

    manifest = {
        "run_id": cfg["run_id"], "stage": args.stage, "seed": cfg["seed"],
        "ontology_version": O.ONTOLOGY_VERSION, "graph_path": str(out),
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "counts": _counts(G),
    }
    write_json(RESULTS_DIR / "graph" / "manifest.json", manifest)
    log(cfg, f"saved graph -> {out}")
    log(cfg, f"manifest -> {RESULTS_DIR / 'graph' / 'manifest.json'}")
    return 0


def _counts(G) -> dict:
    from collections import Counter
    kinds = Counter(G.nodes[n].get("kind", "?") for n in G.nodes)
    rels = Counter(attrs.get("relation", "?") for _, _, attrs in G.edges(data=True))
    return {"by_kind": dict(kinds), "by_relation": dict(rels)}


if __name__ == "__main__":
    sys.exit(main())
