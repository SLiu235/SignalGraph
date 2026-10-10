"""
graph.storage — backend dispatch for the knowledge graph.

Config knob: graph.storage = json | neo4j
  json  -> networkx MultiDiGraph serialized to graph.path (default, offline,
           reproducible). graph/graph_io.py does the deterministic (de)serialization.
  neo4j -> live Neo4j graph via the driver (graph/neo4j_backend.py). A JSON
           snapshot of the same graph is still written to graph.path so offline
           stages and reproducibility checks keep working.

Every stage loads via open_graph(cfg) and saves via save_graph(G, cfg), so the
backend is a config swap and downstream code never touches the DB directly.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from graph import graph_io
from graph import neo4j_backend as NB


def storage_backend(cfg: dict) -> str:
    return str(cfg.get("graph", {}).get("storage", "json")).lower()


def resolve_graph_path(cfg: dict) -> Path:
    p = Path(cfg["graph"]["path"])
    return p if p.is_absolute() else Path(cfg["paths"]["root"]) / p


def open_graph(cfg: dict) -> nx.MultiDiGraph:
    """Load the graph from the configured backend (neo4j or json)."""
    if storage_backend(cfg) == "neo4j":
        return NB.Neo4jBackend(cfg).load_graph()
    return graph_io.load_graph(resolve_graph_path(cfg))


def save_graph(G: nx.MultiGraph, cfg: dict) -> str:
    """Persist the graph to the configured backend(s). A JSON snapshot is always
    written (reproducible mirror); Neo4j is synced when graph.storage == neo4j."""
    out = resolve_graph_path(cfg)
    graph_io.save_graph(G, out)
    if storage_backend(cfg) == "neo4j":
        nb = NB.Neo4jBackend(cfg)
        nb.sync_graph(G)
        nb.close()
    return str(out)
