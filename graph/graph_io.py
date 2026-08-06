"""
graph.graph_io — deterministic JSON serialization of the networkx MultiGraph.
Used by build_graph / update so the graph is re-loadable and reproducible offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ..common import ensure_dir, write_json


def graph_to_json(G: nx.MultiGraph) -> dict:
    nodes = []
    for n, attrs in G.nodes(data=True):
        nodes.append({"id": n, "attrs": _clean(attrs)})
    edges = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        edges.append({"u": u, "v": v, "key": k, "attrs": _clean(attrs)})
    return {"graph_meta": dict(G.graph), "nodes": nodes, "edges": edges}


def graph_from_json(data: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.graph.update(data.get("graph_meta", {}))
    for n in data.get("nodes", []):
        G.add_node(n["id"], **n["attrs"])
    for e in data.get("edges", []):
        G.add_edge(e["u"], e["v"], key=e["key"], **e["attrs"])
    return G


def save_graph(G: nx.MultiGraph, path: Path | str) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    write_json(p, graph_to_json(G))
    return str(p)


def load_graph(path: Path | str) -> nx.MultiGraph:
    with Path(path).open("r") as f:
        data = json.load(f)
    return graph_from_json(data)


def _clean(attrs: dict) -> dict:
    """Make attribute dict JSON-safe (numpy scalars -> floats/ints)."""
    out = {}
    for k, v in attrs.items():
        try:
            if hasattr(v, "item"):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, (list, tuple)):
            v = [_clean_i(x) for x in v]
        out[k] = v
    return out


def _clean_i(x):
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x
