"""
graph.neo4j_backend — Neo4j storage backend behind the same ontology.

Maps the networkx MultiDiGraph to Neo4j deterministically:

  node : (id, kind, attrs...) -> (n:Kind {id})   label = kind, id unique
  edge : (u, v, key, attrs...) -> (u)-[r:KEY {eid, attrs...}]->(v)
        relationship type = edge key (relation), eid = "{u}|{v}|{key}" so
        MERGE upserts are idempotent (a re-sync never duplicates edges).

Complex attribute values (nested dicts / lists, e.g. Company.fundamentals or
price_history) are JSON-encoded with a "__sgjson__:" sentinel because Neo4j
properties are scalars or lists of scalars only; load_graph decodes them back
so the round-trip is faithful.

Operations:
  Neo4jBackend(cfg)        — driver + database from config.graph.neo4j
  .available()             — reachability probe (False when DB is down)
  .sync_graph(G)           — idempotent full write (build/re-build)
  .apply_batch(signals, events) — incremental upsert of just the new items
  .load_graph()            — read Neo4j back into a networkx MultiDiGraph
  .clear()                 — drop all nodes/edges (fresh rebuild)

The storage dispatcher (graph/storage.py) raises a clear error when
graph.storage == "neo4j" but the DB is unreachable, so the live backend never
silently degrades; the offline json backend remains the reproducible default.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import networkx as nx

from graph import ontology as O

# Sentinel prefix marking a JSON-encoded complex property (dict / nested list).
_JSON_PREFIX = "__sgjson__:"


def _encode_value(v: Any) -> Any:
    """Scalars pass through; numpy scalars are unwrapped; dicts / nested lists
    become a JSON string sentinel."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        if hasattr(v, "item"):
            return _encode_value(v.item())  # numpy scalar -> native, then recurse
    except Exception:
        pass
    if isinstance(v, (list, tuple, dict)):
        return _JSON_PREFIX + json.dumps(v, separators=(",", ":"))
    return _JSON_PREFIX + json.dumps(str(v), separators=(",", ":"))


def _decode_value(v: Any) -> Any:
    """Inverse of _encode_value: decode JSON-sentinel strings back to Python.
    A literal string that merely starts with the sentinel is kept as-is."""
    if isinstance(v, str) and v.startswith(_JSON_PREFIX):
        try:
            return json.loads(v[len(_JSON_PREFIX):])
        except Exception:
            return v
    return v


def _encode_props(attrs: dict) -> dict:
    return {k: _encode_value(v) for k, v in attrs.items()}


def _decode_props(props) -> dict:
    out = {}
    for k, v in props.items():
        out[k] = _decode_value(v)
    return out


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _cypher_ident(name: Any, what: str) -> str:
    """Allow-list a label / relationship type for Cypher interpolation. Entity
    kinds and edge relations come from the fixed ontology set; anything else
    (e.g. an integer MultiDiGraph key) is rejected rather than interpolated."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"unsafe {what} for Cypher interpolation: {name!r}")
    return name.title()


def _kind_label(kind: str) -> str:
    """Neo4j label from an entity kind ('company' -> 'Company')."""
    return _cypher_ident(kind, "node label")


def _edge_eid(u: str, v: str, key: str, occurrence: int) -> str:
    """Deterministic relationship id. occurrence disambiguates MultiDiGraph
    edges that share (u, v, key); the ontology keeps these unique so it is 0."""
    return f"{u}|{v}|{key}|{occurrence}"


# Cypher templates. Labels / relationship types are interpolated (fixed
# ontology strings); all data is passed as parameters.
_NODE_MERGE = (
    "MERGE (n:{label} {{id: $id}}) "
    "ON CREATE SET n += $props "
    "ON MATCH SET n += $props"
)
_EDGE_MERGE = (
    "MATCH (a) WHERE a.id = $uid "
    "MATCH (b) WHERE b.id = $vid "
    "MERGE (a)-[r:{reltype} {{eid: $eid}}]->(b) "
    "ON CREATE SET r += $props "
    "ON MATCH SET r += $props"
)
_CONSTRAINT = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"


class Neo4jBackend:
    """Thin driver wrapper. Connection is lazy: the driver is created on first
    use and .available() probes reachability without requiring a live graph."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        conf = cfg.get("graph", {}).get("neo4j", {})
        self.uri = conf.get("uri", "bolt://localhost:7687")
        self.user = conf.get("user", "neo4j")
        self.password = conf.get("password", "") or os.environ.get("NEO4J_PASSWORD", "")
        self.database = conf.get("database", "neo4j")
        self.encrypted = bool(conf.get("encrypted", False))
        self._driver = None

    # ---- connection ----
    def _connect(self):
        if self._driver is None:
            import neo4j
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password), encrypted=self.encrypted
            )
        return self._driver

    def available(self, timeout: float = 3.0) -> bool:
        """Reachability probe. Returns False (never raises) when the DB is down."""
        try:
            d = self._connect()
            d.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self):
        driver = self._connect()
        return driver.session(database=self.database)

    # ---- full write (idempotent) ----
    def sync_graph(self, G: nx.MultiGraph) -> dict:
        if not self.available():
            raise ConnectionError(
                f"Neo4j backend requested (graph.storage=neo4j, uri={self.uri}) "
                "but the database is unreachable. Start Neo4j or set "
                "graph.storage=json for the offline path.")
        kinds = {G.nodes[n].get("kind", "node") for n in G.nodes}
        with self._session() as sess:
            for kind in kinds:
                sess.run(_CONSTRAINT.format(label=_kind_label(kind)))
            # nodes
            for n, attrs in G.nodes(data=True):
                props = _encode_props(dict(attrs))
                sess.run(_NODE_MERGE.format(label=_kind_label(props.get("kind", "node"))),
                         id=n, props=props)
            # edges (dedupe MultiDiGraph occurrences deterministically)
            seen = Counter()
            for u, v, k, attrs in G.edges(keys=True, data=True):
                occurrence = seen[(u, v, k)]
                seen[(u, v, k)] += 1
                reltype = attrs.get("relation") or k
                props = _encode_props({**dict(attrs), "eid": _edge_eid(u, v, k, occurrence)})
                sess.run(_EDGE_MERGE.format(reltype=_cypher_ident(reltype, "relationship type")),
                         uid=u, vid=v, eid=_edge_eid(u, v, k, occurrence), props=props)
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    # ---- incremental upsert of new signals/events ----
    def apply_batch(self, signals: list, events: list) -> dict:
        """Upsert just the new signals/events (+ their edges), reusing the same
        ontology constructors on a temp graph so Cypher stays in one place.

        NOTE: the pipeline (graph.update -> storage.save_graph) uses full
        idempotent sync_graph instead; apply_batch is a convenience for callers
        that want a lighter DB-level upsert without touching the rest of the
        graph (endpoints must already exist)."""
        H = nx.MultiDiGraph()
        for name, spec in O.SIGNAL_TYPES.items():
            O.add_signal_type(H, name, spec)
        for s in signals:
            O.add_signal(H, s)
        for e in events:
            O.add_event(H, e)
        self.sync_graph(H)
        return {"nodes": len(signals) + len(events), "edges": H.number_of_edges()}

    # ---- read back to networkx ----
    def load_graph(self) -> nx.MultiDiGraph:
        if not self.available():
            raise ConnectionError(
                f"Neo4j backend requested (graph.storage=neo4j, uri={self.uri}) "
                "but the database is unreachable. Start Neo4j or set "
                "graph.storage=json for the offline path.")
        G = nx.MultiDiGraph()
        G.graph["ontology_version"] = O.ONTOLOGY_VERSION
        with self._session() as sess:
            for rec in sess.run("MATCH (n) RETURN n.id AS id, n AS n"):
                n = rec["n"]
                # the MERGE pattern wrote `id` as a property; drop it so the
                # node key is not duplicated as an attribute (JSON round-trip
                # stores id as the key only, so backends stay equivalent).
                attrs = _decode_props({k: v for k, v in dict(n).items() if k != "id"})
                G.add_node(rec["id"], **attrs)
            for rec in sess.run(
                    "MATCH (a)-[r]->(b) "
                    "RETURN a.id AS u, b.id AS v, type(r) AS reltype, r AS r"):
                r = rec["r"]
                props = _decode_props({k: v for k, v in dict(r).items() if k != "eid"})
                G.add_edge(rec["u"], rec["v"], key=props.get("relation") or rec["reltype"].lower(),
                           **props)
        return G

    # ---- maintenance ----
    def clear(self) -> None:
        """Drop all nodes and edges (relationships) — used for a fresh rebuild."""
        if not self.available():
            return
        with self._session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")
