"""
graph.ontology — entities and relations for the signaling-game knowledge graph.

The graph represents BOTH:
  * Companies — nodes carrying fundamentals, filings, events, price history.
  * Players   — nodes carrying identity, role, credibility prior, and **signal-access
                edges** describing which signals they can observe, which they can
                transmit, and at what cost.

Entity kinds  : company, player, signal_type, signal, event
Relation kinds: signal_access, transmits, observes, linked_to, about, has_source, instance_of

THE SIGNAL-ACCESS MODEL (the focus)
-----------------------------------
A player's *information set* (what they can observe) and *transmission channel*
(what they can send, and at what cost) are encoded as `signal_access` edges from a
player node to a `signal_type` node. Each edge carries:

    can_observe  : bool    — the player can truthfully observe this signal class
    can_transmit : bool    — the player has a channel to send this signal class
    transmit_cost: float   — the per-signal cost the sender bears (costly vs cheap talk)

Signal types are split by Spence-style *costly* vs Crawford-Sobel *cheap talk*:
    certified_filing / disclosure  -> costly (sender bears real cost -> separates types)
    commentary / media             -> cheap talk (near-zero cost -> credibility by type)

The interpretation of a transmitted signal depends on WHO sends it, WHAT they can
observe (do they actually know it?), and WHAT it costs them (does sending it separate
their type?). A low-cost commentary from an aligned insider and a costly certified
filing revision both move the receiver, but via different game-theoretic channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

ONTOLOGY_VERSION = "1.0"

# ---- entity kinds ----
COMPANY = "company"
PLAYER = "player"
SIGNAL_TYPE = "signal_type"
SIGNAL = "signal"
EVENT = "event"

# ---- relation kinds ----
SIGNAL_ACCESS = "signal_access"   # player -> signal_type {can_observe, can_transmit, transmit_cost, channel}
TRANSMITS = "transmits"           # player -> signal {timestamp}
OBSERVES = "observes"             # player -> signal {timestamp}
LINKED_TO = "linked_to"           # player -> company {relation}
ABOUT = "about"                   # signal -> company
HAS_SOURCE = "has_source"         # company -> signal (filing/news source of record)
INSTANCE_OF = "instance_of"       # signal -> signal_type

# ---- player roles ----
ROLES = [
    "insider", "analyst", "competitor", "supplier",
    "regulator", "journalist", "fund",
]

# ---- signal types (channels) ----
# kind: costly = Spence-style (separates types), cheap_talk = Crawford-Sobel style
SIGNAL_TYPES = {
    "certified_filing": {"kind": "costly", "channel": "filings", "base_cost": 2.0},
    "disclosure":       {"kind": "costly", "channel": "disclosures", "base_cost": 2.0},
    "commentary":       {"kind": "cheap_talk", "channel": "commentary", "base_cost": 0.02},
    "media":            {"kind": "cheap_talk", "channel": "media", "base_cost": 0.02},
}

# Default credibility prior per role (base for the signaling game's receiver prior).
ROLE_CREDIBILITY_PRIOR = {
    "insider": 0.55,
    "analyst": 0.42,
    "competitor": 0.35,
    "supplier": 0.38,
    "regulator": 0.85,
    "journalist": 0.25,
    "fund": 0.60,
}

# Default signal-access profile per role: which signal types the role can observe /
# transmit and at what base cost. Per-player variation is applied by seed_graph.py
# so credibility scores are not degenerate.
ROLE_ACCESS = {
    "insider": [
        {"signal_type": "certified_filing", "can_observe": True,  "can_transmit": True,  "transmit_cost": 1.5},
        {"signal_type": "disclosure",       "can_observe": True,  "can_transmit": False, "transmit_cost": 0.0},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
    ],
    "analyst": [
        {"signal_type": "certified_filing", "can_observe": True,  "can_transmit": False, "transmit_cost": 0.0},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
    ],
    "competitor": [
        {"signal_type": "certified_filing", "can_observe": False, "can_transmit": False, "transmit_cost": 0.0},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
    ],
    "supplier": [
        {"signal_type": "certified_filing", "can_observe": False, "can_transmit": False, "transmit_cost": 0.0},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
    ],
    "regulator": [
        {"signal_type": "disclosure",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 1.5},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": False, "transmit_cost": 0.0},
    ],
    "journalist": [
        {"signal_type": "media",            "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": False, "transmit_cost": 0.0},
    ],
    "fund": [
        {"signal_type": "disclosure",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 1.2},
        {"signal_type": "commentary",       "can_observe": True,  "can_transmit": True,  "transmit_cost": 0.02},
    ],
}

# Which signal types are "costly" (Spence-style separation) vs "cheap talk".
COSTLY_TYPES = {s for s, d in SIGNAL_TYPES.items() if d["kind"] == "costly"}
CHEAP_TALK_TYPES = {s for s, d in SIGNAL_TYPES.items() if d["kind"] == "cheap_talk"}


# ---- entity dataclasses ----
@dataclass
class Company:
    ticker: str
    name: str
    cik: Optional[str] = None
    sector: str = ""
    fundamentals: dict = field(default_factory=dict)
    price_history: list = field(default_factory=list)   # [{"date":..., "price":...}]
    filings: list = field(default_factory=list)         # signal ids (source of record)
    created_at: str = ""

    @property
    def node_id(self) -> str:
        return f"company:{self.ticker}"


@dataclass
class Player:
    player_id: str
    name: str
    role: str
    credibility_prior: float
    informed_prior: float = 0.5   # P(this player is informed)
    aligned_prior: float = 0.5    # P(this player is aligned with receiver objective)
    company_ref: str = ""         # company ticker this player is linked to
    relation: str = ""            # insider_of / covers / competitor_of / supplier_to / regulates / covers_news / invested_in
    access_profile: dict = field(default_factory=dict)  # signal_type -> {can_observe, can_transmit, transmit_cost}

    @property
    def node_id(self) -> str:
        return f"player:{self.player_id}"


@dataclass
class Signal:
    signal_id: str
    signal_type: str          # key in SIGNAL_TYPES
    kind: str                 # costly | cheap_talk
    content: str
    sender: str               # player_id
    company: str              # ticker
    timestamp: str            # ISO date, <= brief time for no-lookahead
    source: str = ""          # filing | news | transcript | disclosure
    source_id: str = ""       # original doc id

    @property
    def node_id(self) -> str:
        return f"signal:{self.signal_id}"


@dataclass
class Event:
    event_id: str
    event_type: str           # filing | disclosure | news | price_move | ...
    timestamp: str
    company: str
    payload: dict = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return f"event:{self.event_id}"


# ---- graph construction helpers ----
def new_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.graph["ontology_version"] = ONTOLOGY_VERSION
    return G


def add_company(G: nx.MultiGraph, c: Company) -> None:
    G.add_node(c.node_id, kind=COMPANY, ticker=c.ticker, name=c.name,
               sector=c.sector, fundamentals=c.fundamentals,
               price_history=c.price_history, filings=c.filings, created_at=c.created_at)


def add_signal_type(G: nx.MultiGraph, name: str, spec: dict) -> None:
    G.add_node(f"stype:{name}", kind=SIGNAL_TYPE, name=name, channel=spec["channel"],
               base_cost=spec["base_cost"], kind_of_signal=spec["kind"])


def add_player(G: nx.MultiGraph, p: Player) -> None:
    G.add_node(p.node_id, kind=PLAYER, player_id=p.player_id, name=p.name, role=p.role,
               credibility_prior=p.credibility_prior, informed_prior=p.informed_prior,
               aligned_prior=p.aligned_prior, company_ref=p.company_ref, relation=p.relation)


def add_signal_access(G: nx.MultiGraph, player_id: str, signal_type: str,
                      can_observe: bool, can_transmit: bool, transmit_cost: float,
                      channel: str) -> None:
    """Adds a signal_access edge player -> signal_type. This is THE signal-access model:
    it encodes what the player can observe, what they can transmit, and at what cost."""
    G.add_edge(f"player:{player_id}", f"stype:{signal_type}", key=SIGNAL_ACCESS,
               relation=SIGNAL_ACCESS, can_observe=can_observe, can_transmit=can_transmit,
               transmit_cost=float(transmit_cost), channel=channel)


def add_signal(G: nx.MultiGraph, s: Signal) -> None:
    G.add_node(s.node_id, kind=SIGNAL, signal_id=s.signal_id, signal_type=s.signal_type,
               kind_of_signal=s.kind, content=s.content, sender=s.sender,
               company=s.company, timestamp=s.timestamp, source=s.source, source_id=s.source_id)
    # signal is an instance of a signal type
    G.add_edge(s.node_id, f"stype:{s.signal_type}", key=INSTANCE_OF, relation=INSTANCE_OF)
    # signal is about a company
    G.add_edge(s.node_id, f"company:{s.company}", key=ABOUT, relation=ABOUT)
    # the transmitting player -> signal (only if that player has transmission access)
    if s.sender:
        G.add_edge(f"player:{s.sender}", s.node_id, key=TRANSMITS, relation=TRANSMITS,
                   timestamp=s.timestamp)
    # filings/news are a company's source of record
    if s.source in ("filing", "news", "transcript", "disclosure"):
        G.add_edge(f"company:{s.company}", s.node_id, key=HAS_SOURCE, relation=HAS_SOURCE,
                   timestamp=s.timestamp)


def add_event(G: nx.MultiGraph, e: Event) -> None:
    G.add_node(e.node_id, kind=EVENT, event_id=e.event_id, event_type=e.event_type,
               timestamp=e.timestamp, company=e.company, payload=e.payload)
    G.add_edge(e.node_id, f"company:{e.company}", key=ABOUT, relation=ABOUT)


def add_linked_to(G: nx.MultiGraph, player_id: str, ticker: str, relation: str) -> None:
    G.add_edge(f"player:{player_id}", f"company:{ticker}", key=LINKED_TO,
               relation=LINKED_TO, relation_type=relation)


# ---- validation ----
def validate(G: nx.MultiGraph) -> list[str]:
    """Consistency checks: every transmits edge must be backed by a signal_access
    edge granting that transmission capability; every signal must have a sender."""
    issues = []
    for u, v, key, attrs in G.edges(keys=True, data=True):
        if key != TRANSMITS:
            continue
        # u is the player node, v is the signal node
        player_node, signal_node = u, v
        if not (G.nodes[player_node].get("kind") == PLAYER and G.nodes[signal_node].get("kind") == SIGNAL):
            issues.append(f"malformed transmits edge {u} -> {v}")
            continue
        # find the signal's type node
        stype = next((n for n in G.neighbors(signal_node)
                      if G.nodes[n].get("kind") == SIGNAL_TYPE), None)
        granted = False
        if stype:
            edge_data = G.get_edge_data(player_node, stype) or {}
            for eattrs in edge_data.values():
                if eattrs.get("relation") == SIGNAL_ACCESS and eattrs.get("can_transmit"):
                    granted = True
        if not granted:
            issues.append(f"{player_node} transmits {signal_node} without signal_access.can_transmit")
    return issues
