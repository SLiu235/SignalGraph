"""Auditable, frozen feature rules. These rules are hypotheses, not fitted forecasts."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import hashlib
import math
import re

from data.pit import parse_utc
from graph import ontology as O
from signaling.signal_game import characterize_equilibrium, credibility_from_regime
from signaling.types import SignalTypeSpec


FEATURE_VERSION = "filing-rules-v1"


def validate_events(events: list[dict]) -> None:
    seen = set()
    for event in events:
        for key in ("event_id", "ticker", "source", "form", "document_text", "document_sha256"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise ValueError(f"event requires nonempty {key}")
        if event["event_id"] in seen:
            raise ValueError(f"duplicate event_id: {event['event_id']}")
        seen.add(event["event_id"])
        accepted = parse_utc(event["event_time"])
        available = parse_utc(event["available_at"])
        retrieved = parse_utc(event["retrieved_at"])
        if available < accepted or retrieved < accepted:
            raise ValueError("availability/retrieval cannot precede acceptance")
        mode = event.get("availability_mode", event.get("provenance", {}).get("availability_mode"))
        if mode == "live" and available < retrieved:
            raise ValueError("live availability cannot precede local retrieval")
        if not re.fullmatch(r"[0-9a-f]{64}", event["document_sha256"]):
            raise ValueError("document_sha256 must identify the original raw bytes")


def _sector_at(event: dict, memberships: list[dict]) -> str:
    at = parse_utc(event["available_at"])
    matches = [m for m in memberships if m["ticker"] == event["ticker"]
               and parse_utc(m["available_at"]) <= at
               and parse_utc(m["valid_from"]) <= at
               and (not m.get("valid_to") or at < parse_utc(m["valid_to"]))]
    if not matches:
        return "unknown"
    return max(matches, key=lambda m: parse_utc(m["available_at"]))["sector"]


def prepare_events(events: list[dict], market: dict, cfg: dict) -> list[dict]:
    """Build features using only earlier issuer documents; equal-time rows are peers.

    A historical replay reconstructs rules at availability time. It is not evidence
    that this software observed the filing live. Raw retrieval time remains intact.
    """
    validate_events(events)
    assumptions = cfg["research"].get("features", {})
    informed = float(assumptions.get("informed_prior", 0.5))
    aligned = float(assumptions.get("aligned_prior", 0.5))
    cost = float(assumptions.get("filing_access_cost", 1.5))
    precision = float(assumptions.get("precision_prior", 0.8))
    lookback = int(assumptions.get("novelty_lookback_days", 365))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (informed, aligned, precision)):
        raise ValueError("feature priors must be finite probabilities")
    if not math.isfinite(cost) or cost < 0 or lookback < 1:
        raise ValueError("invalid filing cost / novelty lookback")
    spec = SignalTypeSpec("certified_filing", "costly", "filings", 2.0)
    regime = characterize_equilibrium(spec, cost, cfg)
    credibility = credibility_from_regime(informed * aligned, regime["regime"], precision)
    history = defaultdict(list)
    output = []
    for event in sorted(events, key=lambda e: (parse_utc(e["available_at"]), e["event_id"])):
        at = parse_utc(event["available_at"])
        text = " ".join(event["document_text"].lower().split())
        tokens = set(re.findall(r"[a-z]{3,}", text))
        previous = [p for p in history[event["ticker"]]
                    if at - timedelta(days=lookback) <= p[0] < at]
        similarity = max((len(tokens & p[1]) / max(1, len(tokens | p[1])) for p in previous), default=0.0)
        positives = len(re.findall(r"\b(raised|improved|increased|strong|growth|profitable)\b", text))
        negatives = len(re.findall(r"\b(lowered|declined|decreased|weak|loss|slowdown)\b", text))
        direction = (positives > negatives) - (positives < negatives)
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        # Imported outcomes/scores must not override this versioned feature rule.
        row = {k: v for k, v in event.items() if k not in ("score", "features", "features_available_at")}
        row.update({
            "sender": f"issuer:{event['cik']}" if event.get("cik") else f"issuer:{event['ticker']}",
            "sector": _sector_at(event, market.get("memberships", [])),
            "features_available_at": event["available_at"],
            "feature_version": FEATURE_VERSION,
            "features": {"direction": direction, "credibility": credibility,
                         "access_cost": cost, "novelty": 1.0 - similarity,
                         "graph_propagation": 0.0,
                         "is_earnings": event["form"].startswith("8-K") and bool(re.search(r"\bitem\s+2\.02\b", text))},
            "feature_provenance": {
                "credibility": "configured signaling-game assumptions; not empirically calibrated",
                "access_cost": "assumed issuer filing cost; not a measured acquisition cost",
                "novelty": "one minus maximum token Jaccard overlap with strictly earlier issuer documents",
                "graph_propagation": "unavailable in single-source EDGAR; zero placeholder, no propagation claim",
                "direction": "positive/negative word counts; unvalidated financial text heuristic",
                "is_earnings": "8-K primary text contains Item 2.02; content label, not an earnings surprise",
                "time": "deterministic reconstruction at source availability; raw retrieval time retained",
                "equilibrium": regime, "prior_documents": len(previous),
                "normalized_text_sha256": fingerprint,
            },
        })
        output.append(row)
        history[event["ticker"]].append((at, tokens))
    return output


def build_graph(events: list[dict], market: dict, cfg: dict):
    """Real issuers and observed documents only; no fabricated analyst population.

    Outcome prices remain in the market bundle, outside the evidence graph. A graph
    snapshot has an explicit cutoff; a complete research graph is a temporal archive.
    """
    G = O.new_graph()
    G.graph.update(data_kind=market["metadata"]["data_kind"],
                   time_basis="UTC availability", feature_version=FEATURE_VERSION,
                   usage="temporal archive; use snapshot_at for decision evidence")
    cutoff = cfg["research"].get("as_of")
    eligible = [e for e in events if not cutoff or parse_utc(e["available_at"]) <= parse_utc(cutoff)]
    for name, spec in O.SIGNAL_TYPES.items():
        O.add_signal_type(G, name, spec)
    for event in eligible:
        ticker, sender = event["ticker"], event["sender"]
        if f"company:{ticker}" not in G:
            O.add_company(G, O.Company(ticker=ticker, name=ticker))
        if f"player:{sender}" not in G:
            O.add_player(G, O.Player(player_id=sender, name=sender, role="issuer",
                                    credibility_prior=event["features"]["credibility"],
                                    company_ref=ticker, relation="files_for"))
        O.add_linked_to(G, sender, ticker, "files_for")
        O.add_signal_access(G, sender, "certified_filing", True, True,
                            event["features"]["access_cost"], "filings")
        O.add_signal(G, O.Signal(signal_id=event["event_id"], signal_type="certified_filing",
                                kind="costly", content=event["document_text"], sender=sender,
                                company=ticker, timestamp=event["available_at"], source="filing",
                                source_id=event.get("accession", event["event_id"])))
        G.nodes[f"signal:{event['event_id']}"].update(
            event_time=event["event_time"], available_at=event["available_at"],
            retrieved_at=event["retrieved_at"], document_sha256=event["document_sha256"],
            document_url=event.get("document_url"), features=event["features"], sector=event["sector"],
            feature_provenance=event["feature_provenance"], provenance=event.get("provenance", {}))
    issues = O.validate(G)
    if issues:
        raise ValueError(f"historical graph validation failed: {issues}")
    G.graph["as_of"] = cutoff
    return G


def snapshot_at(events: list[dict], market: dict, cfg: dict, as_of: str):
    from copy import deepcopy
    selected = deepcopy(cfg)
    selected["research"]["as_of"] = as_of
    parse_utc(as_of)
    return build_graph(events, market, selected)
