"""
signaling.signal_game — the sender–receiver game (Spence / Crawford-Sobel).

For each transmitted signal we compute:
  * sender costs for H and L types (the cost function separates types),
  * the equilibrium regime (separating / pooling) for that signal's type,
  * a credibility score derived from the game: the receiver's Bayes posterior
    P(H | signal) via a noisy-signal update, with reliability fixed by the regime.

Equilibrium characterization (from first principles):
  benefit of being perceived H: b = 1 (normalized).
  separating equilibrium exists for a costly signal iff  c(H) < b <= c(L):
    H sends (cost < benefit), L declines to mimic (cost >= benefit) -> signal
    fully separates types. Cheap talk has c(H) == c(L) ~= 0, so separation is
    impossible -> pooling (Crawford-Sobel): credibility rests on sender priors.

Credibility = receiver's Bayes posterior P(H | m) = (p_H * r) / (p_H*r + (1-p_H)(1-r))
  r = reliability of the signal about sender type:
      separating : r = receiver.precision_prior (signal trusted)
      pooling    : r = 0.5 (uninformative) -> posterior collapses to prior p_H.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, RESULTS_DIR, get_rng
from ..graph import graph_io, ontology as O
from ..graph.ontology import SIGNAL_ACCESS, SIGNAL_TYPE, PLAYER
from . import types as T

BENEFIT = 1.0  # normalized receiver payoff for treating the sender as high type


def signal_access_cost(G, player_node: str, stype: str) -> float | None:
    """Return the player's transmit_cost for this signal type from their
    signal_access edge, or None if they lack transmit access."""
    edge_data = G.get_edge_data(player_node, f"stype:{stype}") or {}
    for attrs in edge_data.values():
        if attrs.get("relation") == SIGNAL_ACCESS:
            return attrs.get("transmit_cost")
    return None


def characterize_equilibrium(spec: T.SignalTypeSpec, access_cost: float) -> dict:
    """Determine the equilibrium regime for a signal given the SENDER'S ACTUAL
    transmit cost (from their signal-access edge) and the type cost multipliers.

    Spence separation condition (benefit of being perceived H normalized to b=1):
        cost_H = access_cost * informed_cost_mult     (H knows truth -> pays less)
        cost_L = access_cost * misaligned_cost_mult   (L pays more to mimic)
        separating iff cost_H < b <= cost_L.
    Because access_cost differs per player, the same signal type can separate for
    some senders (high-cost channel) and pool for others (low-cost channel) — the
    interpretation genuinely depends on WHO transmits it and WHAT it costs them."""
    cost_H = access_cost * _cfg_cache["signaling"]["cost"]["informed_cost_mult"]
    cost_L = access_cost * _cfg_cache["signaling"]["cost"]["misaligned_cost_mult"]
    if spec.kind == T.CHEAP_TALK:
        # cheap talk: both types bear the same near-zero cost -> no Spence separation.
        # Override the type-scaled costs so c(H)==c(L)==access_cost (Crawford-Sobel).
        cost_H = cost_L = access_cost
        regime = "pooling"
        reason = f"cheap talk: c(H)==c(L)~={cost_H:.3f}, no Spence separation (Crawford-Sobel)"
    elif cost_H < BENEFIT <= cost_L:
        regime = "separating"
        reason = f"costly: c(H)={cost_H:.2f} < b={BENEFIT:.0f} <= c(L)={cost_L:.2f} (access cost {access_cost:.2f})"
    else:
        regime = "pooling"
        reason = f"costly: cost differential insufficient ({cost_H:.2f}/{cost_L:.2f}) for this sender's access cost"
    return {"signal_type": spec.name, "kind": spec.kind, "regime": regime,
            "cost_H": cost_H, "cost_L": cost_L, "access_cost": access_cost,
            "reason": reason}


def credibility_from_regime(p_h: float, regime: str, precision_prior: float) -> float:
    """Receiver's Bayes posterior P(H | m) = noisy-signal update.
    reliability r: separating -> precision_prior (trusted), pooling -> 0.5."""
    r = precision_prior if regime == "separating" else 0.5
    num = p_h * r
    den = num + (1 - p_h) * (1 - r)
    return num / den if den > 0 else 0.0


def collect_transmitted_signals(G) -> list[dict]:
    """Gather (signal node, sender player node, signal_type) for every transmits edge."""
    out = []
    for u, v, key, attrs in G.edges(keys=True, data=True):
        if key != O.TRANSMITS:
            continue
        player_node, sig_node = u, v
        sattrs = G.nodes[sig_node]
        pattrs = G.nodes[player_node]
        out.append({
            "signal": sig_node, "signal_id": sattrs["signal_id"],
            "signal_type": sattrs["signal_type"], "content": sattrs["content"],
            "timestamp": sattrs["timestamp"],
            "player": player_node, "player_id": pattrs["player_id"],
            "role": pattrs["role"], "credibility_prior": pattrs["credibility_prior"],
            "informed_prior": pattrs["informed_prior"], "aligned_prior": pattrs["aligned_prior"],
        })
    return out


_cfg_cache = None  # set in main so T.signal_cost has access to config


def score_signals(G, cfg: dict) -> list[dict]:
    global _cfg_cache
    _cfg_cache = cfg
    specs = T.make_signal_type_specs()
    rec = cfg["signaling"]["receiver"]
    precision = float(rec["precision_prior"])
    results = []
    for s in collect_transmitted_signals(G):
        p_h = T.high_type_prior(s["informed_prior"], s["aligned_prior"])
        spec = specs[s["signal_type"]]
        # sender's own access cost (what it actually costs THIS sender to transmit)
        access_cost = signal_access_cost(G, s["player"], s["signal_type"])
        if access_cost is None:
            access_cost = spec.base_cost  # fallback if access edge missing
        eq = characterize_equilibrium(spec, access_cost)
        cred = credibility_from_regime(p_h, eq["regime"], precision)
        results.append({
            "signal_id": s["signal_id"], "company": G.nodes[s["signal"]].get("company"),
            "sender": s["player_id"], "role": s["role"],
            "signal_type": s["signal_type"], "kind": spec.kind,
            "content": s["content"], "timestamp": s["timestamp"],
            "p_H": round(p_h, 4), "access_transmit_cost": access_cost,
            "equilibrium": eq["regime"], "eq_reason": eq["reason"],
            "cost_H": eq["cost_H"], "cost_L": eq["cost_L"],
            "credibility": round(cred, 4),
        })
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="signaling")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gpath = Path(cfg["graph"]["path"])
    if not gpath.is_absolute():
        gpath = Path(cfg["paths"]["root"]) / gpath
    G = graph_io.load_graph(gpath)

    results = score_signals(G, cfg)
    out = RESULTS_DIR / "signaling"
    write_json(out / "credibility.json", {"run_id": cfg["run_id"], "signals": results})

    # equilibrium characterization per signal type (aggregate, representative access cost)
    specs = T.make_signal_type_specs()
    rep_cost = {"costly": 1.5, "cheap_talk": 0.02}
    eq_by_type = {name: characterize_equilibrium(spec, rep_cost[spec.kind])
                  for name, spec in specs.items()}
    write_json(out / "equilibrium.json", {"run_id": cfg["run_id"], "by_type": eq_by_type,
                                         "by_signal": [r["equilibrium"] for r in results]})

    creds = [r["credibility"] for r in results]
    log(cfg, f"scored {len(results)} transmitted signals; "
             f"credibility range [{min(creds):.3f}, {max(creds):.3f}] "
             f"unique={len(set(round(c,3) for c in creds))}")
    from collections import Counter
    regimes = Counter(r["equilibrium"] for r in results)
    log(cfg, f"equilibrium regimes: {dict(regimes)}")
    log(cfg, f"wrote -> {out / 'credibility.json'} , {out / 'equilibrium.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
