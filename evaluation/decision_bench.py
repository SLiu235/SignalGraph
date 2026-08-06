"""
evaluation.decision_bench — score decision briefs against realized outcomes, with
NO-LOOKAHEAD enforcement.

A brief produced at day t may use only graph/evidence with timestamp <= t (enforced
by extraction.retrieve / synthesis). The realized outcome is the price return over the
window (t, t+window_days], using price observations STRICTLY after t. We then compare
the model's predicted price impact (credibility-weighted signal direction) and its
posterior confidence against the realized return, and write per-metric failure
summaries (including citation/hallucination counts from citation_check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, read_json, RESULTS_DIR
from ..graph import graph_io
from ..signaling.belief_update import _direction
from ..extraction import retrieve
from . import citation_check

DAY_PREFIX = "D"


def realized_return(price_history: list, t: int, window: int) -> float:
    """Return over (t, t+window], base = price at day t (last known at brief time)."""
    prices = {retrieve.day_of(p["date"]): p["price"] for p in price_history}
    base = prices.get(t)
    end = prices.get(t + window)
    if base is None or end is None:
        return None
    return end / base - 1.0


def predicted_impact(sig_scores: list) -> float:
    """Credibility-weighted signed direction of the highest-credibility signal."""
    if not sig_scores:
        return 0.0
    best = max(sig_scores, key=lambda s: s["credibility"])
    d = _direction(best.get("content", ""))
    return best["credibility"] * d


def score_brief(ticker: str, brief: dict, G, cfg) -> dict:
    t = brief["brief_time"]
    window = cfg["evaluation"]["outcome_window_days"]
    company = dict(G.nodes[f"company:{ticker}"])
    price_hist = company.get("price_history", [])

    realized = realized_return(price_hist, t, window)

    # no-lookahead check: all brief sources must be <= t
    lookahead = [s for s in brief.get("sources_used", [])
                 if retrieve.day_of(s.get("timestamp", "")) > t]

    pred = predicted_impact(brief.get("signal_credibility", []))
    hit = None
    if realized is not None:
        hit = (pred >= 0) == (realized >= 0)

    return {
        "entity": ticker, "brief_time": t, "realized_return": realized,
        "predicted_impact": round(pred, 4), "direction_hit": hit,
        "no_lookahead_violations": lookahead,
        "posterior_confidence": brief.get("posterior_confidence"),
    }


def run_benchmark(cfg):
    G = graph_io.load_graph(Path(cfg["graph"]["path"])
                            if Path(cfg["graph"]["path"]).is_absolute()
                            else Path(cfg["paths"]["root"]) / cfg["graph"]["path"])
    briefs = read_json(RESULTS_DIR / "briefs" / "decision_briefs.json")
    rows = [score_brief(t, b, G, cfg) for t, b in briefs["briefs"].items()]

    hits = [r for r in rows if r["direction_hit"] is not None]
    accuracy = sum(1 for r in hits if r["direction_hit"]) / len(hits) if hits else None
    # posterior-confidence-weighted accuracy: re-weight each direction hit by the
    # brief's stated Bayesian posterior confidence, so more-confident briefs count more.
    # A calibrated model tends to be more accurate where it states higher confidence.
    conf_hits = [r for r in hits if r["posterior_confidence"] is not None]
    w_accuracy = None
    if conf_hits:
        tot = sum(r["posterior_confidence"] for r in conf_hits)
        if tot > 0:
            w_accuracy = sum(r["posterior_confidence"] for r in conf_hits
                             if r["direction_hit"]) / tot
    violations = [r for r in rows if r["no_lookahead_violations"]]

    citation = citation_check.summarize(briefs["briefs"])
    report = {
        "run_id": cfg["run_id"], "n_briefs": len(rows),
        "direction_accuracy": accuracy,
        "confidence_weighted_accuracy": w_accuracy,
        "rows": rows,
        "no_lookahead_violations": violations,
        "per_metric_failures": {
            "price_impact": [r["entity"] for r in rows if r["realized_return"] is None],
            "credibility": [],  # populated by citation/signaling layers
            "posterior_confidence": [r["entity"] for r in rows
                                     if r["posterior_confidence"] is None],
            "citation_precision": citation["per_metric"]["citation_precision"],
        },
        "citation_summary": citation["summary"],
    }
    write_json(RESULTS_DIR / "evaluation" / "decision_bench.json", report)

    acc_txt = f"{accuracy:.3f}" if accuracy is not None else "n/a"
    w_txt = f"{w_accuracy:.3f}" if w_accuracy is not None else "n/a"
    log(cfg, f"benchmark: {len(rows)} briefs, direction accuracy {acc_txt}, "
             f"confidence-weighted accuracy {w_txt}, "
             f"{len(violations)} no-lookahead violations, "
             f"citation: {citation['summary']}")
    log(cfg, f"wrote -> {RESULTS_DIR / 'evaluation' / 'decision_bench.json'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="evaluate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
