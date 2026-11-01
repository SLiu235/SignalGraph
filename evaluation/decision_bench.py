"""
evaluation.decision_bench — score decision briefs against realized outcomes, with
NO-LOOKAHEAD enforcement.

A brief produced at day t may use only graph/evidence with timestamp <= t (enforced
by extraction.retrieve / synthesis). The realized outcome is the price return over the
window (t, t+window_days], using price observations STRICTLY after t. We then compare
the model's predicted subsequent-return direction (credibility-weighted signal direction)
and its posterior confidence against the realized return, and write per-metric failure
summaries (including citation/hallucination counts from citation_check).
"""
from __future__ import annotations

import argparse
import sys

from common import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config, log, read_json, write_json
from extraction import retrieve
from graph import storage
from signaling.belief_update import _direction
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


def impact_direction(value: float, threshold: float = 0.0) -> int:
    """Classify an impact as positive, negative, or immaterial.

    `threshold` is applied symmetrically to realized returns. A zero-direction model
    prediction is neutral rather than being accidentally counted as bullish.
    """
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def directional_confidence(p_outperform: float | None, predicted_direction: int) -> float | None:
    """Confidence assigned to the direction actually predicted.

    The posterior is P(outperform), so a bearish prediction must use its complement.
    A neutral prediction carries no directional preference and receives 0.5.
    """
    if p_outperform is None:
        return None
    if predicted_direction > 0:
        return p_outperform
    if predicted_direction < 0:
        return 1.0 - p_outperform
    return 0.5


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
    predicted_direction = impact_direction(pred)
    realized_direction = (
        impact_direction(realized, float(cfg["evaluation"].get("price_impact_threshold", 0.0)))
        if realized is not None else None
    )
    hit = None
    if realized is not None:
        hit = predicted_direction == realized_direction

    posterior = brief.get("posterior_confidence")

    return {
        "entity": ticker, "brief_time": t, "realized_return": realized,
        "predicted_impact": round(pred, 4),
        "predicted_impact_direction": predicted_direction,
        "realized_impact_direction": realized_direction,
        "direction_hit": hit,
        "no_lookahead_violations": lookahead,
        "posterior_confidence": posterior,
        "directional_confidence": directional_confidence(posterior, predicted_direction),
    }


def run_benchmark(cfg):
    G = storage.open_graph(cfg)
    briefs = read_json(RESULTS_DIR / "briefs" / "decision_briefs.json")
    rows = [score_brief(t, b, G, cfg) for t, b in briefs["briefs"].items()]

    hits = [r for r in rows if r["direction_hit"] is not None]
    accuracy = sum(1 for r in hits if r["direction_hit"]) / len(hits) if hits else None
    # Directional-confidence-weighted accuracy: weight a bullish call by P(outperform)
    # and a bearish call by 1-P(outperform). The reported posterior is explicitly a
    # heuristic sensitivity score, so this is a diagnostic rather than calibration.
    conf_hits = [r for r in hits if r["directional_confidence"] is not None]
    w_accuracy = None
    if conf_hits:
        tot = sum(r["directional_confidence"] for r in conf_hits)
        if tot > 0:
            w_accuracy = sum(r["directional_confidence"] for r in conf_hits
                             if r["direction_hit"]) / tot
    violations = [r for r in rows if r["no_lookahead_violations"]]

    citation = citation_check.summarize(briefs["briefs"])
    report = {
        "run_id": cfg["run_id"], "n_briefs": len(rows),
        "direction_accuracy": accuracy,
        "directional_confidence_weighted_accuracy": w_accuracy,
        # Kept as a compatibility alias for consumers of older result files.
        "confidence_weighted_accuracy": w_accuracy,
        "rows": rows,
        "no_lookahead_violations": violations,
        "per_metric_failures": {
            "price_impact": [r["entity"] for r in rows if r["realized_return"] is None],
            "credibility": [],  # populated by citation/signaling layers
            "posterior_confidence": [r["entity"] for r in rows
                                     if r["posterior_confidence"] is None],
            "citation_coverage": citation["per_metric"]["citation_coverage"],
        },
        "citation_summary": citation["summary"],
    }
    write_json(RESULTS_DIR / "evaluation" / "decision_bench.json", report)

    acc_txt = f"{accuracy:.3f}" if accuracy is not None else "n/a"
    w_txt = f"{w_accuracy:.3f}" if w_accuracy is not None else "n/a"
    log(cfg, f"benchmark: {len(rows)} briefs, direction accuracy {acc_txt}, "
             f"directional-confidence-weighted accuracy {w_txt}, "
             f"{len(violations)} no-lookahead violations, "
             f"citation: {citation['summary']}")
    log(cfg, f"wrote -> {RESULTS_DIR / 'evaluation' / 'decision_bench.json'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--stage", default="evaluate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
