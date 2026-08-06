"""
signaling.belief_update — Bayesian posterior updating of the receiver's confidence
in a hypothesis (e.g., "this company will outperform") given scored signals.

Each signal carries a credibility = P(H | m) from the game (see signal_game.py).
The receiver also has a prior confidence P(H0) in the hypothesis. For each signal
we form a likelihood of the evidence conditional on the hypothesis vs its negation,
then update sequentially by Bayes' rule:

    P(H0 | E) = P(H0) * P(E|H0) / ( P(H0)*P(E|H0) + (1-P(H0))*P(E|~H0) )

The signal's direction (support in {-1,0,+1}) comes from content sentiment:
    support>0: P(E|H0)=cred,        P(E|~H0)=1-cred
    support<0: P(E|H0)=1-cred,      P(E|~H0)=cred
    support=0: P(E|H0)=P(E|~H0)=0.5 (no update)
This is a principled Bayesian accumulation, not a hand-tuned rubric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, RESULTS_DIR


def _direction(content: str) -> int:
    """Crude deterministic sentiment for synthetic content. Positive/negative words
    -> +1/-1; otherwise 0. Keeps the update grounded in evidence direction."""
    pos = {"raised", "revised up", "bullish", "accumulating", "positive", "upside",
           "overweight", "under control", "held", "strong"}
    neg = {"lowered", "trimming", "negative", "downside", "underweight", "rising",
           "flags", "slowdown"}
    low = content.lower()
    if any(w in low for w in pos):
        return 1
    if any(w in low for w in neg):
        return -1
    return 0


def update_posterior(prior: float, cred: float, direction: int) -> float:
    """One Bayes update step; returns the posterior probability of the hypothesis."""
    if direction == 0:
        return prior
    like_H = cred if direction > 0 else 1 - cred
    like_notH = (1 - cred) if direction > 0 else cred
    num = prior * like_H
    den = num + (1 - prior) * like_notH
    return num / den if den > 0 else prior


def sequential_update(prior: float, signals: list[dict]) -> dict:
    """Accumulate signals (chronologically) into a posterior confidence trajectory."""
    posterior = prior
    steps = []
    for s in signals:
        d = _direction(s.get("content", ""))
        posterior = update_posterior(posterior, s["credibility"], d)
        steps.append({
            "signal_id": s.get("signal_id"), "credibility": s["credibility"],
            "direction": d, "posterior": round(posterior, 4),
        })
    return {"prior": prior, "posterior": round(posterior, 4), "steps": steps}


def build_confidence_report(credibility_records: dict, cfg: dict) -> dict:
    """Group scored signals by company and produce a posterior confidence per company.

    NO-LOOKAHEAD: only signals observed at or before `evaluation.brief_day` are used,
    so a brief produced at time t never updates on signals arriving after t. Signals
    strictly after brief_day are dropped before the sequential Bayes accumulation.
    Prior confidence = 0.5 (indifferent) unless overridden by config."""
    from ..common import day_of
    prior = cfg["evaluation"].get("prior_confidence", 0.5)
    brief_day = cfg["evaluation"]["brief_day"]
    from collections import defaultdict
    by_company = defaultdict(list)
    for s in credibility_records["signals"]:
        # signals at or before brief time only (no future-signal leakage)
        if day_of(s.get("timestamp", "")) <= brief_day:
            by_company[s["company"]].append(s)
    report = {}
    for company, sigs in by_company.items():
        # timestamp-ordered processing so signals at t affect briefs at t, not earlier
        sigs_sorted = sorted(sigs, key=lambda s: s.get("timestamp", ""))
        report[company] = sequential_update(prior, sigs_sorted)
    return {"run_id": credibility_records.get("run_id"), "prior": prior,
            "companies": report}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    ap.add_argument("--stage", default="signaling")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    from ..common import read_json
    cred = read_json(RESULTS_DIR / "signaling" / "credibility.json")
    report = build_confidence_report(cred, cfg)
    write_json(RESULTS_DIR / "signaling" / "posterior.json", report)
    log(cfg, f"wrote posterior confidence for {len(report['companies'])} companies -> "
             f"{RESULTS_DIR / 'signaling' / 'posterior.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
