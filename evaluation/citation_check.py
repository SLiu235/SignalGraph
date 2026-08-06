"""
evaluation.citation_check — per-metric citation / hallucination failure summaries.
For each decision brief we count:
  * total claims, cited claims, uncited (uncited -> LOW-CONFIDENCE),
  * hallucination count (claims that assert facts with no retrieved source),
  * citation precision = cited / total.
A summary is emitted per metric so failures are visible per-metric, not aggregated away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log, write_json, read_json, RESULTS_DIR


def check_brief(brief: dict) -> dict:
    evidence = brief.get("evidence", [])
    total = len(evidence)
    cited = sum(1 for c in evidence if c.get("citations"))
    uncited = total - cited
    halluc = len(brief.get("low_confidence_claims", []))
    precision = cited / total if total else 1.0
    return {
        "total_claims": total, "cited_claims": cited,
        "uncited_claims": uncited, "hallucination_count": halluc,
        "citation_precision": round(precision, 3),
        "failure": uncited > 0 or halluc > 0,
    }


def summarize(briefs: dict) -> dict:
    per_metric = {"citation_precision": {}}
    agg = {"total_claims": 0, "cited": 0, "uncited": 0, "hallucination": 0,
           "failed_briefs": 0}
    for ticker, b in briefs.items():
        r = check_brief(b)
        per_metric["citation_precision"][ticker] = r
        agg["total_claims"] += r["total_claims"]
        agg["cited"] += r["cited_claims"]
        agg["uncited"] += r["uncited_claims"]
        agg["hallucination"] += r["hallucination_count"]
        if r["failure"]:
            agg["failed_briefs"] += 1
    agg["citation_precision"] = (agg["cited"] / agg["total_claims"]
                                 if agg["total_claims"] else 1.0)
    return {"run_id": briefs.get("run_id"), "per_metric": per_metric,
            "aggregate": agg, "summary":
            f"{agg['total_claims']} claims, {agg['cited']} cited, "
            f"{agg['uncited']} uncited (low-confidence), {agg['hallucination']} hallucinated, "
            f"{agg['failed_briefs']} briefs with failures; citation precision {agg['citation_precision']:.3f}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    briefs = read_json(RESULTS_DIR / "briefs" / "decision_briefs.json")
    report = summarize(briefs["briefs"])
    report["run_id"] = briefs.get("run_id")
    write_json(RESULTS_DIR / "evaluation" / "citation_report.json", report)
    log(cfg, f"citation summary: {report['summary']}")
    log(cfg, f"wrote -> {RESULTS_DIR / 'evaluation' / 'citation_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
