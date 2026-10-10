"""
evaluation.citation_check — per-metric citation / hallucination failure summaries.
For each decision brief we count:
  * total claims, cited claims, uncited (uncited -> LOW-CONFIDENCE),
  * hallucination count (claims that assert facts with no retrieved source),
  * citation coverage = cited / total.
A summary is emitted per metric so failures are visible per-metric, not aggregated away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config, log, read_json, write_json


def check_brief(brief: dict) -> dict:
    evidence = brief.get("evidence", [])
    total = len(evidence)
    known = {x["source_id"] for x in brief.get("sources_used", [])}
    cited = sum(1 for c in evidence if c.get("citations") and all(x in known for x in c["citations"]))
    uncited = total - cited
    halluc = len(brief.get("low_confidence_claims", []))
    precision = cited / total if total else 0.0
    return {
        "total_claims": total, "cited_claims": cited,
        "uncited_claims": uncited, "hallucination_count": halluc,
        "citation_coverage": round(precision, 3),
        "semantic_entailment": "not evaluated",
        "failure": uncited > 0 or halluc > 0,
    }


def summarize(briefs: dict) -> dict:
    per_metric = {"citation_coverage": {}}
    agg = {"total_claims": 0, "cited": 0, "uncited": 0, "hallucination": 0,
           "failed_briefs": 0}
    for ticker, b in briefs.items():
        r = check_brief(b)
        per_metric["citation_coverage"][ticker] = r
        agg["total_claims"] += r["total_claims"]
        agg["cited"] += r["cited_claims"]
        agg["uncited"] += r["uncited_claims"]
        agg["hallucination"] += r["hallucination_count"]
        if r["failure"]:
            agg["failed_briefs"] += 1
    agg["citation_coverage"] = (agg["cited"] / agg["total_claims"]
                                 if agg["total_claims"] else 0.0)
    return {"run_id": briefs.get("run_id"), "per_metric": per_metric,
            "aggregate": agg, "summary":
            f"{agg['total_claims']} claims, {agg['cited']} cited, "
            f"{agg['uncited']} uncited (low-confidence), {agg['hallucination']} hallucinated, "
            f"{agg['failed_briefs']} briefs with failures; citation coverage {agg['citation_coverage']:.3f}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
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
