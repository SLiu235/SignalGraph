"""
data._edgar — live SEC EDGAR retrieval (optional). Requires `requests` and a
descriptive user-agent in config (SEC policy). Raises on missing deps so
fetch_filings falls back to synthetic deterministically.

Two entry points:
  fetch_live(cfg, companies)            — full fetch (all filings in seed universe)
  fetch_incremental(cfg, cursor)        — incremental: only filings new since the
        ingestion cursor (data/ingest.py). Tracks the set of seen accession
        numbers plus the latest filed date, so re-runs never re-ingest.
"""
from __future__ import annotations

import time

from graph import ontology as O


def _require_requests():
    try:
        import requests
        return requests
    except ImportError:
        raise RuntimeError("requests not installed (pip install requests)")


def _fetch_raw(cfg: dict, tickers: list) -> list:
    """Network part: returns raw records [{ticker, form, filed, accn}] for the
    requested tickers. Respects SEC rate limiting (>= 100ms per request)."""
    requests = _require_requests()
    ua = cfg["data"]["edgar"]["user_agent"]
    resp = requests.get(cfg["data"]["edgar"]["submissions_url"], headers={"User-Agent": ua})
    resp.raise_for_status()
    cik_by_ticker = {}
    for rec in resp.json():
        cik_by_ticker[rec.get("ticker", "").upper()] = rec.get("cik_str")

    out = []
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if not cik:
            continue
        time.sleep(cfg["data"]["edgar"]["rate_limit_s"])
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers={"User-Agent": ua})
        r.raise_for_status()
        data = r.json()
        for form, filed, accn in zip(data.get("form", []), data.get("filed", []),
                                     data.get("accessionNumber", [])):
            out.append({"ticker": ticker, "form": form, "filed": filed, "accn": accn})
    return out


def _to_signal(rec: dict) -> O.Signal:
    return O.Signal(
        signal_id=f"edgar_{rec['ticker'].lower()}_{rec['accn']}",
        signal_type="certified_filing", kind="costly",
        content=f"SEC {rec['form']} filing {rec['accn']} filed {rec['filed']}",
        sender="", company=rec["ticker"], timestamp=rec["filed"], source="filing",
        source_id=rec["accn"])


def fetch_live(cfg: dict, companies: list):
    """Full EDGAR fetch for the seed universe (used by fetch_filings.py)."""
    ua = cfg["data"]["edgar"]["user_agent"]
    if not ua:
        raise RuntimeError("config data.edgar.user_agent is empty; SEC requires a descriptive UA")
    tickers = {c.ticker for c in companies} if companies else {t for t, *_ in seed_universe()}
    return [_to_signal(r) for r in _fetch_raw(cfg, list(tickers))]


def fetch_incremental(cfg: dict, cursor: dict):
    """Incremental EDGAR ingestion.

    Returns (signals, events, info) where info is a short status string.
    Only filings that are new since the cursor are returned; the cursor's
    `seen` accession set and `last_filed` watermark are updated in place.

    When data.edgar.user_agent is empty (SEC requires one) this skips cleanly
    so the offline ingest path never blocks."""
    ua = cfg["data"]["edgar"]["user_agent"]
    if not ua:
        return [], [], "skipped (data.edgar.user_agent empty)"
    src = cursor.setdefault("sources", {}).setdefault("edgar", {"seen": [], "last_filed": ""})
    seen = set(src.get("seen", []))
    last_filed = src.get("last_filed", "")

    tickers = [t for t, *_ in seed_universe()]
    signals = []
    max_filed = last_filed
    for rec in _fetch_raw(cfg, tickers):
        if rec["form"] not in ("10-K", "8-K"):
            continue
        # Dedupe by accession number (the seen set). SEC files multiple 8-Ks on
        # the same day, so a date-watermark filter would silently drop unseen
        # accessions filed on-or-before last_filed — never skip by date alone.
        if rec["accn"] in seen:
            continue
        signals.append(_to_signal(rec))
        seen.add(rec["accn"])
        if not max_filed or rec["filed"] > max_filed:
            max_filed = rec["filed"]

    src["seen"] = sorted(seen)
    src["last_filed"] = max_filed
    return signals, [], f"{len(signals)} new filings (last_filed={max_filed})"


def seed_universe():
    from data.seed_graph import SEED_COMPANIES
    return SEED_COMPANIES
