"""
data._edgar — live SEC EDGAR retrieval (optional). Requires `requests` and a
descriptive user-agent in config (SEC policy). Raises on missing deps/UA so
fetch_filings falls back to synthetic deterministically.
"""
from __future__ import annotations

import time

from ..graph import ontology as O


def fetch_live(cfg: dict, companies: list):
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed (pip install requests)")

    ua = cfg["data"]["edgar"]["user_agent"]
    if not ua:
        raise RuntimeError("config data.edgar.user_agent is empty; SEC requires a descriptive UA")

    # Minimal ticker -> CIK map from SEC's company_tickers.json.
    # Returns filings only for tickers present in the seed universe.
    tickers = {c.ticker for c in companies} if companies else {t for t, *_ in seed_universe()}
    resp = requests.get(cfg["data"]["edgar"]["submissions_url"], headers={"User-Agent": ua})
    resp.raise_for_status()
    cik_by_ticker = {}
    for rec in resp.json():
        t = rec.get("title", "").upper().split(" ")[-1].strip(".,")
        cik_by_ticker[rec.get("ticker", "").upper()] = rec.get("cik_str")
    out = []
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if not cik:
            continue
        time.sleep(cfg["data"]["edgar"]["rate_limit_s"])
        # submissions endpoint returns filing metadata; keep it lightweight.
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers={"User-Agent": ua})
        r.raise_for_status()
        data = r.json()
        for form, filed, accn in zip(data.get("form", []), data.get("filed", []), data.get("accessionNumber", [])):
            if form in ("10-K", "8-K"):
                out.append(O.Signal(
                    signal_id=f"edgar_{ticker.lower()}_{accn}", signal_type="certified_filing",
                    kind="costly", content=f"SEC {form} filing {accn} filed {filed}",
                    sender="", company=ticker, timestamp=filed, source="filing",
                    source_id=accn))
    return out


def seed_universe():
    from ..data.seed_graph import SEED_COMPANIES
    return SEED_COMPANIES
