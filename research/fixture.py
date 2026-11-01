"""Fabricated calendar, issuers, filings, and prices for plumbing tests only."""
from datetime import date, timedelta
import hashlib

import numpy as np


def create_fixture(seed=42):
    rng = np.random.default_rng(seed)
    dates = []
    day = date(2023, 1, 2)
    while len(dates) < 260:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    factors = [{"session": d, "mkt_rf": float(rng.normal(0, .008)),
                "smb": float(rng.normal(0, .004)), "hml": float(rng.normal(0, .004)),
                "rf": .0001} for d in dates]
    bars, events, memberships, sector_returns = [], [], [], []
    for j, ticker in enumerate(("DEMOA", "DEMOB", "DEMOC", "DEMOD")):
        sector = "fixture-tech" if j < 2 else "fixture-industrial"
        memberships.append({"ticker": ticker, "sector": sector, "valid_from": "2022-01-01T00:00:00Z",
                            "valid_to": None, "available_at": "2022-01-01T00:00:00Z"})
        price = 100.
        for i, (d, f) in enumerate(zip(dates, factors)):
            price *= 1 + f["rf"] + (0.8 + j * .2) * f["mkt_rf"] + .2 * f["smb"] - .1 * f["hml"] + rng.normal(0, .006)
            bars.append({"ticker": ticker, "session": d, "open_at": d + "T14:30:00Z",
                         "close_at": d + "T21:00:00Z", "available_at": d + "T21:00:00Z",
                         "adjusted_close": float(price), "volume": 1_000_000.0})
            if 65 <= i <= 220 and i % 5 == 0:
                # Text is independent of future returns. This fixture intentionally
                # makes no claim to implement a profitable data-generating process.
                word = rng.choice(["improved growth", "declined loss", "unchanged results"])
                text = f"Item 2.02 Results of Operations. {ticker} {word}. Filing {i}."
                accepted = d + "T21:01:00Z"
                events.append({"event_id": f"fixture-{ticker}-{i}", "ticker": ticker, "cik": str(j + 1),
                               "source": "fixture_edgar", "form": "8-K", "event_time": accepted,
                               "available_at": d + "T21:02:00Z", "retrieved_at": "2025-01-01T00:00:00Z",
                               "document_text": text, "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "document_url": "fixture://fabricated-document",
                               "provenance": {"data_kind": "fixture", "availability_mode": "historical"}})
    for f in factors:
        for sector in ("fixture-tech", "fixture-industrial"):
            sector_returns.append({"session": f["session"], "sector": sector,
                                   "return": f["mkt_rf"] + f["rf"]})
    session_schedule = {"version": 1, "calendar_id": "fixture-weekday-calendar-v1",
                        "source": "fabricated fixed-UTC session schedule",
                        "retrieved_at": "2022-01-01T00:00:00Z",
                        "sessions": [{"session": d, "open_at": d + "T14:30:00Z",
                                      "close_at": d + "T21:00:00Z"} for d in dates]}
    market = {"metadata": {"data_kind": "fixture", "source": "fabricated independent-factor fixture",
                           "adjustment": "split_and_dividend", "universe_policy": "four fictional issuers",
                           "retrieved_at": "2025-01-01T00:00:00Z",
                           "calendar": "fabricated weekdays at fixed UTC hours, not an exchange calendar"},
              "session_schedule": session_schedule,
              "bars": bars, "benchmark": [{"session": f["session"], "return": f["mkt_rf"] + f["rf"]} for f in factors],
              "factors": factors, "memberships": memberships, "sector_returns": sector_returns}
    return {"metadata": {"data_kind": "fixture", "source": "fabricated", "availability_mode": "historical"},
            "events": events}, market
