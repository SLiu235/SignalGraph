"""
data.ingest — real incremental event ingestion.

A cursor persists, per source, how far ingestion has advanced
(results/ingest/cursor.json), so each run pulls ONLY new events and applies them
incrementally to the graph (via graph.update.Updater -> graph.storage backend).

Sources (config data.ingest.sources):
  edgar -> real SEC EDGAR filings (data/_edgar.fetch_incremental). Requires a
           descriptive data.edgar.user_agent; skips cleanly when absent. The
           cursor tracks seen accession numbers + last filed date, so re-runs
           never re-ingest.
  feed  -> deterministic synthetic filing feed (offline, reproducible). Each
           poll emits a new batch of filings + price_move events with ids
           feed_{poll}_{i} and timestamps beyond the seed, so incremental
           ingestion is exercised offline exactly like a live feed.
  news  -> deterministic synthetic news feed (same offline design as feed).

Feed timestamps start at data.ingest.feed_start_day, which defaults above
evaluation.brief_day, so ingested events stay strictly post-brief and never
leak into existing decision briefs (no-lookahead preserved).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import DEFAULT_CONFIG_PATH, load_config, log, read_json, write_json, get_rng
from graph import ontology as O
from data.seed_graph import SEED_COMPANIES


def cursor_path(cfg: dict) -> Path:
    p = Path(cfg["data"]["ingest"]["cursor_path"])
    return p if p.is_absolute() else Path(cfg["paths"]["root"]) / p


def load_cursor(cfg: dict) -> dict:
    p = cursor_path(cfg)
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            pass
    return {"sources": {}}


def save_cursor(cfg: dict, cursor: dict) -> Path:
    import datetime
    cursor["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    p = cursor_path(cfg)
    write_json(p, cursor)
    return p


def _feed_poll(cfg: dict, cursor: dict, source: str):
    """Deterministic next batch for the synthetic feed/news source. One signal
    (filing for `feed`, media for `news`) plus one price_move event per item,
    all at timestamps beyond the seed and with ids unique to this poll."""
    src = cursor.setdefault("sources", {}).setdefault(source, {"poll": 0})
    poll = src.get("poll", 0)
    rng = get_rng(cfg, salt=f"ingest_{source}_{poll}")
    per = int(cfg["data"]["ingest"]["feed_events_per_poll"])
    start = int(cfg["data"]["ingest"]["feed_start_day"])
    step = int(cfg["data"]["ingest"]["feed_step_days"])

    signals, events = [], []
    for i in range(per):
        ticker = SEED_COMPANIES[rng.integers(0, len(SEED_COMPANIES))][0]
        day = start + poll * step + i
        ts = f"D{day:04d}"
        if source == "feed":
            sig = O.Signal(
                signal_id=f"feed_{poll}_{i}", signal_type="certified_filing", kind="costly",
                content=(f"Incremental filing for {ticker}: guidance "
                         f"{rng.choice(['raised', 'raised', 'lowered'])}; "
                         f"EPS {rng.uniform(1.0, 6.0):.2f}, costs "
                         f"{rng.choice(['under control', 'rising'])}."),
                sender="", company=ticker, timestamp=ts, source="filing",
                source_id=f"feed-{poll}-{i}")
        else:
            sig = O.Signal(
                signal_id=f"news_{poll}_{i}", signal_type="media", kind="cheap_talk",
                content=(f"Headline on {ticker}: "
                         f"{rng.choice(['positive', 'neutral', 'negative'])} framing "
                         f"of quarterly results."),
                sender="", company=ticker, timestamp=ts, source="news",
                source_id=f"news-{poll}-{i}")
        signals.append(sig)
        events.append(O.Event(
            event_id=f"evt_{source}_{poll}_{i}", event_type="price_move",
            timestamp=ts, company=ticker,
            payload={"move_pct": round(rng.uniform(-0.04, 0.04), 4)}))

    src["poll"] = poll + 1
    return signals, events


def ingest_new(cfg: dict, sources_filter=None, commit: bool = True):
    """Fetch and return (signals, events) that are NEW since the cursor, then
    advance the cursor. `sources_filter` restricts to a subset of enabled
    sources (e.g. ['feed']); None uses every enabled source. When `commit` is
    False (dry-run) the cursor is NOT advanced, so nothing is marked consumed."""
    cursor = load_cursor(cfg)
    enabled = list(cfg["data"]["ingest"].get("sources", ["feed"]))
    if sources_filter:
        enabled = [s for s in enabled if s in sources_filter]

    signals, events = [], []
    for source in enabled:
        if source == "edgar":
            from . import _edgar
            ns, ne, info = _edgar.fetch_incremental(cfg, cursor)
            signals += ns
            events += ne
            log(cfg, f"ingest[edgar]: {info}")
        elif source in ("feed", "news"):
            ns, ne = _feed_poll(cfg, cursor, source)
            signals += ns
            events += ne
            log(cfg, f"ingest[{source}]: {len(ns)} signals + {len(ne)} events (poll "
                      f"{cursor['sources'][source]['poll']})")
        else:
            log(cfg, f"ingest: unknown source '{source}' skipped")

    if commit:
        save_cursor(cfg, cursor)
        log(cfg, f"ingest: cursor -> {cursor_path(cfg)} "
                 f"({len(signals)} signals + {len(events)} events new)")
    else:
        log(cfg, "ingest: dry-run — cursor not advanced")
    return signals, events


def main(argv=None):
    ap = argparse.ArgumentParser(description="incremental event ingestion")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--sources", default=None,
                    help="comma-separated source list to poll (default: all enabled)")
    ap.add_argument("--dry-run", action="store_true", help="fetch but do not persist")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    sources = args.sources.split(",") if args.sources else None
    signals, events = ingest_new(cfg, sources_filter=sources, commit=not args.dry_run)
    if args.dry_run:
        log(cfg, "dry-run: cursor not advanced, nothing persisted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
