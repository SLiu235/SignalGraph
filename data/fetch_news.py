"""
data.fetch_news — news / transcript snippets with a deterministic synthetic fallback.
Synthetic mode returns the same records as seed_graph (offline reproducible).
Live mode is a stub that could be pointed at a news/transcript provider.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log
from ..graph import ontology as O
from ..data import seed_graph


def fetch_news(cfg: dict, companies: list) -> list:
    _, _, signals = seed_graph.build_seed_records(cfg)
    return [s for s in signals if s.source == "news"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    news = fetch_news(cfg, [])
    log(cfg, f"fetched {len(news)} news snippets (synthetic deterministic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
