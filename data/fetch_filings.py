"""
data.fetch_filings — SEC EDGAR filings (10-K / 8-K) with a deterministic synthetic
fallback. In synthetic mode (the default) it returns the same records as seed_graph,
so offline runs reproduce exactly. In edgar mode it attempts live SEC retrieval and
falls back to synthetic on any failure (network, rate-limit, missing user-agent).

SEC requires: a descriptive user-agent and >= 100ms between requests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import load_config, log
from ..graph import ontology as O
from ..data import seed_graph


def fetch_filings(cfg: dict, companies: list) -> list:
    mode = cfg["data"]["mode"]
    if mode == "edgar":
        try:
            from . import _edgar
            return _edgar.fetch_live(cfg, companies)
        except Exception as e:  # network / parsing / UA missing
            log(cfg, f"EDGAR live fetch failed ({e}); falling back to synthetic")
    # synthetic: reuse the deterministic seed filing generator
    _, _, signals = seed_graph.build_seed_records(cfg)
    return [s for s in signals if s.source == "filing"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("quant_signal_game/config/config.yaml")))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    filings = fetch_filings(cfg, [])
    log(cfg, f"fetched {len(filings)} filings (mode={cfg['data']['mode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
