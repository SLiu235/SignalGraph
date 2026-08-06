"""
quant_signal_game.common — shared utilities: config loading, deterministic RNG,
logging, and result writing. Every layer imports from here so seeds and configs
are pinned once and stages stay re-runnable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PKG_ROOT / "results"


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "run_id" in cfg and not cfg["run_id"]:
        cfg["run_id"] = make_run_id(cfg.get("seed", 42))
    cfg.setdefault("paths", {}).setdefault("root", str(PKG_ROOT))
    return cfg


def make_run_id(seed: int) -> str:
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{seed}"


def get_rng(cfg: dict, salt: str = "") -> np.random.Generator:
    """Deterministic RNG seeded from config seed + salt, so sub-stages can be
    re-run in isolation and still reproduce the exact same data."""
    seed = int(cfg.get("seed", 42))
    return np.random.default_rng(seed + sum(ord(c) for c in salt))


def log(cfg: dict, msg: str) -> None:
    if cfg.get("verbose", True):
        print(f"[{cfg.get('run_id', '?')}] {msg}")


def write_json(path: Path | str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def day_of(timestamp: str) -> int:
    """Parse synthetic 'D####' timestamps into an integer day; ISO dates fall back
    to 0 (live data ordering handled upstream). Shared by retrieval and evaluation."""
    if timestamp.startswith("D"):
        try:
            return int(timestamp[1:])
        except ValueError:
            return 0
    return 0
