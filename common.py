"""
SignalGraph common utilities: config loading, deterministic RNG,
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
DEFAULT_CONFIG_PATH = PKG_ROOT / "config" / "config.yaml"


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not cfg.get("run_id"):
        # A Makefile invocation supplies one shared ID so every stage of a pipeline
        # run can be traced together. Direct module execution still gets a fresh ID.
        cfg["run_id"] = os.environ.get("SIGNAL_GRAPH_RUN_ID") or make_run_id(cfg.get("seed", 42))
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
    """Parse synthetic D#### timestamps only; ambiguous real times fail closed."""
    if timestamp.startswith("D"):
        try:
            return int(timestamp[1:])
        except ValueError:
            raise ValueError("Synthetic D#### timestamp required; ISO timestamps need an explicit availability adapter")
    raise ValueError("Synthetic D#### timestamp required; ISO timestamps need an explicit availability adapter")
