"""Our settings (config.yaml) and the organiser baseline config with our overrides applied."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from aaltonomous.baselines.classical import config as baseline

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=None)
def load() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def baseline_config() -> dict[str, Any]:
    """Organiser config (sim profile) with `baseline_overrides` applied; a fresh dict per call."""
    return baseline.load_config(profile="sim", overrides=load()["baseline_overrides"])
