"""Wiring: the sorter built from config.yaml and the three seams (perception, motion, and the mapping given at reset)."""
from __future__ import annotations

from typing import Any

from halal import config
from halal.motion import PickPlanner
from halal.perception import make_detector
from halal.policy.sorter import SortPolicy


def planner_config() -> dict[str, Any]:
    """The organiser config with our overrides and the arm pose named by `policy.look_pose` as its home pose."""
    settings, cfg = config.load(), config.baseline_config()
    name = settings["policy"]["look_pose"]
    if name != "home":
        cfg["home_arm_q"] = settings["poses"][name]
    return cfg


def make_policy(time_limit: float | None = None) -> SortPolicy:
    """The sorter with the configured detector; picks start and end at the look pose; `time_limit` overrides the configured one."""
    settings = config.load()
    detector = make_detector(settings["perception"]["detector"])
    planner = PickPlanner.create(cfg=planner_config())
    return SortPolicy(detector, planner, planner.home, settings["policy"], time_limit or settings["policy"]["time_limit_s"])
