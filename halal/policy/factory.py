"""Wiring: the sorter built from config.yaml and the three seams (perception, motion, and the mapping given at reset)."""
from __future__ import annotations

from halal import config
from halal.motion import PickPlanner
from halal.perception import make_detector
from halal.policy.sorter import SortPolicy


def make_policy(time_limit: float | None = None) -> SortPolicy:
    """The sorter with the configured detector; `time_limit` overrides the configured episode length."""
    settings = config.load()
    detector = make_detector(settings["perception"]["detector"])
    clear_pose = settings["motion"][settings["policy"]["clear_view_pose"]]
    return SortPolicy(detector, PickPlanner.create(), clear_pose, settings["policy"],
                      time_limit or settings["policy"]["time_limit_s"])
