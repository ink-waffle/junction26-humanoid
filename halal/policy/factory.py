"""Wiring: the sorter built from config.yaml and the three seams (perception, motion, and the mapping given at reset)."""
from __future__ import annotations

from typing import Any

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import make_detector
from halal.policy.release import ShiftedReleasePlanner
from halal.policy.sorter import HZ, SortPolicy


def look_pose(cfg: dict[str, Any], spec: JointSpec) -> np.ndarray:
    """The arm pose named by `policy.look_pose` (the organiser home pose or an entry of `poses`), checked against the limits."""
    settings = config.load()
    name = settings["policy"]["look_pose"]
    if name != "home":
        cfg = {**cfg, "home_arm_q": settings["poses"][name]}
    return home_arm(cfg, spec)


def make_planner() -> ShiftedReleasePlanner:
    """The pick planner on the organiser config (sim profile plus our overrides); its picks start and end at the look pose."""
    cfg = config.baseline_config()
    spec = JointSpec()
    return ShiftedReleasePlanner(cfg, Scene.load(), hand_postures(cfg, spec), look_pose(cfg, spec),
                                 ArmKinematics(cfg, spec), SafetyConfig(dt=1.0 / HZ), spec)


def make_policy(time_limit: float | None = None) -> SortPolicy:
    """The sorter with the configured detector; `time_limit` overrides the configured episode length."""
    settings = config.load()
    detector = make_detector(settings["perception"]["detector"])
    planner = make_planner()
    return SortPolicy(detector, planner, planner.home, settings["policy"], time_limit or settings["policy"]["time_limit_s"])
