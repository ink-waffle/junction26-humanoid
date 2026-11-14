"""Wiring: the sorter built from config.yaml and the three seams (perception, motion, and the mapping given at reset)."""
from __future__ import annotations

from aaltonomous.baselines.classical.config import hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from halal import config
from halal.motion import PickPlanner
from halal.perception import make_detector
from halal.policy.sorter import HZ, SortPolicy


def make_planner() -> PickPlanner:
    """The pick planner on the organiser config (sim profile plus our overrides)."""
    cfg = config.baseline_config()
    spec = JointSpec()
    return PickPlanner(cfg, Scene.load(), hand_postures(cfg, spec), home_arm(cfg, spec), ArmKinematics(cfg, spec),
                       SafetyConfig(dt=1.0 / HZ), spec)


def make_policy(time_limit: float | None = None) -> SortPolicy:
    """The sorter with the configured detector; `time_limit` overrides the configured episode length."""
    settings = config.load()
    detector = make_detector(settings["perception"]["detector"])
    return SortPolicy(detector, make_planner(), settings["policy"], time_limit or settings["policy"]["time_limit_s"])
