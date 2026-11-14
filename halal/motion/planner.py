"""Pick-place planning: the organiser planner driven by our Grasp strategies."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, home_arm, mirror_left_to_right
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan, Planner
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from halal import config
from halal.motion.grasps import Grasp

MIRROR_Y = np.array([1.0, -1.0, 1.0])


class PickPlanner(Planner):
    @classmethod
    def create(cls, safety: SafetyConfig | None = None, cfg: dict[str, Any] | None = None) -> PickPlanner:
        """Planner on the organiser config with our overrides (or `cfg`), the scene file and the sim home pose."""
        cfg, spec = cfg or config.baseline_config(), JointSpec()
        return cls(cfg, Scene.load(), hand_postures(cfg, spec), home_arm(cfg, spec), ArmKinematics(cfg, spec), safety, spec)

    def plan_pick(self, det: Any, grasp: Grasp, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray) -> Plan:
        """Full pick -> bin -> home plan for one arm with one grasp; `Plan.ok` is False with a reason on failure."""
        target = replace(det, top=det.top + np.r_[grasp.center_shift, 0.0])
        with self._grasp_applied(grasp, side):
            if grasp.finger_yaw is None:
                return self._plan_side_hints(target, side, bin_name, arm_q, hand_q)
            plan = self._plan_side(target, side, bin_name, arm_q, hand_q)
            return self._check_finger_yaw(plan, side, grasp.finger_yaw)

    @contextmanager
    def _grasp_applied(self, grasp: Grasp, side: str) -> Iterator[None]:
        """Swap the grasp's overrides into the planner and kinematics state for the duration of one plan."""
        kin, g = self.kin, self.cfg["grasp"]
        saved = (kin.hint, kin.w_hint, dict(kin.offset), dict(self.post), g["height_frac"])
        if grasp.finger_yaw is not None:
            kin.hint = np.array([np.cos(grasp.finger_yaw), np.sin(grasp.finger_yaw), 0.0])
            kin.w_hint = config.load()["motion"]["finger_yaw_weight"]
        if grasp.point_in_palm is not None:
            p = np.asarray(grasp.point_in_palm, float)
            kin.offset = {"left": p, "right": p * MIRROR_Y}
        for name, left in grasp.postures.items():
            self.post[name] = np.r_[np.asarray(left, float), mirror_left_to_right(left)]
        if grasp.height_frac is not None:
            g["height_frac"] = grasp.height_frac
        try:
            yield
        finally:
            kin.hint, kin.w_hint, kin.offset, post, g["height_frac"] = saved
            self.post.clear()
            self.post.update(post)

    def _check_finger_yaw(self, plan: Plan, side: str, finger_yaw: float) -> Plan:
        """Reject a plan whose fingers at the grasp point away from the requested heading."""
        if not plan.ok:
            return plan
        grasp_tick = next(end for name, _, end in plan.phases if name == "descend") - 1
        self.kin._fk(plan.arm[grasp_tick])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        error = abs((np.arctan2(x[1], x[0]) - finger_yaw + np.pi) % (2 * np.pi) - np.pi)
        if error > config.load()["motion"]["finger_yaw_tolerance"]:
            plan.ok, plan.reason = False, f"fingers {np.degrees(error):.0f} deg off the requested heading"
        return plan
