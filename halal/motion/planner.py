"""Pick-place planning: the organiser planner driven by our Grasp strategies, with shorter joint-space paths."""
from __future__ import annotations

import copy
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
SHORTCUT_SAMPLES = 12


class PickPlanner(Planner):
    drop_shift = np.zeros(2)

    @classmethod
    def create(cls, safety: SafetyConfig | None = None, cfg: dict[str, Any] | None = None) -> PickPlanner:
        """Planner on the organiser config with our overrides (or `cfg`), the scene file and the sim home pose."""
        cfg, spec = cfg or config.baseline_config(), JointSpec()
        return cls(cfg, Scene.load(), hand_postures(cfg, spec), home_arm(cfg, spec), ArmKinematics(cfg, spec), safety, spec)

    def plan_pick(self, det: Any, grasp: Grasp, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray, *,
                  drop_shift: tuple[float, float] = (0.0, 0.0), end: str = "home",
                  place_xy: tuple[float, float] | None = None) -> Plan:
        """Pick -> bin release point + `drop_shift` (or counter spot `place_xy`) -> `end` (home | retreat); `Plan.ok` False with a reason on failure."""
        if place_xy is None and not self.reaches_bin(side, bin_name):
            return Plan(False, side=side, color=det.color, reason=f"{side} arm cannot reach the {bin_name} bin")
        target = replace(det, top=det.top + np.r_[grasp.center_shift, 0.0])
        self.drop_shift = np.asarray(drop_shift, float)
        with self._grasp_applied(grasp, side), self._counter_under(target):
            if grasp.finger_yaw is None:
                plan = self._plan_side_hints(target, side, bin_name, arm_q, hand_q, place_xy=place_xy)
            else:
                plan = self._plan_side(target, side, bin_name, arm_q, hand_q, place_xy=place_xy)
                plan = self._check_finger_yaw(plan, side, grasp.finger_yaw)
        return self._ending_at(plan, end)

    def reaches_bin(self, side: str, bin_name: str) -> bool:
        """Whether this arm can hold the grasp point above the bin's release point at carry height (cached)."""
        cache = self.__dict__.setdefault("_reach", {})
        if (side, bin_name) not in cache:
            g, bins = self.cfg["grasp"], self.scene.cfg["bins"]
            z = self.scene.counter_z + max(g["carry_height"], float(bins["wall_height"]) + self.height + 0.03)
            cache[side, bin_name] = bool(self.kin.solve(side, np.r_[super().drop_xy(bin_name), z], self.home)[1])
        return cache[side, bin_name]

    def drop_xy(self, name: str) -> np.ndarray:
        return super().drop_xy(name) + self.drop_shift

    def bin_clearance(self, side: str, arm_q: np.ndarray, hand_q: np.ndarray, margin: float | None = None) -> float:
        margin = config.load()["motion"]["bin_margin"] if margin is None else margin
        return super().bin_clearance(side, arm_q, hand_q, margin=margin)

    def _cartesian(self, side: str, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, name: str, plan: Plan) -> list | None:
        """Carry in joint space when it stays above the release height; other segments become one joint move when straight enough."""
        if name == "carry":
            reason = plan.reason
            carry = self._joint_carry(side, q0, p0, p1, self._release_height(), name, plan)
            if carry is not None:
                return carry
            plan.reason = reason
        waypoints = super()._cartesian(side, p0, p1, q0, name, plan)
        if waypoints and len(waypoints) > 1 and self._is_straight(side, p0, p1, q0, waypoints[-1], name):
            return [waypoints[-1]]
        return waypoints

    def _release_height(self) -> float:
        """Grasp point height at which objects are released over a bin."""
        bins, g = self.scene.cfg["bins"], self.cfg["grasp"]
        return self.scene.counter_z + float(bins["wall_height"]) + self.height / 2 + g["drop_above_rim"]

    def _is_straight(self, side: str, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, name: str) -> bool:
        """Whether the joint-space move q0 -> q1 keeps the grasp point near the line p0 -> p1."""
        tolerances = config.load()["motion"]["shortcut_tolerance"]
        tolerance = tolerances.get(name, tolerances["default"])
        p0, line = np.asarray(p0, float), np.asarray(p1, float) - np.asarray(p0, float)
        for f in np.linspace(0, 1, SHORTCUT_SAMPLES)[1:-1]:
            p = self.kin.grasp_pose(side, q0 + f * (q1 - q0))[0]
            along = np.clip((p - p0) @ line / (line @ line), 0, 1)
            if np.linalg.norm(p - (p0 + along * line)) > tolerance:
                return False
        return True

    @staticmethod
    def _ending_at(plan: Plan, end: str) -> Plan:
        """Cut an ok plan after its phase `end`; `home` keeps all of it."""
        if not plan.ok or end == "home":
            return plan
        last = next(stop for name, _, stop in plan.phases if name == end)
        plan.arm, plan.hand = plan.arm[:last], plan.hand[:last]
        plan.phases = [phase for phase in plan.phases if phase[2] <= last]
        return plan

    @contextmanager
    def _counter_under(self, det: Any) -> Iterator[None]:
        """Plan against the counter height the detection implies (object top minus object height), not the scene file's."""
        nominal = self.scene
        if config.load()["motion"]["counter_from_detection"]:
            cfg = copy.deepcopy(nominal.cfg)
            cfg["counter"]["mat_origin_in_pelvis"][2] = float(det.top[2]) - self.height
            self.scene = Scene(cfg)
        try:
            yield
        finally:
            self.scene = nominal

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
