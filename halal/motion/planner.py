"""Pick-place planning: the organiser planner driven by our Grasp strategies, with shorter joint-space paths."""
from __future__ import annotations

import copy
import itertools
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Sequence

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan, Planner, ramp, time_scale
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from halal import config
from halal.motion import sequence
from halal.motion.grasps import Grasp, right_hand
from halal.motion.obstacles import distance, outline, top_of
from halal.perception import SceneLayout

MIRROR_Y = np.array([1.0, -1.0, 1.0])
SHORTCUT_SAMPLES = 12
BIN_PHASES = ("carry", "lower", "retreat")
HELD_PHASES = ("lift", "carry")
NEAR_OBJECTS_PHASES = ("descend", "close")
OBSTACLE_CHECK_EVERY = 3
LATE_FINGER = slice(5, 7)
DIGIT_SAMPLES = np.linspace(0.0, 1.0, 5)


class PickPlanner(Planner):
    drop_shift = np.zeros(2)
    near_inset = 0.0
    low = False
    obstacles: Sequence[Any] = ()

    @classmethod
    def create(cls, safety: SafetyConfig | None = None, cfg: dict[str, Any] | None = None) -> PickPlanner:
        """Planner on the organiser config with our overrides (or `cfg`), the scene file and the sim home pose."""
        cfg, spec = cfg or config.baseline_config(), JointSpec()
        planner = cls(cfg, Scene.load(), hand_postures(cfg, spec), home_arm(cfg, spec), ArmKinematics(cfg, spec), safety, spec)
        planner.use_scene(planner.scene)
        return planner

    @property
    def motion(self) -> dict[str, Any]:
        """Our motion settings from config; each planner holds its own copy so that benchmarks can vary them."""
        return self.__dict__.setdefault("_motion", dict(config.load()["motion"]))

    def plan_pick(self, det: Any, grasp: Grasp, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray, *,
                  drop_shift: tuple[float, float] = (0.0, 0.0), end: str = "home",
                  place_xy: tuple[float, float] | None = None, squeeze: float | None = None,
                  obstacles: Sequence[Any] = ()) -> Plan:
        """Pick (closing `squeeze` rad past `closed`; None = the grasp's own depth) -> bin release point + `drop_shift` (or counter spot `place_xy`) -> `end` (home | retreat), keeping the hand off `obstacles` (other objects: detections, or bare top points of unknown outline)."""
        self.drop_shift = np.asarray(drop_shift, float)
        self.obstacles = [o for o in obstacles if np.linalg.norm(top_of(o)[:2] - det.top[:2]) > self.motion["obstacles"]["radius"]]
        insets = [0.0] if place_xy is not None else self._reachable_insets(side, bin_name)
        plan = Plan(False, side=side, color=det.color, reason=f"{side} arm cannot reach the {bin_name} bin")
        target = replace(det, top=det.top + np.r_[grasp.center_shift, 0.0])
        depth = grasp.squeeze if squeeze is None else squeeze
        with self._grasp_applied(grasp, side, depth), self._sized_as(det), self._counter_under(target):
            for self.near_inset, self.low in itertools.product(insets, self._sequences(place_xy)):
                plan = self._plan_with(grasp, target, side, bin_name, arm_q, hand_q, place_xy)
                if plan.ok:
                    break
        return self._ending_at(self._staggered_release(self._carried_gently(plan, grasp), side), end)

    def _carried_gently(self, plan: Plan, grasp: Grasp) -> Plan:
        """The plan with lift and carry re-timed to the grasp's own arm acceleration; unchanged when the grasp sets none."""
        if not plan.ok or grasp.carry_acc is None:
            return plan
        parts = []
        for name, first, stop in plan.phases:
            arm, hand = plan.arm[first:stop], plan.hand[first:stop]
            if name in HELD_PHASES:
                arm = time_scale(plan.arm[first - 1:stop], self.cfg["motion"]["arm_max_vel"], grasp.carry_acc, self.dt)
                hand = np.repeat(hand[-1][None], len(arm), 0)
            parts.append((name, arm, hand))
        return self._assembled(plan, parts)

    def _sequences(self, place_xy: tuple[float, float] | None) -> list[bool]:
        """Whether to try our low sequence, per configured sequence in order; counter placements only have the organiser's."""
        names = ["organiser"] if place_xy is not None else self.motion["sequences"]
        return [name == "low" for name in names]

    def _staggered_release(self, plan: Plan, side: str) -> Plan:
        """The plan with its release rebuilt so that one finger starts opening after the rest of the hand; unchanged when the stagger is 0."""
        if not plan.ok or not self.motion["release_stagger_s"]:
            return plan
        cols = ArmKinematics.sl(side)
        first, stop = next((a, b) for name, a, b in plan.phases if name == "open")
        opening = self._opening(plan.hand[first - 1, cols], plan.hand[stop - 1, cols])
        parts = [(name, *((plan.arm[a:b], plan.hand[a:b]) if name != "open" else self._held(plan, first, cols, opening)))
                 for name, a, b in plan.phases]
        return self._assembled(plan, parts)

    def _opening(self, closed: np.ndarray, release: np.ndarray) -> np.ndarray:
        """One hand's rows (7 joints) from closed to release and the hold after; with a stagger one finger starts that much later."""
        motion, delay = self.cfg["motion"], int(round(self.motion["release_stagger_s"] / self.dt))
        together = ramp(closed, release, motion["hand_max_vel"], self.dt)
        rows = np.vstack([together, np.repeat(release[None], delay + int(round(motion["open_hold_s"] / self.dt)), 0)])
        rows[:delay + len(together), LATE_FINGER] = np.vstack([np.repeat(closed[None], delay, 0), together])[:, LATE_FINGER]
        return rows

    def _plan_with(self, grasp: Grasp, det: Any, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray,
                   place_xy: tuple[float, float] | None) -> Plan:
        """One sequence (low or organiser) with the grasp's finger heading, or with each default finger hint in turn."""
        if grasp.finger_yaw is None:
            return self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=place_xy)
        plan = self._plan_side(det, side, bin_name, arm_q, hand_q, place_xy=place_xy)
        return self._check_finger_yaw(plan, side, grasp.finger_yaw)

    def resqueezed(self, plan: Plan, grasp: Grasp, side: str, squeeze: float) -> Plan:
        """The plan with a deeper closure inserted after its last close / squeeze phase: ramp, hold, then carry and release from the new posture."""
        cols, motion = ArmKinematics.sl(side), self.cfg["motion"]
        with self._grasp_applied(grasp, side, squeeze):
            closed, release = self.post["closed"][cols], self.post[self.cfg["grasp"]["release_posture"]][cols]
        cut = max(stop for name, _, stop in plan.phases if name in ("close", "squeeze"))
        hold = np.repeat(closed[None], int(round(motion["close_hold_s"] / self.dt)), 0)
        deeper = np.vstack([ramp(plan.hand[cut - 1, cols], closed, motion["hand_max_vel"], self.dt), hold])
        opening = self._opening(closed, release)
        parts = [(name, plan.arm[first:stop], plan.hand[first:stop]) for name, first, stop in plan.phases if stop <= cut]
        parts.append(("squeeze", *self._held(plan, cut - 1, cols, deeper)))
        for name, first, stop in plan.phases:
            if first >= cut and name == "open":
                parts.append((name, *self._held(plan, first, cols, opening)))
            elif first >= cut:
                hand = plan.hand[first:stop].copy()
                if not any(part[0] == "open" for part in parts):
                    hand[:, cols] = closed
                parts.append((name, plan.arm[first:stop], hand))
        return self._assembled(plan, parts)

    def aborted(self, plan: Plan, side: str) -> Plan:
        """The plan cut after its last close / squeeze phase: the hand reopens to the preshape and the arm goes back up to the pregrasp."""
        cols = ArmKinematics.sl(side)
        cut = max(stop for name, _, stop in plan.phases if name in ("close", "squeeze"))
        first, stop = next((a, b) for name, a, b in plan.phases if name == "descend")
        reopen = ramp(plan.hand[cut - 1, cols], plan.hand[stop - 1, cols], self.cfg["motion"]["hand_max_vel"], self.dt)
        parts = [(name, plan.arm[a:b], plan.hand[a:b]) for name, a, b in plan.phases if b <= cut]
        parts.append(("reopen", *self._held(plan, cut - 1, cols, reopen)))
        ascent = plan.arm[first - 1:stop - 1][::-1]
        parts.append(("ascend", ascent, np.repeat(parts[-1][2][-1][None], len(ascent), 0)))
        return self._assembled(plan, parts)

    @staticmethod
    def _held(plan: Plan, tick: int, cols: slice, fingers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arm and hand rows that keep everything as at `tick` while one hand runs through `fingers`."""
        arm, hand = np.repeat(plan.arm[tick][None], len(fingers), 0), np.repeat(plan.hand[tick][None], len(fingers), 0)
        hand[:, cols] = fingers
        return arm, hand

    def _assembled(self, plan: Plan, parts: list[tuple[str, np.ndarray, np.ndarray]]) -> Plan:
        """A copy of `plan` with the trajectory rebuilt from named parts and dry-run through the safety filter again."""
        arm, hand = np.vstack([a for _, a, _ in parts]), np.vstack([h for _, _, h in parts])
        stops = np.cumsum([len(a) for _, a, _ in parts])
        phases = [(name, int(stop - len(a)), int(stop)) for (name, a, _), stop in zip(parts, stops)]
        clips = self.dry_run_safety(arm[0], hand[0], arm, hand)
        return replace(plan, arm=arm, hand=hand, phases=phases, safety_clips=clips, ok=clips == 0,
                       reason="" if clips == 0 else f"{clips} safety clips in dry run")

    def set_layout(self, layout: SceneLayout) -> None:
        """Plan against the counter height and bin positions perception reports for this episode."""
        file_scene = self.__dict__.setdefault("file_scene", self.scene)
        cfg = copy.deepcopy(file_scene.cfg)
        cfg["counter"]["mat_origin_in_pelvis"][2] = float(layout.counter_z)
        for name, xy in layout.bins.items():
            cfg["bins"]["centers"][name] = file_scene.pelvis_to_mat(np.r_[xy, 0.0])[:2].tolist()
        self.use_scene(Scene(cfg))

    def use_scene(self, scene: Scene) -> None:
        """Replace the geometry (counter height, bins) all later plans work against; bin walls count as at least `min_wall_height` tall."""
        self.__dict__.setdefault("file_scene", self.scene)
        cfg = copy.deepcopy(scene.cfg)
        cfg["bins"]["wall_height"] = max(float(cfg["bins"]["wall_height"]), self.motion["min_wall_height"])
        self.scene = Scene(cfg)
        self.__dict__.pop("_reach", None)

    def reaches_bin(self, side: str, bin_name: str) -> bool:
        """Whether this arm can hold the grasp point over the bin's release point at carry and release height (cached)."""
        cache = self.__dict__.setdefault("_reach", {})
        key = (side, bin_name, self.near_inset)
        if key not in cache:
            xy = self.drop_xy(bin_name) - self.drop_shift
            heights = (self.carry_height(), self.release_height())
            cache[key] = all(self.kin.solve(side, np.r_[xy, z], self.home)[1] for z in heights)
        return cache[key]

    def _reachable_insets(self, side: str, bin_name: str) -> list[float]:
        """The configured distances from the bin's near wall that pass the reach check for this arm, largest first."""
        reachable = []
        for self.near_inset in self.motion["drop_inset"]["near"]:
            if self.reaches_bin(side, bin_name):
                reachable.append(self.near_inset)
        return reachable

    def drop_xy(self, name: str) -> np.ndarray:
        """Release point (pelvis xy): the point inside the bin's inset inner walls nearest the pelvis, plus the drop shift."""
        inner = self.scene.mat_to_pelvis(np.array(self.scene.bin_inner_box(name)))[:, :2]
        inset = np.array([self.near_inset, self.motion["drop_inset"]["side"]])
        low, high = inner.min(0) + inset, inner.max(0) - inset
        return np.clip(np.zeros(2), low, np.maximum(low, high)) + self.drop_shift

    def bin_clearance(self, side: str, arm_q: np.ndarray, hand_q: np.ndarray, margin: float | None = None) -> float:
        margin = self.motion["bin_margin"] if margin is None else margin
        return super().bin_clearance(side, arm_q, hand_q, margin=margin)

    def _cartesian(self, side: str, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, name: str, plan: Plan) -> list | None:
        """Carry in joint space when it stays above the release height; other segments become one joint move when straight enough."""
        with self._heading_free(name in BIN_PHASES):
            if name == "carry":
                reason = plan.reason
                carry = self._joint_carry(side, q0, p0, p1, self.release_height(), name, plan)
                if carry is not None:
                    return carry
                plan.reason = reason
            waypoints = super()._cartesian(side, p0, p1, q0, name, plan)
        if waypoints and len(waypoints) > 1 and self._is_straight(side, p0, p1, q0, waypoints[-1], name):
            return [waypoints[-1]]
        return waypoints

    @contextmanager
    def _heading_free(self, free: bool) -> Iterator[None]:
        """Drop the grasp's finger heading from the IK: on the way to the bin the hand may turn as the arm likes."""
        weight = self.kin.w_hint
        if free:
            self.kin.w_hint = 0.0
        try:
            yield
        finally:
            self.kin.w_hint = weight

    def release_height(self) -> float:
        """Grasp point height at which objects are released over a bin."""
        bins, g = self.scene.cfg["bins"], self.cfg["grasp"]
        return self.scene.counter_z + float(bins["wall_height"]) + self.height / 2 + g["drop_above_rim"]

    def carry_height(self) -> float:
        """Grasp point height above a bin before lowering to the release height."""
        clear_of_rim = float(self.scene.cfg["bins"]["wall_height"]) + self.height + self.motion["carry_margin"]
        return self.scene.counter_z + max(self.cfg["grasp"]["carry_height"], clear_of_rim)

    def object_clears_bins(self, p: np.ndarray) -> bool:
        """Whether an object held with its centre at p is above the rim of every bin it overlaps."""
        bins, margin = self.scene.cfg["bins"], self.motion["rim_clearance"]
        reach = np.asarray(bins["outer_size"], float) / 2 + self.height / 2 + margin
        centre = self.scene.pelvis_to_mat(p)[:2]
        over = any(np.all(np.abs(centre - np.asarray(c, float)) <= reach) for c in bins["centers"].values())
        return not over or p[2] - self.height / 2 >= self.scene.counter_z + float(bins["wall_height"]) + margin

    def _plan_side(self, det: Any, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray,
                   place_xy: tuple[float, float] | None = None) -> Plan:
        """Our low sequence, or the organiser's high one (the fallback, and the only one that places on the counter)."""
        if self.low:
            return self._clear_of_obstacles(sequence.plan_low(self, det, side, bin_name, arm_q, hand_q), side)
        return self._clear_of_obstacles(super()._plan_side(det, side, bin_name, arm_q, hand_q, place_xy=place_xy), side)

    def _clear_of_obstacles(self, plan: Plan, side: str) -> Plan:
        """Fail a plan whose fingers or thumb, while down among the objects, come too close to another object."""
        limits = self.motion["obstacles"]
        for obstacle in self.obstacles if plan.ok else ():
            if self.digit_gap(plan, side, obstacle) < limits["finger_radius"] + limits["margin"]:
                plan.ok = False
                plan.reason = "hand would touch the object at ({:+.3f},{:+.3f})".format(*top_of(obstacle)[:2])
                break
        return plan

    def digit_gap(self, plan: Plan, side: str, obstacle: Any) -> float:
        """Smallest horizontal distance (m) from a digit's centre line, while below an obstacle's top during descend and close, to its outline; inf if none comes down."""
        limits = self.motion["obstacles"]
        corners, radius = outline(obstacle, self.height, limits["radius"])
        ceiling, gap = top_of(obstacle)[2] + limits["margin"], np.inf
        for name, first, stop in plan.phases:
            for tick in range(first, stop, OBSTACLE_CHECK_EVERY) if name in NEAR_OBJECTS_PHASES else ():
                points = self._digit_points(side, plan.arm[tick], plan.hand[tick])
                low = points[points[:, 2] < ceiling]
                if len(low):
                    gap = min(gap, float(distance(low[:, :2], corners).min()) - radius)
        return gap

    def _digit_points(self, side: str, arm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
        """Points along each distal finger link (index, middle, thumb) from its joint to its tip in the pelvis frame, (15, 3)."""
        points = self.kin.hand_points(side, arm_q, hand_q)
        joints, tips = points[2:5], points[5:8]
        return np.vstack([joints + f * (tips - joints) for f in DIGIT_SAMPLES])

    def _is_straight(self, side: str, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, name: str) -> bool:
        """Whether the joint-space move q0 -> q1 keeps the grasp point near the line p0 -> p1."""
        tolerances = self.motion["shortcut_tolerance"]
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
    def _sized_as(self, det: Any) -> Iterator[None]:
        """Plan for the height the detection measured (`det.size`); the configured object height when it has none."""
        nominal = self.height
        self.height = float(getattr(det, "size", None) or nominal)
        try:
            yield
        finally:
            self.height = nominal

    @contextmanager
    def _counter_under(self, det: Any) -> Iterator[None]:
        """Plan against the counter height the detection implies (object top minus object height), not the scene file's."""
        nominal = self.scene
        if self.motion["counter_from_detection"]:
            cfg = copy.deepcopy(nominal.cfg)
            cfg["counter"]["mat_origin_in_pelvis"][2] = float(det.top[2]) - self.height
            self.scene = Scene(cfg)
        try:
            yield
        finally:
            self.scene = nominal

    @contextmanager
    def _grasp_applied(self, grasp: Grasp, side: str, squeeze: float) -> Iterator[None]:
        """Swap the grasp's overrides and the squeeze depth into the planner and kinematics state for one plan."""
        kin, g = self.kin, self.cfg["grasp"]
        saved = (kin.hint, kin.w_hint, dict(kin.offset), dict(self.post), g["height_frac"])
        if grasp.finger_yaw is not None:
            kin.hint = np.array([np.cos(grasp.finger_yaw), np.sin(grasp.finger_yaw), 0.0])
            kin.w_hint = self.motion["finger_yaw_weight"]
        if grasp.point_in_palm is not None:
            p = np.asarray(grasp.point_in_palm, float)
            kin.offset = {"left": p, "right": p * MIRROR_Y}
        for name, left in grasp.postures.items():
            self.post[name] = np.r_[np.asarray(left, float), right_hand(left)]
        if grasp.height_frac is not None:
            g["height_frac"] = grasp.height_frac
        self.post["closed"] = self._squeezed(self.post["closed"], grasp, squeeze)
        try:
            yield
        finally:
            kin.hint, kin.w_hint, kin.offset, post, g["height_frac"] = saved
            self.post.clear()
            self.post.update(post)

    def _squeezed(self, closed: np.ndarray, grasp: Grasp, squeeze: float) -> np.ndarray:
        """Closed posture (14,) pushed `squeeze` rad further along the grasp's squeeze axis, kept inside the filter's range."""
        axis = np.asarray(grasp.squeeze_axis or self.motion["squeeze_axis"], float)
        margin = self.motion["hand_limit_margin"]
        deeper = closed + squeeze * np.r_[axis, right_hand(axis)]
        return np.clip(deeper, self.spec.hand_lower + margin, self.spec.hand_upper - margin)

    def _check_finger_yaw(self, plan: Plan, side: str, finger_yaw: float) -> Plan:
        """Reject a plan whose fingers at the grasp point away from the requested heading."""
        if not plan.ok:
            return plan
        grasp_tick = next(end for name, _, end in plan.phases if name == "descend") - 1
        self.kin._fk(plan.arm[grasp_tick])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        error = abs((np.arctan2(x[1], x[0]) - finger_yaw + np.pi) % (2 * np.pi) - np.pi)
        if error > self.motion["finger_yaw_tolerance"]:
            plan.ok, plan.reason = False, f"fingers {np.degrees(error):.0f} deg off the requested heading"
        return plan
