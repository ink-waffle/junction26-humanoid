"""Our pick sequence: pregrasp and lift just above the objects, then a rising joint-space carry to the bin."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan, ramp, time_scale

if TYPE_CHECKING:
    from halal.motion.planner import PickPlanner

CHECK_EVERY = 3
CARRY_SAMPLES = 20
CLOSING_CLEARANCE = 0.003
RIM_TOUCH = 0.003


@dataclass
class Timeline:
    """30 Hz arm and hand targets grown phase by phase from a start state."""
    planner: PickPlanner
    arm: np.ndarray
    hand: np.ndarray
    parts: list[tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)

    def move(self, name: str, waypoints: list[np.ndarray], hand_to: np.ndarray | None = None) -> None:
        """Arm through joint waypoints at the speed limits; the hand may ramp to a posture meanwhile."""
        motion = self.planner.cfg["motion"]
        arm = time_scale([self.arm, *waypoints], motion["arm_max_vel"], motion["arm_max_acc"], self.planner.dt)
        fingers = np.empty((0, len(self.hand))) if hand_to is None else ramp(self.hand, hand_to, motion["hand_max_vel"], self.planner.dt)
        ticks = max(len(arm), len(fingers))
        self._add(name, _padded(arm, ticks, self.arm), _padded(fingers, ticks, self.hand))

    def grip(self, name: str, posture: np.ndarray, hold_s: float) -> None:
        """Hand to a posture with the arm still, then hold."""
        fingers = ramp(self.hand, posture, self.planner.cfg["motion"]["hand_max_vel"], self.planner.dt)
        ticks = len(fingers) + int(round(hold_s / self.planner.dt))
        self._add(name, _padded(np.empty((0, len(self.arm))), ticks, self.arm), _padded(fingers, ticks, posture))

    def _add(self, name: str, arm: np.ndarray, hand: np.ndarray) -> None:
        if len(arm):
            self.parts.append((name, arm, hand))
            self.arm, self.hand = arm[-1].copy(), hand[-1].copy()


def _padded(rows: np.ndarray, ticks: int, fill: np.ndarray) -> np.ndarray:
    """Rows extended to `ticks` by repeating their last one (or `fill` when there are none)."""
    last = rows[-1] if len(rows) else np.asarray(fill, float)
    return np.vstack([rows, np.repeat(last[None], ticks - len(rows), 0)])


def plan_low(planner: PickPlanner, det: Any, side: str, bin_name: str, arm_q: np.ndarray, hand_q: np.ndarray) -> Plan:
    """Pick -> bin -> home with a low pregrasp and lift; `ok` False with a reason when a step or a check fails."""
    plan = Plan(False, side=side, color=det.color, target_bin=bin_name)
    path = _joint_path(planner, det, side, bin_name, arm_q, plan)
    if path is None:
        return plan
    _timed(planner, plan, path, side, arm_q, hand_q)
    plan.reason = _violation(planner, plan, side, arm_q, hand_q)
    plan.ok = not plan.reason
    return plan


def _joint_path(planner: PickPlanner, det: Any, side: str, bin_name: str, arm_q: np.ndarray, plan: Plan) -> dict[str, list] | None:
    """Joint waypoints of every arm move, None (with `plan.reason`) when one cannot be solved."""
    grasp = grasp_solution(planner, det, side, arm_q, plan)
    if grasp is None:
        return None
    points = _points(planner, grasp[1], bin_name)
    pregrasp, ok = planner._ik(side, points["pregrasp"], grasp[0], "pregrasp", plan)
    down = ok and planner._cartesian(side, points["pregrasp"], points["grasp"], pregrasp, "descend", plan)
    up = down and planner._cartesian(side, points["grasp"], points["lift"], down[-1], "lift", plan)
    carry = up and _rising_carry(planner, side, up[-1], points, plan)
    lower = carry and planner._cartesian(side, points["above_bin"], points["drop"], carry[-1], "lower", plan)
    retreat = lower and planner._cartesian(side, points["drop"], points["retreat"], lower[-1], "retreat", plan)
    if not retreat:
        plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
        return None
    plan.waypoints.update(grasp=points["grasp"], drop=points["drop"])
    return {"to_pregrasp": [pregrasp], "descend": down, "lift": up, "carry": carry + lower, "retreat": retreat}


def grasp_solution(planner: PickPlanner, det: Any, side: str, arm_q: np.ndarray, plan: Plan) -> tuple[np.ndarray, np.ndarray] | None:
    """Arm solution and grasp point: object mid-height, raised until the preshaped fingertips clear the counter."""
    g, counter, cols = planner.cfg["grasp"], planner.scene.counter_z, ArmKinematics.sl(side)
    preshape = np.zeros(14)
    preshape[cols] = planner.post["preshape"][cols]
    xy = det.top[:2] + np.asarray(g["arm_xy_offset"][side], float)
    z = counter + g["height_frac"] * planner.height
    while z <= counter + planner.height - planner.motion["min_grip_below_top"]:
        q, ok = planner._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
        if not ok:
            plan.reason = f"IK grasp {side}"
            return None
        clearance = planner.clearance(side, q, preshape, fingers_only=True)
        if clearance >= g["fingertip_clearance"] - 1e-3:
            return q, np.r_[xy, z]
        z += g["fingertip_clearance"] - clearance
    plan.reason = "fingers cannot reach below the object's top without touching the counter"
    return None


def _points(planner: PickPlanner, grasp: np.ndarray, bin_name: str) -> dict[str, np.ndarray]:
    """Grasp-point targets of the sequence in the pelvis frame."""
    motion, release = planner.motion, planner.drop_xy(bin_name)
    up = np.array([0.0, 0.0, 1.0])
    return {"pregrasp": grasp + motion["approach_height"] * up, "grasp": grasp, "lift": grasp + motion["lift_height"] * up,
            "above_bin": np.r_[release, planner.carry_height()], "drop": np.r_[release, planner.release_height()],
            "retreat": np.r_[release, planner.release_height() + planner.cfg["grasp"]["retreat_up"]]}


def _rising_carry(planner: PickPlanner, side: str, lifted: np.ndarray, points: dict[str, np.ndarray], plan: Plan) -> list | None:
    """One joint-space move from the low lift to above the bin, if the held object never dips and clears the bin walls."""
    with planner._heading_free(True):
        above, ok = planner._ik(side, points["above_bin"], lifted, "carry", plan)
    floor = points["lift"][2] - planner.motion["carry_dip"]
    path = [planner.kin.grasp_pose(side, lifted + f * (above - lifted))[0] for f in np.linspace(0, 1, CARRY_SAMPLES)]
    if ok and all(p[2] >= floor and planner.object_clears_bins(p) for p in path):
        return [above]
    plan.reason = "no rising joint-space carry"
    return None


def _timed(planner: PickPlanner, plan: Plan, path: dict[str, list], side: str, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
    """Fill the plan's 30 Hz trajectory and phases from the joint path."""
    motion, cols = planner.cfg["motion"], ArmKinematics.sl(side)
    posture = {name: np.r_[hand_q[:cols.start], planner.post[key][cols], hand_q[cols.stop:]]
               for name, key in (("preshape", "preshape"), ("closed", "closed"), ("open", "open"),
                                 ("release", planner.cfg["grasp"]["release_posture"]))}
    home = np.r_[arm_q[:cols.start], planner.home[cols], arm_q[cols.stop:]]
    line = Timeline(planner, np.asarray(arm_q, float).copy(), np.asarray(hand_q, float).copy())
    line.move("to_pregrasp", path["to_pregrasp"], hand_to=posture["preshape"])
    line.move("descend", path["descend"])
    line.grip("close", posture["closed"], motion["close_hold_s"])
    line.move("lift", path["lift"])
    line.move("carry", path["carry"])
    line.grip("open", posture["release"], motion["open_hold_s"])
    line.move("retreat", path["retreat"])
    line.move("home", [home], hand_to=posture["open"])
    plan.arm, plan.hand = np.vstack([a for _, a, _ in line.parts]), np.vstack([h for _, _, h in line.parts])
    stops = np.cumsum([len(a) for _, a, _ in line.parts])
    plan.phases = [(name, int(stop - len(a)), int(stop)) for (name, a, _), stop in zip(line.parts, stops)]


def _violation(planner: PickPlanner, plan: Plan, side: str, arm_q: np.ndarray, hand_q: np.ndarray) -> str:
    """First failed check of the timed trajectory: counter clearance, bin rims, safety filter; empty when all pass."""
    for name, first, stop in plan.phases:
        needed = _counter_clearance(planner, name)
        for tick in range(first, stop, CHECK_EVERY):
            height = planner.clearance(side, plan.arm[tick], plan.hand[tick])
            if height < needed - 2e-3:
                return f"hand {100 * height:.1f} cm above counter during {name} (need {100 * needed:.1f})"
            if name != "open" and planner.bin_clearance(side, plan.arm[tick], plan.hand[tick]) < RIM_TOUCH:
                return f"hand below a bin rim during {name}"
    plan.safety_clips = planner.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
    return f"{plan.safety_clips} safety clips in dry run" if plan.safety_clips else ""


def _counter_clearance(planner: PickPlanner, phase: str) -> float:
    """Height above the counter the hand must keep in a phase: over any object on the way in, fingertip margin at the pick."""
    g, motion = planner.cfg["grasp"], planner.motion
    if phase == "to_pregrasp":
        return planner.height + motion["over_object_clearance"]
    if phase == "close":
        return CLOSING_CLEARANCE
    return g["fingertip_clearance"] if phase in ("descend", "lift") else planner.cfg["motion"]["counter_clearance"]
