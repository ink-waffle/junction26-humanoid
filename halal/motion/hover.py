"""Hover before the final descent: the hand beside and above the grasp point, both in the head camera's view, for a closed-loop correction."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan

from halal.motion.grasps import INWARD, Grasp
from halal.motion.planner import SIDES, PickPlanner
from halal.perception import ObjectDetection
from halal.perception.geometry import Camera
from halal.perception.hands import HandSurface

CHECK_EVERY = 3
RIM_TOUCH = 0.003


@dataclass(frozen=True)
class Hover:
    """Commands to a hover pose beside the grasp point and what the head camera sees from it."""
    arm: np.ndarray      # (T, 14) commands from (arm_q, hand_q) to the hover pose; the picking hand ramps to the plan's preshape on the way
    hand: np.ndarray     # (T, 14)
    offset: np.ndarray   # (3,) hover grasp point minus the target grasp point (pelvis xyz)
    hidden: float        # FK-predicted share of the target's top face the hand hides from `camera` at the hover pose
    low: float           # lowest hand point above the counter at the hover pose (m)


@lru_cache(maxsize=1)
def _surface() -> HandSurface:
    """Forearm and hand surface model, loaded once (it reads the URDF)."""
    return HandSurface()


def plan_hover(planner: PickPlanner, plan: Plan, grasp: Grasp, side: str, det: ObjectDetection, camera: Camera,
               arm_q: np.ndarray, hand_q: np.ndarray, target: np.ndarray | None = None) -> Hover | None:
    """First configured offset beside the grasp point that the arm reaches, moves to cleanly and from which the target stays in view; None if none."""
    settings = planner.motion["hover"]
    aim = np.asarray(plan.waypoints["grasp"] if target is None else target, float)
    top = det.top + aim - plan.waypoints["grasp"]
    seed, preshape = _phase_row(plan.arm, plan, "to_pregrasp", -1), _phase_row(plan.hand, plan, "descend", 0)
    hand = _with_side(hand_q, preshape, side)
    with planner._grasp_applied(grasp, side, grasp.squeeze), planner._sized_as(det), planner._counter_under(det):
        planner.kin.hint, planner.kin.w_hint = _finger_heading(planner, plan, side), planner.motion["finger_yaw_weight"]
        for offset in np.asarray(settings["offsets"], float):
            hover = _hover_at(planner, side, camera, arm_q, hand_q, seed, hand, aim, top, offset * [1.0, -INWARD[side], 1.0])
            if hover is not None:
                return hover
    return None


def grasp_point(planner: PickPlanner, grasp: Grasp, side: str, arm_q: np.ndarray) -> np.ndarray:
    """FK grasp point (pelvis xyz) of this arm pose with the grasp's point_in_palm applied."""
    with planner._grasp_applied(grasp, side, grasp.squeeze):
        return planner.kin.grasp_pose(side, np.asarray(arm_q, float))[0].copy()


def _hover_at(planner: PickPlanner, side: str, camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray,
              seed: np.ndarray, hand: np.ndarray, aim: np.ndarray, top: np.ndarray, offset: np.ndarray) -> Hover | None:
    """The hover at one pelvis-frame offset from the aimed grasp point, None when a check fails."""
    settings = planner.motion["hover"]
    q, ok, _, _ = planner.kin.solve(side, aim + offset, seed, top_down=True, extra_seeds=[planner.home[ArmKinematics.sl(side)]])
    if not ok:
        return None
    pose = _with_side(arm_q, q, side)
    low = planner.clearance(side, pose, hand)
    if low < settings["min_low"]:
        return None
    hidden = hidden_share(camera, pose, hand, top, settings)
    if hidden > settings["max_hidden"]:
        return None
    for stops in ([pose], [_with_side(arm_q, seed, side), pose]):   # direct, else through the plan's pregrasp
        arm, fingers = _through(planner, arm_q, hand_q, stops, hand)
        if _clean(planner, arm, fingers) and not planner.dry_run_safety(arm_q, hand_q, arm, fingers):
            return Hover(arm, fingers, offset, hidden, low)
    return None


def _through(planner: PickPlanner, arm_q: np.ndarray, hand_q: np.ndarray, stops: list[np.ndarray], hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint moves through each stop in turn (the picking hand ramps to `hand` on the first), as (T, 14) arm and hand commands."""
    arms, hands, at = [], [], np.asarray(arm_q, float)
    for k, stop in enumerate(stops):
        arm, fingers = planner.go_to(at, hand_q if k == 0 else hand, stop, hand)
        arms.append(arm)
        hands.append(fingers)
        at = stop
    return np.vstack(arms), np.vstack(hands)


def hidden_share(camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray, top: np.ndarray, settings: dict[str, Any]) -> float:
    """Share of a square on the object's top face that the posed forearms and hands hide from the camera, or that lies outside its image."""
    face = _face(top, settings["face"], settings["face_samples"])
    seen = camera.project(face)
    width, height = settings["image_px"]
    outside = (seen[:, 0] < 0) | (seen[:, 0] >= width) | (seen[:, 1] < 0) | (seen[:, 1] >= height)
    points, _ = _surface().posed(arm_q, hand_q, fingers=True)
    body = camera.project(points)
    body = body[body[:, 2] > 0]
    if not len(body):
        return float(outside.mean())
    near = cKDTree(body[:, :2]).query_ball_point(seen[:, :2], settings["hidden_px"])
    covered = np.array([bool(len(i)) and body[i, 2].min() < z - settings["hidden_depth"] for i, z in zip(near, seen[:, 2])])
    return float((outside | covered).mean())


def _face(top: np.ndarray, side: float, samples: int) -> np.ndarray:
    """Grid points over a horizontal square of that side length centred on the top face."""
    grid = np.linspace(-side / 2, side / 2, samples)
    return np.array([[top[0] + a, top[1] + b, top[2]] for a in grid for b in grid])


def _clean(planner: PickPlanner, arm: np.ndarray, hand: np.ndarray) -> bool:
    """Whether both hands stay an object height plus margin above the counter and off the bin rims on the way."""
    needed = planner.height + planner.motion["over_object_clearance"]
    return all(planner.clearance(side, arm[tick], hand[tick]) >= needed and planner.bin_clearance(side, arm[tick], hand[tick]) >= RIM_TOUCH
               for tick in range(0, len(arm), CHECK_EVERY) for side in SIDES)


def _finger_heading(planner: PickPlanner, plan: Plan, side: str) -> np.ndarray:
    """Unit finger direction (pelvis xy) the plan has at the grasp: the hover keeps it."""
    planner.kin._fk(_phase_row(plan.arm, plan, "descend", -1))
    x = planner.kin.data.oMf[planner.kin.palm[side]].rotation[:, 0]
    return np.r_[x[:2] / np.linalg.norm(x[:2]), 0.0]


def _phase_row(rows: np.ndarray, plan: Plan, phase: str, at: int) -> np.ndarray:
    """The first (`at` 0) or last (`at` -1) row of a plan phase."""
    first, stop = next((a, b) for name, a, b in plan.phases if name == phase)
    return rows[first if at == 0 else stop - 1].copy()


def _with_side(full: np.ndarray, values: np.ndarray, side: str) -> np.ndarray:
    """`full` (14,) with one side's 7 entries taken from `values` (14,)."""
    cols, out = ArmKinematics.sl(side), np.asarray(full, float).copy()
    out[cols] = values[cols]
    return out
