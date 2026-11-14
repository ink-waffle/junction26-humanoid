"""Pick selection: which object next, with which arm, grasp and release point; the first plan that passes wins."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.motion import TOUCHING, Grasp, PickPlanner, candidate_grasps
from halal.perception import ObjectDetection
from halal.policy.background import Cancelled
from halal.policy.release import ReleaseSlots, Shift

log = logging.getLogger(__name__)
SIDES = ("left", "right")
Obstacles = Sequence[np.ndarray]    # top centres (pelvis xyz) of every detection of the look; the planner skips the target's own


def grasp_kind(det: ObjectDetection) -> str:
    """What motion chooses an object's grasps by: `lying` for a prism on a side, else the detected shape, `unknown` without one."""
    return "lying" if det.lying else det.shape or "unknown"


@dataclass(frozen=True)
class Target:
    """A detected object, the bin the instruction sends it to and the picks already tried on it."""
    detection: ObjectDetection
    bin_name: str
    attempts: int = 0

    @property
    def xy(self) -> np.ndarray:
        return self.detection.top[:2]

    def __str__(self) -> str:
        d = self.detection
        return f"{d.color} {'lying ' * d.lying}{d.shape or 'object'} at ({d.top[0]:+.3f},{d.top[1]:+.3f}) -> {self.bin_name}"


@dataclass(frozen=True)
class PlanStart:
    """What every plan of one look starts from: release counts per bin, usable arm -> hand class, the commanded pose."""
    released: Mapping[str, int]
    hands: Mapping[str, str]
    arm_q: np.ndarray
    hand_q: np.ndarray
    cancelled: Cancelled


@dataclass(frozen=True)
class Pick:
    target: Target
    side: str
    grasp: Grasp
    plan: Plan


class PickChooser:
    def __init__(self, planner: PickPlanner, slots: ReleaseSlots, settings: Mapping[str, Any]) -> None:
        self.planner, self.slots, self.settings = planner, slots, settings

    def choose(self, targets: Sequence[Target], released: Mapping[str, int], hands: Mapping[str, str], arm_q: np.ndarray,
               hand_q: np.ndarray, obstacles: Obstacles, cancelled: Cancelled) -> tuple[Pick | None, list[Target]]:
        """The first target whose plan stays clear of the other objects, and the failed ones; a target with a neighbour in the way waits."""
        start = PlanStart(released, hands, arm_q, hand_q, cancelled)
        unplannable: list[Target] = []
        waiting: list[Target] = []
        for target in sorted(targets, key=self._order):
            pick, touching = self._plan(target, start, obstacles)
            if pick is not None:
                return pick, unplannable
            (waiting if touching else unplannable).append(target)
        pick = self._as_if_alone(waiting, start, obstacles)
        return pick, unplannable if pick is not None else unplannable + waiting

    def _as_if_alone(self, waiting: Sequence[Target], start: PlanStart, obstacles: Obstacles) -> Pick | None:
        """No target is clear of its neighbours: of the least tried, the one whose plan as if it stood alone keeps furthest from them."""
        picks = [pick for pick, _ in (self._plan(target, start, []) for target in waiting) if pick is not None]
        pick = min(picks, key=lambda pick: (*self._order(pick.target)[:2], -self._room(pick, obstacles)), default=None)
        if pick is not None:
            log.warning("no target is clear of its neighbours: picking %s as if it stood alone, fingers %.0f mm from the nearest",
                        pick.target, 1000 * self._room(pick, obstacles))
        return pick

    def _room(self, pick: Pick, obstacles: Obstacles) -> float:
        """Smallest distance (m) between the pick's fingers, while they are down, and the centre of another object."""
        own = self.planner.motion["obstacles"]["radius"]
        others = [top for top in obstacles if float(np.linalg.norm(top[:2] - pick.target.xy)) > own]
        return min((self.planner.digit_gap(pick.plan, pick.side, top) for top in others), default=np.inf)

    def _order(self, target: Target) -> tuple[int, bool, float]:
        """Sort key: least tried first, kinds with a grasp of their own before the others, then nearest its bin."""
        return target.attempts, grasp_kind(target.detection) in self.settings["last_kinds"], self._distance_to_bin(target)

    def _plan(self, target: Target, start: PlanStart, obstacles: Obstacles) -> tuple[Pick | None, bool]:
        """`_first_plan`; whatever one target's data breaks on the way counts as no plan, so the other targets keep their turn."""
        try:
            return self._first_plan(target, start, obstacles)
        except Exception:
            log.exception("no options for %s", target)
            return None, False

    def _first_plan(self, target: Target, start: PlanStart, obstacles: Obstacles) -> tuple[Pick | None, bool]:
        """The first arm, grasp and release slot with a plan; none, and True, if that plan would touch a neighbouring object."""
        options = self._options(target, start.released.get(target.bin_name, 0), start.hands)
        if not options:
            log.info("no plan for %s: no arm in use reaches the %s bin with a grasp", target, target.bin_name)
        for side, grasp, shift in options:
            if start.cancelled():
                break
            plan = self._plan_pick(target, side, grasp, shift, start, obstacles)
            if plan.ok:
                return Pick(target, side, grasp, plan), False
            log.info("no plan for %s with the %s arm, grasp %s, release shift %s: %s", target, side, grasp.name,
                     shift, plan.reason)
            if TOUCHING in plan.reason:
                return None, True
        return None, False

    def _options(self, target: Target, released: int, hands: Mapping[str, str]) -> list[tuple[str, Grasp, Shift]]:
        """Arm, grasp and slot in the order to try: a retry starts with the next arm and grasp and moves the grasp point a little; an arm left with last resorts only goes last."""
        det, turns = target.detection, target.attempts
        offsets = self.settings["retry_offsets"]
        dx, dy = offsets[turns % len(offsets)]
        shifts = self.slots.candidates(target.bin_name, released)
        arms = _rotated([side for side in self._arms(target) if side in hands], turns)
        grasps = {side: [replace(grasp, center_shift=(grasp.center_shift[0] + dx, grasp.center_shift[1] + dy))
                         for grasp in self._in_turn(candidate_grasps(det.shape, det.yaw, side, hands[side], det.lying), turns)]
                  for side in arms}
        arms = sorted(arms, key=lambda side: all(grasp.last_resort for grasp in grasps[side]))
        return [(side, grasp, shift) for shift in shifts for side in arms for grasp in grasps[side]]

    @staticmethod
    def _in_turn(grasps: list[Grasp], turns: int) -> list[Grasp]:
        """The candidates starting `turns` places later; motion's last resorts stay behind all the others."""
        return _rotated([grasp for grasp in grasps if not grasp.last_resort], turns) + [grasp for grasp in grasps if grasp.last_resort]

    def _plan_pick(self, target: Target, side: str, grasp: Grasp, shift: Shift, start: PlanStart, obstacles: Obstacles) -> Plan:
        """One planner call; an exception inside the planner counts as no plan."""
        try:
            return self.planner.plan_pick(target.detection, grasp, side, target.bin_name, start.arm_q, start.hand_q,
                                          drop_shift=shift, obstacles=obstacles)
        except Exception as error:
            log.exception("planning failed for %s", target)
            return Plan(False, side=side, color=target.detection.color, reason=repr(error))

    def _arms(self, target: Target) -> list[str]:
        """Arms that reach the bin: a side bin's own arm first, for the centre bin the arm on the object's side."""
        near = target.bin_name if target.bin_name in SIDES else ("left" if target.xy[1] > 0 else "right")
        far = "right" if near == "left" else "left"
        return [side for side in (near, far) if self.planner.reaches_bin(side, target.bin_name, target.detection)]

    def _distance_to_bin(self, target: Target) -> float:
        return float(np.linalg.norm(target.xy - self.planner.bin_xy(target.bin_name)))


def _rotated(options: list, turns: int) -> list:
    """The same options starting `turns` places later."""
    turns %= max(len(options), 1)
    return options[turns:] + options[:turns]
