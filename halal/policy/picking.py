"""Pick selection: which object next, with which arm, grasp and release point; the first plan that passes wins."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.motion import Grasp, PickPlanner, candidate_grasps
from halal.perception import ObjectDetection
from halal.policy.background import Cancelled
from halal.policy.release import ReleaseSlots, Shift

log = logging.getLogger(__name__)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Target:
    """A detected object, the bin the instruction sends it to and the picks already tried on it."""
    detection: ObjectDetection
    bin_name: str
    attempts: int = 0
    neighbours: int = 0         # other detections close enough to be knocked over by a grasp

    @property
    def xy(self) -> np.ndarray:
        return self.detection.top[:2]

    def __str__(self) -> str:
        d = self.detection
        return f"{d.color} {d.shape or 'object'} at ({d.top[0]:+.3f},{d.top[1]:+.3f}) -> {self.bin_name}"


@dataclass(frozen=True)
class Pick:
    target: Target
    side: str
    grasp: Grasp
    plan: Plan


class PickChooser:
    def __init__(self, planner: PickPlanner, slots: ReleaseSlots, retry_offsets: Sequence[Sequence[float]]) -> None:
        self.planner, self.slots, self.retry_offsets = planner, slots, retry_offsets

    def choose(self, targets: Sequence[Target], released: Mapping[str, int], hands: Mapping[str, str], arm_q: np.ndarray,
               hand_q: np.ndarray, cancelled: Cancelled) -> tuple[Pick | None, list[Target]]:
        """The first plannable target and the targets before it without a plan; `hands` = usable arm -> hand class."""
        unplannable: list[Target] = []
        for target in sorted(targets, key=self._order):
            pick = self._plan(target, released.get(target.bin_name, 0), hands, arm_q, hand_q, cancelled)
            if pick is not None:
                return pick, unplannable
            unplannable.append(target)
        return None, unplannable

    def _order(self, target: Target) -> tuple[int, bool, int, float]:
        """Sort key: least tried first, a known shape before an unknown one, fewest close neighbours, then nearest its bin."""
        return target.attempts, target.detection.shape is None, target.neighbours, self._distance_to_bin(target)

    def _plan(self, target: Target, released: int, hands: Mapping[str, str], arm_q: np.ndarray, hand_q: np.ndarray,
              cancelled: Cancelled) -> Pick | None:
        """The first arm, grasp and release slot with a plan."""
        for side, grasp, shift in self._options(target, released, hands):
            if cancelled():
                break
            plan = self._plan_pick(target, side, grasp, shift, arm_q, hand_q)
            if plan.ok:
                return Pick(target, side, grasp, plan)
            log.info("no plan for %s with the %s arm, grasp %s, release shift %s: %s", target, side, grasp.name,
                     shift, plan.reason)
        return None

    def _options(self, target: Target, released: int, hands: Mapping[str, str]) -> list[tuple[str, Grasp, Shift]]:
        """Arm, grasp and slot in the order to try; a retry starts with the next arm and grasp and moves the grasp point a little."""
        det, turns = target.detection, target.attempts
        dx, dy = self.retry_offsets[turns % len(self.retry_offsets)]
        shifts = self.slots.candidates(target.bin_name, released)
        arms = _rotated([side for side in self._arms(target) if side in hands], turns)
        grasps = {side: [replace(grasp, center_shift=(grasp.center_shift[0] + dx, grasp.center_shift[1] + dy))
                         for grasp in _rotated(candidate_grasps(det.shape, det.yaw, side, hands[side]), turns)]
                  for side in arms}
        return [(side, grasp, shift) for shift in shifts for side in arms for grasp in grasps[side]]

    def _plan_pick(self, target: Target, side: str, grasp: Grasp, shift: Shift, arm_q: np.ndarray,
                   hand_q: np.ndarray) -> Plan:
        """One planner call; an exception inside the planner counts as no plan."""
        try:
            return self.planner.plan_pick(target.detection, grasp, side, target.bin_name, arm_q, hand_q, drop_shift=shift)
        except Exception as error:
            log.exception("planning failed for %s", target)
            return Plan(False, side=side, color=target.detection.color, reason=repr(error))

    def _arms(self, target: Target) -> list[str]:
        """Arms that reach the bin: a side bin's own arm first, for the centre bin the arm on the object's side."""
        near = target.bin_name if target.bin_name in SIDES else ("left" if target.xy[1] > 0 else "right")
        far = "right" if near == "left" else "left"
        return [side for side in (near, far) if self.planner.reaches_bin(side, target.bin_name)]

    def _distance_to_bin(self, target: Target) -> float:
        return float(np.linalg.norm(target.xy - self.planner.bin_xy(target.bin_name)))


def _rotated(options: list, turns: int) -> list:
    """The same options starting `turns` places later."""
    turns %= max(len(options), 1)
    return options[turns:] + options[:turns]
