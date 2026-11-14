"""Pick selection: which object next, with which arm, grasp and release point; the first plan that passes the planner's checks wins."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.motion import Grasp, PickPlanner, candidate_grasps
from halal.perception import ObjectDetection
from halal.policy.background import Cancelled
from halal.policy.release import ReleaseSlots

log = logging.getLogger(__name__)
SIDES = ("left", "right")


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
        return f"{d.color} {d.shape or 'object'} at ({d.top[0]:+.3f},{d.top[1]:+.3f}) -> {self.bin_name}"


@dataclass(frozen=True)
class Pick:
    target: Target
    side: str
    grasp: Grasp | None    # None = the planner's default grasp (relocations)
    plan: Plan

    @property
    def relocation(self) -> bool:
        return self.plan.target_bin != self.target.bin_name


class PickChooser:
    def __init__(self, planner: PickPlanner, slots: ReleaseSlots, relocate: bool) -> None:
        self.planner, self.slots, self.relocate = planner, slots, relocate

    def choose(self, targets: Sequence[Target], obstacles: Sequence[ObjectDetection], released: Mapping[str, int],
               squeeze: Mapping[str, float], arm_q: np.ndarray, hand_q: np.ndarray,
               cancelled: Cancelled) -> tuple[Pick | None, list[Target]]:
        """The first plannable target, least tried and then nearest its bin first, and the targets before it without a plan."""
        unplannable: list[Target] = []
        for target in sorted(targets, key=lambda target: (target.attempts, self._distance_to_bin(target))):
            pick = self._plan(target, obstacles, released.get(target.bin_name, 0), squeeze, arm_q, hand_q, cancelled)
            if pick is not None:
                return pick, unplannable
            unplannable.append(target)
        return None, unplannable

    def _plan(self, target: Target, obstacles: Sequence[ObjectDetection], released: int, squeeze: Mapping[str, float],
              arm_q: np.ndarray, hand_q: np.ndarray, cancelled: Cancelled) -> Pick | None:
        try:
            return (self._direct(target, released, squeeze, arm_q, hand_q, cancelled)
                    or self._relocation(target, obstacles, arm_q, hand_q, cancelled))
        except Exception:
            log.exception("planning failed for %s", target)
            return None

    def _direct(self, target: Target, released: int, squeeze: Mapping[str, float], arm_q: np.ndarray,
                hand_q: np.ndarray, cancelled: Cancelled) -> Pick | None:
        det = target.detection
        shifts = self.slots.candidates(target.bin_name, released)
        attempts = [(side, grasp, shift) for shift in shifts for side in self._arms(target)
                    for grasp in _rotated(candidate_grasps(det.shape, det.yaw, side), target.attempts)]
        for side, grasp, shift in attempts:
            if cancelled():
                break
            plan = self.planner.plan_pick(det, grasp, side, target.bin_name, arm_q, hand_q, drop_shift=shift,
                                          squeeze=squeeze[side])
            if plan.ok:
                return Pick(target, side, grasp, plan)
            log.info("no plan for %s with the %s arm, grasp %s, release shift %s: %s", target, side, grasp.name,
                     shift, plan.reason)
        return None

    def _relocation(self, target: Target, obstacles: Sequence[ObjectDetection], arm_q: np.ndarray, hand_q: np.ndarray,
                    cancelled: Cancelled) -> Pick | None:
        """Cross-body fallback: put the object down where an arm that reaches the bin can pick it up."""
        if not self.relocate or cancelled():
            return None
        others = [o.top[:2] for o in obstacles if o is not target.detection]
        plan = self.planner.plan_relocation(target.detection, target.bin_name, arm_q, hand_q, others)
        if not plan.ok:
            log.info("no relocation for %s: %s", target, plan.reason)
            return None
        return Pick(target, plan.side, None, plan)

    def _arms(self, target: Target) -> list[str]:
        """Arms that reach the bin: a side bin's own arm first, for the centre bin the arm on the object's side; a retry starts with the next one."""
        near = target.bin_name if target.bin_name in SIDES else ("left" if target.xy[1] > 0 else "right")
        far = "right" if near == "left" else "left"
        arms = [side for side in (near, far) if self.planner.reaches_bin(side, target.bin_name)]
        return _rotated(arms, target.attempts)

    def _distance_to_bin(self, target: Target) -> float:
        return float(np.linalg.norm(target.xy - self.planner.bin_xy(target.bin_name)))


def _rotated(options: list, turns: int) -> list:
    """The same options starting `turns` places later, so that a retry does not repeat the last choice."""
    turns %= max(len(options), 1)
    return options[turns:] + options[:turns]
