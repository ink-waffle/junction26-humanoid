"""Pick selection: which object next, with which arm and grasp; the first plan that passes the planner's checks wins."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.motion import Grasp, PickPlanner, candidate_grasps
from halal.perception import ObjectDetection
from halal.policy.background import Cancelled

log = logging.getLogger(__name__)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Target:
    """A detected object and the bin the instruction sends it to."""
    detection: ObjectDetection
    bin_name: str

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
    def __init__(self, planner: PickPlanner, relocate: bool) -> None:
        self.planner, self.relocate = planner, relocate
        bins = planner.scene.bin_names
        self._reach = {(side, name): self._reaches(side, name) for side in SIDES for name in bins}

    def choose(self, targets: Sequence[Target], obstacles: Sequence[ObjectDetection], arm_q: np.ndarray,
               hand_q: np.ndarray, cancelled: Cancelled) -> tuple[Pick | None, list[Target]]:
        """The first plannable target, nearest its bin first, and the targets before it that have no plan."""
        unplannable: list[Target] = []
        for target in sorted(targets, key=self._distance_to_bin):
            pick = self._plan(target, obstacles, arm_q, hand_q, cancelled)
            if pick is not None:
                return pick, unplannable
            unplannable.append(target)
        return None, unplannable

    def _plan(self, target: Target, obstacles: Sequence[ObjectDetection], arm_q: np.ndarray, hand_q: np.ndarray,
              cancelled: Cancelled) -> Pick | None:
        try:
            return (self._direct(target, arm_q, hand_q, cancelled)
                    or self._relocation(target, obstacles, arm_q, hand_q, cancelled))
        except Exception:
            log.exception("planning failed for %s", target)
            return None

    def _direct(self, target: Target, arm_q: np.ndarray, hand_q: np.ndarray, cancelled: Cancelled) -> Pick | None:
        det = target.detection
        attempts = [(side, grasp) for side in self._arms(target) for grasp in candidate_grasps(det.shape, det.yaw, side)]
        for side, grasp in attempts:
            if cancelled():
                break
            plan = self.planner.plan_pick(det, grasp, side, target.bin_name, arm_q, hand_q)
            if plan.ok:
                return Pick(target, side, grasp, plan)
            log.info("no plan for %s with the %s arm, grasp %s: %s", target, side, grasp.name, plan.reason)
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
        """Arms that reach the bin: a side bin's own arm first, for the centre bin the arm on the object's side."""
        near = target.bin_name if target.bin_name in SIDES else ("left" if target.xy[1] > 0 else "right")
        far = "right" if near == "left" else "left"
        return [side for side in (near, far) if self._reach[(side, target.bin_name)]]

    def _reaches(self, side: str, bin_name: str) -> bool:
        """Whether this arm can hold an object over the bin's release point at carry height."""
        planner = self.planner
        z = planner.scene.counter_z + planner.cfg["grasp"]["carry_height"]
        return bool(planner.kin.solve(side, np.r_[planner.drop_xy(bin_name), z], planner.home)[1])

    def _distance_to_bin(self, target: Target) -> float:
        return float(np.linalg.norm(target.xy - self.planner.bin_xy(target.bin_name)))
