"""Joint-space moves outside pick plans (to the look pose), checked like the planner checks its own trajectories."""
from __future__ import annotations

import numpy as np

from halal.motion import PickPlanner

SIDES = ("left", "right")
CHECK_EVERY_TICKS = 3


def checked_move(planner: PickPlanner, arm_q: np.ndarray, hand_q: np.ndarray, arm_to: np.ndarray, hand_to: np.ndarray,
                 clearance: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Straight joint-space move, or None if the safety filter would clip it or a hand comes near the counter or a bin rim."""
    arm, hand = planner.go_to(arm_q, hand_q, arm_to, hand_to)
    if planner.dry_run_safety(arm_q, hand_q, arm, hand):
        return None
    for k in range(0, len(arm), CHECK_EVERY_TICKS):
        for side in SIDES:
            lowest = min(planner.clearance(side, arm[k], hand[k]), planner.bin_clearance(side, arm[k], hand[k]))
            if lowest < clearance:
                return None
    return arm, hand
