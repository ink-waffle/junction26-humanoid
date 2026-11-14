"""Grasp verification: finger joints that stop short of their commanded closure mean something is in the hand."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from aaltonomous.baselines.classical.planner import Plan, ramp

from halal.policy.commands import OBS_LAG_TICKS


def hand_slice(side: str) -> slice:
    return slice(0, 7) if side == "left" else slice(7, 14)


def phase(plan: Plan, name: str) -> tuple[int, int]:
    """First tick and end tick (exclusive) of a named phase of the plan."""
    return next((start, end) for phase_name, start, end in plan.phases if phase_name == name)


@dataclass(frozen=True)
class GripWindow:
    """The ticks of a pick plan during which the picking hand should be holding the object."""
    side: str
    closed: np.ndarray       # commanded posture of the closed hand (7,)
    direction: np.ndarray    # per joint +1 / -1 / 0: the way it moves when the hand closes
    first_tick: int          # the lift is over and the hand command has been still for the settle time
    last_tick: int           # last tick before the hand opens

    @classmethod
    def of(cls, plan: Plan, side: str, settle_ticks: int) -> GripWindow:
        joints = hand_slice(side)
        close_start, close_end = phase(plan, "close")
        closed, before = plan.hand[close_end - 1, joints], plan.hand[close_start - 1, joints]
        still_from = close_start + int(np.argmax(np.all(plan.hand[close_start:close_end, joints] == closed, axis=1)))
        first_tick = max(still_from + settle_ticks, phase(plan, "lift")[1] - 1)
        return cls(side, closed, np.sign(closed - before), first_tick, phase(plan, "open")[0] - 1)


class GripMonitor:
    """Scores how far the fingers stay short of the closed posture, relative to the same hand closed on nothing."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._empty: dict[tuple[str, bytes], np.ndarray] = {}

    def calibrated(self, side: str, closed: np.ndarray) -> bool:
        return (side, closed.tobytes()) in self._empty

    def calibrate(self, side: str, closed: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
        """Store this hand's tracking error at `closed` with nothing in it (bias, scale error and sag in one)."""
        error = closed - np.asarray(hand_q, float)[hand_slice(side)]
        self._empty[(side, closed.tobytes())] = error
        return error

    def score(self, window: GripWindow, hand_q: np.ndarray) -> float:
        """Summed shortfall of the joints behind the closed posture (rad); about zero for an empty hand."""
        error = window.closed - np.asarray(hand_q, float)[hand_slice(window.side)]
        empty = self._empty.get((window.side, window.closed.tobytes()), 0.0)
        return float(np.maximum(window.direction * (error - empty), 0.0).sum())

    def holding(self, window: GripWindow, hand_q: np.ndarray) -> bool:
        return self.score(window, hand_q) >= self.threshold


def with_empty_closing(plan: Plan, side: str, hand_start: np.ndarray, settle_ticks: int, hand_vel: float,
                       dt: float) -> tuple[Plan, int]:
    """The plan with the picking hand closing on nothing during the approach, and the tick at which to read its error."""
    joints = hand_slice(side)
    approach_end = phase(plan, "to_pregrasp")[1]
    closed, preshape = plan.hand[phase(plan, "close")[1] - 1, joints], plan.hand[approach_end - 1, joints]
    closing = ramp(hand_start[joints], closed, hand_vel, dt)
    still = np.repeat(closed[None], settle_ticks + OBS_LAG_TICKS, 0)
    probe = np.vstack([closing, still, ramp(closed, preshape, hand_vel, dt)])
    wait = max(len(probe) - approach_end, 0)   # an approach shorter than the probe waits at the pregrasp
    arm = np.insert(plan.arm, approach_end, np.repeat(plan.arm[approach_end - 1][None], wait, 0), axis=0)
    hand = np.insert(plan.hand, approach_end, np.repeat(plan.hand[approach_end - 1][None], wait, 0), axis=0)
    hand[:approach_end + wait, joints] = preshape
    hand[:len(probe), joints] = probe
    phases = [(name, start + wait * (start > 0), end + wait) for name, start, end in plan.phases]
    return replace(plan, arm=arm, hand=hand, phases=phases), len(closing) + len(still) - 1
