"""Hand calibration and grasp verification: fingers that stop short of their commanded closure hold something."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from aaltonomous.baselines.classical.planner import Plan, ramp

from halal.motion.planner import LATE_FINGER
from halal.policy.commands import OBS_LAG_TICKS


SIDES = ("left", "right")


def hand_slice(side: str) -> slice:
    return slice(0, 7) if side == "left" else slice(7, 14)


def phase(plan: Plan, name: str) -> tuple[int, int]:
    """First tick and end tick (exclusive) of a named phase of the plan."""
    return next((start, end) for phase_name, start, end in plan.phases if phase_name == name)


@dataclass(frozen=True)
class HandResponse:
    """How one hand answered a close on nothing: its steady tracking error and how fast it follows a command."""
    empty_error: np.ndarray   # (7,) commanded minus measured, closed and at rest with nothing in the hand
    bias: float               # mean size of that error over the joints: ~0 for the sim's hand, 0.02-0.04 rad for the real one
    lag_s: float              # mean delay of the joints behind the command (sim ~0.1 s, real ~0.2 s)
    stiff: bool               # the sim's stiff actuator (no steady error); False = the soft real hand


def calibration_close(hand_q: np.ndarray, opened: np.ndarray, closed: np.ndarray, still_ticks: int, hand_vel: float,
                      dt: float) -> tuple[np.ndarray, int, int]:
    """Hand trajectory (T, 14): open if needed, close on nothing, hold, reopen; with its first closing and last sampled tick."""
    start = hand_q if np.abs(closed - hand_q).sum() >= np.abs(closed - opened).sum() else opened
    opening = ramp(hand_q, start, hand_vel, dt)
    closing = ramp(start, closed, hand_vel, dt)
    still = np.repeat(closed[None], still_ticks + 2 * OBS_LAG_TICKS, 0)   # closed past the last sample: observations can be newer
    trajectory = np.vstack([opening, closing, still, ramp(closed, opened, hand_vel, dt)])
    return trajectory, len(opening), len(opening) + len(closing) + still_ticks - 1


class CalibrationTrace:
    """Both hands during a calibration close, sampled from the observations of successive `act()` calls."""

    def __init__(self, hand: np.ndarray, closing_from: int, last_sample: int) -> None:
        self.hand, self.closing_from, self.last_sample = hand, closing_from, last_sample
        self.times: list[float] = []
        self.commanded: list[np.ndarray] = []
        self.measured: list[np.ndarray] = []

    def add_tick(self, t: float, tick: int, hand_q: np.ndarray) -> None:
        """Sample for trajectory tick `tick`; only the open hand just before the close, the close and the hold count."""
        if self.closing_from - 1 <= tick <= self.last_sample:
            self.times.append(t)
            self.commanded.append(self.hand[max(tick, 0)])
            self.measured.append(np.asarray(hand_q, float))

    def responses(self, stiff_bias: float) -> dict[str, HandResponse]:
        times, commanded, measured = np.array(self.times), np.array(self.commanded), np.array(self.measured)
        return {side: hand_response(times, commanded[:, hand_slice(side)], measured[:, hand_slice(side)], stiff_bias)
                for side in SIDES}


def hand_response(times: np.ndarray, commanded: np.ndarray, measured: np.ndarray, stiff_bias: float) -> HandResponse:
    """One hand's calibration trace (samples x 7, open -> closed -> still) as its empty error, bias, lag and class."""
    error = commanded - measured
    travel = commanded[-1] - commanded[0]
    behind = (error - error[-1]) @ np.sign(travel)   # summed over the joints, beyond the steady error
    area = float(np.sum(0.5 * (behind[1:] + behind[:-1]) * np.diff(times)))
    moved = float(np.abs(travel - error[-1]).sum())
    lag_s = area / moved if moved > 0 else 0.0   # time integral of a step's tracking error = step x mean delay
    bias = float(np.abs(error[-1]).mean())
    return HandResponse(error[-1], bias, lag_s, bias < stiff_bias)


@dataclass(frozen=True)
class GripWindow:
    """The ticks of a pick plan during which the picking hand should be holding the object."""
    side: str
    closed: np.ndarray       # commanded posture of the closed hand (7,)
    direction: np.ndarray    # per joint +1 / -1 / 0: the way it moves when the hand closes
    first_tick: int          # the lift is over and the hand command has been still for the settle time
    last_tick: int           # a little before the hand opens: an observation may be newer than assumed
    released: np.ndarray     # commanded posture of the hand after letting go (7,)

    @classmethod
    def of(cls, plan: Plan, side: str, settle_ticks: int) -> GripWindow:
        joints = hand_slice(side)
        close_start, close_end = phase(plan, "close")
        closed, before = plan.hand[close_end - 1, joints], plan.hand[close_start - 1, joints]
        still_from = close_start + int(np.argmax(np.all(plan.hand[close_start:close_end, joints] == closed, axis=1)))
        first_tick = max(still_from + settle_ticks, phase(plan, "lift")[1] - 1)
        return cls(side, closed, np.sign(closed - before), first_tick, phase(plan, "open")[0] - 1 - OBS_LAG_TICKS,
                   plan.hand[phase(plan, "retreat")[1] - 1, joints])

    def closing_to(self, closed: np.ndarray, opened: np.ndarray) -> GripWindow:
        """The same hand watched at another closed posture (7,), e.g. the one its empty-hand error was measured in."""
        return replace(self, closed=closed, direction=np.sign(closed - opened))

    def both_hands(self, hand_q: np.ndarray, posture: np.ndarray) -> np.ndarray:
        """`hand_q` (14,) with this window's hand at `posture` (7,)."""
        hand = np.array(hand_q, float)
        hand[hand_slice(self.side)] = posture
        return hand

    def score(self, hand_q: np.ndarray, empty_error: np.ndarray | float, joints: Sequence[int]) -> float:
        """Summed shortfall of `joints` behind the closed posture beyond the empty-hand error (rad); ~0 = empty."""
        error = self.closed - np.asarray(hand_q, float)[hand_slice(self.side)]
        return float(np.maximum(self.direction * (error - empty_error), 0.0)[list(joints)].sum())


@dataclass(frozen=True)
class ReleaseCheck:
    """One more close on nothing after letting go, arm above the bin: a hand that still holds something shows it."""
    hand: np.ndarray    # (n, 14): close, stay closed
    read_tick: int      # row at which the grip score is read

    @classmethod
    def closing(cls, hand_q: np.ndarray, window: GripWindow, settle_ticks: int, not_before: int, hand_vel: float,
                dt: float) -> ReleaseCheck:
        """The check for the hand of `window` from the commanded `hand_q`; the score is read at row `not_before` or later."""
        closed = window.both_hands(hand_q, window.closed)
        closing = ramp(hand_q, closed, hand_vel, dt)
        read_tick = max(len(closing) + settle_ticks - 1, not_before)
        still = np.repeat(closed[None], read_tick + 1 + 2 * OBS_LAG_TICKS - len(closing), 0)
        return cls(np.vstack([closing, still]), read_tick)


def opening_in_turn(hand_q: np.ndarray, released: np.ndarray, side: str, pause_ticks: int, hand_vel: float,
                    dt: float) -> np.ndarray:
    """Hand rows (n, 14) from `hand_q` to `released`: thumb and one finger open and rest, only then the last finger."""
    late = np.arange(14)[hand_slice(side)][LATE_FINGER]
    first = np.array(released, float)
    first[late] = hand_q[late]
    return np.vstack([ramp(hand_q, first, hand_vel, dt), np.repeat(first[None], pause_ticks, 0),
                      ramp(first, released, hand_vel, dt)])


def catching_up(hand_q: np.ndarray, hand: np.ndarray, hand_vel: float, dt: float) -> np.ndarray:
    """The hand rows `hand` (n, 14) entered from `hand_q`: every joint closes its gap at `hand_vel` at most."""
    rows, q = np.empty((len(hand), 14)), np.asarray(hand_q, float)
    for k, target in enumerate(hand):
        q = q + np.clip(target - q, -hand_vel * dt, hand_vel * dt)
        rows[k] = q
    return rows
