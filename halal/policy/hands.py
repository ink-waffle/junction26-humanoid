"""The once-per-episode calibration close: both hands close on nothing and tell what kind of hand they are."""
from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

import numpy as np

from halal.motion import PickPlanner
from halal.policy.commands import OBS_LAG_TICKS, CommandStream
from halal.policy.grip import SIDES, CalibrationTrace, HandResponse, calibration_close

log = logging.getLogger(__name__)
HZ = 30.0


class HandCalibrator:
    def __init__(self, planner: PickPlanner, settings: Mapping[str, Any]) -> None:
        self.planner, self.settings = planner, settings
        self.finished = threading.Event()
        self.reset(enabled=False)

    def reset(self, enabled: bool) -> None:
        """New episode: nothing is known about the hands; a look still waiting for the last result is let go."""
        self.finished.set()
        self.finished = threading.Event()
        self.enabled, self.running = enabled, False
        self.hands: dict[str, HandResponse] = {}
        self.trace: CalibrationTrace | None = None

    def due(self, arm_q: np.ndarray) -> bool:
        """Not done yet this episode, and both hands could close where they are without touching anything."""
        if not self.enabled or self.trace is not None:
            return False
        closed = self.planner.post["closed"]
        return all(self.planner.clearance(side, arm_q, closed) >= self.settings["calibration_clearance"] for side in SIDES)

    def start(self, stream: CommandStream) -> np.ndarray:
        """Queue the close with the arms held still; returns the hand posture it ends in (open)."""
        planner, every = self.planner, self.settings["calibration_sample_ticks"]
        hand, closing_from, last_sample = calibration_close(
            stream.hand_cmd, planner.post["open"], planner.post["closed"], round(self.settings["calibration_hold_s"] * HZ),
            planner.cfg["motion"]["hand_max_vel"], 1.0 / HZ)
        self.trace, self.running = CalibrationTrace(hand, closing_from, last_sample), True
        before_close = closing_from + OBS_LAG_TICKS - 1   # replies end where the next observation is a wanted sample
        stream.load(np.repeat(stream.arm_cmd[None], len(hand), 0), hand,
                    cuts=[before_close, *range(last_sample + OBS_LAG_TICKS, before_close, -every)])
        return hand[-1]

    def sample(self, stream: CommandStream, obs: Mapping[str, Any]) -> None:
        self.trace.add_tick(float(obs["t"]), stream.observed_tick, obs["hand_q"])

    def finish(self, t: float) -> None:
        """The close has run: classify both hands and release whoever waits for the result."""
        self.hands = self.trace.responses(self.settings["stiff_hand_bias"], self.settings["stiff_hand_lag_s"])
        for side, hand in self.hands.items():
            log.info("t=%.1f %s hand is %s: steady bias %.4f rad, lag %.3f s, empty-hand error %s", t, side,
                     "stiff" if hand.stiff else "soft", hand.bias, hand.lag_s, np.round(hand.empty_error, 3).tolist())
        self.stop()

    def stop(self) -> None:
        """End the calibration with whatever is known (nothing, if it did not finish)."""
        self.running = False
        self.finished.set()

    def hand_class(self, side: str) -> str:
        """`stiff` or `soft` from the calibration close, `unknown` without one."""
        hand = self.hands.get(side)
        return "unknown" if hand is None else "stiff" if hand.stiff else "soft"

    def grasp_class(self, side: str) -> str:
        """The class grasps are built for: an uncalibrated hand is treated as stiff (shallower, never ejects)."""
        return "soft" if self.hand_class(side) == "soft" else "stiff"

    def empty_error(self, side: str) -> np.ndarray | float:
        hand = self.hands.get(side)
        return 0.0 if hand is None else hand.empty_error
