"""Command stream: the trajectory being executed, handed out in chunks, and holds of the last command in between."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from aaltonomous.interface.safety import JointSpec

Action = dict[str, Any]
OBS_LAG_TICKS = 2   # the runner asks for a chunk one tick early: its observation is from two ticks before that chunk


class CommandStream:
    def __init__(self, spec: JointSpec, chunk_ticks: int, hold_ticks: int) -> None:
        self.spec, self.chunk_ticks, self.hold_ticks = spec, chunk_ticks, hold_ticks
        self.arm_cmd: np.ndarray | None = None
        self.hand_cmd: np.ndarray | None = None
        self.load(np.empty((0, 14)), np.empty((0, 14)))

    @property
    def started(self) -> bool:
        return self.arm_cmd is not None

    @property
    def busy(self) -> bool:
        return self.position < len(self._arm)

    @property
    def observed_tick(self) -> int:
        """Index of the newest trajectory tick the observation of the current `act()` call can reflect."""
        return self.position - OBS_LAG_TICKS

    def start(self, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
        """First command = the measured pose, clamped into the joint limits (resting fingers read slightly past zero)."""
        self.arm_cmd = np.clip(np.asarray(arm_q, float), self.spec.arm_lower, self.spec.arm_upper)
        self.hand_cmd = np.clip(np.asarray(hand_q, float), self.spec.hand_lower, self.spec.hand_upper)

    def load(self, arm: np.ndarray, hand: np.ndarray, settle_ticks: int = 0, cuts: Iterable[int] = ()) -> None:
        """Replace the trajectory: it starts at the last command, holds its end for `settle_ticks`, chunks end at `cuts`."""
        arm, hand = np.asarray(arm, float).reshape(-1, 14), np.asarray(hand, float).reshape(-1, 14)
        end_arm = arm[-1] if len(arm) else self.arm_cmd
        end_hand = hand[-1] if len(hand) else self.hand_cmd
        self._arm = np.vstack([arm, np.repeat(end_arm[None], settle_ticks, 0)]) if settle_ticks else arm
        self._hand = np.vstack([hand, np.repeat(end_hand[None], settle_ticks, 0)]) if settle_ticks else hand
        self._cuts = sorted(cuts)
        self.position = 0

    def next_chunk(self) -> Action:
        """The next ticks of the trajectory, up to the next cut; a short tail is merged in so no reply is tiny."""
        start, left = self.position, len(self._arm) - self.position
        end = start + (left if left < 2 * self.chunk_ticks else self.chunk_ticks)
        end = min([end] + [cut for cut in self._cuts if start < cut])
        arm, hand = self._arm[start:end], self._hand[start:end]
        self.position = end
        self.arm_cmd, self.hand_cmd = arm[-1].copy(), hand[-1].copy()
        return {"arm_q_target": arm.copy(), "hand_q_target": hand.copy(), "done": False}

    def hold(self, done: bool = False) -> Action:
        """Repeat the last command; the final reply of an episode is a single tick with `done`."""
        k = 1 if done else self.hold_ticks
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": done}
