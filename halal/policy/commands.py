"""Command stream: the trajectory being executed, handed out in chunks, and holds of the last command in between."""
from __future__ import annotations

from typing import Any

import numpy as np

from aaltonomous.interface.safety import JointSpec

Action = dict[str, Any]


class CommandStream:
    def __init__(self, spec: JointSpec, chunk_ticks: int, hold_ticks: int) -> None:
        self.spec, self.chunk_ticks, self.hold_ticks = spec, chunk_ticks, hold_ticks
        self.arm_cmd: np.ndarray | None = None
        self.hand_cmd: np.ndarray | None = None
        self._arm = np.empty((0, 14))
        self._hand = np.empty((0, 14))

    @property
    def started(self) -> bool:
        return self.arm_cmd is not None

    @property
    def busy(self) -> bool:
        return len(self._arm) > 0

    def start(self, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
        """First command = the measured pose, clamped into the joint limits (resting fingers read slightly past zero)."""
        self.arm_cmd = np.clip(np.asarray(arm_q, float), self.spec.arm_lower, self.spec.arm_upper)
        self.hand_cmd = np.clip(np.asarray(hand_q, float), self.spec.hand_lower, self.spec.hand_upper)

    def load(self, arm: np.ndarray, hand: np.ndarray, settle_ticks: int = 0) -> None:
        """Queue a trajectory that starts at the last command, followed by `settle_ticks` of holding its end."""
        arm, hand = np.asarray(arm, float).reshape(-1, 14), np.asarray(hand, float).reshape(-1, 14)
        end_arm = arm[-1] if len(arm) else self.arm_cmd
        end_hand = hand[-1] if len(hand) else self.hand_cmd
        self._arm = np.vstack([arm, np.repeat(end_arm[None], settle_ticks, 0)])
        self._hand = np.vstack([hand, np.repeat(end_hand[None], settle_ticks, 0)])

    def next_chunk(self) -> Action:
        """The next ticks of the trajectory; a short tail is merged into the last chunk so no reply is tiny."""
        k = len(self._arm) if len(self._arm) < 2 * self.chunk_ticks else self.chunk_ticks
        arm, hand = self._arm[:k], self._hand[:k]
        self._arm, self._hand = self._arm[k:], self._hand[k:]
        self.arm_cmd, self.hand_cmd = arm[-1].copy(), hand[-1].copy()
        return {"arm_q_target": arm.copy(), "hand_q_target": hand.copy(), "done": False}

    def hold(self, done: bool = False) -> Action:
        """Repeat the last command; the final reply of an episode is a single tick with `done`."""
        k = 1 if done else self.hold_ticks
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": done}
