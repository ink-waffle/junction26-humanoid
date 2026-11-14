"""Release points: objects dropped into one bin are spread over several points so that they do not stack above the rim."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from halal.motion import PickPlanner


class ReleaseSlots:
    def __init__(self, shifts: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self.shifts = {name: [np.asarray(shift, float) for shift in slots] for name, slots in shifts.items()}

    def candidates(self, bin_name: str, released: int) -> list[np.ndarray]:
        """Shifts of the bin's default release point, the slot after the last used one first."""
        slots = self.shifts[bin_name]
        return [slots[(released + i) % len(slots)] for i in range(len(slots))]


class ShiftedReleasePlanner(PickPlanner):
    """Pick planner whose release point is moved by `drop_shift`; stands in for a `plan_pick` argument."""
    drop_shift = np.zeros(2)

    def drop_xy(self, name: str) -> np.ndarray:
        return super().drop_xy(name) + self.drop_shift
