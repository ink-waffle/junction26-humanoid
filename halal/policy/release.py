"""Release points: objects dropped into one bin are spread over several points so that they do not stack above the rim."""
from __future__ import annotations

from typing import Mapping, Sequence

Shift = tuple[float, float]


class ReleaseSlots:
    def __init__(self, shifts: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self.shifts = {name: [(float(x), float(y)) for x, y in slots] for name, slots in shifts.items()}

    def candidates(self, bin_name: str, released: int) -> list[Shift]:
        """Shifts of the bin's default release point, the slot after the last used one first."""
        slots = self.shifts[bin_name]
        return [slots[(released + i) % len(slots)] for i in range(len(slots))]
