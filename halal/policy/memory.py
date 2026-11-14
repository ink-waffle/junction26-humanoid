"""Episode memory: where picks were tried and where planning failed, with objects matched by position."""
from __future__ import annotations

import numpy as np


class Attempts:
    def __init__(self, radius: float, max_picks: int, max_plan_failures: int) -> None:
        self.radius, self.max_picks, self.max_plan_failures = radius, max_picks, max_plan_failures
        self._picked: list[np.ndarray] = []
        self._unplanned: list[np.ndarray] = []

    def record_pick(self, xy: np.ndarray) -> None:
        """A pick was started on the object at `xy`."""
        self._picked.append(np.array(xy, float))

    def record_plan_failure(self, xy: np.ndarray) -> None:
        """No plan was found for the object at `xy` in one look."""
        self._unplanned.append(np.array(xy, float))

    def count(self, xy: np.ndarray) -> int:
        """Picks started and planning failures so far for the object at `xy`."""
        return self._near(self._picked, xy) + self._near(self._unplanned, xy)

    def exhausted(self, xy: np.ndarray) -> bool:
        """Whether the object at `xy` has used up its picks or failed planning too often."""
        return (self._near(self._picked, xy) >= self.max_picks
                or self._near(self._unplanned, xy) >= self.max_plan_failures)

    def _near(self, spots: list[np.ndarray], xy: np.ndarray) -> int:
        return sum(float(np.linalg.norm(spot - xy)) < self.radius for spot in spots)
