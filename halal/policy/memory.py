"""Episode memory: where picks were tried and where planning failed, with objects matched by position."""
from __future__ import annotations

import numpy as np


class Attempts:
    def __init__(self, radius: float, limit: int) -> None:
        self.radius, self.limit = radius, limit
        self._tried: list[np.ndarray] = []
        self._abandoned: list[np.ndarray] = []

    def record(self, xy: np.ndarray) -> None:
        """A pick was started on the object at `xy`."""
        self._tried.append(np.array(xy, float))

    def abandon(self, xy: np.ndarray) -> None:
        """No plan exists for the object at `xy`."""
        self._abandoned.append(np.array(xy, float))

    def exhausted(self, xy: np.ndarray) -> bool:
        """Whether the object at `xy` was abandoned or has used up its attempts."""
        return self._near(self._abandoned, xy) > 0 or self._near(self._tried, xy) >= self.limit

    def _near(self, spots: list[np.ndarray], xy: np.ndarray) -> int:
        return sum(float(np.linalg.norm(spot - xy)) < self.radius for spot in spots)
