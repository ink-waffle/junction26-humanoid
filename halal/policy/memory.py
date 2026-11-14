"""Episode memory: picks tried, planning failures and the clearest view of each object, matched by position."""
from __future__ import annotations

import numpy as np


class Attempts:
    def __init__(self, radius: float) -> None:
        self.radius = radius
        self._picked: list[np.ndarray] = []
        self._unplanned: list[np.ndarray] = []

    def record_pick(self, xy: np.ndarray) -> None:
        """A pick was started on the object at `xy`."""
        self._picked.append(np.array(xy, float))

    def record_plan_failure(self, xy: np.ndarray) -> None:
        """No plan was found for the object at `xy` in one look."""
        self._unplanned.append(np.array(xy, float))

    def picks(self, xy: np.ndarray) -> int:
        """Picks started so far on the object at `xy`."""
        return self._near(self._picked, xy)

    def count(self, xy: np.ndarray) -> int:
        """Picks started and planning failures so far for the object at `xy`: the more, the later it is tried again."""
        return self.picks(xy) + self._near(self._unplanned, xy)

    def _near(self, spots: list[np.ndarray], xy: np.ndarray) -> int:
        return sum(float(np.linalg.norm(spot - xy)) < self.radius for spot in spots)


class Shapes:
    """The shape each object showed in the clearest look at it so far."""

    def __init__(self, radius: float) -> None:
        self.radius = radius
        self._seen: list[tuple[np.ndarray, str, float]] = []

    def clearest(self, xy: np.ndarray, shape: str | None, visible: float) -> str | None:
        """Record one detection and return the shape from the look that saw most of the object at `xy`."""
        if shape is not None:
            self._seen.append((np.array(xy, float), shape, visible))
        near = [(seen_visible, seen_shape) for spot, seen_shape, seen_visible in self._seen
                if float(np.linalg.norm(spot - xy)) < self.radius]
        return max(near, key=lambda item: item[0])[1] if near else None
