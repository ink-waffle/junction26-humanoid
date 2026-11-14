"""Episode memory: picks tried, planning failures and the clearest view of each object, matched by position."""
from __future__ import annotations

import numpy as np


class Attempts:
    def __init__(self, radius: float) -> None:
        self.radius = radius
        self._picked: list[np.ndarray] = []
        self._unplanned: list[np.ndarray] = []
        self._broken: list[np.ndarray] = []

    def record_pick(self, xy: np.ndarray) -> None:
        """A pick was started on the object at `xy`."""
        self._picked.append(np.array(xy, float))

    def record_plan_failure(self, xy: np.ndarray) -> None:
        """No plan was found for the object at `xy` in one look."""
        self._unplanned.append(np.array(xy, float))

    def record_broken(self, xy: np.ndarray) -> None:
        """Building the grasp options for the object at `xy` raised in one look."""
        self._broken.append(np.array(xy, float))

    def broken(self, xy: np.ndarray) -> int:
        """Looks in which the options for the object at `xy` could not be built."""
        return self._near(self._broken, xy)

    def follow(self, old_xy: np.ndarray, new_xy: np.ndarray) -> None:
        """The object tried at `old_xy` now lies at `new_xy`: its tries move with it, none stay behind to be counted twice."""
        for spots in (self._picked, self._unplanned):
            spots[:] = [np.array(new_xy, float) if float(np.linalg.norm(spot - old_xy)) < self.radius else spot for spot in spots]

    def picks(self, xy: np.ndarray) -> int:
        """Picks started so far on the object at `xy`."""
        return self._near(self._picked, xy)

    def count(self, xy: np.ndarray) -> int:
        """Picks started and planning failures so far for the object at `xy`: the more, the later it is tried again."""
        return self.picks(xy) + self._near(self._unplanned, xy)

    def _near(self, spots: list[np.ndarray], xy: np.ndarray) -> int:
        return sum(float(np.linalg.norm(spot - xy)) < self.radius for spot in spots)


class Shapes:
    """The shape each object showed in the clearest look at it so far, matched by colour and position."""

    def __init__(self, radius: float) -> None:
        self.radius = radius
        self._seen: list[tuple[np.ndarray, str | None, str, float]] = []

    def clearest(self, xy: np.ndarray, color: str | None, shape: str | None, visible: float) -> str | None:
        """Record one detection and return the shape from the look that saw most of the object of this colour at `xy`."""
        if shape is not None:
            self._seen.append((np.array(xy, float), color, shape, visible))
        near = [(seen_visible, seen_shape) for spot, seen_color, seen_shape, seen_visible in self._seen
                if seen_color == color and float(np.linalg.norm(spot - xy)) < self.radius]
        return max(near, key=lambda item: item[0])[1] if near else None

    def forget(self, xy: np.ndarray) -> None:
        """A pick started on the object at `xy`: it leaves the spot, falls or is pushed, so what was seen there no longer holds."""
        self._seen = [seen for seen in self._seen if float(np.linalg.norm(seen[0] - xy)) >= self.radius]
