"""Where a pickable object can be: on the counter and clear of the bins; reachability is left to the planner."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from aaltonomous.scene.scene import Scene


class ObjectArea:
    """The counter top inside an edge margin, minus a clearance strip around every bin footprint (pelvis frame)."""

    def __init__(self, scene: Scene, edge_margin: float, bin_clearance: float) -> None:
        counter, bins = scene.cfg["counter"], scene.cfg["bins"]
        mat, size = np.asarray(counter["mat_size"], float), np.asarray(counter["counter_size"], float)
        corners = scene.mat_to_pelvis([mat / 2 - size / 2, mat / 2 + size / 2])[:, :2]
        self.lo, self.hi = corners.min(0) + edge_margin, corners.max(0) - edge_margin
        self.bin_half = np.asarray(bins["outer_size"], float)[::-1] / 2 + bin_clearance   # mat (x, y) = pelvis (y, x)

    def contains(self, xy: np.ndarray, bin_centers: Mapping[str, np.ndarray]) -> bool:
        """Whether `xy` is on the counter and away from the bins standing at `bin_centers` this episode."""
        on_counter = bool(np.all(xy >= self.lo) and np.all(xy <= self.hi))
        at_bin = any(np.all(np.abs(xy - center) <= self.bin_half) for center in bin_centers.values())
        return on_counter and not at_bin
