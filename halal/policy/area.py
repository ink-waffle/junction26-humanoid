"""Where an object on the counter can be: detections elsewhere (bin rims, things behind the bins) are not picked."""
from __future__ import annotations

import numpy as np

from aaltonomous.scene.scene import Scene


class ObjectArea:
    """The layout band widened by a margin, minus a clearance strip around every bin footprint (pelvis frame)."""

    def __init__(self, scene: Scene, band_margin: float, bin_clearance: float) -> None:
        layout, bins = scene.cfg["layout"], scene.cfg["bins"]
        corners = scene.mat_to_pelvis([[layout["region_x"][i], layout["region_y"][i]] for i in (0, 1)])[:, :2]
        self.lo, self.hi = corners.min(0) - band_margin, corners.max(0) + band_margin
        self.bin_centers = [scene.mat_to_pelvis(center)[:2] for center in bins["centers"].values()]
        self.bin_half = np.asarray(bins["outer_size"], float)[::-1] / 2 + bin_clearance   # mat (x, y) = pelvis (y, x)

    def contains(self, xy: np.ndarray) -> bool:
        in_band = bool(np.all(xy >= self.lo) and np.all(xy <= self.hi))
        at_bin = any(np.all(np.abs(xy - center) <= self.bin_half) for center in self.bin_centers)
        return in_band and not at_bin
