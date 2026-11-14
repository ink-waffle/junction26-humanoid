"""The three bins found by geometry alone: box walls of roughly the known footprint standing on the counter, any colour."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, PlaneFit


class BinFinder:
    """Looks for each bin's front wall in the lane of its scene position and measures where that wall starts and stops."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.size = np.asarray(scene.cfg["bins"]["outer_size"], float)
        self.expected = {name: scene.mat_to_pelvis([*c, 0.0])[:2] for name, c in scene.cfg["bins"]["centers"].items()}

    def find(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> dict[str, np.ndarray]:
        """Centre xy of every bin; bins that are not seen are placed from the seen ones by the scene's spacing."""
        fronts = {name: self._front_x(depth, camera, counter, lane[1]) for name, lane in self.expected.items()}
        row = np.nanmedian(list(fronts.values())) if np.isfinite(list(fronts.values())).any() else np.nan
        seen = {name: self._centre(depth, camera, counter, x, self.expected[name][1]) for name, x in fronts.items()
                if abs(x - row) <= self.cfg["row_tolerance"]}
        seen = {name: centre for name, centre in seen.items() if centre is not None}
        offset = np.mean([seen[n] - self.expected[n] for n in seen], axis=0) if seen else np.zeros(2)
        return {name: seen.get(name, self.expected[name] + offset) for name in self.expected}

    def _wall_labels(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, x: float, y: np.ndarray) -> np.ndarray:
        """Per y: +1 a vertical surface stands at this x, -1 the camera sees past it, 0 hidden; averaged over the heights."""
        rows = []
        for h in self.cfg["heights"]:
            xy = np.c_[np.full_like(y, x), y]
            z = (counter.offset - xy @ counter.normal[:2]) / counter.normal[2] + h
            rows.append(camera.surface_labels(depth, np.c_[xy, z], self.cfg["depth_tolerance"]))
        return np.mean(rows, axis=0)

    def _front_x(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, lane_y: float) -> float:
        """x of the nearest wall that fills what is visible of the bin's lane (objects and fingers are narrower); NaN if none."""
        c = self.cfg
        y = lane_y + np.arange(-c["lane_half"], c["lane_half"], c["cell"]) + c["cell"] / 2
        for x in np.arange(c["x_range"][0], c["x_range"][1], c["cell"]):
            labels = self._wall_labels(depth, camera, counter, x, y)
            wall, open_ = int((labels > 0.5).sum()), int((labels < -0.5).sum())
            if wall + open_ >= c["min_visible"] and wall >= c["min_fill"] * (wall + open_):
                return float(x)
        return float("nan")

    def _centre(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, front: float, lane_y: float) -> np.ndarray | None:
        """Bin centre from its front wall: sideways from the wall's seen ends, forwards by half its width."""
        c = self.cfg
        y = lane_y + np.arange(-c["span_reach"], c["span_reach"], c["span_step"]) + c["span_step"] / 2
        labels = self._wall_labels(depth, camera, counter, front + c["cell"] / 2, y)
        middle = len(y) // 2
        below, above = np.flatnonzero(labels[:middle] < -0.5), np.flatnonzero(labels[middle:] < -0.5)
        lo = y[below[-1]] + c["span_step"] / 2 if len(below) and self._solid(labels[below[-1] + 1:middle]) else np.nan
        hi = y[middle + above[0]] - c["span_step"] / 2 if len(above) and self._solid(labels[middle:middle + above[0]]) else np.nan
        width = hi - lo if np.isfinite(hi - lo) else self.size[0]
        if not c["width_range"][0] <= width <= c["width_range"][1]:
            return None
        centre_y = np.nanmean([lo + width / 2, hi - width / 2]) if np.isfinite([lo, hi]).any() else lane_y
        return np.array([front + width / 2, centre_y])

    @staticmethod
    def _solid(stretch: np.ndarray) -> bool:
        """True when the stretch between the lane centre and a wall end is seen as wall, not hidden."""
        return len(stretch) > 0 and float(np.mean(stretch > 0.5)) > 0.8
