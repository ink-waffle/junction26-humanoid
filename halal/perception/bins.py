"""The three bins found by geometry alone: a row of box walls of roughly the known footprint standing on the counter, any colour."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, PlaneFit


class BinFinder:
    """Looks for the row's front wall over the whole width, then for each box's stretch of it and that stretch's ends."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.size = np.asarray(scene.cfg["bins"]["outer_size"], float)
        self.expected = {name: scene.mat_to_pelvis([*c, 0.0])[:2] for name, c in scene.cfg["bins"]["centers"].items()}

    def find(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> dict[str, np.ndarray]:
        """Centre xy of every bin; bins that are not seen are placed from the seen ones by the scene's spacing."""
        row = self._row_x(depth, camera, counter)
        seen = {} if row is None else self._boxes(depth, camera, counter, row)
        offset = np.mean([seen[n] - self.expected[n] for n in seen], axis=0) if seen else np.zeros(2)
        return {name: seen.get(name, self.expected[name] + offset) for name in self.expected}

    def rim_z(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, bins: Mapping[str, np.ndarray]) -> float | None:
        """Pelvis z of the highest bin rim: where the walls over the bin footprints stop, lifted to the row's highest corner."""
        c = self.cfg["rim"]
        points = camera.cloud(depth, c["pixel_step"]).reshape(-1, 3)
        half = self.size[::-1] / 2 + c["margin"]
        over = np.any([np.all(np.abs(points[:, :2] - centre) < half, axis=1) for centre in bins.values()], axis=0)
        counts = np.histogram(counter.height(points[over]), np.arange(c["band"][0], c["band"][1] + c["step"], c["step"]))[0]
        base = np.median(counts[:int(round((c["surely_wall"] - c["band"][0]) / c["step"]))])
        if base < c["min_points"]:
            return None
        top = c["band"][0] + c["step"] * (np.flatnonzero(counts >= c["wall_share"] * base)[-1] + 1)
        corners = np.concatenate([centre + half * signs for centre in bins.values() for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
        return float(max(counter.z_at(corner) for corner in corners.reshape(-1, 2)) + top * counter.normal[2])

    def _wall_labels(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, x: float, y: np.ndarray) -> np.ndarray:
        """Per y: +1 a vertical surface stands at this x, -1 the camera sees past it, 0 hidden; averaged over the heights."""
        rows = []
        for h in self.cfg["heights"]:
            xy = np.c_[np.full_like(y, x), y]
            z = (counter.offset - xy @ counter.normal[:2]) / counter.normal[2] + h
            rows.append(camera.surface_labels(depth, np.c_[xy, z], self.cfg["depth_tolerance"]))
        return np.mean(rows, axis=0)

    def _filled(self, labels: np.ndarray, needed: int) -> bool:
        """True when enough of a stretch is seen and a wall fills what is seen of it (objects and fingers are narrower)."""
        wall, open_ = int((labels > 0.5).sum()), int((labels < -0.5).sum())
        return wall + open_ >= needed and wall >= self.cfg["min_fill"] * (wall + open_)

    def _front_x(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, y: np.ndarray, x_range: tuple[float, float],
                 needed: int) -> float | None:
        """x of the nearest wall in `x_range` that fills what is visible of the stretch `y`."""
        for x in np.arange(x_range[0], x_range[1], self.cfg["cell"]):
            if self._filled(self._wall_labels(depth, camera, counter, x, y), needed):
                return float(x)
        return None

    def _row_x(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> float | None:
        """x of the row's front wall: the nearest wall that fills the lanes of all three bins taken together."""
        c = self.cfg
        lane = np.arange(-c["lane_half"], c["lane_half"], c["cell"]) + c["cell"] / 2
        y = np.concatenate([centre[1] + lane for centre in self.expected.values()])
        return self._front_x(depth, camera, counter, y, c["x_range"], c["min_visible"])

    def _boxes(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, row: float) -> dict[str, np.ndarray]:
        """Centre of every bin whose own stretch of the row's front wall is seen."""
        c, found = self.cfg, {}
        lane = np.arange(-c["lane_half"], c["lane_half"], c["cell"]) + c["cell"] / 2
        for name, expected in self.expected.items():
            window = (row - c["row_window"], row + c["row_window"] + c["cell"] / 2)
            front = self._front_x(depth, camera, counter, expected[1] + lane, window, c["min_lane_visible"])
            centre = None if front is None else self._centre(depth, camera, counter, front, expected[1])
            if centre is not None:
                found[name] = centre
        return found

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
