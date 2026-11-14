"""The three bins found by geometry alone: a row of box walls of roughly the known footprint standing on the counter, any colour."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, PlaneFit


class BinFinder:
    """Looks for the row's front wall, then lays three equal boxes with one gap width along it; the scene file gives the box size."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.size = np.asarray(scene.cfg["bins"]["outer_size"], float)
        self.expected = {name: scene.mat_to_pelvis([*c, 0.0])[:2] for name, c in scene.cfg["bins"]["centers"].items()}
        self.names = sorted(self.expected, key=lambda name: self.expected[name][1])
        self.middle = float(np.mean([xy[1] for xy in self.expected.values()]))
        self.y = np.arange(-cfg["row_half"], cfg["row_half"], cfg["span_step"]) + cfg["span_step"] / 2
        slides = np.arange(-cfg["side_reach"], cfg["side_reach"] + 1e-9, cfg["span_step"])
        gaps = np.arange(cfg["gap_range"][0], cfg["gap_range"][1] + 1e-9, cfg["span_step"])
        self.rows = [(self.middle + slide, self.size[0] + gap) for gap in gaps for slide in slides]
        self.combs = np.stack([self._comb(centre, pitch) for centre, pitch in self.rows])
        scene_pitch = float(np.ptp([xy[1] for xy in self.expected.values()])) / (len(self.names) - 1)
        self.prior = cfg["prior"] * np.array([abs(centre - self.middle) + abs(pitch - scene_pitch) for centre, pitch in self.rows])

    def find(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> dict[str, np.ndarray]:
        """Centre xy of every bin as seen; the scene file's where no row of boxes is found."""
        row = self._row_x(depth, camera, counter)
        placed = None if row is None else self._placed(depth, camera, counter, row)
        return dict(self.expected) if placed is None else self._boxes(depth, camera, counter, row, placed)

    def _comb(self, centre: float, pitch: float) -> np.ndarray:
        """Over `self.y`: +1 where the front wall of a row of boxes centred at `centre` with this pitch stands, else -1."""
        index = np.arange(len(self.names)) - (len(self.names) - 1) / 2
        on = np.any(np.abs(self.y[:, None] - (centre + index * pitch)) < self.size[0] / 2, axis=1)
        return np.where(on, 1.0, -1.0)

    def _placed(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, row: float) -> tuple[float, float] | None:
        """Middle and pitch of the row of boxes that best explains wall and seen-past cells along the front wall; None if none does."""
        c = self.cfg
        labels = np.sign(np.round(self._wall_labels(depth, camera, counter, row + c["cell"] / 2, self.y)))
        scores = self.combs @ labels - self.prior
        best = int(np.argmax(scores))
        if not scores[best] >= c["min_agreement"] * np.abs(labels).sum() > 0:
            return None
        fine = np.arange(-c["span_step"], c["span_step"] + 1e-9, c["fine_step"])
        near = [(self.rows[best][0] + slide, self.rows[best][1] + stretch) for stretch in fine for slide in fine]
        agreement = [self._comb(centre, pitch) @ labels for centre, pitch in near]
        return tuple(np.mean([row for row, score in zip(near, agreement) if score == max(agreement)], axis=0))

    def _boxes(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, row: float, placed: tuple[float, float]) -> dict[str, np.ndarray]:
        """Centre of every box of the placed row, each forwards from its own front wall where that is seen near the row's."""
        c, found = self.cfg, {}
        lane = np.arange(-c["lane_half"], c["lane_half"], c["cell"]) + c["cell"] / 2
        window = (row - c["row_window"], row + c["row_window"] + c["cell"] / 2)
        for k, name in enumerate(self.names):
            y = placed[0] + (k - (len(self.names) - 1) / 2) * placed[1]
            front = self._front_x(depth, camera, counter, y + lane, window, (c["min_lane_visible"], c["min_fill"]))
            found[name] = np.array([(row if front is None else front) + self.size[1] / 2, y])
        return found

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

    def _front_x(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, y: np.ndarray, x_range: tuple[float, float],
                 needed: tuple[int, float]) -> float | None:
        """x of the nearest wall in `x_range` over the stretch `y`: `needed` = seen cells, and the share of them that is wall."""
        for x in np.arange(x_range[0], x_range[1], self.cfg["cell"]):
            labels = self._wall_labels(depth, camera, counter, x, y)
            wall, open_ = int((labels > 0.5).sum()), int((labels < -0.5).sum())
            if wall + open_ >= needed[0] and wall >= needed[1] * (wall + open_):
                return float(x)
        return None

    def _row_x(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> float | None:
        """x of the row's front wall: the nearest wall over half of what is seen of the row's width (objects fill a third)."""
        c, ys = self.cfg, [centre[1] for centre in self.expected.values()]
        y = np.arange(min(ys) - self.size[0] / 2, max(ys) + self.size[0] / 2, c["cell"]) + c["cell"] / 2
        return self._front_x(depth, camera, counter, y, c["x_range"], (c["min_visible"], c["row_fill"]))
