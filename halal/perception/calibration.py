"""Self-calibration of the head camera from depth alone: the counter plane levels it, the bin walls place it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, fit_line, fit_plane, rotation_between, yaw_rotation

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class CameraFit:
    """The camera to deproject with and the references it was corrected against (False = nominal pose kept)."""
    camera: Camera
    levelled: bool
    placed: bool


class SelfCalibration:
    """Corrects `T_pelvis_optical` so that the counter is level at its known height and the bins stand where the scene says."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.counter_z = scene.counter_z
        bins = scene.cfg["bins"]
        centres = [scene.mat_to_pelvis([*c, 0.0]) for c in bins["centers"].values()]
        self.wall_x = float(centres[0][0] - bins["outer_size"][1] / 2)
        self.wall_spans = [(c[1] - bins["outer_size"][0] / 2, c[1] + bins["outer_size"][0] / 2) for c in centres]

    def fit(self, depth: np.ndarray, nominal: Camera) -> CameraFit:
        """Level against the counter, then place against the bins; each stage falls back to what it was given."""
        levelled = self._level(depth, nominal)
        if levelled is None:
            return CameraFit(nominal, False, False)
        placed = self._place(depth, levelled)
        return CameraFit(placed or levelled, True, placed is not None)

    def _level(self, depth: np.ndarray, camera: Camera) -> Camera | None:
        """Roll, pitch and height from the counter plane in front of the bins."""
        c = self.cfg["plane"]
        points = camera.cloud(depth, c["pixel_step"]).reshape(-1, 3)
        near = ((points[:, 0] > c["x_range"][0]) & (points[:, 0] < c["x_range"][1]) & (np.abs(points[:, 1]) < c["y_half"])
                & (np.abs(points[:, 2] - self.counter_z) < c["search"]))
        if near.sum() < c["min_points"]:
            return None
        plane = fit_plane(points[near], c["tolerances"])
        centre = camera.T[:3, 3]
        lift = self.counter_z - (plane.offset - plane.normal @ centre + centre[2])
        tilt = np.degrees(np.arccos(min(1.0, plane.normal[2])))
        if plane.inliers < c["min_points"] or plane.rms > c["max_rms"] or tilt > self.cfg["max_turn_deg"] or abs(lift) > self.cfg["max_shift"]:
            return None
        return camera.moved(rotation_between(plane.normal, UP), lift * UP)

    def _place(self, depth: np.ndarray, camera: Camera) -> Camera | None:
        """Yaw and x from the front wall line of the bin row, y from where that wall starts and stops."""
        squared = self._square(depth, camera)
        if squared is None:
            return None
        slide = self._slide(depth, squared)
        if slide is None:
            return None
        return squared.moved(np.eye(3), np.array([0.0, slide, 0.0]))

    def _square(self, depth: np.ndarray, camera: Camera) -> Camera | None:
        c = self.cfg["wall"]
        points = camera.cloud(depth, c["pixel_step"]).reshape(-1, 3)
        height = points[:, 2] - self.counter_z
        front = ((height > c["height_band"][0]) & (height < c["height_band"][1]) & (np.abs(points[:, 0] - self.wall_x) < c["search"])
                 & (np.abs(points[:, 1]) < c["y_half"]))
        if front.sum() < c["min_points"]:
            return None
        line = fit_line(points[front, 1], points[front, 0], c["tolerances"])
        angle = float(np.arctan(line.slope))
        centre = camera.T[:3, 3]
        turned_x = np.cos(angle) * (line.intercept - centre[0]) + np.sin(angle) * centre[1] + centre[0]
        shift = self.wall_x - turned_x
        if line.inliers < c["min_points"] or line.rms > c["max_rms"] or abs(np.degrees(angle)) > self.cfg["max_turn_deg"] or abs(shift) > self.cfg["max_shift"]:
            return None
        return camera.moved(yaw_rotation(angle), np.array([shift, 0.0, 0.0]))

    def _slide(self, depth: np.ndarray, camera: Camera) -> float | None:
        """The y shift that best lines the seen wall segments and gaps up with the three bins."""
        c = self.cfg["wall"]
        y = np.arange(-c["y_half"], c["y_half"], c["profile_step"]) + c["profile_step"] / 2
        rows = [camera.surface_labels(depth, np.c_[np.full_like(y, self.wall_x), y, np.full_like(y, self.counter_z + h)],
                                      c["depth_tolerance"]) for h in c["profile_heights"]]
        profile = np.mean(rows, axis=0)
        reach = int(round(self.cfg["max_shift"] / c["profile_step"]))
        offsets = np.arange(-reach, reach + 1)
        scores = np.array([profile @ self._expected_wall(y - k * c["profile_step"]) for k in offsets])
        best = int(np.argmax(scores))
        if np.abs(profile).sum() < c["min_profile"] or scores[best] < c["min_agreement"] * np.abs(profile).sum() or best in (0, len(offsets) - 1):
            return None
        a, b, d = scores[best - 1:best + 2]
        peak = offsets[best] + (0.5 * (a - d) / (a - 2 * b + d) if a - 2 * b + d < 0 else 0.0)
        return float(-peak * c["profile_step"])

    def _expected_wall(self, y: np.ndarray) -> np.ndarray:
        """+1 where a bin's front wall stands at this y, -1 in the gaps and beside the row."""
        on = np.zeros(len(y), bool)
        for lo, hi in self.wall_spans:
            on |= (y >= lo) & (y <= hi)
        return np.where(on, 1.0, -1.0)
