"""Head camera pose per episode: the robot's own hands tie it to the arm frame, counter plane and bin walls to the scene file."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, PlaneFit, fit_line, fit_plane, rotation_between, yaw_rotation
from halal.perception.hands import HandReference

UP = np.array([0.0, 0.0, 1.0])
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFit:
    """The camera to deproject with, the counter plane in its frame, and what fixed the pose."""
    camera: Camera
    counter: PlaneFit
    source: str             # "scene": counter and bins stand where the scene file says | "hands" | "nominal"


class SelfCalibration:
    """Scene-anchored pose when the hands confirm (or cannot contradict) it, else the pose the hands give, else nominal."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.counter_z = scene.counter_z
        self.hands = HandReference(cfg["hands"])
        bins = scene.cfg["bins"]
        centres = [scene.mat_to_pelvis([*c, 0.0]) for c in bins["centers"].values()]
        self.wall_x = float(centres[0][0] - bins["outer_size"][1] / 2)
        self.wall_spans = [(c[1] - bins["outer_size"][0] / 2, c[1] + bins["outer_size"][0] / 2) for c in centres]
        region = scene.layout_kwargs()["region"]
        corners = [scene.mat_to_pelvis([x, y, scene.object_height() / 2]) for x in region[0] for y in region[1]]
        self.band = np.array([*corners, np.mean(corners, axis=0)])   # object band corners and, last, its centre

    def fit(self, depth: np.ndarray, nominal: Camera, arm_q: np.ndarray, hand_q: np.ndarray) -> CameraFit:
        """Pose and counter plane for this look; every stage that fails its checks leaves the decision to the next."""
        floor = self._counter(depth, nominal)
        by_scene = self._anchored(depth, nominal, floor) if floor else None
        by_hands = self._by_hands(depth, nominal, arm_q, hand_q, floor)
        if by_scene is not None and (by_hands is None or self._apart(by_scene, by_hands) <= self.cfg["agree_within"]):
            camera, source = by_scene, "scene"
        elif by_hands is not None:
            camera, source = by_hands, "hands"
        else:
            camera, source = nominal, "nominal"
        LOG.info("head camera fixed by %s (scene fit %s, hand fit %s)", source, by_scene is not None, by_hands is not None)
        return CameraFit(camera, self.counter(depth, camera), source)

    def counter(self, depth: np.ndarray, camera: Camera) -> PlaneFit:
        """The counter plane as this camera sees it now; the scene file's level plane when it cannot be fitted."""
        return self._counter(depth, camera) or PlaneFit(UP, self.counter_z)

    def _apart(self, a: Camera, b: Camera) -> float:
        """Largest distance between where two poses put the same pixels of the object band."""
        return float(np.linalg.norm(b.deproject(a.project(self.band)) - self.band, axis=1).max())

    def _counter(self, depth: np.ndarray, camera: Camera) -> PlaneFit | None:
        """The counter plane in front of the bins as this camera sees it."""
        c = self.cfg["plane"]
        points = camera.cloud(depth, c["pixel_step"]).reshape(-1, 3)
        near = ((points[:, 0] > c["x_range"][0]) & (points[:, 0] < c["x_range"][1]) & (np.abs(points[:, 1]) < c["y_half"])
                & (np.abs(points[:, 2] - self.counter_z) < c["search"]))
        if near.sum() < c["min_points"]:
            return None
        plane = fit_plane(points[near], c["tolerances"])
        return plane if plane.inliers >= c["min_points"] and plane.rms <= c["max_rms"] else None

    def _by_hands(self, depth: np.ndarray, camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray, floor: PlaneFit | None) -> Camera | None:
        """The pose that puts the seen arms on their kinematic model, if enough of them is seen to trust it."""
        c = self.cfg["hands"]
        fit = self.hands.fit(depth, camera, arm_q, hand_q, floor, self.band[-1])
        trusted = fit is not None and fit.rms <= c["max_rms"] and fit.spread.max() <= c["max_spread"]
        return fit.camera if trusted else None

    def _anchored(self, depth: np.ndarray, camera: Camera, floor: PlaneFit) -> Camera | None:
        """The pose that makes the counter level at its scene height and puts the bin row where the scene file has it."""
        levelled = self._level(camera, floor)
        squared = self._square(depth, levelled) if levelled else None
        slide = self._slide(depth, squared) if squared else None
        return None if slide is None else squared.moved(np.eye(3), np.array([0.0, slide, 0.0]))

    def _level(self, camera: Camera, floor: PlaneFit) -> Camera | None:
        """Roll, pitch and height from the counter plane."""
        centre = camera.T[:3, 3]
        lift = self.counter_z - (floor.offset - floor.normal @ centre + centre[2])
        tilt = np.degrees(np.arccos(min(1.0, floor.normal[2])))
        if tilt > self.cfg["max_turn_deg"] or abs(lift) > self.cfg["max_shift"]:
            return None
        return camera.moved(rotation_between(floor.normal, UP), lift * UP)

    def _square(self, depth: np.ndarray, camera: Camera) -> Camera | None:
        """Yaw and x from the front wall line of the bin row."""
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
