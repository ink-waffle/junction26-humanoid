"""Head camera pose: the robot's own hands tie it to the arm frame, counter plane and bin walls to the scene file."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from aaltonomous.scene.scene import Scene

from halal.perception.geometry import Camera, PlaneFit, fit_line, fit_plane, rotation_between, yaw_rotation
from halal.perception.hands import HandFit, HandReference

UP = np.array([0.0, 0.0, 1.0])
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFit:
    """The camera to deproject with and what fixed its pose."""
    camera: Camera
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

    def decide(self, nominal: Camera, by_scene: Camera | None, by_hands: Camera | None) -> CameraFit:
        """The scene anchor unless the hands contradict it by more than `agree_within`; then the hands; with neither, nominal."""
        apart = self.apart(by_scene, by_hands) if by_scene and by_hands else None
        if by_scene is not None and (apart is None or apart <= self.cfg["agree_within"]):
            fit = CameraFit(by_scene, "scene")
        else:
            fit = CameraFit(by_hands, "hands") if by_hands is not None else CameraFit(nominal, "nominal")
        LOG.debug("head camera by %s: scene anchor %s, arms %s, apart %s", fit.source, "fits" if by_scene else "does not fit",
                  "seen" if by_hands else "not seen", "n/a" if apart is None else f"{1e3 * apart:.1f} mm")
        return fit

    def scene_anchor(self, depth: np.ndarray, nominal: Camera) -> Camera | None:
        """The pose that makes the counter level at its scene height and puts the bin row where the scene file has it."""
        floor = self.counter_fit(depth, nominal)
        return self._anchored(depth, nominal, floor) if floor else None

    def arm_pose(self, depth: np.ndarray, nominal: Camera, joints: tuple[np.ndarray, np.ndarray]) -> Camera | None:
        """The pose that puts the seen arms on their kinematic model, if what was matched is provably an arm."""
        return self._by_hands(depth, nominal, joints, self.counter_fit(depth, nominal))

    def consensus(self, poses: list[Camera]) -> Camera | None:
        """Of several believed arm poses of one episode, the one nearest to all others at the object band; None of none."""
        if not poses:
            return None
        apart = [[self.apart(a, b) for b in poses] for a in poses]
        return poses[int(np.argmin(np.sum(apart, axis=1)))]

    def counter(self, depth: np.ndarray, camera: Camera) -> PlaneFit:
        """The counter plane as this camera sees it now; the scene file's level plane when it cannot be fitted."""
        return self.counter_fit(depth, camera) or PlaneFit(UP, self.counter_z)

    def apart(self, a: Camera, b: Camera) -> float:
        """Largest distance between where two poses put the same pixels of the object band."""
        return float(np.linalg.norm(b.deproject(a.project(self.band)) - self.band, axis=1).max())

    def counter_fit(self, depth: np.ndarray, camera: Camera) -> PlaneFit | None:
        """The counter plane in front of the bins as this camera sees it; None when no plane is found there."""
        c = self.cfg["plane"]
        points = camera.cloud(depth, c["pixel_step"]).reshape(-1, 3)
        near = ((points[:, 0] > c["x_range"][0]) & (points[:, 0] < c["x_range"][1]) & (np.abs(points[:, 1]) < c["y_half"])
                & (np.abs(points[:, 2] - self.counter_z) < c["search"]))
        if near.sum() < c["min_points"]:
            return None
        plane = fit_plane(points[near], c["tolerances"])
        return plane if plane.inliers >= c["min_points"] and plane.rms <= c["max_rms"] else None

    def _by_hands(self, depth: np.ndarray, camera: Camera, joints: tuple[np.ndarray, np.ndarray], floor: PlaneFit | None) -> Camera | None:
        fit = self.hands.fit(depth, camera, *joints, floor, self.band[-1])
        if fit is None:
            LOG.info("arms: too little of them in view")
            return None
        checks, (turn, shift) = self.hand_checks(fit, camera), self.displacement(fit, camera)
        LOG.info("arms: %d px, rms %.1f mm, model covered %.2f of the %.2f that returned depth (judged on %d of %d visible points, those clear "
                 "of the counter), pixels on model %.2f, spread %.1f mm, camera turned %.1f deg, shifted %.0f mm: %s", fit.points, 1e3 * fit.rms,
                 fit.coverage, fit.known, fit.clear, fit.visible, fit.inliers, 1e3 * fit.spread.max(), turn, 1e3 * shift,
                 "trusted" if all(checks.values()) else f"rejected {checks}")
        return fit.camera if all(checks.values()) else None

    @staticmethod
    def displacement(fit: HandFit, start: Camera) -> tuple[float, float]:
        """How far a hand fit moved the camera from the pose it started at: turn (deg) and shift (m)."""
        turn = np.degrees(np.linalg.norm(Rotation.from_matrix(fit.camera.T[:3, :3] @ start.T[:3, :3].T).as_rotvec()))
        return float(turn), float(np.linalg.norm(fit.camera.T[:3, 3] - start.T[:3, 3]))

    def hand_checks(self, fit: HandFit, start: Camera) -> dict[str, bool]:
        """Every test a hand fit must pass to be believed, by name."""
        c, (turn, shift) = self.cfg["hands"], self.displacement(fit, start)
        return {"rms": fit.rms <= c["max_rms"], "spread": fit.spread.max() <= c["max_spread"], "coverage": fit.coverage >= c["min_coverage"],
                "inliers": fit.inliers >= c["min_inliers"], "turn": turn <= c["max_turn_deg"], "shift": shift <= c["max_shift"]}

    def _anchored(self, depth: np.ndarray, camera: Camera, floor: PlaneFit) -> Camera | None:
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
