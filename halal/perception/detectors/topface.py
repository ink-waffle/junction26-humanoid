"""Depth-first detector: calibrated camera, top faces at object height, outline fit for shape, yaw and centre."""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import ObjectDetection, SceneLayout, scene_layout
from halal.perception.bins import BinFinder
from halal.perception.calibration import CameraFit, SelfCalibration
from halal.perception.color import ColorClassifier
from halal.perception.geometry import Camera, PlaneFit
from halal.perception.shapes import OutlineFitter, TopFace


class TopFaceDetector:
    name = "topface"

    def __init__(self) -> None:
        cfg = config.load()["perception"]
        self.cfg = cfg["topface"]
        self.scene = Scene.load()
        self.nominal = Camera.head()
        self.calibration = SelfCalibration(self.scene, cfg["calibration"])
        self.bins = BinFinder(self.scene, cfg["bins"])
        self.fitter = OutlineFitter(self.scene, self.cfg["outline"])
        self.colors = ColorClassifier(self.cfg["color"])
        self.band_centre = self.calibration.band[-1, :2]
        self.pose: CameraFit | None = None
        self.seen = scene_layout(self.scene)
        self.last_t = -np.inf

    @property
    def extrinsics(self) -> np.ndarray:
        """`T_pelvis_optical` in use: the episode's calibrated pose once there is one, else nominal."""
        return (self.pose.camera if self.pose else self.nominal).T

    def layout(self) -> SceneLayout:
        return self.seen

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]:
        depth, rgb = obs["head_depth"], obs["head_rgb"]
        if not depth.any():
            return []
        camera = self._camera(obs)
        counter = self.calibration.counter(depth, camera)
        self.seen = self._layout(depth, camera, counter)
        depth = cv2.medianBlur(depth, self.cfg["depth_median"])
        cloud = camera.cloud(depth)
        height = counter.height(cloud)
        gains = self.colors.white_balance(rgb[self._bare_counter(cloud, height)])
        faces = self._faces(depth, camera, cloud, height, counter, rgb, gains)
        found = [self._describe(face, *self._body(face, camera, rgb, cloud, height), gains) for face in faces]
        return [d for d in found if d.confidence >= self.cfg["min_confidence"]]

    def _camera(self, obs: Mapping[str, Any]) -> Camera:
        """The episode's camera pose: refitted while no arm has been seen, replaced only by a fit with better evidence."""
        t = float(obs.get("t", 0.0))
        if t < self.last_t:
            self.pose = None
        self.last_t = t
        if self.pose is None or self.pose.evidence < self.cfg["settled_evidence"]:
            still = np.abs(obs.get("arm_dq", 0.0)).max() < self.cfg["still_speed"]
            fit = self.calibration.fit(obs["head_depth"], self.nominal, (obs["arm_q"], obs["hand_q"]) if still else None)
            if self.pose is None or fit.evidence > self.pose.evidence:
                self.pose = fit
        return self.pose.camera

    def _layout(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> SceneLayout:
        """Counter height and bins: the scene file's when the camera is anchored to it, as seen when the hands fixed it."""
        if self.pose is None or self.pose.source != "hands":
            return SceneLayout(self.scene.counter_z, scene_layout(self.scene).bins, self.pose.source if self.pose else "nominal")
        return SceneLayout(counter.z_at(self.band_centre), self.bins.find(depth, camera, counter), "hands")

    def _on_counter(self, cloud: np.ndarray) -> np.ndarray:
        """Pixels whose point lies inside the workspace and outside every bin footprint."""
        (x0, x1), y_half = self.cfg["workspace_x"], self.cfg["workspace_y_half"]
        ok = (cloud[..., 0] > x0) & (cloud[..., 0] < x1) & (np.abs(cloud[..., 1]) < y_half)
        half = np.asarray(self.scene.cfg["bins"]["outer_size"])[::-1] / 2 + self.cfg["bin_margin"]
        for centre in self.seen.bins.values():
            ok &= ~np.all(np.abs(cloud[..., :2] - centre) < half, axis=-1)
        return ok

    def _faces(self, depth: np.ndarray, camera: Camera, cloud: np.ndarray, height: np.ndarray, counter: PlaneFit,
               rgb: np.ndarray, gains: np.ndarray) -> list[TopFace]:
        """Top faces of all objects: per colour first (touching objects of two colours part there), then what is left."""
        c = self.cfg["seed"]
        top = (np.abs(height - self.scene.object_height()) < c["height_band"]) & self._on_counter(cloud)
        kernel = np.ones((c["open_px"], c["open_px"]), np.uint8)
        mask = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        colour = np.full(mask.shape, -1, np.int8)
        colour[mask > 0] = self.colors.pixel_classes(rgb[mask > 0], gains)
        faces: list[TopFace] = []
        for index in range(len(self.cfg["color"]["basins"])):
            own = cv2.morphologyEx((colour == index).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            rival = (colour >= 0) & (colour != index)
            faces += [face for seed in self._seeds(own, cloud, counter) for face in self.fitter.fit(depth, camera, seed, counter, rival)]
        faces = self._distinct(faces)
        for face in faces:
            window = self._window(face, camera)
            mask[window][np.linalg.norm(cloud[window][..., :2] - face.best.centre[:2], axis=-1) < c["claim_radius"]] = 0
        rest = [face for seed in self._seeds(mask, cloud, counter) for face in self.fitter.fit(depth, camera, seed, counter)]
        return self._distinct(faces + rest)

    def _window(self, face: TopFace, camera: Camera) -> tuple[slice, slice]:
        """Image rows and columns that contain everything of the object at the fitted centre."""
        reach = self.cfg["window_px"]
        u, v = np.round(camera.project(face.best.centre[None])[0, :2]).astype(int)
        return slice(max(v - reach, 0), max(v + reach, 0)), slice(max(u - reach, 0), max(u + reach, 0))

    def _seeds(self, mask: np.ndarray, cloud: np.ndarray, counter: PlaneFit) -> list[np.ndarray]:
        """Rough top-plane centres of the blobs left in the top-height mask."""
        tall = self.scene.object_height()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = [np.nanmean(cloud[labels == i], axis=0) for i in range(1, count)
                 if stats[i, cv2.CC_STAT_AREA] >= self.cfg["seed"]["min_px"]]
        return [blob - (counter.height(blob) - tall) * counter.normal for blob in blobs]

    def _distinct(self, faces: list[TopFace]) -> list[TopFace]:
        """Drop fits that landed on an object another, better seen, fit already explains."""
        kept: list[TopFace] = []
        for face in sorted(faces, key=lambda f: -f.best.score):
            if all(np.linalg.norm(face.best.centre - k.best.centre) > self.cfg["min_separation"] for k in kept):
                kept.append(face)
        return kept

    def _bare_counter(self, cloud: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Pixels of the bare counter in front of the bins: the white reference."""
        c = self.cfg["color"]
        return (np.abs(height) < c["mat_band"]) & (cloud[..., 0] > c["mat_x_range"][0]) & (cloud[..., 0] < c["mat_x_range"][1])

    def _body(self, face: TopFace, camera: Camera, rgb: np.ndarray, cloud: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, int]:
        """Colours of the pixels on the object (footprint, counter to top) and how many pixels stand just above its top."""
        c, window = self.cfg["color"], self._window(face, camera)
        over = np.linalg.norm(cloud[window][..., :2] - face.best.centre[:2], axis=-1) < c["body_radius"]
        tall = height[window]
        low, top = self.cfg["overhang_band"]
        overhang = over & (tall > self.scene.object_height() + low) & (tall < self.scene.object_height() + top)
        return rgb[window][over & (tall > c["body_heights"][0]) & (tall < c["body_heights"][1])], int(overhang.sum())

    def _describe(self, face: TopFace, body_rgb: np.ndarray, overhang: int, gains: np.ndarray) -> ObjectDetection:
        """Turn one top-face fit into a detection with honest shape, yaw and confidence."""
        c, best = self.cfg, face.best
        if len(body_rgb) < c["min_support_px"] or overhang > c["max_overhang"] * len(body_rgb):
            return ObjectDetection(color="red", top=best.centre, confidence=0.0)
        color, color_conf = self.colors.classify(body_rgb, gains)
        shaped = face.margin >= c["min_shape_margin"]
        turned = shaped and best.yaw is not None and best.yaw_contrast >= c["min_yaw_contrast"]
        low, full = c["score_range"]
        fit_conf = float(np.clip((best.score / best.area - low) / (full - low), 0.0, 1.0))
        return ObjectDetection(color=color, top=best.centre, shape=best.shape if shaped else None, yaw=best.yaw if turned else None,
                               confidence=fit_conf * color_conf, pixels=len(face.pixels), visible=best.visible)
