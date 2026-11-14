"""Depth-first detector: self-calibrated camera, top faces at object height, outline fit for shape, yaw and centre."""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import ObjectDetection
from halal.perception.calibration import CameraFit, SelfCalibration
from halal.perception.color import ColorClassifier
from halal.perception.geometry import Camera
from halal.perception.shapes import OutlineFitter, TopFace


class TopFaceDetector:
    name = "topface"

    def __init__(self) -> None:
        cfg = config.load()["perception"]
        self.cfg = cfg["topface"]
        self.scene = Scene.load()
        self.nominal = Camera.head()
        self.calibration = SelfCalibration(self.scene, cfg["calibration"])
        self.fitter = OutlineFitter(self.scene, self.cfg["outline"])
        self.colors = ColorClassifier(self.scene, self.cfg["color"])
        self.camera_fit: CameraFit | None = None
        self.last_t = -np.inf

    @property
    def extrinsics(self) -> np.ndarray:
        """`T_pelvis_optical` in use: self-calibrated once the episode had a successful fit, else nominal."""
        return (self.camera_fit.camera if self.camera_fit else self.nominal).T

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]:
        depth, rgb = obs["head_depth"], obs["head_rgb"]
        if not depth.any():
            return []
        camera = self._camera(depth, float(obs.get("t", 0.0)))
        depth = cv2.medianBlur(depth, self.cfg["depth_median"])
        cloud = camera.cloud(depth)
        height = cloud[..., 2] - self.scene.counter_z
        faces = self._distinct([self.fitter.fit(depth, camera, seed) for seed in self._seeds(cloud, height)])
        gains = self.colors.white_balance(rgb[self._mat_mask(cloud, height)])
        found = [self._describe(face, self._body_rgb(face, camera, rgb, cloud, height), gains) for face in faces]
        return [d for d in found if d.confidence >= self.cfg["min_confidence"]]

    def _camera(self, depth: np.ndarray, t: float) -> Camera:
        """The episode's camera: fitted at the first look that allows it, nominal until then."""
        if t < self.last_t:
            self.camera_fit = None
        self.last_t = t
        if self.camera_fit is None:
            fit = self.calibration.fit(depth, self.nominal)
            if not fit.placed:
                return fit.camera
            self.camera_fit = fit
        return self.camera_fit.camera

    def _on_counter(self, cloud: np.ndarray) -> np.ndarray:
        """Pixels whose point lies over the counter and outside every bin footprint."""
        mat = self.scene.pelvis_to_mat(cloud)
        size = np.asarray(self.scene.cfg["counter"]["mat_size"])
        ok = np.all((mat[..., :2] > 0) & (mat[..., :2] < size), axis=-1)
        half = np.asarray(self.scene.cfg["bins"]["outer_size"]) / 2 + self.cfg["bin_margin"]
        for centre in self.scene.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(mat[..., :2] - np.asarray(centre)) < half, axis=-1)
        return ok

    def _seeds(self, cloud: np.ndarray, height: np.ndarray) -> list[np.ndarray]:
        """Rough xy centres of blobs at object-top height on the counter."""
        c = self.cfg["seed"]
        top = (np.abs(height - self.scene.object_height()) < c["height_band"]) & self._on_counter(cloud)
        mask = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN, np.ones((c["open_px"], c["open_px"]), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [np.nanmean(cloud[labels == i][:, :2], axis=0) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= c["min_px"]]

    def _distinct(self, faces: list[TopFace]) -> list[TopFace]:
        """Drop fits that landed on an object another, better seen, fit already explains."""
        kept: list[TopFace] = []
        for face in sorted(faces, key=lambda f: -f.best.score):
            if all(np.linalg.norm(face.best.centre - k.best.centre) > self.cfg["min_separation"] for k in kept):
                kept.append(face)
        return kept

    def _mat_mask(self, cloud: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Pixels of the bare mat in front of the bins: the white reference."""
        c = self.cfg["color"]
        return (np.abs(height) < c["mat_band"]) & (cloud[..., 0] > c["mat_x_range"][0]) & (cloud[..., 0] < c["mat_x_range"][1])

    def _body_rgb(self, face: TopFace, camera: Camera, rgb: np.ndarray, cloud: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Colours of the pixels on the object itself: over its footprint, between the counter and its top."""
        c = self.cfg["color"]
        u, v = np.round(camera.project(np.r_[face.best.centre, self.fitter.top_z][None])[0, :2]).astype(int)
        rows = slice(max(v - c["window_px"], 0), max(v + c["window_px"], 0))
        cols = slice(max(u - c["window_px"], 0), max(u + c["window_px"], 0))
        reach = np.linalg.norm(cloud[rows, cols, :2] - face.best.centre, axis=-1)
        tall = height[rows, cols]
        on_body = (reach < c["body_radius"]) & (tall > c["body_heights"][0]) & (tall < c["body_heights"][1])
        return rgb[rows, cols][on_body]

    def _describe(self, face: TopFace, body_rgb: np.ndarray, gains: np.ndarray) -> ObjectDetection:
        """Turn one top-face fit into a detection with honest shape, yaw and confidence."""
        c, best = self.cfg, face.best
        top = np.r_[best.centre, self.fitter.top_z]
        if len(body_rgb) < c["min_support_px"]:
            return ObjectDetection(color="red", top=top, confidence=0.0)
        color, color_conf = self.colors.classify(body_rgb, gains)
        shaped = face.margin >= c["min_shape_margin"]
        turned = shaped and best.yaw is not None and best.yaw_contrast >= c["min_yaw_contrast"]
        low, full = c["score_range"]
        fit_conf = float(np.clip((best.score / best.area - low) / (full - low), 0.0, 1.0))
        return ObjectDetection(color=color, top=top, shape=best.shape if shaped else None, yaw=best.yaw if turned else None,
                               confidence=fit_conf * color_conf, pixels=len(face.pixels), visible=best.visible)
