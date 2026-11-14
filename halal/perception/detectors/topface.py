"""Depth-first detector: calibrated camera, top faces at object height, outline fit for shape, yaw and centre."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import ObjectDetection, SceneLayout, scene_layout
from halal.perception.bins import BinFinder
from halal.perception.calibration import CameraFit, SelfCalibration
from halal.perception.color import ColorClassifier
from halal.perception.geometry import Camera, PlaneFit, levelness, plan_depth
from halal.perception.roof import Roof, fit_roof
from halal.perception.shapes import OutlineFitter, TopFace


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Draft:
    """A detection whose colour is not named yet, and the colour sampled on it (None: geometry has rejected it)."""
    detection: ObjectDetection
    colour: np.ndarray | None = None


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
        self.pose = CameraFit(self.nominal, "nominal")
        self.seen = scene_layout(self.scene)
        self._forget()

    def _forget(self) -> None:
        """Drop what earlier looks established: a new episode has its own camera pose and scene."""
        self.anchor: Camera | None = None
        self.arms: Camera | None = None
        self.arm_attempts = 0
        self.split: float | None = None
        self.last_t = -np.inf
        self.last_joints: np.ndarray | None = None

    @property
    def extrinsics(self) -> np.ndarray:
        """`T_pelvis_optical` used at the last look."""
        return self.pose.camera.T

    def layout(self) -> SceneLayout:
        return self.seen

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]:
        depth, rgb = obs["head_depth"], obs["head_rgb"]
        if not depth.any():
            return []
        anchor, arms = self._references(obs)
        pose = self.calibration.decide(self.nominal, anchor or self.anchor, arms)
        if pose.source != self.pose.source:
            LOG.info("head camera now fixed by %s (was %s)", pose.source, self.pose.source)
        self.pose, camera = pose, pose.camera
        counter = self.calibration.counter(depth, camera)
        self.seen = self._layout(depth, camera, counter)
        depth = cv2.medianBlur(depth, self.cfg["depth_median"])
        cloud = camera.cloud(depth)
        height = counter.height(cloud)
        free = self._on_counter(cloud)
        gains = self.colors.white_balance(rgb[self._bare_counter(cloud, height)])
        faces = [face for face in self._faces(depth, camera, cloud, height, free, counter, rgb, gains) if self._level(face, cloud, counter)]
        drafts = [self._describe(face, counter, *self._body(face, camera, rgb, cloud, height, counter), gains) for face in faces]
        drafts += self._stragglers(faces, camera, cloud, height, free, rgb, gains, counter)
        if self.split is None:
            self.split = self.colors.split([draft.colour for draft in drafts if draft.colour is not None])
        found = [d for d in map(self._named, drafts) if d.confidence >= self.cfg["min_confidence"]]
        if found:
            self.anchor, self.arms = anchor or self.anchor, arms
        return found

    def _references(self, obs: Mapping[str, Any]) -> tuple[Camera | None, Camera | None]:
        """This look's scene anchor, and the arm pose: the episode's once a look with objects kept one, else a new attempt."""
        t = float(obs.get("t", 0.0))
        if t < self.last_t:
            self._forget()
        still = self._still(obs, t)
        anchor = self.calibration.scene_anchor(obs["head_depth"], self.nominal)
        if self.arms is not None or not still or self.arm_attempts >= self.cfg["arm_attempts"]:
            return anchor, self.arms
        self.arm_attempts += 1
        return anchor, self.calibration.arm_pose(obs["head_depth"], self.nominal, (obs["arm_q"], obs["hand_q"]))

    def _still(self, obs: Mapping[str, Any], t: float) -> bool:
        """True when arms and fingers stand where an earlier look of this episode saw them: their image shows the joints."""
        joints = np.r_[obs["arm_q"], obs["hand_q"]]
        waited = t - self.last_t >= self.cfg["still_time"]
        same = self.last_joints is not None and np.abs(joints - self.last_joints).max() < self.cfg["still_tolerance"]
        if waited or self.last_joints is None:
            self.last_t, self.last_joints = t, joints
        return bool(waited and same and np.abs(obs.get("arm_dq", 0.0)).max() < self.cfg["still_speed"])

    def _layout(self, depth: np.ndarray, camera: Camera, counter: PlaneFit) -> SceneLayout:
        """Counter height and bins: the scene file's when the camera is anchored to it, as seen when the hands fixed it."""
        if self.pose.source != "hands":
            return SceneLayout(self.scene.counter_z, scene_layout(self.scene).bins, self.pose.source)
        bins = self.bins.find(depth, camera, counter)
        return SceneLayout(counter.z_at(self.band_centre), bins, "hands", self.bins.rim_z(depth, camera, counter, bins))

    def _on_counter(self, cloud: np.ndarray) -> np.ndarray:
        """Pixels whose point lies inside the workspace and outside every bin footprint."""
        (x0, x1), y_half = self.cfg["workspace_x"], self.cfg["workspace_y_half"]
        ok = (cloud[..., 0] > x0) & (cloud[..., 0] < x1) & (np.abs(cloud[..., 1]) < y_half)
        half = np.asarray(self.scene.cfg["bins"]["outer_size"])[::-1] / 2 + self.cfg["bin_margin"]
        for centre in self.seen.bins.values():
            ok &= ~np.all(np.abs(cloud[..., :2] - centre) < half, axis=-1)
        return ok

    def _faces(self, depth: np.ndarray, camera: Camera, cloud: np.ndarray, height: np.ndarray, free: np.ndarray,
               counter: PlaneFit, rgb: np.ndarray, gains: np.ndarray) -> list[TopFace]:
        """Top faces of all objects: per colour first (touching objects of two colours part there), then what is left."""
        c, (small, large) = self.cfg["seed"], self.cfg["size_range"]
        top = (height > small - c["height_band"]) & (height < large + c["height_band"]) & free
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

    def _unanchored(self) -> bool:
        """True when the scene did not fix the camera: bin footprints are then guesses and walls can reach the candidates."""
        return self.pose.source != "scene"

    def _level(self, face: TopFace, cloud: np.ndarray, counter: PlaneFit) -> bool:
        """False for a fit whose top cells form no level patch (a wall face); only judged while the camera is unanchored."""
        if not self._unanchored():
            return True
        inside = np.all((face.pixels >= 0) & (face.pixels < cloud.shape[1::-1]), axis=1)
        points = cloud[face.pixels[inside, 1], face.pixels[inside, 0]]
        points = points[np.isfinite(points).all(axis=1)]
        return len(points) >= 3 and levelness(points, counter.normal) >= self.cfg["wall_guard"]["min_level"]

    def _window(self, face: TopFace, camera: Camera) -> tuple[slice, slice]:
        """Image rows and columns that contain everything of the object at the fitted centre."""
        reach = self.cfg["window_px"]
        u, v = np.round(camera.project(face.best.centre[None])[0, :2]).astype(int)
        return slice(max(v - reach, 0), max(v + reach, 0)), slice(max(u - reach, 0), max(u + reach, 0))

    def _named(self, draft: Draft) -> ObjectDetection:
        """The draft with its colour family and confidence, by the episode's red / yellow boundary where one was seen."""
        if draft.colour is None:
            return draft.detection
        color, confidence = self.colors.name(draft.colour, self.split)
        return replace(draft.detection, color=color, confidence=confidence)

    def _stragglers(self, faces: list[TopFace], camera: Camera, cloud: np.ndarray, height: np.ndarray, free: np.ndarray,
                    rgb: np.ndarray, gains: np.ndarray, counter: PlaneFit) -> list[Draft]:
        """Coloured blobs on the counter that no top-face fit explains: a prism lying on a side, else colour and a rough centre."""
        c = self.cfg["straggler"]
        mask = ((height > c["heights"][0]) & (height < c["heights"][1]) & free).astype(np.uint8)
        for face in faces:
            window = self._window(face, camera)
            mask[window][np.linalg.norm(cloud[window][..., :2] - face.best.centre[:2], axis=-1) < c["claim_radius"]] = 0
        coloured = np.zeros_like(mask)
        coloured[mask > 0] = self.colors.pixel_classes(rgb[mask > 0], gains) >= 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(coloured, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
        found = []
        for i in range(1, count):
            points = cloud[labels == i]
            if stats[i, cv2.CC_STAT_AREA] < c["min_px"] or np.ptp(points[:, :2], axis=0).max() > c["max_extent"]:
                continue
            if self._unanchored() and plan_depth(points) < self.cfg["wall_guard"]["min_depth"]:
                continue
            blob = ObjectDetection(color="red", top=np.r_[points[:, :2].mean(axis=0), points[:, 2].max()], confidence=0.0,
                                   pixels=int(stats[i, cv2.CC_STAT_AREA]), visible=0.0)
            roof = fit_roof(points, height[labels == i], counter.normal, counter.offset, self.cfg["roof"])
            found.append(Draft(blob if roof is None else self._lying(roof, blob.pixels), self.colors.sample(rgb[labels == i], gains)))
        return found

    @staticmethod
    def _lying(roof: Roof, pixels: int) -> ObjectDetection:
        """A prism lying on a side, still to be coloured: ridge centre, ridge direction and the ridge height as measured."""
        return ObjectDetection(color="red", top=roof.centre, shape="triangle", yaw=roof.yaw, confidence=0.0, pixels=pixels,
                               visible=roof.support, size=roof.height, lying=True)

    def _seeds(self, mask: np.ndarray, cloud: np.ndarray, counter: PlaneFit) -> list[np.ndarray]:
        """Rough centres of the blobs left in the top-height mask, each on the top plane of an object as tall as the blob lies."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = [cloud[labels == i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.cfg["seed"]["min_px"]]
        centres = [np.nanmean(blob, axis=0) for blob in blobs]
        talls = [float(np.nanmedian(counter.height(blob))) for blob in blobs]
        return [centre - (counter.height(centre) - self._tall(tall)) * counter.normal for centre, tall in zip(centres, talls)]

    def _tall(self, measured: float) -> float:
        """Object height to work with: the scene file's unless the measured one clearly differs, kept inside the size range."""
        scene = self.scene.object_height()
        return scene if abs(measured - scene) < self.cfg["size_snap"] else float(np.clip(measured, *self.cfg["size_range"]))

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

    def _body(self, face: TopFace, camera: Camera, rgb: np.ndarray, cloud: np.ndarray, height: np.ndarray,
              counter: PlaneFit) -> tuple[np.ndarray, int]:
        """Colours of the pixels on the object (footprint, counter to top) and how many pixels stand just above its top."""
        c, window = self.cfg["color"], self._window(face, camera)
        over = np.linalg.norm(cloud[window][..., :2] - face.best.centre[:2], axis=-1) < c["body_radius"]
        tall, own = height[window], float(counter.height(face.best.centre))
        low, top = self.cfg["overhang_band"]
        overhang = over & (tall > own + low) & (tall < own + top)
        return rgb[window][over & (tall > c["body_heights"][0]) & (tall < c["body_heights"][1])], int(overhang.sum())

    def _describe(self, face: TopFace, counter: PlaneFit, body_rgb: np.ndarray, overhang: int, gains: np.ndarray) -> Draft:
        """Turn one top-face fit into a detection still to be coloured: `visible` is about the pose and the size."""
        c, best = self.cfg, face.best
        tall, measured = float(counter.height(best.centre)), best.visible >= c["min_size_visible"]
        odd = abs(tall - self.scene.object_height()) >= c["size_snap"] and not measured   # an unexpected size needs a well seen top
        if len(body_rgb) < c["min_support_px"] or overhang > c["max_overhang"] * len(body_rgb) or odd:
            return Draft(ObjectDetection(color="red", top=best.centre, confidence=0.0))
        shaped = face.margin >= c["min_shape_margin"] and best.visible >= c["min_shape_visible"]
        turned = shaped and best.yaw is not None and best.yaw_contrast >= c["min_yaw_contrast"]
        size = tall if measured else None
        seen = ObjectDetection(color="red", top=best.centre, shape=best.shape if shaped else None, yaw=best.yaw if turned else None,
                               confidence=0.0, pixels=len(face.pixels), visible=best.visible, size=size)
        return Draft(seen, self.colors.sample(body_rgb, gains))
