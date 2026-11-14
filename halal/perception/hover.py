"""One look from the grasp hover pose: both arms turn the camera, the picking hand against its FK surface shifts it, the target top is re-fitted."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.spatial import cKDTree

from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import ObjectDetection
from halal.perception.calibration import SelfCalibration
from halal.perception.geometry import MM, Camera, PlaneFit
from halal.perception.hands import ASSET, CURL_JOINTS, HandReference
from halal.perception.shapes import OutlineFit, OutlineFitter


@dataclass(frozen=True)
class HoverLook:
    """What one hover look measured; `failed` names the guards that did not pass."""
    object_top: np.ndarray | None   # (3,) target top centre in the arm (FK) frame = seen_top + hand_shift; None when either is missing
    seen_top: np.ndarray | None     # (3,) target top centre in this frame, through `camera` when set, else through the camera given
    hand_shift: np.ndarray | None   # (3,) pelvis-frame shift that puts the seen picking hand onto its FK surface
    visible: float                  # top-face share seen in this frame (0 when no face found)
    rms: float                      # hand fit point-to-plane residual (m)
    coverage: float                 # share of the visible hand model with depth at its pixel that has a matched pixel
    inliers: float                  # share of the depth pixels around the hand that lie on the model
    points: int                     # matched pixels at the last gate
    spread: float                   # predicted std of the shift (m, largest axis)
    failed: str                     # "" when every guard passed, else the failed guards comma-separated ("visible,inliers")
    seconds: float
    camera: Camera | None = None    # this look's own believed arm fit, which `seen_top` was measured through; None: the camera given


@dataclass(frozen=True)
class ShiftFit:
    """Translation-only registration of the seen hand onto its FK surface, with its support."""
    shift: np.ndarray
    rms: float
    coverage: float
    inliers: float
    points: int
    spread: float


class HoverMeasurer:
    """Measures the target from the hover pose relative to the picking hand; holds no episode state."""

    def __init__(self) -> None:
        cfg = config.load()["perception"]
        self.cfg, self.hands = cfg["hover"], cfg["calibration"]["hands"]
        scene = Scene.load()
        self.calibration = SelfCalibration(scene, cfg["calibration"])
        self.reference = self.calibration.hands
        self.fitter = OutlineFitter(scene, cfg["topface"]["outline"])
        self.median = cfg["topface"]["depth_median"]
        self.bin_half = np.asarray(scene.cfg["bins"]["outer_size"])[::-1] / 2 + self.cfg["bin_margin"]
        names = [str(name) for name in np.load(ASSET)["links"]]
        self.own = {side: self._side_links(names, side) for side in ("left", "right")}

    def _side_links(self, names: list[str], side: str) -> np.ndarray:
        """True for the surface samples of one side's links named in `hover.links`."""
        chosen = [i for i, name in enumerate(names) if name.startswith(f"{side}_") and name[len(side) + 1:].startswith(tuple(self.cfg["links"]))]
        return np.isin(self.reference.surface.link, chosen)

    def measure(self, obs: Mapping[str, Any], camera: Camera, side: str, target: ObjectDetection,
                bins: Mapping[str, np.ndarray]) -> HoverLook:
        """The target top in the arm frame from one still look at the hover pose, with every guard by name."""
        start, depth = time.perf_counter(), obs["head_depth"]
        arm_q, hand_q = np.asarray(obs["arm_q"]), np.asarray(obs["hand_q"])
        turned = self._turned(depth, camera, arm_q, hand_q)
        expected = np.asarray(target.top, float) if turned is None else turned.deproject(camera.project(np.asarray(target.top, float)[None]))[0]
        camera = turned or camera
        counter = self.calibration.counter(depth, camera)
        face = self._top_face(depth, camera, counter, expected)
        fit = self._shift(depth, camera, counter, side, arm_q, hand_q, bins)
        seen = None if face is None else face.centre
        visible = 0.0 if face is None else face.visible
        shift = None if fit is None else fit.shift
        return HoverLook(None if seen is None or shift is None else seen + shift, seen, shift, visible,
                         np.nan if fit is None else fit.rms, 0.0 if fit is None else fit.coverage, 0.0 if fit is None else fit.inliers,
                         0 if fit is None else fit.points, np.nan if fit is None else fit.spread, ",".join(self._failed(visible, fit)),
                         time.perf_counter() - start, turned)

    def _turned(self, depth: np.ndarray, camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray) -> Camera | None:
        """This look's own believed 6-DOF fit of both arms (held up at the hover, clear of the floor gate); None when off or not believed."""
        if not self.cfg["arm_fit"]:
            return None
        fit = self.calibration.hand_fit(depth, camera, (arm_q, hand_q), self.calibration.counter_fit(depth, camera))
        return fit.camera if fit is not None and all(self.calibration.hand_checks(fit, camera).values()) else None

    def _failed(self, visible: float, fit: ShiftFit | None) -> list[str]:
        """Names of the guards a look does not pass."""
        c = self.cfg
        failed = ["visible"] if visible < c["min_visible"] else []
        if fit is None:
            return failed + ["points"]
        checks = {"points": fit.points >= c["min_points"], "rms": fit.rms <= c["max_rms"], "coverage": fit.coverage >= c["min_coverage"],
                  "inliers": fit.inliers >= c["min_inliers"], "spread": fit.spread <= c["max_spread"],
                  "shift": float(np.linalg.norm(fit.shift)) <= c["max_shift"]}
        return failed + [name for name, ok in checks.items() if not ok]

    def _top_face(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, expected: np.ndarray) -> OutlineFit | None:
        """The best outline fit of the top face nearest where `camera` puts the target's top, seeded there; None when none lies within reach."""
        faces = self.fitter.fit(cv2.medianBlur(depth, self.median), camera, expected, counter)
        near = [face.best for face in faces if np.linalg.norm(face.best.centre[:2] - expected[:2]) <= self.cfg["match_radius"]]
        return min(near, key=lambda fit: np.linalg.norm(fit.centre[:2] - expected[:2]), default=None)

    def _model(self, camera: Camera, side: str, arm_q: np.ndarray, hand_q: np.ndarray,
               shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Both arms' surface (points, normals) as far as it faces the camera, and True on the picking side's wrist and hand."""
        surface = self.reference.surface
        fingers = bool(np.abs(hand_q[list(CURL_JOINTS)]).max() <= self.hands["open_curl"])
        points, normals = surface.posed(arm_q, hand_q, fingers)
        own = self.own[side] if fingers else self.own[side][surface.rigid]
        front = self.reference.facing(points, normals, camera, shape)
        return points[front], normals[front], own[front]

    def _evidence(self, camera: Camera, counter: PlaneFit, bins: Mapping[str, np.ndarray], uvz: np.ndarray) -> np.ndarray:
        """True for rows (u, v, depth) that can be hand: above the floor gate and outside every bin footprint."""
        points = camera.deproject(uvz)
        ok = (uvz[:, 2] > 0) & (counter.height(points) > self.cfg["floor_clearance"])
        for centre in bins.values():
            ok &= ~np.all(np.abs(points[:, :2] - centre) < self.bin_half, axis=1)
        return ok

    def _pixels(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, bins: Mapping[str, np.ndarray], model: np.ndarray) -> np.ndarray:
        """Rows (u, v, depth) of the hand evidence in the padded image box of the model."""
        uv, pad, step = camera.project(model)[:, :2], self.hands["pad_px"], self.hands["pixel_step"]
        lo = np.maximum(np.floor(uv.min(axis=0)).astype(int) - pad, 0)
        hi = np.ceil(uv.max(axis=0)).astype(int) + pad
        v, u = np.mgrid[lo[1]:min(hi[1], depth.shape[0]):step, lo[0]:min(hi[0], depth.shape[1]):step]
        uvz = np.c_[u.ravel(), v.ravel(), depth[v, u].ravel() * MM]
        return uvz[self._evidence(camera, counter, bins, uvz)]

    def _shift(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, side: str, arm_q: np.ndarray, hand_q: np.ndarray,
               bins: Mapping[str, np.ndarray]) -> ShiftFit | None:
        """Translation-only point-to-plane ICP of the seen picking hand onto its model; None with too little of either in view."""
        model, normals, own = self._model(camera, side, arm_q, hand_q, depth.shape)
        if own.sum() < self.cfg["min_points"]:
            return None
        pixels = self._pixels(depth, camera, counter, bins, model[own])
        if len(pixels) < self.cfg["min_points"]:
            return None
        seen, tree, shift = camera.deproject(pixels), cKDTree(model), np.zeros(3)
        prior = np.eye(3) / self.cfg["prior_shift"] ** 2
        for gate in self.hands["gates"]:
            distance, index = tree.query(seen + shift, distance_upper_bound=gate)
            near = np.isfinite(distance)
            near[near] = own[index[near]]   # pixels nearest the forearm or the other arm are theirs, not the hand's
            if near.sum() < self.cfg["min_points"]:
                return None
            n = normals[index[near]]
            along = np.einsum("ij,ij->i", n, model[index[near]] - seen[near] - shift)
            weight = min(1.0, self.hands["independent_points"] / len(n)) / self.hands["noise"] ** 2
            information = weight * n.T @ n
            shift = shift + np.linalg.solve(information + prior, weight * n.T @ along - prior @ shift)
        rms = float(np.sqrt(np.mean(along ** 2)))
        spread = float(np.sqrt(np.diag(np.linalg.inv(information + prior))).max())
        coverage, inliers = self._support(depth, camera.moved(np.eye(3), shift), tree, model, own, seen + shift)
        return ShiftFit(shift, rms, coverage, inliers, int(near.sum()), spread)

    def _support(self, depth: np.ndarray, moved: Camera, tree: cKDTree, model: np.ndarray, own: np.ndarray,
                 seen: np.ndarray) -> tuple[float, float]:
        """Coverage of the hand model by matched pixels (of its points whose pixel returned depth) and share of the pixels around it on it."""
        first, last = self.hands["gates"][0], self.hands["gates"][-1]
        distance, index = tree.query(seen, distance_upper_bound=first)
        around = np.isfinite(distance)
        around[around] = own[index[around]]
        on_model = distance[around] <= last
        hand = model[own]
        matched = seen[around][on_model]
        covered = np.isfinite(cKDTree(matched).query(hand, distance_upper_bound=last)[0]) if len(matched) else np.zeros(len(hand), bool)
        known = covered | HandReference._returned(depth, moved, hand)
        return float(covered.sum() / max(known.sum(), 1)), float(on_model.mean()) if len(on_model) else 0.0
