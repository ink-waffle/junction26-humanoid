"""Top-face outlines of the three object shapes and their fit to what a depth image shows of one top face."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene, shape_polygon

from halal.perception import SHAPES
from halal.perception.geometry import Camera, PlaneFit

PERIOD_DEG = {"cube": 90, "triangle": 120, "cylinder": 1}
HEADING = {"cube": 0.0, "triangle": np.pi / 2, "cylinder": 0.0}   # reported yaw minus the `shape_polygon` rotation
OVERSAMPLE = 4
COARSE = 2


@dataclass(frozen=True)
class OutlineFit:
    """Best placement of one shape's outline on a top-face view; areas are in view cells."""
    shape: str
    centre: np.ndarray      # top-face centre, pelvis frame (3,)
    yaw: float | None
    area: float             # of the whole outline
    score: float            # seen-as-top minus seen-as-empty area inside the outline
    seen: float             # outline area the depth image shows as top face
    worst_yaw_score: float  # score of the least fitting yaw: equal to `score` when the view does not fix the yaw
    cells: np.ndarray       # view cells (flat indices) that are top face and inside the outline

    @property
    def visible(self) -> float:
        """Fraction of the outline seen as top face."""
        return self.seen / self.area

    @property
    def yaw_contrast(self) -> float:
        """Outline fraction the fit loses at the worst yaw."""
        return (self.score - self.worst_yaw_score) / self.area


@dataclass(frozen=True)
class TopFace:
    """The outline fits of every shape to one top-face view, best first."""
    fits: tuple[OutlineFit, ...]
    pixels: np.ndarray      # (N, 2) image u, v of the best fit's cells

    @property
    def best(self) -> OutlineFit:
        return self.fits[0]

    @property
    def margin(self) -> float:
        """Lead of the best shape over the runner-up as a fraction of the best outline's area."""
        return (self.fits[0].score - self.fits[1].score) / self.fits[0].area


def render_outline(shape: str, apothem: float, fillet: float, yaw: float, cell: float, size: int) -> np.ndarray:
    """Soft mask (size, size) of the rounded outline around the grid centre; rows run along pelvis x, columns along y."""
    fine = cell / OVERSAMPLE
    n = size * OVERSAMPLE
    corners = shape_polygon(shape, apothem - fillet, yaw - HEADING[shape]) / fine + (n - 1) / 2
    core = np.zeros((n, n), np.uint8)
    cv2.fillPoly(core, [np.round(corners[:, ::-1] * 16).astype(np.int32)], 255, shift=4)
    rounded = cv2.distanceTransform(255 - core, cv2.DIST_L2, 5) <= fillet / fine
    return cv2.resize(rounded.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def wrap_yaw(shape: str, degrees: int) -> float | None:
    """Template angle -> yaw in rad within half a symmetry period of zero; None for the cylinder."""
    period = PERIOD_DEG[shape]
    return None if shape == "cylinder" else float(np.radians((degrees + period / 2) % period - period / 2))


class OutlineFitter:
    """Correlates every shape at every yaw with a metric top-plane view: +1 top face seen, -1 seen empty, 0 unknown."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.cell = cfg["cell"]
        self.height = scene.object_height()
        self.size = cfg["outline_cells"]
        self.grid = cfg["view_cells"]
        self.fillet = scene.cfg["objects"]["fillet"]
        self.masks = {s: np.stack([render_outline(s, scene.object_dims(s)["apothem"], self.fillet, np.radians(a), self.cell, self.size)
                                   for a in range(PERIOD_DEG[s])]) for s in SHAPES}
        half = (self.size // COARSE, self.size // COARSE)
        self.coarse = {s: np.stack([cv2.resize(m, half, interpolation=cv2.INTER_AREA) for m in self.masks[s]]) for s in SHAPES}
        self.area = {s: float(self.masks[s][0].sum()) for s in SHAPES}
        offsets = (np.arange(self.grid) - (self.grid - 1) / 2) * self.cell
        self.offsets = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)

    def view(self, depth: np.ndarray, camera: Camera, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
        """Top-plane evidence grid around `origin` (a point of the top plane): +1 top face, -1 empty, 0 hidden or unknown."""
        labels = camera.surface_labels(depth, origin + self.offsets @ axes, self.cfg["depth_tolerance"])
        return labels.reshape(self.grid, self.grid).astype(np.float32)

    def fit(self, depth: np.ndarray, camera: Camera, origin: np.ndarray, counter: PlaneFit) -> TopFace:
        """Fit all shapes to the top-face view around a rough centre on the top plane."""
        axes = counter.axes()
        view = self.view(depth, camera, origin, axes)
        small = cv2.resize(view, (self.grid // COARSE, self.grid // COARSE), interpolation=cv2.INTER_AREA)
        start = origin - self._silhouette_shift(camera, origin, counter.normal)
        fits = sorted((self._fit_shape(s, view, small, start, axes) for s in SHAPES), key=lambda f: -f.score)
        pixels = camera.project(origin + self.offsets[fits[0].cells] @ axes)[:, :2]
        return TopFace(tuple(fits), np.round(pixels).astype(int))

    def _silhouette_shift(self, camera: Camera, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """How far the seen top face sits towards the camera: front-face rays within tolerance, far fillet seen past."""
        ray = origin - camera.T[:3, 3]
        away = ray - (ray @ normal) * normal
        slope = np.arctan2(-ray @ normal, np.linalg.norm(away))
        near = self.cfg["depth_tolerance"] * np.cos(slope)
        far = self.fillet * ((1 - np.sin(slope)) + (1 - np.cos(slope)) / np.tan(slope))
        return -0.5 * (near + far) * away / np.linalg.norm(away)

    def _fit_shape(self, shape: str, view: np.ndarray, small: np.ndarray, start: np.ndarray, axes: np.ndarray) -> OutlineFit:
        step = min(self.cfg["coarse_yaw_deg"], PERIOD_DEG[shape])
        coarse = [self._peak(small, self.coarse[shape][a]) for a in range(0, PERIOD_DEG[shape], step)]
        k = int(np.argmax([score for score, _ in coarse]))
        angles = [(k * step + d) % PERIOD_DEG[shape] for d in range(1 - step, step)]
        fine = [self._refine(view, self.masks[shape][a], coarse[k][1]) for a in angles]
        j = int(np.argmax([score for score, _ in fine]))
        score, (row, col) = fine[j]
        placed = np.zeros_like(view)
        placed[row:row + self.size, col:col + self.size] = self.masks[shape][angles[j]]
        top = placed * (view > 0)
        worst = COARSE ** 2 * min(score for score, _ in coarse)
        shift = (np.array([row, col]) - (self.grid - self.size) / 2) * self.cell
        return OutlineFit(shape, start + shift @ axes, wrap_yaw(shape, angles[j]), self.area[shape], score, float(top.sum()),
                          worst, np.flatnonzero(top > self.cfg["support_coverage"]))

    @staticmethod
    def _peak(view: np.ndarray, mask: np.ndarray) -> tuple[float, tuple[int, int]]:
        """Highest correlation of the mask over all placements inside the view, with its top-left cell."""
        _, score, _, (col, row) = cv2.minMaxLoc(cv2.matchTemplate(view, mask, cv2.TM_CCORR))
        return float(score), (row, col)

    def _refine(self, view: np.ndarray, mask: np.ndarray, coarse_cell: tuple[int, int]) -> tuple[float, tuple[int, int]]:
        """Best placement at full resolution within a few cells of a half-resolution peak."""
        reach, limit = self.cfg["refine_cells"], self.grid - self.size
        row0, col0 = (int(np.clip(COARSE * c - reach, 0, limit)) for c in coarse_cell)
        row1, col1 = (int(np.clip(COARSE * c + reach, 0, limit)) for c in coarse_cell)
        score, (row, col) = self._peak(view[row0:row1 + self.size, col0:col1 + self.size], mask)
        return score, (row0 + row, col0 + col)
