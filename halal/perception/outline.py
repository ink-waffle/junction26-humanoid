"""The black hands in the head colour image against their model posed by FK: how far a camera pose puts them, in pixels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from halal.perception.geometry import Camera
from halal.perception.hands import LINKS, HandReference

SIDES = ("left", "right")


@dataclass(frozen=True)
class Outline:
    """How the visible black links of one hand lie on the dark pixels: as projected, and after the image shift that fits best."""
    points: int                 # visible model points of the black links
    on_dark: float              # share of them on dark pixels as projected
    residual_px: float          # their mean distance to the nearest dark pixel (clipped)
    shift: tuple[int, int]      # image shift (u, v) that puts them nearest to dark pixels
    shifted_on_dark: float
    shifted_px: float


class HandOutline:
    """Per hand, the image shift that best lays the black links of the FK-posed model onto the dark pixels of the head image."""

    def __init__(self, cfg: Mapping[str, Any], hands: HandReference) -> None:
        self.cfg, self.hands = cfg, hands
        names = np.array(LINKS)[hands.surface.link]
        self.black = np.array([name.endswith(tuple(cfg["black_parts"])) for name in names])
        self.side = np.array([name.split("_", 1)[0] for name in names])

    def fit(self, rgb: np.ndarray, camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray) -> dict[str, Outline]:
        """Every hand with enough visible black model points, by side."""
        points, normals = self.hands.surface.posed(arm_q, hand_q, True)
        front = self.hands.facing(points, normals, camera, rgb.shape[:2])
        uv = np.round(camera.project(points[front])[:, :2]).astype(int)
        away = self.distance(rgb)
        own = {side: self.black[front] & (self.side[front] == side) for side in SIDES}
        return {side: self._shifted(away, uv[mask]) for side, mask in own.items() if mask.sum() >= self.cfg["min_points"]}

    def distance(self, rgb: np.ndarray) -> np.ndarray:
        """Distance (px) of every pixel to the nearest dark one, clipped: 0 on dark pixels."""
        dark = (rgb.max(axis=-1) < self.cfg["dark_value"]).astype(np.uint8)
        return np.minimum(cv2.distanceTransform(1 - dark, cv2.DIST_L2, 5), self.cfg["clip_px"])

    def _shifted(self, away: np.ndarray, uv: np.ndarray) -> Outline:
        """The outline as projected and at its best shift: a coarse search over the reach, then every pixel around the best."""
        reach, step = self.cfg["reach_px"], self.cfg["step_px"]
        coarse = [(du, dv) for du in range(-reach, reach + 1, step) for dv in range(-reach, reach + 1, step)]
        best = min(coarse, key=lambda shift: self._distances(away, uv, shift).mean())
        fine = [(best[0] + du, best[1] + dv) for du in range(1 - step, step) for dv in range(1 - step, step)]
        best = min(fine, key=lambda shift: self._distances(away, uv, shift).mean())
        before, after = self._distances(away, uv, (0, 0)), self._distances(away, uv, best)
        return Outline(len(uv), float(np.mean(before == 0)), float(before.mean()), best, float(np.mean(after == 0)), float(after.mean()))

    @staticmethod
    def _distances(away: np.ndarray, uv: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
        """Distance to the nearest dark pixel of every model point moved by `shift` (points leaving the image are held at its edge)."""
        u = np.clip(uv[:, 0] + shift[0], 0, away.shape[1] - 1)
        v = np.clip(uv[:, 1] + shift[1], 0, away.shape[0] - 1)
        return away[v, u]
