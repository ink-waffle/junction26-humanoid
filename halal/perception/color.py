"""Object colour from its most saturated pixels: hue against the scene's reference colours, white-balanced on the mat."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aaltonomous.scene.scene import Scene

SQRT3 = np.sqrt(3.0)


def hue_chroma(rgb: np.ndarray) -> tuple[float, float]:
    """Hue angle (deg, 0 = red, 120 = green) and chroma relative to brightness of one RGB triple."""
    r, g, b = (float(c) for c in rgb)
    hue = np.degrees(np.arctan2(SQRT3 * (g - b), 2 * r - g - b)) % 360.0
    return float(hue), float((max(r, g, b) - min(r, g, b)) / max(r, g, b, 1e-6))


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class ColorClassifier:
    """Names a colour sample by the nearest reference hue and says how sure that is."""

    def __init__(self, scene: Scene, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.hues = {name: hue_chroma(np.asarray(rgb, float))[0] for name, rgb in scene.colors.items()}

    @staticmethod
    def white_balance(mat_rgb: np.ndarray) -> np.ndarray:
        """Per-channel gains that make the (grey) mat neutral; ones when there are no mat pixels."""
        if not len(mat_rgb):
            return np.ones(3)
        grey = np.median(mat_rgb, axis=0).astype(float) + 1e-6
        return grey.mean() / grey

    def classify(self, samples: np.ndarray, gains: np.ndarray) -> tuple[str, float]:
        """Colour name and confidence in [0, 1] for RGB samples (N, 3) of one object."""
        balanced = samples * gains
        spread = (balanced.max(axis=1) - balanced.min(axis=1)) / np.maximum(balanced.max(axis=1), 1.0)
        vivid = balanced[spread >= np.quantile(spread, 1.0 - self.cfg["vivid_fraction"])]
        hue, chroma = hue_chroma(np.median(vivid, axis=0))
        gaps = sorted((hue_gap(hue, ref), name) for name, ref in self.hues.items())
        (nearest, name), (second, _) = gaps[0], gaps[1]
        sure, boundary = self.cfg["hue_sure_deg"], (nearest + second) / 2
        by_hue = np.clip((boundary - nearest) / max(boundary - sure, 1e-6), 0.0, 1.0)
        low, full = self.cfg["chroma_range"]
        by_chroma = np.clip((chroma - low) / (full - low), 0.0, 1.0)
        return name, float(by_hue * by_chroma)
