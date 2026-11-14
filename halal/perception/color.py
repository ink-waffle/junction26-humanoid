"""Object colour from its most saturated pixels: hue basins around the three object colours, white-balanced on the counter."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

SQRT3 = np.sqrt(3.0)


def hue_chroma(rgb: np.ndarray) -> tuple[float, float]:
    """Hue angle (deg, 0 = red, 120 = green) and chroma relative to brightness of one RGB triple."""
    r, g, b = (float(c) for c in rgb)
    hue = np.degrees(np.arctan2(SQRT3 * (g - b), 2 * r - g - b)) % 360.0
    return float(hue), float((max(r, g, b) - min(r, g, b)) / max(r, g, b, 1e-6))


def outside(hue: float, basin: tuple[float, float]) -> float:
    """Degrees by which a hue lies outside the basin running from basin[0] up to basin[1] (0 inside)."""
    lo, hi = basin
    if (hue - lo) % 360.0 <= (hi - lo) % 360.0:
        return 0.0
    return float(min((lo - hue) % 360.0, (hue - hi) % 360.0))


class ColorClassifier:
    """Names a colour sample by the hue basin it falls into; hues between basins are nobody's and get no confidence."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg

    @staticmethod
    def white_balance(counter_rgb: np.ndarray) -> np.ndarray:
        """Per-channel gains that make the counter neutral; ones when there are no counter pixels."""
        if not len(counter_rgb):
            return np.ones(3)
        grey = np.median(counter_rgb, axis=0).astype(float) + 1e-6
        return grey.mean() / grey

    def pixel_classes(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Per RGB sample (N, 3): index of the basin its hue is in, -1 when it is in none or too pale to say."""
        r, g, b = (samples * gains).T
        hue = np.degrees(np.arctan2(SQRT3 * (g - b), 2 * r - g - b)) % 360.0
        vivid = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) >= self.cfg["chroma_range"][0] * np.maximum(np.maximum(r, g), b)
        classes = np.full(len(samples), -1, np.int8)
        for index, (lo, hi) in enumerate(self.cfg["basins"].values()):
            classes[vivid & ((hue - lo) % 360.0 <= (hi - lo) % 360.0)] = index
        return classes

    def classify(self, samples: np.ndarray, gains: np.ndarray) -> tuple[str, float]:
        """Colour name and confidence in [0, 1] for RGB samples (N, 3) of one object."""
        balanced = samples * gains
        spread = (balanced.max(axis=1) - balanced.min(axis=1)) / np.maximum(balanced.max(axis=1), 1.0)
        vivid = balanced[spread >= np.quantile(spread, 1.0 - self.cfg["vivid_fraction"])]
        hue, chroma = hue_chroma(np.median(vivid, axis=0))
        miss, name = min((outside(hue, basin), name) for name, basin in self.cfg["basins"].items())
        low, full = self.cfg["chroma_range"]
        by_chroma = np.clip((chroma - low) / (full - low), 0.0, 1.0)
        return name, float(np.clip(1.0 - miss / self.cfg["taper_deg"], 0.0, 1.0) * by_chroma)
