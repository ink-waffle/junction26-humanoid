"""Object colour from its most saturated pixels: hue basins of the three colour families, white-balanced on the counter."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

SQRT3 = np.sqrt(3.0)


def hue_angles(rgb: np.ndarray) -> np.ndarray:
    """Hue angle (deg, 0 = red, 120 = green) of RGB triples (..., 3)."""
    r, g, b = np.moveaxis(np.asarray(rgb, float), -1, 0)
    return np.degrees(np.arctan2(SQRT3 * (g - b), 2 * r - g - b)) % 360.0


def chromas(rgb: np.ndarray) -> np.ndarray:
    """Chroma relative to brightness, (max - min) / max, of RGB triples (..., 3) on the 0..255 scale."""
    rgb = np.asarray(rgb, float)
    return (rgb.max(axis=-1) - rgb.min(axis=-1)) / np.maximum(rgb.max(axis=-1), 1.0)


def outside(hue: float, basin: tuple[float, float]) -> float:
    """Degrees by which a hue lies outside the basin running from basin[0] up to basin[1] (0 inside)."""
    lo, hi = basin
    if (hue - lo) % 360.0 <= (hi - lo) % 360.0:
        return 0.0
    return float(min((lo - hue) % 360.0, (hue - hi) % 360.0))


class ColorClassifier:
    """Names a colour sample by the hue basin (colour family) it falls into; hues between basins are nobody's."""

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
        """Per RGB sample (N, 3): index of the basin its hue is in, -1 when it is in none or too pale or dark to say."""
        balanced = samples * gains
        hue = hue_angles(balanced)
        vivid = (chromas(balanced) >= self.cfg["chroma_range"][0]) & (balanced.max(axis=-1) >= self.cfg["min_value"])
        classes = np.full(len(samples), -1, np.int8)
        for index, (lo, hi) in enumerate(self.cfg["basins"].values()):
            classes[vivid & ((hue - lo) % 360.0 <= (hi - lo) % 360.0)] = index
        return classes

    def vivid(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """White-balanced median colour of the most saturated share of RGB samples (N, 3): the colour that gets named."""
        balanced = samples * gains
        spread = chromas(balanced)
        return np.median(balanced[spread >= np.quantile(spread, 1.0 - self.cfg["vivid_fraction"])], axis=0)

    def name(self, colour: np.ndarray) -> tuple[str, float]:
        """Colour family and confidence in [0, 1] of one white-balanced RGB colour."""
        hue, chroma = float(hue_angles(colour)), float(chromas(colour))
        miss, name = min((outside(hue, basin), name) for name, basin in self.cfg["basins"].items())
        low, full = self.cfg["chroma_range"]
        by_chroma = np.clip((chroma - low) / (full - low), 0.0, 1.0) * (colour.max() >= self.cfg["min_value"])
        return name, float(np.clip(1.0 - miss / self.cfg["taper_deg"], 0.0, 1.0) * by_chroma)

    def own(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """The samples (N, 3) of the family most of them are in, when that is a fair share: a touching neighbour's edge is left out."""
        classes = self.pixel_classes(samples, gains)
        counts = np.bincount(classes[classes >= 0], minlength=len(self.cfg["basins"]))
        return samples[classes == counts.argmax()] if counts.max() >= self.cfg["own_share"] * len(samples) else samples

    def classify(self, samples: np.ndarray, gains: np.ndarray) -> tuple[str, float]:
        """Colour family and confidence in [0, 1] for RGB samples (N, 3) of one object."""
        return self.name(self.vivid(self.own(samples, gains), gains))
