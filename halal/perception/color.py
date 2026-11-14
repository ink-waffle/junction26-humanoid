"""Object colour from its most saturated pixels: hue basins of the three colour families, white-balanced on the counter."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
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


def turned(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """RGB image or samples (uint8, (..., 3)) with every hue turned by `degrees`."""
    if not rgb.size:
        return rgb
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + degrees) % 360.0
    return np.round(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0).astype(np.uint8).reshape(rgb.shape)


def apart(a: float, b: float) -> float:
    """Degrees between two hues, the short way round."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


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

    def white_balance(self, counter_rgb: np.ndarray) -> np.ndarray:
        """Per-channel gains that make a grey counter neutral; ones without counter pixels or on a coloured counter (wood)."""
        if not len(counter_rgb):
            return np.ones(3)
        grey = np.median(counter_rgb, axis=0).astype(float) + 1e-6
        return grey.mean() / grey if self.grey(grey) else np.ones(3)

    def grey(self, colour: np.ndarray) -> bool:
        """True for a counter colour (median RGB) pale enough to be the white reference: the sim mat or drop 2's table, not wood."""
        return float(chromas(colour)) <= self.cfg["grey_chroma"]

    def dark_share(self, samples: np.ndarray, counter_rgb: np.ndarray) -> float:
        """Share of an object's RGB samples (N, 3) that are dark against the bare counter's brightness: 1 for a black object."""
        if not len(samples) or not len(counter_rgb):
            return 0.0
        limit = self.cfg["black"]["value_share"] * float(np.median(counter_rgb.max(axis=-1)))
        return float(np.mean(samples.max(axis=-1) < limit))

    def black(self, share: float) -> float:
        """Confidence in [0, 1] that an object with this share of dark pixels is black."""
        low, full = self.cfg["black"]["share_range"]
        return float(np.clip((share - low) / (full - low), 0.0, 1.0))

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

    def own(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """The samples (N, 3) of the family most of them are in, when that is a fair share: a touching neighbour's edge is left out."""
        classes = self.pixel_classes(samples, gains)
        counts = np.bincount(classes[classes >= 0], minlength=len(self.cfg["basins"]))
        return samples[classes == counts.argmax()] if counts.max() >= self.cfg["own_share"] * len(samples) else samples

    def sample(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """The white-balanced colour that stands for RGB samples (N, 3) of one object."""
        return self.vivid(self.own(samples, gains), gains)

    def boundary(self) -> float:
        """Hue (deg) midway between the red and the yellow basin as configured."""
        red, yellow = self.cfg["basins"]["red"], self.cfg["basins"]["yellow"]
        return (red[1] + (yellow[0] - red[1]) % 360.0 / 2) % 360.0

    def split(self, colours: Sequence[np.ndarray]) -> float | None:
        """Hue (deg) where red ends and yellow begins, from the one clear gap between a look's object colours that has a real yellow above it."""
        c, (start, red_end), (yellow_start, end) = self.cfg["warm_split"], self.cfg["basins"]["red"], self.cfg["basins"]["yellow"]
        up = lambda hue: (hue - start) % 360.0
        vivid = [x for x in colours if chromas(x) >= self.cfg["chroma_range"][0] and x.max() >= self.cfg["min_value"]]
        warm = sorted(u for u in (up(float(hue_angles(x))) for x in vivid) if u <= up(end))
        least = lambda lo: max(lo - up(red_end), -c["max_shift"])       # smallest turn of all hues that leaves `lo` in the red basin
        most = lambda hi: min(hi - up(yellow_start), c["max_shift"])    # largest that leaves `hi` in the yellow basin
        gaps = [(lo, hi) for lo, hi in zip(warm, warm[1:]) if hi - lo >= c["min_gap"] and least(lo) <= most(hi)]
        if any(least(lo) <= 0.0 <= most(hi) for lo, hi in gaps):
            return self.boundary()
        yellow_above = len(gaps) == 1 and gaps[0][1] >= up(yellow_start)   # two oranges, or tape and an orange, must not move it
        return (sum(gaps[0]) / 2 + start) % 360.0 if yellow_above else None

    def basins(self, split: float | None) -> Mapping[str, tuple[float, float]]:
        """The configured basins, red and yellow meeting around `split` when the episode has one."""
        fixed = self.cfg["basins"]
        if split is None:
            return fixed
        half = (fixed["yellow"][0] - fixed["red"][1]) % 360.0 / 2
        return {**fixed, "red": (fixed["red"][0], (split - half) % 360.0), "yellow": ((split + half) % 360.0, fixed["yellow"][1])}

    def name(self, colour: np.ndarray, split: float | None = None) -> tuple[str, float]:
        """Family and confidence in [0, 1] of one white-balanced colour; `split` None: no hue gap vouches for the red / yellow boundary."""
        hue, chroma = float(hue_angles(colour)), float(chromas(colour))
        miss, name = min((outside(hue, basin), name) for name, basin in self.basins(split).items())
        low, full = self.cfg["chroma_range"]
        by_chroma = np.clip((chroma - low) / (full - low), 0.0, 1.0) * (colour.max() >= self.cfg["min_value"])
        unvouched = 1.0 if split is not None else apart(hue, self.boundary()) / self.cfg["taper_deg"]
        return name, float(np.clip(min(1.0 - miss / self.cfg["taper_deg"], unvouched), 0.0, 1.0) * by_chroma)

    def classify(self, samples: np.ndarray, gains: np.ndarray, split: float | None = None) -> tuple[str, float]:
        """Colour family and confidence in [0, 1] for RGB samples (N, 3) of one object."""
        return self.name(self.sample(samples, gains), split)
