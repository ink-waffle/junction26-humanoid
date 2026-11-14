"""An Intel RealSense D435i depth stream modelled on a rendered depth image aligned to colour (sources: logs/perception.md)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from halal.perception.geometry import MM
from halal.perception.hands import LINKS, HandSurface

LEVELS = ("mild", "typical", "bad")


class ArmPixels:
    """Pixels of a depth image that show chosen links of the robot's arms: FK through the camera that took the frame."""

    def __init__(self) -> None:
        self.surface = HandSurface()
        self.link_names = LINKS

    def mask(self, depth_mm: np.ndarray, camera: Any, arm_q: np.ndarray, hand_q: np.ndarray, parts: Sequence[str],
             reach: float) -> np.ndarray:
        """True where the depth point lies within `reach` of the posed surface of a link whose name ends in one of `parts`."""
        points = self.surface.posed(arm_q, hand_q, True)[0]
        wanted = np.array([bool(parts) and name.endswith(tuple(parts)) for name in self.link_names], bool)
        chosen = points[wanted[self.surface.link]]
        out = np.zeros(depth_mm.shape, bool)
        v, u = np.nonzero(depth_mm > 0)
        if len(chosen) and len(u):
            seen = camera.deproject(np.c_[u, v, depth_mm[v, u] * MM])
            out[v, u] = np.isfinite(cKDTree(chosen).query(seen, distance_upper_bound=reach)[0])
        return out


def coherent(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise field that varies over about `sigma` pixels."""
    field = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return field / field.std()


def patches(shape: tuple[int, int], share: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Random blobs about `sigma` pixels wide that cover `share` of the image."""
    field = coherent(shape, sigma, rng)
    return field < np.quantile(field, share)


def metres(disparity: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    """Depth of a disparity image; infinite where there is none."""
    with np.errstate(divide="ignore"):
        return np.where(disparity > 0, c["depth_focal"] * c["baseline"] / disparity, np.inf)


def matched(disparity: np.ndarray, c: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    """What block matching leaves of a disparity image: edges moved by the window, foreground fattened there, thin parts smoothed."""
    v, u = np.mgrid[:disparity.shape[0], :disparity.shape[1]].astype(np.float32)
    du, dv = (c["edge_jitter_px"] * coherent(disparity.shape, c["window_px"], rng) for _ in range(2))
    moved = cv2.remap(disparity, u + du, v + dv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    kernel = np.ones((2 * c["fatten_px"] + 1,) * 2, np.uint8)
    fat = cv2.dilate(moved, kernel)
    with np.errstate(invalid="ignore"):
        at_edge = metres(cv2.erode(moved, kernel), c) - metres(fat, c) > c["edge_step"]
    return cv2.medianBlur(np.where(at_edge, fat, moved), c["smooth_px"])


def shadowed(depth_m: np.ndarray, focal: float, c: Mapping[str, Any]) -> np.ndarray:
    """Pixels the right imager cannot see: left of every nearer surface, as wide as the disparity step across its edge."""
    shift = np.where(depth_m > 0, focal * c["shadow_baseline"] / np.maximum(depth_m, 1e-3), 0.0)
    landing = np.arange(depth_m.shape[1]) - shift
    nearest = np.minimum.accumulate(landing[:, ::-1], axis=1)[:, ::-1]
    hidden = np.zeros(depth_m.shape, bool)
    hidden[:, :-1] = nearest[:, 1:] < landing[:, :-1] - c["shadow_margin_px"]
    return hidden


def edge_pixels(depth_m: np.ndarray, c: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels at a depth step, with the nearest and farthest depth among their neighbours."""
    kernel = np.ones((3, 3), np.uint8)
    near, far = cv2.erode(depth_m, kernel), cv2.dilate(depth_m, kernel)
    return (far - near > c["edge_step"]) & (near > 0), near, far


def degrade(depth_mm: np.ndarray, black: np.ndarray, dark: np.ndarray, focal: float, cfg: Mapping[str, Any], level: str,
            rng: np.random.Generator) -> np.ndarray:
    """The depth image (uint16 mm, 0 = invalid) as a D435i would deliver it at this severity; `black` and `dark` pixels return poorly."""
    c = {**cfg, **cfg["levels"][level]}
    shape, scale = depth_mm.shape, c["depth_focal"] * c["baseline"]
    disparity = np.where(depth_mm > 0, scale / (np.maximum(depth_mm, 1) * MM), 0.0).astype(np.float32)
    disparity = matched(disparity, c, rng)
    clean = np.where(disparity > 0, scale / np.maximum(disparity, 1e-6), 0.0).astype(np.float32)
    grain = c["correlated_share"] * coherent(shape, c["window_px"], rng) + (1 - c["correlated_share"] ** 2) ** 0.5 * rng.standard_normal(shape)
    noisy = disparity + c["subpixel_rms"] * np.where(black | dark, c["dark_noise_gain"], 1.0) * grain + c["disparity_bias"] * rng.choice([-1.0, 1.0])
    noisy = np.round(noisy * c["disparity_steps"]) / c["disparity_steps"]
    depth = scale / np.maximum(noisy, 1e-6)
    edge, near, far = edge_pixels(clean, c)
    draw = rng.uniform(size=shape)
    flying = edge & (draw < c["fly_rate"])
    depth[flying] = (near + rng.uniform(size=shape) * (far - near))[flying]
    lost = (clean <= 0) | shadowed(clean, focal, c) | (edge & ~flying & (draw < c["fly_rate"] + c["edge_drop"]))
    lost |= black & ~patches(shape, c["black_fill"], c["window_px"], rng)
    lost |= dark & ~black & ~patches(shape, c["dark_fill"], c["window_px"], rng)
    lost |= patches(shape, c["hole_share"], c["hole_px"], rng)
    return np.where(lost, 0, np.clip(np.round(depth / MM), 0, 65535)).astype(np.uint16)
