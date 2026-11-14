"""Edge length of the foam cubes of drop 2 from RGB alone: the 21 cm centre box is the ruler that places the camera.

    python -m halal.eval.real.drop2_size   -> camera over the table per view, cube edges in mm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares

from halal.eval.real import colours, dataset, drop2_colours, drop2_frames
from halal.perception.geometry import Camera

BOX = 0.21
READING_PX = 3.0
SQUARE = 0.15       # edges of a cube's top face differ by less than this share of their mean
# Centre box in the first head frame, read by eye: rim back left, back right, front left, front right, foot left, foot right.
BOX_OUTLINE_PX: dict[int, tuple[tuple[int, int], ...]] = {
    38: ((352, 2), (556, 5), (333, 138), (570, 165), (336, 196), (562, 212)),
    17: ((232, 22), (424, 10), (235, 165), (458, 150), (238, 206), (450, 206)),
}
VIEW_EPISODES = {38: range(36, 48), 17: range(16, 19)}   # episodes that share the camera view of the labelled frame
PITCH_STARTS = (35.0, 50.0, 65.0)


def rotation(pitch: float, roll: float) -> np.ndarray:
    """Columns: the camera's right, down and forward axes in a table frame (x ahead, y left, z up), looking down by `pitch`."""
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    turn = cv2.Rodrigues(forward * roll)[0]
    return np.stack([turn @ right, turn @ np.cross(forward, right), forward], axis=1)


def project(points: np.ndarray, K: np.ndarray, pitch: float, roll: float, height: float) -> np.ndarray:
    """Pixels (N, 2) of table-frame points (N, 3) for a camera `height` above the table origin."""
    seen = (points - [0.0, 0.0, height]) @ rotation(pitch, roll)
    return (seen[:, :2] / seen[:, 2:]) * [K[0, 0], K[1, 1]] + K[:2, 2]


def on_plane(pixels: np.ndarray, K: np.ndarray, pitch: float, roll: float, height: float, level: float) -> np.ndarray:
    """Table-frame xy (N, 2) of the points at height `level` above the table that the pixels (N, 2) show."""
    rays = np.c_[(pixels - K[:2, 2]) / [K[0, 0], K[1, 1]], np.ones(len(pixels))] @ rotation(pitch, roll).T
    return (rays * ((level - height) / rays[:, 2:]))[:, :2]


def box_model(x: float, y: float, yaw: float, wall: float) -> np.ndarray:
    """Outline corners (6, 3) of a box seen from the front: rim back left, back right, front left, front right, foot left, right."""
    c, s = np.cos(yaw), np.sin(yaw)
    corner = lambda ahead, left, z: [x + c * ahead * BOX / 2 - s * left * BOX / 2, y + s * ahead * BOX / 2 + c * left * BOX / 2, z]
    return np.array([corner(1, 1, wall), corner(1, -1, wall), corner(-1, 1, wall), corner(-1, -1, wall), corner(-1, 1, 0.0), corner(-1, -1, 0.0)])


def fit_camera(outline: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pitch, roll, camera height, and the box's x, y, yaw and wall height that reproduce its six outline corners."""
    residual = lambda p: (project(box_model(*p[3:]), K, *p[:3]) - outline).ravel()
    fits = [least_squares(residual, [np.radians(pitch), 0.0, 0.55, 0.3, 0.0, 0.0, 0.07]) for pitch in PITCH_STARTS]
    return min(fits, key=lambda fit: fit.cost).x


def top_quad(rgb: np.ndarray, blob: drop2_colours.Blob) -> np.ndarray | None:
    """Image corners (4, 2) of the lit top face of a cube blob; None when the lit region is no quadrilateral."""
    value = np.where(blob.mask, rgb.max(axis=-1), 0).astype(np.uint8)
    threshold, _ = cv2.threshold(value[blob.mask][:, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = cv2.morphologyEx((value > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quads = [c for eps in np.arange(0.01, 0.12, 0.005) if len(c := cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)) == 4]
    return quads[0][:, 0].astype(float) if quads else None


def edges(quad: np.ndarray, K: np.ndarray, camera: np.ndarray, level: float) -> np.ndarray:
    """The four edge lengths (m) of an image quadrilateral taken to lie `level` above the table."""
    corners = on_plane(quad, K, *camera, level)
    return np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)


def cube_size(quad: np.ndarray, K: np.ndarray, camera: np.ndarray) -> float:
    """Edge of a cube (m) from its top face: the height at which the face, placed there, is as wide as it is high."""
    return float(brentq(lambda level: edges(quad, K, camera, level).mean() - level, 0.0, 0.9 * camera[2]))


def camera_spread(outline: np.ndarray, K: np.ndarray, draws: int = 60) -> list[np.ndarray]:
    """Camera fits to the outline with every corner moved by the reading error: how far the ruler can be off."""
    rng = np.random.default_rng(0)
    return [fit_camera(outline + rng.normal(0.0, READING_PX, outline.shape), K)[:3] for _ in range(draws)]


def cubes(rgb: np.ndarray, K: np.ndarray, camera: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Colour and top-face quadrilateral of every single blob whose lit face is a square at its own height: the cubes."""
    blobs, _ = drop2_colours.segment(rgb)
    quads = [(name, top_quad(rgb, blob)) for name, group in blobs.items() for blob in group if blob.single]
    square = lambda quad: np.ptp(edges(quad, K, camera, size := cube_size(quad, K, camera))) < SQUARE * size
    return [(name, quad) for name, quad in quads if quad is not None and square(quad)]


def measure(view: int, frames: dict[int, np.ndarray], K: np.ndarray) -> list[float]:
    """Print the camera fit of one view and the cube sizes in its episodes; returns the sizes (m)."""
    outline = np.array(BOX_OUTLINE_PX[view], float)
    fit = fit_camera(outline, K)
    error = np.abs(project(box_model(*fit[3:]), K, *fit[:3]) - outline).max()
    print(f"view of ep{view:02d}: optical axis {np.degrees(fit[0]):.1f} deg below the table plane, roll {np.degrees(fit[1]):.1f} deg, camera "
          f"{fit[2] * 1e3:.0f} mm above the table, box wall {fit[6] * 1e3:.0f} mm, worst corner {error:.1f} px")
    spread, sizes = camera_spread(outline, K), []
    for episode in VIEW_EPISODES[view]:
        for name, quad in cubes(frames[episode], K, fit[:3]):
            sizes.append(cube_size(quad, K, fit[:3]))
            low, high = np.percentile([cube_size(quad, K, camera) for camera in spread], [16, 84]) * 1e3
            print(f"   ep{episode:02d} {name:7s} cube edge {sizes[-1] * 1e3:5.1f} mm   (box corners read {READING_PX:.0f} px off: {low:.1f}..{high:.1f})")
    return sizes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=drop2_frames.ROOT)
    args = ap.parse_args(argv)
    frames = colours.first_frames(dataset.load(args.root))
    sizes = np.array([size for view in BOX_OUTLINE_PX for size in measure(view, frames, Camera.head().K)]) * 1e3
    print(f"{len(sizes)} cube top faces: edge median {np.median(sizes):.1f} mm, p16..p84 {np.percentile(sizes, 16):.1f}..{np.percentile(sizes, 84):.1f}, "
          f"min {sizes.min():.1f}, max {sizes.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
