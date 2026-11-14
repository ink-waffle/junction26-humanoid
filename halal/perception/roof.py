"""A triangular prism lying on a rectangular side, from depth: one or two sloped faces that meet in a level ridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Face:
    """A plane `normal . p = offset` (unit normal, pointing up) and the points of a blob that lie on it."""
    normal: np.ndarray
    offset: float
    on: np.ndarray          # (N,) bool


@dataclass(frozen=True)
class Roof:
    """A lying prism as its sloped faces show it."""
    centre: np.ndarray      # middle of the ridge, pelvis frame (3,)
    yaw: float              # ridge direction about pelvis z, rad, in (-pi/2, pi/2]
    height: float           # ridge above the counter, m
    support: float          # share of the blob's points on the sloped faces
    faces: int              # sloped faces seen: with 2 the ridge is where they meet, with 1 the top of the face


def sloped_face(points: np.ndarray, up: np.ndarray, cfg: Mapping[str, Any], facing: np.ndarray | None = None) -> Face | None:
    """The plane with most of the points (N, 3) on it among those tilted like a prism's side; `facing`: only planes leaning against it."""
    if len(points) < cfg["min_points"]:
        return None
    picks = np.random.default_rng(0).integers(len(points), size=(cfg["trials"], 3))
    a, b, c = (points[picks[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1)
    normals = normals[length > 0] / length[length > 0, None]
    normals *= np.sign(normals @ up)[:, None]
    tilt = np.degrees(np.arccos(np.clip(normals @ up, -1.0, 1.0)))
    ok = (tilt > cfg["tilt_deg"][0]) & (tilt < cfg["tilt_deg"][1])
    if facing is not None:
        ok &= (normals - np.outer(normals @ up, up)) @ facing < 0.0
    if not ok.any():
        return None
    normals, offsets = normals[ok], np.einsum("ij,ij->i", normals, a[length > 0])[ok]
    on = np.abs(points @ normals.T - offsets) < cfg["tolerance"]
    best = on[:, int(on.sum(axis=0).argmax())]
    if best.sum() < cfg["min_points"]:
        return None
    centre = points[best].mean(axis=0)
    normal = np.linalg.svd(points[best] - centre, full_matrices=False)[2][2]
    normal = normal if normal @ up > 0 else -normal
    off = np.abs(points @ normal - centre @ normal)
    on = off < cfg["tolerance"]
    flat = on.sum() >= cfg["min_flat"] * (off < cfg["flat_band"]).sum()
    sloped = cfg["tilt_deg"][0] < np.degrees(np.arccos(min(1.0, normal @ up))) < cfg["tilt_deg"][1]   # the refit can leave the range
    return Face(normal, float(centre @ normal), on) if flat and sloped else None


def meet(face: Face, level: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray]:
    """A point and the unit direction of the line in which a face meets the plane `level . p = target`."""
    along = np.cross(face.normal, level)
    along /= np.linalg.norm(along)
    return np.linalg.solve(np.stack([face.normal, level, along]), [face.offset, target, 0.0]), along


def fit_roof(points: np.ndarray, heights: np.ndarray, up: np.ndarray, base: float, cfg: Mapping[str, Any]) -> Roof | None:
    """The lying prism in a blob's points (N, 3) with their heights above the counter (whose plane is `up . p = base`); None if none."""
    first = sloped_face(points, up, cfg)
    if first is None:
        return None
    lean = first.normal - (first.normal @ up) * up
    rest = sloped_face(points[~first.on], up, cfg, facing=lean)
    second = None if rest is None else Face(rest.normal, rest.offset, np.abs(points @ rest.normal - rest.offset) < cfg["tolerance"])
    on = first.on if second is None else first.on | second.on
    if second is None:
        point, along = meet(first, up, base + float(np.percentile(heights[first.on], cfg["top_percentile"])))
    else:
        point, along = meet(first, second.normal, second.offset)
    reach = (points[on] - point) @ along
    low, high = np.percentile(reach, [2.0, 98.0])
    centre = point + along * (low + high) / 2
    height = float(centre @ up - base)
    slope = np.ptp((points[first.on] - point) @ np.cross(along, first.normal))
    fits = (cfg["ridge_height"][0] < height < cfg["ridge_height"][1] and cfg["length"][0] < high - low < cfg["length"][1]
            and slope > cfg["min_slope"] and on.mean() >= cfg["min_support"])
    if not fits:
        return None
    yaw = float(np.arctan2(along[1], along[0]))
    return Roof(centre, (yaw + np.pi / 2) % np.pi - np.pi / 2, height, float(on.mean()), 1 if second is None else 2)
