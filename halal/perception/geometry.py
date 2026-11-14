"""Head camera geometry: the pinhole model between pixels and the pelvis frame, and trimmed plane and line fits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from aaltonomous.interface.calibration import Calibration

MM = 1e-3


@lru_cache(maxsize=4)
def _pixel_rays(intrinsics: tuple[float, ...], shape: tuple[int, int]) -> np.ndarray:
    """Optical-frame direction of every pixel scaled to unit depth, (H, W, 3)."""
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[:shape[0], :shape[1]]
    return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(shape)], axis=-1)


@dataclass(frozen=True, eq=False)
class Camera:
    """Pinhole camera: intrinsics K and the pose `T_pelvis_optical` of its optical frame."""
    K: np.ndarray
    T: np.ndarray

    @classmethod
    def head(cls) -> Camera:
        """The head camera as calibration.yaml describes it."""
        cam = Calibration.load().cams["head"]
        return cls(cam.K, cam.T_pelvis_optical)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pelvis-frame points (N, 3) -> rows of (u, v, optical depth)."""
        p = (points - self.T[:3, 3]) @ self.T[:3, :3]
        return np.c_[self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2], self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2], p[:, 2]]

    def deproject(self, uvz: np.ndarray) -> np.ndarray:
        """Rows of (u, v, optical depth) -> pelvis-frame points."""
        x = (uvz[:, 0] - self.K[0, 2]) / self.K[0, 0] * uvz[:, 2]
        y = (uvz[:, 1] - self.K[1, 2]) / self.K[1, 1] * uvz[:, 2]
        return np.c_[x, y, uvz[:, 2]] @ self.T[:3, :3].T + self.T[:3, 3]

    def cloud(self, depth_mm: np.ndarray, step: int = 1) -> np.ndarray:
        """Depth image -> pelvis-frame points of every `step`-th pixel (h, w, 3), NaN where the depth is invalid."""
        rays = _pixel_rays((self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]), depth_mm.shape)[::step, ::step]
        sampled = depth_mm[::step, ::step]
        z = np.where(sampled > 0, sampled * MM, np.nan)
        return (rays * z[..., None]) @ self.T[:3, :3].T + self.T[:3, 3]

    def surface_labels(self, depth_mm: np.ndarray, points: np.ndarray, tolerance: float) -> np.ndarray:
        """Per point of a hypothetical surface: +1 the depth image shows it, -1 it sees past it, 0 hidden or unknown."""
        uvz = self.project(points)
        u, v = np.round(uvz[:, 0]).astype(int), np.round(uvz[:, 1]).astype(int)
        inside = (u >= 0) & (u < depth_mm.shape[1]) & (v >= 0) & (v < depth_mm.shape[0]) & (uvz[:, 2] > 0)
        seen = np.zeros(len(points))
        seen[inside] = depth_mm[v[inside], u[inside]] * MM
        gap = np.where(seen > 0, seen - uvz[:, 2], -np.inf)
        return np.where(np.abs(gap) <= tolerance, 1.0, np.where(gap > tolerance, -1.0, 0.0))

    def moved(self, rotation: np.ndarray, shift: np.ndarray) -> Camera:
        """The camera turned about its own optical centre by a pelvis-frame rotation, then shifted."""
        T = self.T.copy()
        T[:3, :3] = rotation @ self.T[:3, :3]
        T[:3, 3] = self.T[:3, 3] + shift
        return replace(self, T=T)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation that turns unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + c)


def yaw_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class PlaneFit:
    """Plane `normal . p = offset` with an upward unit normal and the support it was fitted on."""
    normal: np.ndarray
    offset: float
    inliers: int
    rms: float


def fit_plane(points: np.ndarray, tolerances: tuple[float, ...]) -> PlaneFit:
    """Least-squares plane through near-horizontal points, refitted on the inliers of each successive tolerance."""
    normal, offset = np.array([0.0, 0.0, 1.0]), float(np.median(points[:, 2]))
    for tolerance in tolerances:
        keep = points[np.abs(points @ normal - offset) < tolerance]
        centre = keep.mean(axis=0)
        normal = np.linalg.svd(keep - centre, full_matrices=False)[2][2]
        normal = normal if normal[2] > 0 else -normal
        offset = float(centre @ normal)
    residual = keep @ normal - offset
    return PlaneFit(normal, offset, len(keep), float(np.sqrt(np.mean(residual ** 2))))


@dataclass(frozen=True)
class LineFit:
    """Line `value = intercept + slope * along` and the support it was fitted on."""
    intercept: float
    slope: float
    inliers: int
    rms: float


def fit_line(along: np.ndarray, value: np.ndarray, tolerances: tuple[float, ...]) -> LineFit:
    """Least-squares line refitted on the inliers of each successive tolerance, starting from the median value."""
    intercept, slope = float(np.median(value)), 0.0
    for tolerance in tolerances:
        keep = np.abs(value - intercept - slope * along) < tolerance
        slope, intercept = (float(c) for c in np.polyfit(along[keep], value[keep], 1))
    residual = (value - intercept - slope * along)[keep]
    return LineFit(intercept, slope, int(keep.sum()), float(np.sqrt(np.mean(residual ** 2))))
