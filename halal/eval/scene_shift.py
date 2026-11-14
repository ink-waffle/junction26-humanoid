"""A recorded sim frame re-rendered with the table moved against the robot, as the real recordings show it (bench only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from aaltonomous.scene.scene import Scene

from halal.eval.depth_sensor import ArmPixels
from halal.perception.geometry import MM
from halal.perception.hands import PARTS

FAR = 1e9


@dataclass(frozen=True)
class Displacement:
    """Rigid motion of the table with everything on it (4x4, pelvis frame) and the further slide of the bin row along its x."""
    table: np.ndarray
    bins_x: float

    @classmethod
    def from_config(cls, scene: Scene, c: Mapping[str, float]) -> Displacement:
        """Pitch about the pelvis y axis through the middle of the object band (far edge up), then a lift."""
        region = scene.layout_kwargs()["region"]
        pivot = scene.mat_to_pelvis([np.mean(region[0]), np.mean(region[1]), 0.0])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("y", -c["pitch_deg"], degrees=True).as_matrix()
        T[:3, 3] = pivot - T[:3, :3] @ pivot + [0.0, 0.0, c["counter_z"]]
        return cls(T, float(c["bins_x"]))

    def moved(self, points: np.ndarray, bins: bool = False) -> np.ndarray:
        """Points (N, 3) of the table, or of the bin row, where the displaced scene has them."""
        return (points + [self.bins_x if bins else 0.0, 0.0, 0.0]) @ self.table[:3, :3].T + self.table[:3, 3]


def bin_footprints(scene: Scene, points: np.ndarray, margin: float) -> np.ndarray:
    """True for points (N, 3) over the outer footprint of a bin."""
    half = np.asarray(scene.cfg["bins"]["outer_size"])[::-1] / 2 + margin
    centres = [scene.mat_to_pelvis([*c, 0.0])[:2] for c in scene.cfg["bins"]["centers"].values()]
    return np.any([np.all(np.abs(points[:, :2] - c) < half, axis=1) for c in centres], axis=0)


def on_table(scene: Scene, points: np.ndarray) -> np.ndarray:
    """True for pelvis-frame points (N, 3) over the table top."""
    origin, c = scene.mat_to_pelvis([0.0, 0.0, 0.0]), scene.cfg["counter"]
    mat = np.c_[points[:, 1] - origin[1], points[:, 0] - origin[0]]
    return np.all(np.abs(mat - np.asarray(c["mat_size"]) / 2) < np.asarray(c["counter_size"]) / 2, axis=1)


def splat(target: tuple[np.ndarray, np.ndarray], uvz: np.ndarray, colours: np.ndarray) -> None:
    """Draw points into (depth m, rgb), each at its nearest pixel, the nearest point winning."""
    depth, rgb = target
    u, v = np.round(uvz[:, 0]).astype(int), np.round(uvz[:, 1]).astype(int)
    ok = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0]) & (uvz[:, 2] > 0)
    order = np.argsort(-uvz[ok, 2])
    u, v, z, colours = u[ok][order], v[ok][order], uvz[ok, 2][order], colours[ok][order]
    nearer = z < depth[v, u]
    depth[v[nearer], u[nearer]], rgb[v[nearer], u[nearer]] = z[nearer], colours[nearer]


def close_cracks(depth: np.ndarray, rgb: np.ndarray) -> None:
    """Fill empty pixels mostly surrounded by drawn ones (resampling cracks) from their neighbourhood's median, twice."""
    for _ in range(2):
        drawn = (depth < FAR).astype(np.uint8)
        crack = (drawn == 0) & (cv2.boxFilter(drawn, -1, (3, 3), normalize=False) >= 5)
        depth[crack] = cv2.medianBlur(np.where(depth < FAR, depth, 0.0).astype(np.float32), 3)[crack]
        rgb[crack] = cv2.medianBlur(rgb, 3)[crack]


def bare_counter(scene: Scene, camera: Any, move: Displacement, shape: tuple[int, int]) -> np.ndarray:
    """Depth (m) at which every pixel's ray meets the displaced table top; FAR where it misses it."""
    v, u = np.mgrid[:shape[0], :shape[1]]
    rays = camera.deproject(np.c_[u.ravel(), v.ravel(), np.ones(u.size)]) - camera.T[:3, 3]
    normal, point = move.table[:3, 2], move.moved(scene.mat_to_pelvis([0.0, 0.0, 0.0])[None])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (point - camera.T[:3, 3]) @ normal / (rays @ normal)
    hits = camera.T[:3, 3] + rays * depth[:, None]
    back = (hits - move.table[:3, 3]) @ move.table[:3, :3]
    return np.where((depth > 0) & on_table(scene, back), depth, FAR).reshape(shape)


def displaced(obs: Mapping[str, Any], camera: Any, scene: Scene, move: Displacement, arms: ArmPixels,
              c: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Head RGB and depth with table and bins displaced; arms, background and invalid pixels stay, uncovered counter is drawn bare."""
    holes, rgb = obs["head_depth"] == 0, obs["head_rgb"]
    depth_mm = np.where(holes, cv2.medianBlur(obs["head_depth"], 5), obs["head_depth"])
    v, u = np.nonzero(depth_mm > 0)
    points = camera.deproject(np.c_[u, v, depth_mm[v, u] * MM])
    height = points[:, 2] - scene.counter_z
    arm = arms.mask(depth_mm, camera, obs["arm_q"], obs["hand_q"], PARTS, c["arm_reach"])[v, u] | (height > c["arm_height"])
    table = ~arm & (height > -c["below"]) & on_table(scene, points)
    bins = table & bin_footprints(scene, points, c["bin_margin"])
    mat = table & ~bins & (np.abs(height) < c["below"])
    out_depth, out_rgb = np.full(depth_mm.shape, FAR), np.zeros_like(rgb)
    for part, is_bin in ((table & ~bins, False), (bins, True)):
        splat((out_depth, out_rgb), camera.project(move.moved(points[part], is_bin)), rgb[v[part], u[part]])
    close_cracks(out_depth, out_rgb)
    counter = bare_counter(scene, camera, move, depth_mm.shape)
    bare = (out_depth >= FAR) & (counter < FAR)
    out_depth[bare], out_rgb[bare] = counter[bare], np.median(rgb[v[mat], u[mat]], axis=0)
    stays = np.zeros(depth_mm.shape, bool)
    stays[v[~table], u[~table]] = True
    stays &= depth_mm * MM < out_depth
    out_depth[stays], out_rgb[stays] = depth_mm[stays] * MM, rgb[stays]
    return out_rgb, np.where((out_depth < FAR) & ~holes, np.round(out_depth / MM), 0).astype(np.uint16)
