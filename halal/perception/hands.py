"""The robot's own forearms and hands as a camera reference: their surface by forward kinematics against the depth image.

    python -m halal.perception.hands --regen     resamples assets/hand_surface.npz from the URDF meshes under third_party/
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec

from halal.perception.geometry import MM, Camera, PlaneFit

ASSET = Path(__file__).with_name("assets") / "hand_surface.npz"
PARTS = ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "hand_palm_link", "hand_thumb_0_link",
         "hand_thumb_1_link", "hand_thumb_2_link", "hand_middle_0_link", "hand_middle_1_link", "hand_index_0_link",
         "hand_index_1_link")
LINKS = tuple(f"{side}_{part}" for side in ("left", "right") for part in PARTS)
SAMPLE_AREA = 16e-6     # one surface sample per this many square metres
COVER_REACH = 0.05      # a sample whose outward ray meets its own link within this distance is not on the outside


class HandSurface:
    """Sampled outer surface of both forearms and hands, posed in the pelvis frame from measured joint angles."""

    def __init__(self) -> None:
        spec = JointSpec()
        self.model = pin.buildModelFromUrdf(str(DEFAULT_URDF))
        self.data = self.model.createData()
        self.arm_iq = [self.model.joints[self.model.getJointId(n)].idx_q for n in spec.arm_names]
        self.hand_iq = [self.model.joints[self.model.getJointId(n)].idx_q for n in spec.hand_names]
        asset = np.load(ASSET)
        self.frames = [self.model.getFrameId(str(name)) for name in asset["links"]]
        self.link, self.points, self.normals = asset["link"], asset["points"].astype(float), asset["normals"].astype(float)

    def posed(self, arm_q: np.ndarray, hand_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Surface points and outward normals (N, 3) in the pelvis frame."""
        q = pin.neutral(self.model)
        q[self.arm_iq], q[self.hand_iq] = arm_q, hand_q
        pin.framesForwardKinematics(self.model, self.data, q)
        rotations = np.stack([self.data.oMf[f].rotation for f in self.frames])[self.link]
        offsets = np.stack([self.data.oMf[f].translation for f in self.frames])[self.link]
        return np.einsum("nij,nj->ni", rotations, self.points) + offsets, np.einsum("nij,nj->ni", rotations, self.normals)


@dataclass(frozen=True)
class HandFit:
    """A camera pose that puts the seen forearms and hands onto their kinematic model, with its support."""
    camera: Camera
    points: int             # depth pixels matched to the model at the final gate
    rms: float              # their point-to-surface distance, m
    spread: np.ndarray      # predicted standard deviation (xyz, m) this pose leaves on a point at `anchor`


class HandReference:
    """Point-to-plane registration of the depth pixels near the posed hand surface: 6 DOF, held to the given pose by a prior."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.surface = HandSurface()
        self.prior = np.r_[np.full(3, np.radians(cfg["prior_turn_deg"])), np.full(3, cfg["prior_shift"])] ** -2.0

    def fit(self, depth: np.ndarray, camera: Camera, arm_q: np.ndarray, hand_q: np.ndarray, floor: PlaneFit | None,
            anchor: np.ndarray) -> HandFit | None:
        """Register starting from `camera`; None when too little of the arms is in view."""
        model, normals = self._facing(*self.surface.posed(arm_q, hand_q), camera, depth.shape)
        pixels = self._pixels(depth, camera, model, floor)
        if len(model) < self.cfg["min_points"] or len(pixels) < self.cfg["min_points"]:
            return None
        tree, pose, moved = cKDTree(model), np.zeros(6), camera
        for gate in self.cfg["gates"]:
            system = self._normal_equations(tree, model, normals, moved, pixels, gate)
            if system is None:
                return None
            information, gradient, points, rms = system
            pose = pose + np.linalg.solve(information + np.diag(self.prior), gradient - self.prior * pose)
            moved = camera.moved(Rotation.from_rotvec(pose[:3]).as_matrix(), pose[3:])
        lever = np.c_[-skew(anchor - camera.T[:3, 3]), np.eye(3)]
        covariance = lever @ np.linalg.inv(information + np.diag(self.prior)) @ lever.T
        return HandFit(moved, points, rms, np.sqrt(np.diag(covariance)))

    def _facing(self, model: np.ndarray, normals: np.ndarray, camera: Camera, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """The model points inside the image that face the camera and are the nearest in their image cell."""
        view = model - camera.T[:3, 3]
        uvz = camera.project(model)
        u, v = uvz[:, 0], uvz[:, 1]
        cosine = np.einsum("ij,ij->i", normals, view) / np.linalg.norm(view, axis=1)
        edge = self.cfg["border_px"]
        ok = np.flatnonzero((cosine < -self.cfg["min_facing"]) & (u >= edge) & (u < shape[1] - edge) & (v >= edge) & (v < shape[0] - edge))
        cell = (v[ok] // self.cfg["cell_px"]).astype(int) * shape[1] + (u[ok] // self.cfg["cell_px"]).astype(int)
        nearest = np.full(shape[0] * shape[1], np.inf)
        np.minimum.at(nearest, cell, uvz[ok, 2])
        front = ok[uvz[ok, 2] <= nearest[cell] + self.cfg["cell_depth"]]
        return model[front], normals[front]

    def _pixels(self, depth: np.ndarray, camera: Camera, model: np.ndarray, floor: PlaneFit | None) -> np.ndarray:
        """Rows (u, v, depth) of valid pixels in the padded image box of the model, clear of the counter."""
        if not len(model):
            return np.zeros((0, 3))
        c = self.cfg
        uv = camera.project(model)[:, :2]
        lo = np.maximum(np.floor(uv.min(axis=0)).astype(int) - c["pad_px"], 0)
        hi = np.ceil(uv.max(axis=0)).astype(int) + c["pad_px"]
        v, u = np.mgrid[lo[1]:min(hi[1], depth.shape[0]):c["pixel_step"], lo[0]:min(hi[0], depth.shape[1]):c["pixel_step"]]
        uvz = np.c_[u.ravel(), v.ravel(), depth[v, u].ravel() * MM]
        uvz = uvz[uvz[:, 2] > 0]
        if floor is None:
            return uvz
        return uvz[camera.deproject(uvz) @ floor.normal - floor.offset > c["floor_clearance"]]

    def _normal_equations(self, tree: cKDTree, model: np.ndarray, normals: np.ndarray, camera: Camera, pixels: np.ndarray,
                          gate: float) -> tuple[np.ndarray, np.ndarray, int, float] | None:
        """Point-to-plane information matrix and gradient for a small turn about the optical centre and a shift."""
        seen = camera.deproject(pixels)
        distance, index = tree.query(seen, distance_upper_bound=gate)
        near = np.isfinite(distance)
        if near.sum() < self.cfg["min_points"]:
            return None
        n, arm = normals[index[near]], seen[near] - camera.T[:3, 3]
        along = np.einsum("ij,ij->i", n, model[index[near]] - seen[near])
        jacobian = np.c_[np.cross(arm, n), n]
        weight = min(1.0, self.cfg["independent_points"] / len(n)) / self.cfg["noise"] ** 2
        rms = float(np.sqrt(np.mean(along ** 2)))
        return weight * jacobian.T @ jacobian, weight * jacobian.T @ along, int(near.sum()), rms


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def load_stl(path: Path) -> np.ndarray:
    """Triangles (N, 3, 3) of a binary STL file, metres."""
    raw = np.fromfile(path, dtype=np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84)
    return raw["corners"].astype(float)


def link_triangles(urdf: Path, link: str) -> np.ndarray:
    """Triangles of a link's visual mesh in the link frame."""
    visual = next(el for el in ET.parse(urdf).getroot().iter("link") if el.get("name") == link).find("visual")
    origin = visual.find("origin")
    turn = Rotation.from_euler("xyz", [float(x) for x in origin.get("rpy").split()]).as_matrix()
    offset = np.array([float(x) for x in origin.get("xyz").split()])
    return load_stl(urdf.parent / visual.find("geometry/mesh").get("filename")) @ turn.T + offset


def covered(points: np.ndarray, normals: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """True for samples whose outward ray hits the mesh again within COVER_REACH (Moeller-Trumbore, one ray at a time)."""
    a, edge1, edge2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    hit = np.zeros(len(points), bool)
    for i, (origin, ray) in enumerate(zip(points + 1e-5 * normals, normals)):
        p = np.cross(ray, edge2)
        det = np.einsum("ij,ij->i", edge1, p)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = origin - a
            u = np.einsum("ij,ij->i", s, p) / det
            q = np.cross(s, edge1)
            v = q @ ray / det
            t = np.einsum("ij,ij->i", edge2, q) / det
            hit[i] = np.any((np.abs(det) > 1e-14) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < COVER_REACH))
    return hit


def sample_surface(triangles: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform samples of the outside of a mesh with their unit normals."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    pick = rng.choice(len(triangles), size=int(area.sum() / SAMPLE_AREA), p=area / area.sum())
    r1, r2 = np.sqrt(rng.uniform(size=len(pick))), rng.uniform(size=len(pick))
    t = triangles[pick]
    points = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
    normals = cross[pick] / (2 * area[pick])[:, None]
    outside = ~covered(points, normals, triangles[area > 0])
    return points[outside], normals[outside]


def regen() -> None:
    """Sample every link of PARTS on both sides and write the asset."""
    rng = np.random.default_rng(0)
    sampled = [sample_surface(link_triangles(Path(DEFAULT_URDF), link), rng) for link in LINKS]
    index = np.concatenate([np.full(len(points), i, np.int16) for i, (points, _) in enumerate(sampled)])
    ASSET.parent.mkdir(exist_ok=True)
    np.savez_compressed(ASSET, links=np.array(LINKS), link=index,
                        points=np.concatenate([p for p, _ in sampled]).astype(np.float32),
                        normals=np.concatenate([n for _, n in sampled]).astype(np.float16))
    print(f"{len(index)} surface samples of {len(LINKS)} links -> {ASSET} ({ASSET.stat().st_size // 1024} kB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true", required=True)
    ap.parse_args(argv)
    regen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
