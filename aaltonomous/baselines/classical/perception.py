"""Head RGB-D -> coloured objects in the pelvis frame. Shape-agnostic: an object is a coloured blob, its grasp
target is the centre of its top face.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw), ...]

Per colour: HSV threshold (config.yaml perception.hsv) -> opening -> connected components. Per blob: erode, take
the pixels with valid depth, deproject them (calibration.yaml intrinsics + T_pelvis_optical), keep the top face
(points within `top_band` of a robust max height) and use the median of those as the top centre. With too few valid
depth pixels (holes, or no depth stream) the blob centroid ray is intersected with the plane at the known object
height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong height are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "plane"
    yaw: float | None = None   # in-plane orientation of the top face (rad), info only
    uv: tuple = field(default=(0.0, 0.0))

    def __repr__(self):
        x, y, z = self.top
        return (f"{self.color:<6} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]")


class Perception:
    def __init__(self, cfg, scene, calibration):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # pelvis-frame xy correction added to every detection (head extrinsics error; tuned on the robot)
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        out = {}
        for color, ranges in self.cfg["hsv"].items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def detect(self, rgb, depth, debug=False):
        import cv2
        c = self.cfg
        dets, rejected = [], []
        dlo, dhi = c["depth_range_mm"]
        keep = self._keep_mask(depth)
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < c["min_area_px"]:
                    continue
                blob = (lab == i).astype(np.uint8)
                e = int(c["erode_px"])
                core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                if core.sum() < 0.25 * area:
                    core = blob
                v, u = np.nonzero(core)
                d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
                ok = (d >= dlo) & (d <= dhi)
                det = None
                if ok.sum() >= c["min_valid_depth_px"]:
                    det = self._from_depth(color, u[ok], v[ok], d[ok], area)
                if det is None:
                    det = self._from_plane(color, cent[i], area, int(ok.sum()))
                why = self._reject(det)
                if why:
                    rejected.append((det, why))
                else:
                    dets.append(det)
        return (dets, rejected) if debug else dets

    def _keep_mask(self, depth):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        keep = self._bin_hulls_mask(depth.shape if depth is not None else (480, 640))
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        ok = (h >= 0.25) & (h <= c["height_range"][1])
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _from_depth(self, color, u, v, d, area):
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        xy = xy + self.xy_offset
        return Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                         (float(np.median(u)), float(np.median(v))))

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.cal._T(self.cam)
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det):
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        return None


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
