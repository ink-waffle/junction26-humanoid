"""A tiny kinematic "world" for the mock backend so the classical baseline can be tested end to end offline.

The stock MockBackend returns blank images and has no objects. RenderMockBackend adds:
  * a ray-cast head RGB-D render (counter, coloured bins, coloured objects) through calibration.yaml,
  * kinematic grasping: an object within `attach_radius` of a closing hand's grasp point sticks to it, and drops
    straight down (into a bin or onto the counter) when the hand opens.
No physics, no contacts, no occlusion by the arms. Not a substitute for Isaac (WP3); it only exercises the
perception -> IK -> trajectory -> AaltoEnv/SafetyFilter path. AaltoEnv itself is used unchanged:

    env = make_mock_env(seed=0)       # AaltoEnv(backend="mock") with its backend swapped for this one
"""
from __future__ import annotations

import numpy as np

from aaltonomous.interface.backends import H, W, MockBackend
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

FLOOR_Z = -0.75
MAT_RGB = (175, 175, 170)
FLOOR_RGB = (60, 60, 60)


def _yaw_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Renderer:
    def __init__(self, scene: Scene, cal: Calibration, cam="head"):
        self.scene, self.cal, self.cam = scene, cal, cam
        c = cal.cams[cam]
        T = cal._T(cam)
        self.K, self.R, self.o = c.K, T[:3, :3], T[:3, 3]
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        ray_opt = np.stack([(u - c.K[0, 2]) / c.K[0, 0], (v - c.K[1, 2]) / c.K[1, 1], np.ones_like(u)], -1)
        self.rays = ray_opt @ self.R.T             # pelvis-frame directions; ray param t == optical depth (m)
        self.colors = {k: np.asarray(v, float) for k, v in scene.colors.items()}
        self._bg = self._background()

    # ---------------------------------------------------------------- primitives
    def _plane(self, z):
        dz = self.rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self.o[2]) / dz
        return np.where((dz < -1e-9) & (t > 0), t, np.inf)

    def _background(self):
        sc = self.scene
        rgb = np.zeros((H, W, 3), float)
        zb = np.full((H, W), np.inf)
        t_floor = self._plane(FLOOR_Z)
        rgb[np.isfinite(t_floor)] = FLOOR_RGB
        zb = np.minimum(zb, t_floor)
        t = self._plane(sc.counter_z)
        p = self.o + t[..., None] * self.rays
        pm = sc.pelvis_to_mat(np.where(np.isfinite(p), p, 0.0))
        mat = np.asarray(sc.cfg["counter"]["mat_size"], float)
        size = np.asarray(sc.cfg["counter"]["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        on = np.isfinite(t) & np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1) & (t < zb)
        rgb[on] = MAT_RGB
        rgb[on] += ((np.floor(pm[on, 0] / 0.05) + np.floor(pm[on, 1] / 0.05)) % 2)[:, None] * 12.0  # grid
        zb[on] = t[on]
        b = sc.cfg["bins"]
        ox, oy = np.asarray(b["outer_size"], float) / 2
        wt, wh, ft = b["wall_thickness"], b["wall_height"], b["floor_thickness"]
        for name, (cx, cy) in b["centers"].items():
            col = self.colors[b["color"][name]]
            c = sc.mat_to_pelvis([cx, cy, 0.0])
            parts = [((0, 0, ft / 2), (oy, ox, ft / 2)),                 # mat x = pelvis y, mat y = pelvis x
                     ((oy - wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)), ((-oy + wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)),
                     ((0, ox - wt / 2, wh / 2), (oy, wt / 2, wh / 2)), ((0, -ox + wt / 2, wh / 2), (oy, wt / 2, wh / 2))]
            for off, half in parts:
                self._box(rgb, zb, c + np.asarray(off), np.eye(3), np.asarray(half), col)
        return rgb, zb

    def _pixels(self, center, R, half):
        corners = center + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half) @ R.T
        uvz = self.cal.project(self.cam, corners)
        if np.any(uvz[:, 2] <= 0.01):
            return None
        u0, v0 = np.floor(uvz[:, :2].min(0)).astype(int) - 1
        u1, v1 = np.ceil(uvz[:, :2].max(0)).astype(int) + 2
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if u0 >= u1 or v0 >= v1:
            return None
        return slice(v0, v1), slice(u0, u1)

    def _box(self, rgb, zb, center, R, half, col):
        win = self._pixels(center, R, half)
        if win is None:
            return
        d = self.rays[win] @ R                    # into the box frame
        o = (self.o - center) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin, tmax = tn.max(-1), tf.min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        axis = tn.argmax(-1)
        nz = np.abs((np.eye(3)[axis] @ R.T)[..., 2])
        self._write(rgb, zb, win, hit, tmin, nz, col)

    def _cylinder(self, rgb, zb, center, radius, hz, col):
        half = np.array([radius, radius, hz])
        win = self._pixels(center, np.eye(3), half)
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2 * (o[0] * d[..., 0] + o[1] * d[..., 1])
        cc = o[0] ** 2 + o[1] ** 2 - radius ** 2
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = (-b - np.sqrt(disc)) / (2 * a)
            tcap = (hz - o[2]) / d[..., 2]
        side = (disc >= 0) & (ts > 0) & (np.abs(o[2] + ts * d[..., 2]) <= hz)
        pc = o + tcap[..., None] * d
        cap = (tcap > 0) & (pc[..., 0] ** 2 + pc[..., 1] ** 2 <= radius ** 2)
        t = np.where(cap & (~side | (tcap < ts)), tcap, np.where(side, ts, np.inf))
        nz = np.where(t == tcap, 1.0, 0.0)
        self._write(rgb, zb, win, np.isfinite(t), t, nz, col)

    @staticmethod
    def _write(rgb, zb, win, hit, t, nz, col):
        z = zb[win]
        m = hit & (t < z)
        if not m.any():
            return
        shade = 0.6 + 0.4 * nz
        sub = rgb[win]
        sub[m] = col * shade[m][:, None]
        z[m] = t[m]
        rgb[win], zb[win] = sub, z

    # ---------------------------------------------------------------- frame
    def render(self, objects, rng=None, depth_noise_mm=0.0, hole_frac=0.0):
        rgb, zb = self._bg[0].copy(), self._bg[1].copy()
        for ob in objects:
            col = self.colors[ob["color"]]
            s, h = ob["size"], ob.get("height", ob["size"])
            if ob.get("shape", "cube") in ("cylinder", "circle"):
                self._cylinder(rgb, zb, ob["pos"], s / 2, h / 2, col)
            else:  # cube; triangle prisms are drawn as their bounding box (a blob is a blob for this baseline)
                self._box(rgb, zb, ob["pos"], _yaw_R(ob.get("yaw", 0.0)), np.array([s / 2, s / 2, h / 2]), col)
        depth = np.where(np.isfinite(zb), zb * 1000.0, 0.0)
        if rng is not None and depth_noise_mm > 0:
            depth = depth + rng.normal(0, depth_noise_mm, depth.shape)
        if rng is not None and hole_frac > 0:
            depth[rng.random(depth.shape) < hole_frac] = 0
        return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(depth, 0, 65535).astype(np.uint16)


class RenderMockBackend(MockBackend):
    name = "mock"

    def __init__(self, scene, cal, kin, postures, attach_radius=0.04, depth_noise_mm=1.0, hole_frac=0.02, seed=0,
                 **kw):
        super().__init__(**kw)
        self.scene, self.kin, self.post = scene, kin, postures
        self.renderer = Renderer(scene, cal)
        self.attach_radius = attach_radius
        self.depth_noise_mm, self.hole_frac = depth_noise_mm, hole_frac
        self.rng = np.random.default_rng(seed)
        self.objects, self.held = [], {"left": None, "right": None}
        self._cache = None

    def reset_scene(self, layout, instruction):
        super().reset_scene(layout, instruction)
        sc = self.scene
        self.objects = [{"color": c["color"], "shape": c.get("shape", "cube"), "yaw": c.get("yaw", 0.0), "size": sc.size,
                         "height": sc.object_height(c.get("shape")),
                         "pos": sc.mat_to_pelvis(sc.cube_spawn_mat(c))} for c in layout]
        self.held = {"left": None, "right": None}
        self._cache = None

    def _closure(self, side):
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        o, c = self.post["open"][sl], self.post["closed"][sl]
        return float((self.hand_q[sl] - o) @ (c - o) / ((c - o) @ (c - o)))

    def advance(self, dt):
        super().advance(dt)
        for side in ("left", "right"):
            g, _ = self.kin.grasp_pose(side, self.arm_q)
            f = self._closure(side)
            h = self.held[side]
            if h is None and f > 0.6:
                free = [i for i, ob in enumerate(self.objects) if i not in self.held.values()]
                if free:
                    i = min(free, key=lambda i: np.linalg.norm(self.objects[i]["pos"] - g))
                    if np.linalg.norm(self.objects[i]["pos"] - g) < self.attach_radius:
                        self.held[side] = (i, self.objects[i]["pos"] - g)
            elif h is not None:
                i, off = h
                self.objects[i]["pos"] = g + off
                if f < 0.3:
                    self._drop(self.objects[i])
                    self.held[side] = None

    def _drop(self, ob):
        sc = self.scene
        pm = sc.pelvis_to_mat(ob["pos"])
        for name in sc.bin_names:
            lo, hi = sc.bin_inner_box(name)
            if np.all(pm[:2] >= lo[:2]) and np.all(pm[:2] <= hi[:2]):
                ob["pos"] = sc.mat_to_pelvis([pm[0], pm[1], lo[2] + ob["height"] / 2])
                return
        z = ob["height"] / 2 if sc.on_counter(np.r_[pm[:2], 0.0]) else -1.0
        ob["pos"] = sc.mat_to_pelvis([pm[0], pm[1], z])

    def images(self):
        key = tuple(np.round(np.concatenate([o["pos"] for o in self.objects]), 4)) if self.objects else ()
        if self._cache is None or self._cache[0] != key:
            rgb, depth = self.renderer.render(self.objects, self.rng, self.depth_noise_mm, self.hole_frac)
            self._cache = (key, rgb, depth)
        out = super().images()
        out["head_rgb"], out["head_depth"] = self._cache[1].copy(), self._cache[2].copy()
        return out

    def outcomes(self, mapping):
        return [self.scene.classify_cube(o["pos"], 0.0, o["color"], mapping) for o in self.objects]  # colour rule (rung 1)


def make_mock_env(scene_path=None, seed=0, cfg=None, realtime=False, **backend_kw):
    """AaltoEnv(backend="mock") with the render/grasp mock backend. scene_path overrides scene.yaml."""
    from aaltonomous.interface.env import AaltoEnv
    from .config import hand_postures, load_config
    from .kinematics import ArmKinematics
    cfg = cfg or load_config()
    env = AaltoEnv(backend="mock", task="sort_fixed", seed=seed, realtime=realtime)
    if scene_path is not None:
        env.scene = Scene.load(scene_path)
    env.backend = RenderMockBackend(env.scene, Calibration.load(), ArmKinematics(cfg), hand_postures(cfg),
                                    seed=seed, **backend_kw)
    return env
