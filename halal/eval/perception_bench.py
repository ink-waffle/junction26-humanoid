"""Perception benchmark: one detector against privileged ground truth over a collected frame set.

    python -m halal.eval.perception_bench --detector baseline --frames runs/frames/dr_train [--split heldout]
    python -m halal.eval.perception_bench --armless 10      rendered scenes without arms: the hand fit must not be believed
    --stale   every episode's detector first sees a frame of another episode (a stale first frame must not stick)
    --looks 2 every frame is shown twice, 0.2 s apart (a robot standing still)
    --depth typical --depth-seed 0   the depth image as a RealSense D435i would deliver it (bench.d435: mild | typical | bad)
    --shift real                     the table re-rendered where the real recordings have it (bench.shifts)
    --still   only frames in which the arms stand where the frame before had them: the looks of a policy, episode by episode
    A set recorded from a displaced sim carries `scene.json` ({"offsets": {"counter": [x, y, z], "bins": [x, y]}, "boxes":
    {"wall_height": h}}, as halal/ops/real_sim.yaml): ground truth and the layout figures then use that placement.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from aaltonomous.baselines.classical.mock_world import Renderer
from aaltonomous.interface import tasks
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene, shape_polygon

from halal import config
from halal.eval.depth_sensor import LEVELS, ArmPixels, degrade
from halal.eval.scene_shift import Displacement, displaced
from halal.perception import SHAPES, Detector, ObjectDetection, SceneLayout, make_detector, scene_layout
from halal.perception.color import turned

REPO = Path(__file__).resolve().parents[2]
CAMERA = "head"
SYMMETRY = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3}
APEX_OFFSET = {"cube": 0.0, "triangle": np.pi / 2}
SPLITS = ("all", "dev", "heldout")
LOOK_INTERVAL = 0.2
Row = dict[str, Any]


@dataclass(frozen=True)
class Pinhole:
    """A camera model: intrinsics K and the pose `T_pelvis_optical`."""
    K: np.ndarray
    T: np.ndarray

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pelvis-frame points (N, 3) -> (u, v, optical depth)."""
        p = (points - self.T[:3, 3]) @ self.T[:3, :3]
        return np.c_[self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2], self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2], p[:, 2]]

    def deproject(self, uvz: np.ndarray) -> np.ndarray:
        """(u, v, optical depth) rows -> pelvis-frame points."""
        x = (uvz[:, 0] - self.K[0, 2]) / self.K[0, 0] * uvz[:, 2]
        y = (uvz[:, 1] - self.K[1, 2]) / self.K[1, 1] * uvz[:, 2]
        return np.c_[x, y, uvz[:, 2]] @ self.T[:3, :3].T + self.T[:3, 3]


@dataclass(frozen=True)
class Truth:
    """One ground-truth object lying on the counter."""
    name: str
    color: str
    shape: str
    top: np.ndarray
    yaw: float | None
    view: str
    toppled: bool = False       # a prism lying on a side: `top` is the middle of its ridge, `yaw` the ridge direction


@dataclass(frozen=True)
class Frame:
    """One recorded observation with its ground truth and the camera that really took it."""
    seed: int
    tick: int
    obs: dict[str, Any]
    truths: list[Truth]
    ignored: list[np.ndarray]   # xy of objects that are neither lying still on the counter nor clearly off it
    camera: Pinhole
    dr: dict[str, Any]
    arms: bool = True           # False: the image shows no robot arm, whatever the joint angles say
    move: Displacement | None = None    # where the table stands against the scene file, when the frame was re-rendered


def sim_camera(nominal: Pinhole, camera_dr: Mapping[str, Any], cfg: Mapping[str, Any]) -> Pinhole:
    """The camera the sim renders with: DR pose offset in the world frame (as `_apply_dr`), principal point as measured."""
    K, T = nominal.K.copy(), nominal.T.copy()
    K[1, 1] = K[0, 0]
    K[:2, 2] = cfg["sim_principal_point"]
    T[:3, :3] = Rotation.from_rotvec(camera_dr.get("rot_offset_rad", [0.0] * 3)).as_matrix() @ T[:3, :3]
    T[:3, 3] += np.asarray(camera_dr.get("pos_offset", [0.0] * 3))
    return Pinhole(K, T)


def reobserve(point: np.ndarray, true: Pinhole, used: Pinhole) -> np.ndarray:
    """Where a point imaged by the true camera lands in the pelvis frame when deprojected with the model in use."""
    return used.deproject(true.project(point[None]))[0]


def body_yaw(quat_xyzw: Sequence[float]) -> float:
    """Rotation about pelvis z of an (almost) upright object."""
    return float(Rotation.from_quat(quat_xyzw).as_euler("zyx")[0])


def top_face_points(scene: Scene, shape: str, top: np.ndarray, yaw: float) -> np.ndarray:
    """Sample points spread over the object's top face, pelvis frame."""
    outline = shape_polygon(shape, scene.object_dims(shape)["apothem"], yaw)
    rings = np.concatenate([s * outline for s in (0.3, 0.6, 0.9)])
    return np.c_[top[:2] + rings, np.full(len(rings), top[2])]


def view_class(points: np.ndarray, camera: Pinhole, depth: np.ndarray, cfg: Mapping[str, float]) -> str:
    """`border` when the top face leaves the image, `hidden` when something is in front of it, else `clear`."""
    uvz = camera.project(points)
    u, v = np.round(uvz[:, 0]).astype(int), np.round(uvz[:, 1]).astype(int)
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    if inside.mean() < cfg["min_inside"]:
        return "border"
    seen = depth[v[inside], u[inside]] / 1000.0
    blocked = (seen > 0) & (seen < uvz[inside, 2] - cfg["occlusion_gap"])
    return "hidden" if blocked.mean() > cfg["max_blocked"] else "clear"


def presence(scene: Scene, obj: Mapping[str, Any], cfg: Mapping[str, float]) -> str:
    """`lying` still on the counter outside every bin, `absent` (in a bin, held up, off the counter) or `unclear`."""
    mat = scene.pelvis_to_mat(np.asarray(obj["pos"], float))
    lift = mat[2] - scene.object_height() / 2
    if not scene.on_counter(mat) or scene.bin_of(mat) is not None or lift > cfg["held_height"]:
        return "absent"
    still = obj.get("speed", 0.0) <= scene.cfg["success"]["at_rest_speed"]
    return "lying" if still and abs(lift) < cfg["resting_tolerance"] else "unclear"


def toppled_truth(obj: Mapping[str, Any], scene: Scene, spin: Rotation) -> Truth:
    """Ground truth of a prism lying on a side: the uppermost edge is its ridge."""
    corners = spin.apply(np.c_[shape_polygon("triangle", scene.object_dims("triangle")["apothem"], 0.0), np.zeros(3)])
    axis = spin.apply([0.0, 0.0, 1.0])
    ridge = np.asarray(obj["pos"], float) + corners[np.argmax(corners[:, 2])]
    return Truth(obj["name"], obj["color"], "triangle", ridge, float(np.arctan2(axis[1], axis[0])), "toppled", toppled=True)


def make_truth(obj: Mapping[str, Any], scene: Scene, camera: Pinhole, depth: np.ndarray, cfg: Mapping[str, float]) -> Truth:
    spin = Rotation.from_quat(obj["quat_xyzw"])
    if obj["shape"] == "triangle" and abs(spin.apply([0.0, 0.0, 1.0])[2]) < 0.5:
        return toppled_truth(obj, scene, spin)
    centre = np.asarray(obj["pos"], float)
    top = centre + [0.0, 0.0, scene.object_height() / 2]
    yaw = body_yaw(obj["quat_xyzw"])
    view = view_class(top_face_points(scene, obj["shape"], top, yaw), camera, depth, cfg)
    apex = yaw + APEX_OFFSET[obj["shape"]] if obj["shape"] in APEX_OFFSET else None
    return Truth(obj["name"], obj["color"], obj["shape"], top, apex, view)


def placed(scene: Scene, placement: Mapping[str, Any]) -> tuple[Scene, Displacement]:
    """The scene as a recorded set had it (its `scene.json`), and the same as a displacement of the scene file's."""
    counter, bins = np.asarray(placement["offsets"]["counter"], float), np.asarray(placement["offsets"]["bins"], float)
    cfg = copy.deepcopy(scene.cfg)
    cfg["counter"]["mat_origin_in_pelvis"] = (scene.origin + counter).tolist()
    cfg["bins"]["centers"] = {name: [c[0] + bins[1], c[1] + bins[0]] for name, c in cfg["bins"]["centers"].items()}
    cfg["bins"]["wall_height"] = placement["boxes"]["wall_height"]
    table = np.eye(4)
    table[:3, 3] = counter
    return Scene(cfg), Displacement(table, {name: bins for name in cfg["bins"]["centers"]})


def load_frame(path: Path, scene: Scene, nominal: Pinhole, cfg: Mapping[str, Any], move: Displacement | None = None) -> Frame:
    """One recorded frame; `scene` is the scene as it stood when recorded, `move` its displacement from the scene file."""
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    obs = {k: data[k] for k in data.files if k != "meta"}
    obs.update(arm_dq=np.zeros(14, np.float32), instruction=meta["instruction"], t=float(meta.get("t", 0.0)))
    dr = meta["dr"] or {}
    camera = sim_camera(nominal, dr.get("camera", {}), cfg)
    state = [presence(scene, o, cfg) for o in meta["objects"]]
    truths = [make_truth(o, scene, camera, obs["head_depth"], cfg) for o, s in zip(meta["objects"], state) if s == "lying"]
    ignored = [np.asarray(o["pos"][:2], float) for o, s in zip(meta["objects"], state) if s == "unclear"]
    return Frame(int(meta["seed"]), int(meta.get("tick", 0)), obs, truths, ignored, camera, dr, move=move)


def armless_frames(scene: Scene, nominal: Pinhole, layouts: int, cfg: Mapping[str, Any]) -> list[Frame]:
    """Scenes rendered by the organisers' mock world (no arms drawn) under joint angles that put the hands in view."""
    renderer, rng, frames = Renderer(scene, Calibration.load()), np.random.default_rng(cfg["armless_seed"]), []
    for i in range(layouts):
        layout = tasks.sample_layout(rng, shapes=("cube", "cylinder"), **scene.layout_kwargs())
        objects = [{"name": f"obj_{o['color']}_{o['shape']}_{k}", "color": o["color"], "shape": o["shape"], "speed": 0.0,
                    "pos": scene.mat_to_pelvis(scene.cube_spawn_mat(o)), "yaw": o["yaw"], "size": scene.size,
                    "height": scene.object_height(), "quat_xyzw": Rotation.from_euler("z", o["yaw"]).as_quat()}
                   for k, o in enumerate(layout)]
        rgb, depth = renderer.render(objects, rng, cfg["armless_depth_noise_mm"])
        for j, arm_q in enumerate(cfg["armless_arm_q"]):
            obs = {"head_rgb": rgb, "head_depth": depth, "arm_q": np.asarray(arm_q, np.float32), "hand_q": np.zeros(14, np.float32),
                   "arm_dq": np.zeros(14, np.float32), "instruction": "", "t": 0.0}
            truths = [make_truth(o, scene, nominal, depth, cfg) for o in objects]
            frames.append(Frame(len(cfg["armless_arm_q"]) * i + j, 0, obs, truths, [], nominal, {}, arms=False))
    return frames


def silhouette(scene: Scene, truth: Truth, camera: Pinhole, shape: tuple[int, int]) -> np.ndarray:
    """Image mask of one lying object: the convex outline of its top and bottom faces through the camera that took the frame."""
    turn = (truth.yaw or 0.0) - APEX_OFFSET.get(truth.shape, 0.0)
    outline = truth.top[:2] + shape_polygon(truth.shape, scene.object_dims(truth.shape)["apothem"], turn)
    corners = np.concatenate([np.c_[outline, np.full(len(outline), z)] for z in (truth.top[2], truth.top[2] - scene.object_height())])
    hull = cv2.convexHull(np.round(camera.project(corners)[:, :2]).astype(np.int32))
    return cv2.fillConvexPoly(np.zeros(shape, np.uint8), hull, 1) > 0


def repainted(frame: Frame, scene: Scene, turns: Mapping[str, float]) -> Frame:
    """The frame with every lying object's pixels turned in hue by its colour's angle (deg): the real foam palette in the sim's light."""
    rgb = frame.obs["head_rgb"].copy()
    for truth in frame.truths:
        mask = silhouette(scene, truth, frame.camera, rgb.shape[:2])
        rgb[mask] = turned(rgb[mask], turns[truth.color])
    return replace(frame, obs={**frame.obs, "head_rgb": rgb})


def rotated(frame: Frame, degrees: float) -> Frame:
    """The frame with the hue of the whole image turned: a colour cast the counter's white balance does not remove."""
    return replace(frame, obs={**frame.obs, "head_rgb": turned(frame.obs["head_rgb"], degrees)})


def shifted(frame: Frame, scene: Scene, move: Displacement, arms: ArmPixels, cfg: Mapping[str, float]) -> Frame:
    """The frame with table, objects and bins displaced against the robot, ground truth moved along."""
    rgb, depth = displaced(frame.obs, frame.camera, scene, move, arms, cfg)
    truths = [replace(t, top=move.moved(t.top[None])[0]) for t in frame.truths]
    ignored = [move.moved(np.r_[xy, scene.counter_z][None])[0, :2] for xy in frame.ignored]
    return replace(frame, obs={**frame.obs, "head_rgb": rgb, "head_depth": depth}, truths=truths, ignored=ignored, move=move)


def dark_pixels(frame: Frame, arms: ArmPixels, c: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Pixels that return little to an IR stereo camera: the black arm links by FK, and whatever is dark in the image."""
    obs = frame.obs
    parts = c["black_parts"] if frame.arms else []
    black = arms.mask(obs["head_depth"], frame.camera, obs["arm_q"], obs["hand_q"], parts, c["arm_reach"])
    return black, obs["head_rgb"].max(axis=2) < c["dark_value"]


def sensed(frame: Frame, arms: ArmPixels, cfg: Mapping[str, Any], level: str, seed: int) -> Frame:
    """The frame with its depth image as a D435i would deliver it; the noise is drawn from the frame's seed, tick and `seed`."""
    rng = np.random.default_rng([seed, frame.seed, frame.tick])
    black, dark = dark_pixels(frame, arms, {**cfg, **cfg["levels"][level]})
    depth = degrade(frame.obs["head_depth"], black, dark, float(frame.camera.K[0, 0]), cfg, level, rng)
    return replace(frame, obs={**frame.obs, "head_depth": depth})


def conditioned(frames: Sequence[Frame], scene: Scene, cfg: Mapping[str, Any], shift: str | None, depth: str | None,
                seed: int) -> list[Frame]:
    """The frames with the table displaced (`shift`) and then seen through the depth sensor model (`depth`), as asked."""
    arms = ArmPixels()
    if shift:
        move = Displacement.from_config(scene, cfg["shifts"][shift])
        frames = [shifted(frame, scene, move, arms, cfg["shift"]) for frame in frames]
    return [sensed(frame, arms, cfg["d435"], depth, seed) for frame in frames] if depth else list(frames)


def still(frames: Sequence[Frame], tolerance: float) -> list[Frame]:
    """The frames of each episode whose arm and hand joints stand where its previous recorded frame had them."""
    ordered = sorted(frames, key=lambda f: (f.seed, f.tick))
    joints = [np.r_[f.obs["arm_q"], f.obs["hand_q"]] for f in ordered]
    return [f for k, f in enumerate(ordered) if k and ordered[k - 1].seed == f.seed and np.abs(joints[k] - joints[k - 1]).max() < tolerance]


def layout_row(frame: Frame, layout: SceneLayout, scene: Scene, wall_height: float) -> Row:
    """One look's layout against where counter, bins and rims really stand (`scene`: the scene file's)."""
    file, half = scene_layout(scene), np.asarray(scene.cfg["bins"]["outer_size"])[::-1] / 2
    truly = (lambda points, name=None: points) if frame.move is None else frame.move.moved
    bins = {name: truly(np.r_[xy, scene.counter_z][None], name)[0] for name, xy in file.bins.items()}
    band = scene.mat_to_pelvis([*np.mean(scene.layout_kwargs()["region"], axis=1), 0.0])
    rims = [truly(np.r_[xy + half * signs, scene.counter_z + wall_height][None], name)[0, 2]
            for name, xy in file.bins.items() for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    return {"seed": frame.seed, "tick": frame.tick, "source": layout.source,
            "bins_mm": [1e3 * float(np.linalg.norm(layout.bins[name] - bins[name][:2])) for name in file.bins],
            "counter_mm": 1e3 * float(layout.counter_z - truly(band[None])[0, 2]),
            "rim_mm": None if layout.rim_z is None else 1e3 * float(layout.rim_z - max(rims))}


def in_split(seed: int, split: str, heldout_every: int) -> bool:
    heldout = seed % heldout_every == heldout_every - 1
    return split == "all" or heldout == (split == "heldout")


def match(dets: Sequence[ObjectDetection], truths: Sequence[Truth], radius: float) -> dict[int, int]:
    """Optimal one-to-one assignment truth index -> detection index within `radius` (xy of the top centre)."""
    if not dets or not truths:
        return {}
    cost = np.array([[np.linalg.norm(d.top[:2] - t.top[:2]) for d in dets] for t in truths])
    rows, cols = linear_sum_assignment(np.where(cost <= radius, cost, 1e3))
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= radius}


def yaw_error(shape: str, detected: float, true: float) -> float:
    """Absolute yaw difference modulo the shape's rotational symmetry, rad."""
    period = SYMMETRY[shape]
    return float(abs((detected - true + period / 2) % period - period / 2))


def factors(frame: Frame, truth: Truth, nominal: Pinhole) -> Row:
    """The episode and object conditions the breakdowns group by."""
    shift = reobserve(truth.top, frame.camera, nominal) - truth.top
    tint = frame.dr.get("objects", {}).get(truth.name, {})
    return {"seed": frame.seed, "color": truth.color, "shape": truth.shape, "view": truth.view, "toppled": truth.toppled,
            "nominal_camera_mm": 1e3 * float(np.linalg.norm(shift[:2])),
            "gain": frame.dr.get("camera", {}).get("rgb_gain", 1.0),
            "key_light": frame.dr.get("lighting", {}).get("key_intensity_scale", 1.0),
            "hue_shift_deg": 360.0 * abs(tint.get("dh", 0.0))}


def outcome(frame: Frame, truth: Truth, det: ObjectDetection | None, used: Pinhole) -> Row:
    """What the detector made of one ground-truth object; `own` is its error with the camera model's part removed."""
    if det is None:
        return {"matched": False}
    error = det.top - truth.top
    own = det.top - reobserve(truth.top, frame.camera, used)
    shape_ok = det.shape == truth.shape
    has_yaw = shape_ok and det.yaw is not None and truth.yaw is not None and not truth.toppled
    ridge = truth.toppled and det.lying and det.yaw is not None
    return {"matched": True, "color_ok": det.color == truth.color, "confidence": float(det.confidence),
            "xy_mm": 1e3 * float(np.linalg.norm(error[:2])), "z_mm": 1e3 * float(abs(error[2])),
            "own_xy_mm": 1e3 * float(np.linalg.norm(own[:2])),
            "camera_mm": 1e3 * float(np.linalg.norm((error - own)[:2])), "det_shape": det.shape, "visible": float(det.visible),
            "size_mm": None if det.size is None else 1e3 * float(det.size), "det_lying": bool(det.lying),
            "ridge_deg": float(np.degrees(abs((det.yaw - truth.yaw + np.pi / 2) % np.pi - np.pi / 2))) if ridge else None,
            "yaw_deg": float(np.degrees(yaw_error(truth.shape, det.yaw, truth.yaw))) if has_yaw else None}


def score_frame(frame: Frame, dets: Sequence[ObjectDetection], nominal: Pinhole, used: Pinhole,
                radius: float) -> tuple[list[Row], list[float]]:
    """One row per ground-truth object plus the confidences of the phantoms: detections of nothing that lies there."""
    pairs = match(dets, frame.truths, radius)
    rows = [{**factors(frame, t, nominal), **outcome(frame, t, dets[pairs[i]] if i in pairs else None, used)}
            for i, t in enumerate(frame.truths)]
    spare = [d for k, d in enumerate(dets) if k not in pairs.values()]
    phantoms = [float(d.confidence) for d in spare if all(np.linalg.norm(d.top[:2] - xy) > radius for xy in frame.ignored)]
    return rows, phantoms


@dataclass
class Run:
    """Everything one pass of a detector over a frame set produced."""
    rows: list[Row]
    phantoms: list[float]
    runtimes: list[float]
    sources: list[str]
    layouts: list[Row]            # per look: the layout the detector reported against the truth
    believed_absent_arms: int     # frames without arms in the image whose layout nevertheless came from the hands


def run(detector_name: str, frames: Sequence[Frame], nominal: Pinhole, radius: float, looks: int = 1, stale: bool = False,
        wall_height: float | None = None) -> Run:
    """One detector per seed (= one episode), its frames in recording order, as the policy would look; the last look counts."""
    out, scene = Run([], [], [], [], [], 0), Scene.load()
    wall_height = scene.cfg["bins"]["wall_height"] if wall_height is None else wall_height
    detectors: dict[int, Detector] = {}
    ordered = sorted(frames, key=lambda f: (f.seed, f.tick))
    first = {f.seed: f for f in reversed(ordered)}
    seeds = sorted(first)
    for frame in ordered:
        if frame.seed not in detectors:
            detectors[frame.seed] = make_detector(detector_name)
            if stale:
                detectors[frame.seed].detect(first[seeds[seeds.index(frame.seed) - 1]].obs)
        detector = detectors[frame.seed]
        for look in range(looks):
            start = time.perf_counter()
            dets = detector.detect({**frame.obs, "t": frame.obs["t"] + LOOK_INTERVAL * look})
        out.runtimes.append(1e3 * (time.perf_counter() - start))
        used = Pinhole(nominal.K, getattr(detector, "extrinsics", nominal.T))
        rows, phantoms = score_frame(frame, dets, nominal, used, radius)
        out.rows += rows
        out.phantoms += phantoms
        out.sources.append(detector.layout().source)
        out.layouts.append(layout_row(frame, detector.layout(), scene, wall_height))
        out.believed_absent_arms += out.sources[-1] == "hands" and not frame.arms
    return out


def spread(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    if not len(a):
        return dict.fromkeys(("mean", "median", "p95", "max"), float("nan"))
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def ratio(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def summarise(rows: Sequence[Row]) -> Row:
    """Accuracy figures over a group of ground-truth rows."""
    hit = [r for r in rows if r["matched"]]
    shaped = [r for r in hit if r["det_shape"] is not None]
    turned = [r["yaw_deg"] for r in hit if r["yaw_deg"] is not None]
    wanted = [r for r in hit if r["shape"] in SYMMETRY]
    sized = [r["size_mm"] for r in hit if r["size_mm"] is not None and not r["det_lying"]]
    down = [r for r in rows if r["toppled"]]
    lying = [r for r in down if r["matched"] and r["det_lying"]]
    return {"objects": len(rows), "recall": ratio(len(hit), len(rows)),
            "color_acc": ratio(sum(r["color_ok"] for r in hit), len(hit)),
            "xy_mm": spread([r["xy_mm"] for r in hit]), "z_mm": spread([r["z_mm"] for r in hit]),
            "own_xy_mm": spread([r["own_xy_mm"] for r in hit]), "camera_mm": spread([r["camera_mm"] for r in hit]),
            "shape_decided": ratio(len(shaped), len(hit)),
            "shape_acc": ratio(sum(r["det_shape"] == r["shape"] for r in shaped), len(shaped)),
            "yaw_decided": ratio(len(turned), len(wanted)), "yaw_deg": spread(turned),
            "size_decided": ratio(len(sized), len(hit)), "size_mm": spread(sized),
            "toppled": len(down), "toppled_lying": ratio(len(lying), len(down)), "ridge_deg": spread([r["ridge_deg"] for r in lying]),
            "ridge_xy_mm": spread([r["xy_mm"] for r in lying]), "ridge_z_mm": spread([r["z_mm"] for r in lying]),
            "false_lying": sum(r["det_lying"] for r in hit if not r["toppled"])}


def confusion(rows: Sequence[Row]) -> dict[str, dict[str, int]]:
    """True shape -> detected shape counts over matched objects (`none` = not classified)."""
    table = {s: dict.fromkeys((*SHAPES, "none"), 0) for s in SHAPES}
    for r in rows:
        if r["matched"]:
            table[r["shape"]][r["det_shape"] or "none"] += 1
    return table


def bucket(edges: Sequence[float], unit: str) -> Callable[[float], str]:
    """Label function that puts a value into its interval between consecutive edges; labels sort by interval."""
    bounds = [-np.inf, *edges, np.inf]

    def label(value: float) -> str:
        i = int(np.searchsorted(edges, value, side="right"))
        return f"{i + 1}: {bounds[i]:g}..{bounds[i + 1]:g} {unit}"
    return label


def breakdowns(rows: Sequence[Row], cfg: Mapping[str, Any]) -> dict[str, dict[str, Row]]:
    """Summaries per shape, view class and binned calibration / lighting / tint factor."""
    keys: dict[str, Callable[[Row], str]] = {
        "shape": lambda r: r["shape"], "view": lambda r: r["view"], "color": lambda r: r["color"],
        "nominal camera error at object": lambda r: bucket(cfg["camera_edges_mm"], "mm")(r["nominal_camera_mm"]),
        "rgb gain": lambda r: bucket(cfg["gain_edges"], "x")(r["gain"]),
        "key light": lambda r: bucket(cfg["key_light_edges"], "x")(r["key_light"]),
        "object hue shift": lambda r: bucket(cfg["hue_edges_deg"], "deg")(r["hue_shift_deg"])}
    out: dict[str, dict[str, Row]] = {}
    for title, key in keys.items():
        groups: dict[str, list[Row]] = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        out[title] = {g: summarise(groups[g]) for g in sorted(groups)}
    return out


def confidence_table(rows: Sequence[Row], phantoms: Sequence[float], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Confidence histograms of right-colour, wrong-colour and phantom detections, and what each threshold keeps."""
    groups = {"right colour": [r["confidence"] for r in rows if r["matched"] and r["color_ok"]],
              "wrong colour": [r["confidence"] for r in rows if r["matched"] and not r["color_ok"]], "phantom": list(phantoms)}
    edges = [0.0, *cfg["confidence_edges"], 1.0 + 1e-9]
    histogram = {g: np.histogram(v, edges)[0].tolist() for g, v in groups.items()}
    kept = {str(t): {g: int(np.sum(np.asarray(v) >= t)) for g, v in groups.items()} for t in cfg["decision_thresholds"]}
    return {"edges": edges[:-1], "histogram": histogram, "kept": kept, "totals": {g: len(v) for g, v in groups.items()}}


def report(result: Run, cfg: Mapping[str, Any]) -> Row:
    overall = summarise(result.rows)
    detections = sum(r["matched"] for r in result.rows) + len(result.phantoms)
    overall.update(frames=len(result.runtimes), detections=detections, false_positives=len(result.phantoms),
                   precision=ratio(detections - len(result.phantoms), detections),
                   nominal_camera_mm=spread([r["nominal_camera_mm"] for r in result.rows]), runtime_ms=spread(result.runtimes),
                   sources={s: result.sources.count(s) for s in sorted(set(result.sources))},
                   believed_absent_arms=result.believed_absent_arms)
    return {"overall": overall, "confusion": confusion(result.rows), "breakdowns": breakdowns(result.rows, cfg),
            "confidence": confidence_table(result.rows, result.phantoms, cfg), "layout": layout_report(result.layouts)}


def layout_report(layouts: Sequence[Row]) -> list[str]:
    """Per layout source: looks, bin centre, counter height and rim error (mm); and how often an episode changed its source."""
    out, episodes = [], {}
    for row in layouts:
        episodes.setdefault(row["seed"], []).append(row["source"])
    changes = {seed: [f"{a}>{b}" for a, b in zip(seq[:-1], seq[1:]) if a != b] for seed, seq in episodes.items()}
    for source in sorted({row["source"] for row in layouts}):
        rows = [row for row in layouts if row["source"] == source]
        bins, counter = spread([v for row in rows for v in row["bins_mm"]]), spread([abs(row["counter_mm"]) for row in rows])
        rim = [row["rim_mm"] for row in rows if row["rim_mm"] is not None]
        out.append(f"layout by {source}: {len(rows)} looks  bins mm median {bins['median']:.1f} p95 {bins['p95']:.1f} max {bins['max']:.1f}  "
                   f"looks with a bin over 30 mm off {sum(max(row['bins_mm']) > 30 for row in rows)}  counter mm median {counter['median']:.1f} "
                   f"max {counter['max']:.1f}  rim mm, signed: " + (f"median {np.median(rim):+.1f} min {min(rim):+.1f} max {max(rim):+.1f}" if rim else "none"))
    out.append("layout source changes inside an episode: " + (", ".join(f"seed {seed}: {' '.join(c)}" for seed, c in changes.items() if c) or "none"))
    return out


def line(label: str, s: Row) -> str:
    """One fixed-width table row of a summary."""
    return (f"{label:<26}{s['objects']:>5}{s['recall']:>8.3f}{s['color_acc']:>8.3f}{s['xy_mm']['median']:>8.1f}"
            f"{s['xy_mm']['p95']:>8.1f}{s['own_xy_mm']['median']:>8.1f}{s['own_xy_mm']['p95']:>8.1f}"
            f"{s['shape_decided']:>8.2f}{s['shape_acc']:>8.3f}{s['yaw_deg']['median']:>8.1f}{s['yaw_deg']['p95']:>8.1f}")


def render(title: str, result: Row) -> str:
    """The report as plain text."""
    o = result["overall"]
    head = (f"{'':<26}{'n':>5}{'recall':>8}{'colour':>8}{'xy med':>8}{'xy p95':>8}{'own med':>8}{'own p95':>8}"
            f"{'shaped':>8}{'shape':>8}{'yaw med':>8}{'yaw p95':>8}")
    out = [title, f"frames {o['frames']}  objects {o['objects']}  detections {o['detections']}  "
           f"recall {o['recall']:.3f}  precision {o['precision']:.3f}  colour {o['color_acc']:.3f}"]
    for key, unit in (("xy_mm", "top xy error, mm"), ("own_xy_mm", "  detector part, mm"), ("camera_mm", "  camera model part, mm"),
                      ("nominal_camera_mm", "nominal camera model, mm"), ("z_mm", "top z error, mm"),
                      ("yaw_deg", "yaw error, deg"), ("runtime_ms", "runtime per frame, ms")):
        s = o[key]
        out.append(f"{unit:<30} mean {s['mean']:6.1f}  median {s['median']:6.1f}  p95 {s['p95']:6.1f}  max {s['max']:6.1f}")
    out.append(f"shape decided {o['shape_decided']:.3f}  correct of decided {o['shape_acc']:.3f}  yaw decided {o['yaw_decided']:.3f}")
    out.append(f"size reported {o['size_decided']:.3f}  median {o['size_mm']['median']:.2f} mm  max {o['size_mm']['max']:.2f} mm")
    out.append(f"prisms lying on a side {o['toppled']}  seen as such {o['toppled_lying']:.3f}  ridge direction, deg: median "
               f"{o['ridge_deg']['median']:.1f} p95 {o['ridge_deg']['p95']:.1f} max {o['ridge_deg']['max']:.1f}  ridge centre xy, mm: median "
               f"{o['ridge_xy_mm']['median']:.1f} max {o['ridge_xy_mm']['max']:.1f}  z median {o['ridge_z_mm']['median']:.1f}  "
               f"called lying without being one {o['false_lying']}")
    out.append("shape confusion (true -> detected): " + "; ".join(
        f"{t}: " + " ".join(f"{d} {n}" for d, n in row.items() if n) for t, row in result["confusion"].items()))
    out.append("layout source per frame: " + ", ".join(f"{s} {n}" for s, n in o["sources"].items())
               + f"; hand fit believed although no arm is in the image: {o['believed_absent_arms']}")
    out += result["layout"]
    c = result["confidence"]
    out.append("confidence from " + " ".join(f"{e:.1f}" for e in c["edges"]) + ": " + "; ".join(
        f"{g} {h}" for g, h in c["histogram"].items()))
    out += [f"  acting at confidence >= {t}: " + ", ".join(f"{g} {n}/{c['totals'][g]}" for g, n in k.items())
            for t, k in c["kept"].items()]
    out += ["", head, line("all", o)]
    for name, groups in result["breakdowns"].items():
        out += [f"by {name}"] + [line("  " + g, s) for g, s in groups.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default=config.load()["perception"]["detector"])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path)
    source.add_argument("--armless", type=int, metavar="LAYOUTS")
    ap.add_argument("--split", choices=SPLITS, default="all")
    ap.add_argument("--looks", type=int, default=1, help="show every frame this many times, 0.2 s apart (a robot standing still)")
    ap.add_argument("--stale", action="store_true", help="show each episode's detector a frame of another episode first")
    ap.add_argument("--repaint", action="store_true", help="turn the objects' hues to the real foam palette (bench.repaint_deg)")
    ap.add_argument("--rotate", type=float, default=0.0, metavar="DEG", help="then turn the hue of the whole image by this much")
    ap.add_argument("--depth", choices=LEVELS, help="deliver the depth image as a RealSense D435i would, at this severity")
    ap.add_argument("--depth-seed", type=int, default=0)
    ap.add_argument("--shift", choices=sorted(config.load()["perception"]["bench"]["shifts"]), help="re-render the table displaced")
    ap.add_argument("--still", action="store_true", help="only frames whose joints stand where the frame before had them")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "perception_bench")
    args = ap.parse_args(argv)

    cfg = config.load()["perception"]["bench"]
    head = Calibration.load().cams[CAMERA]
    nominal = Pinhole(head.K, head.T_pelvis_optical)
    scene, label = Scene.load(), args.frames.name if args.frames else "armless"
    placement = args.frames / "scene.json" if args.frames else None
    there, move = placed(scene, json.loads(placement.read_text())) if placement and placement.exists() else (scene, None)
    frames = armless_frames(scene, nominal, args.armless, cfg) if args.armless else [
        load_frame(p, there, nominal, cfg, move) for p in sorted(args.frames.glob("seed_*.npz"))
        if in_split(int(p.stem.split("_")[1]), args.split, cfg["heldout_every"])]
    frames = [repainted(frame, scene, cfg["repaint_deg"]) for frame in frames] if args.repaint else frames
    frames = [rotated(frame, args.rotate) for frame in frames] if args.rotate else frames
    frames = still(frames, cfg["still_tolerance"]) if args.still else frames
    frames = conditioned(frames, scene, cfg, args.shift, args.depth, args.depth_seed)
    passed = run(args.detector, frames, nominal, cfg["match_radius"], args.looks, args.stale, there.cfg["bins"]["wall_height"])
    result = report(passed, cfg)
    print(render(f"{args.detector} on {label} [{args.split}]", result))
    args.out.mkdir(parents=True, exist_ok=True)
    label = "_".join(filter(None, [label, args.shift, args.depth]))
    (args.out / f"{args.detector}_{label}_{args.split}.json").write_text(json.dumps({**result, "rows": passed.rows, "layouts": passed.layouts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
