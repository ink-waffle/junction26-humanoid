"""Perception benchmark: one detector against privileged ground truth over a collected frame set.

    python -m halal.eval.perception_bench --detector baseline --frames runs/frames/dr_train [--split heldout]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene, shape_polygon

from halal import config
from halal.perception import SHAPES, Detector, ObjectDetection, make_detector

REPO = Path(__file__).resolve().parents[2]
CAMERA = "head"
SYMMETRY = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3}
APEX_OFFSET = {"cube": 0.0, "triangle": np.pi / 2}
SPLITS = ("all", "dev", "heldout")
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


@dataclass(frozen=True)
class Frame:
    """One recorded observation with its ground truth and the camera that really took it."""
    seed: int
    obs: dict[str, Any]
    truths: list[Truth]
    camera: Pinhole
    dr: dict[str, Any]


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


def on_counter(scene: Scene, centre: np.ndarray, cfg: Mapping[str, float]) -> bool:
    """True for an object resting on the counter outside every bin."""
    mat = scene.pelvis_to_mat(centre)
    resting = abs(mat[2] - scene.object_height() / 2) < cfg["resting_tolerance"]
    return bool(resting and scene.on_counter(mat) and scene.bin_of(mat) is None)


def make_truth(obj: Mapping[str, Any], scene: Scene, camera: Pinhole, depth: np.ndarray, cfg: Mapping[str, float]) -> Truth:
    centre = np.asarray(obj["pos"], float)
    top = centre + [0.0, 0.0, scene.object_height() / 2]
    yaw = body_yaw(obj["quat_xyzw"])
    view = view_class(top_face_points(scene, obj["shape"], top, yaw), camera, depth, cfg)
    apex = yaw + APEX_OFFSET[obj["shape"]] if obj["shape"] in APEX_OFFSET else None
    return Truth(obj["name"], obj["color"], obj["shape"], top, apex, view)


def load_frame(path: Path, scene: Scene, nominal: Pinhole, cfg: Mapping[str, Any]) -> Frame:
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    obs = {k: data[k] for k in data.files if k != "meta"}
    obs.update(arm_dq=np.zeros(14, np.float32), instruction=meta["instruction"], t=0.0)
    dr = meta["dr"] or {}
    camera = sim_camera(nominal, dr.get("camera", {}), cfg)
    truths = [make_truth(o, scene, camera, obs["head_depth"], cfg) for o in meta["objects"]
              if on_counter(scene, np.asarray(o["pos"], float), cfg)]
    return Frame(int(meta["seed"]), obs, truths, camera, dr)


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
    return {"seed": frame.seed, "color": truth.color, "shape": truth.shape, "view": truth.view,
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
    has_yaw = shape_ok and det.yaw is not None and truth.yaw is not None
    return {"matched": True, "color_ok": det.color == truth.color, "confidence": float(det.confidence),
            "xy_mm": 1e3 * float(np.linalg.norm(error[:2])), "z_mm": 1e3 * float(abs(error[2])),
            "own_xy_mm": 1e3 * float(np.linalg.norm(own[:2])),
            "camera_mm": 1e3 * float(np.linalg.norm((error - own)[:2])), "det_shape": det.shape,
            "yaw_deg": float(np.degrees(yaw_error(truth.shape, det.yaw, truth.yaw))) if has_yaw else None}


def score_frame(frame: Frame, dets: Sequence[ObjectDetection], nominal: Pinhole, used: Pinhole,
                radius: float) -> tuple[list[Row], int]:
    """One row per ground-truth object plus the number of detections that match none."""
    pairs = match(dets, frame.truths, radius)
    rows = [{**factors(frame, t, nominal), **outcome(frame, t, dets[pairs[i]] if i in pairs else None, used)}
            for i, t in enumerate(frame.truths)]
    return rows, len(dets) - len(pairs)


def run(detector_name: str, frames: Sequence[Frame], nominal: Pinhole, radius: float) -> tuple[list[Row], int, list[float]]:
    """Detect every frame with a fresh detector (one frame = one episode); rows, false positives, runtimes in ms."""
    rows, false_positives, runtimes = [], 0, []
    for frame in frames:
        detector: Detector = make_detector(detector_name)
        start = time.perf_counter()
        dets = detector.detect(frame.obs)
        runtimes.append(1e3 * (time.perf_counter() - start))
        used = Pinhole(nominal.K, getattr(detector, "extrinsics", nominal.T))
        frame_rows, extra = score_frame(frame, dets, nominal, used, radius)
        rows += frame_rows
        false_positives += extra
    return rows, false_positives, runtimes


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
    return {"objects": len(rows), "recall": ratio(len(hit), len(rows)),
            "color_acc": ratio(sum(r["color_ok"] for r in hit), len(hit)),
            "xy_mm": spread([r["xy_mm"] for r in hit]), "z_mm": spread([r["z_mm"] for r in hit]),
            "own_xy_mm": spread([r["own_xy_mm"] for r in hit]), "camera_mm": spread([r["camera_mm"] for r in hit]),
            "shape_decided": ratio(len(shaped), len(hit)),
            "shape_acc": ratio(sum(r["det_shape"] == r["shape"] for r in shaped), len(shaped)),
            "yaw_decided": ratio(len(turned), len(wanted)), "yaw_deg": spread(turned)}


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


def report(rows: Sequence[Row], false_positives: int, runtimes: Sequence[float], cfg: Mapping[str, Any]) -> Row:
    overall = summarise(rows)
    detections = sum(r["matched"] for r in rows) + false_positives
    overall.update(frames=len(runtimes), detections=detections, false_positives=false_positives,
                   precision=ratio(detections - false_positives, detections),
                   nominal_camera_mm=spread([r["nominal_camera_mm"] for r in rows]), runtime_ms=spread(runtimes))
    return {"overall": overall, "confusion": confusion(rows), "breakdowns": breakdowns(rows, cfg)}


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
    out.append("shape confusion (true -> detected): " + "; ".join(
        f"{t}: " + " ".join(f"{d} {n}" for d, n in row.items() if n) for t, row in result["confusion"].items()))
    out += ["", head, line("all", o)]
    for name, groups in result["breakdowns"].items():
        out += [f"by {name}"] + [line("  " + g, s) for g, s in groups.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default=config.load()["perception"]["detector"])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--split", choices=SPLITS, default="all")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "perception_bench")
    args = ap.parse_args(argv)

    cfg = config.load()["perception"]["bench"]
    head = Calibration.load().cams[CAMERA]
    nominal = Pinhole(head.K, head.T_pelvis_optical)
    frames = [load_frame(p, Scene.load(), nominal, cfg) for p in sorted(args.frames.glob("seed_*.npz"))
              if in_split(int(p.stem.split("_")[1]), args.split, cfg["heldout_every"])]
    rows, false_positives, runtimes = run(args.detector, frames, nominal, cfg["match_radius"])
    result = report(rows, false_positives, runtimes, cfg)
    print(render(f"{args.detector} on {args.frames.name} [{args.split}]", result))
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.detector}_{args.frames.name}_{args.split}.json"
    (args.out / name).write_text(json.dumps({**result, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
