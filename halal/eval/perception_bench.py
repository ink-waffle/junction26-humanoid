"""Perception benchmark: one detector against privileged ground truth over a collected frame set.

    python -m halal.eval.perception_bench --detector baseline --frames runs/frames/dr_train [--split heldout]
    python -m halal.eval.perception_bench --armless 10      rendered scenes without arms: the hand fit must not be believed
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

from aaltonomous.baselines.classical.mock_world import Renderer
from aaltonomous.interface import tasks
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
    tick: int
    obs: dict[str, Any]
    truths: list[Truth]
    ignored: list[np.ndarray]   # xy of objects that are neither lying still on the counter nor clearly off it
    camera: Pinhole
    dr: dict[str, Any]
    arms: bool = True           # False: the image shows no robot arm, whatever the joint angles say


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
    obs.update(arm_dq=np.zeros(14, np.float32), instruction=meta["instruction"], t=float(meta.get("t", 0.0)))
    dr = meta["dr"] or {}
    camera = sim_camera(nominal, dr.get("camera", {}), cfg)
    state = [presence(scene, o, cfg) for o in meta["objects"]]
    truths = [make_truth(o, scene, camera, obs["head_depth"], cfg) for o, s in zip(meta["objects"], state) if s == "lying"]
    ignored = [np.asarray(o["pos"][:2], float) for o, s in zip(meta["objects"], state) if s == "unclear"]
    return Frame(int(meta["seed"]), int(meta.get("tick", 0)), obs, truths, ignored, camera, dr)


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
    believed_absent_arms: int     # frames without arms in the image whose layout nevertheless came from the hands


def run(detector_name: str, frames: Sequence[Frame], nominal: Pinhole, radius: float) -> Run:
    """One detector per seed (= one episode), its frames in recording order, as the policy would look."""
    out = Run([], [], [], [], 0)
    detectors: dict[int, Detector] = {}
    for frame in sorted(frames, key=lambda f: (f.seed, f.tick)):
        detector = detectors.setdefault(frame.seed, make_detector(detector_name))
        start = time.perf_counter()
        dets = detector.detect(frame.obs)
        out.runtimes.append(1e3 * (time.perf_counter() - start))
        used = Pinhole(nominal.K, getattr(detector, "extrinsics", nominal.T))
        rows, phantoms = score_frame(frame, dets, nominal, used, radius)
        out.rows += rows
        out.phantoms += phantoms
        out.sources.append(detector.layout().source)
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
            "confidence": confidence_table(result.rows, result.phantoms, cfg)}


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
    out.append("layout source per frame: " + ", ".join(f"{s} {n}" for s, n in o["sources"].items())
               + f"; hand fit believed although no arm is in the image: {o['believed_absent_arms']}")
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
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "perception_bench")
    args = ap.parse_args(argv)

    cfg = config.load()["perception"]["bench"]
    head = Calibration.load().cams[CAMERA]
    nominal = Pinhole(head.K, head.T_pelvis_optical)
    scene, label = Scene.load(), args.frames.name if args.frames else "armless"
    frames = armless_frames(scene, nominal, args.armless, cfg) if args.armless else [
        load_frame(p, scene, nominal, cfg) for p in sorted(args.frames.glob("seed_*.npz"))
        if in_split(int(p.stem.split("_")[1]), args.split, cfg["heldout_every"])]
    passed = run(args.detector, frames, nominal, cfg["match_radius"])
    result = report(passed, cfg)
    print(render(f"{args.detector} on {label} [{args.split}]", result))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.detector}_{label}_{args.split}.json").write_text(json.dumps({**result, "rows": passed.rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
