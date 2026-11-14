"""Calibration stages against ground truth, one first look per frame: counter plane, scene anchor, hand fit and its checks, bins.

    python -m halal.eval.calibration_bench --frames runs/frames/dr_train --split dev [--shift real] [--depth typical]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from halal import config
from halal.eval.depth_sensor import LEVELS
from halal.eval.perception_bench import CAMERA, REPO, SPLITS, Frame, Pinhole, conditioned, in_split, load_frame, spread
from halal.perception import scene_layout
from halal.perception.bins import BinFinder
from halal.perception.calibration import SelfCalibration
from halal.perception.geometry import Camera, PlaneFit, fit_plane

Row = dict[str, Any]


def truly(frame: Frame, points: np.ndarray, bin_name: str | None = None) -> np.ndarray:
    """Scene-file points (N, 3) of the table, or of one bin, where this frame really has them."""
    return points if frame.move is None else frame.move.moved(points, bin_name)


def offset_mm(frame: Frame, used: Camera, points: np.ndarray) -> np.ndarray:
    """How far (mm, per point) `used` puts true points from where they are, imaged by the camera that took the frame."""
    return 1e3 * np.linalg.norm(used.deproject(frame.camera.project(points)) - points, axis=1)


def plane_row(frame: Frame, plane: PlaneFit | None, nominal: Camera, surface: np.ndarray) -> Row:
    """Counter plane fitted through the nominal camera against the true counter (`surface` points) seen through the same camera."""
    if plane is None:
        return {"plane_ok": False}
    seen = nominal.deproject(frame.camera.project(truly(frame, surface)))
    true = fit_plane(seen, (1.0,))
    return {"plane_ok": True, "plane_tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(plane.normal @ true.normal))))),
            "plane_height_mm": 1e3 * float(plane.z_at(seen[-1, :2]) - seen[-1, 2]), "plane_rms_mm": 1e3 * plane.rms}


def hands_row(frame: Frame, calibration: SelfCalibration, nominal: Camera, plane: PlaneFit | None, band: np.ndarray) -> Row:
    """The hand fit of one look: every check by name, whether it is believed, and how far it puts the object band off."""
    obs = frame.obs
    fit = calibration.hands.fit(obs["head_depth"], nominal, obs["arm_q"], obs["hand_q"], plane, calibration.band[-1])
    if fit is None:
        return {"hands_fit": False, "hands_ok": False}
    checks, (turn, shift) = calibration.hand_checks(fit, nominal), calibration.displacement(fit, nominal)
    return {"hands_fit": True, "hands_ok": all(checks.values()), **{f"check_{k}": bool(v) for k, v in checks.items()},
            "hands_rms_mm": 1e3 * fit.rms, "hands_points": fit.points, "hands_spread_mm": 1e3 * float(fit.spread.max()), "hands_coverage": fit.coverage, "hands_known": fit.known, "hands_inliers": fit.inliers,
            "hands_clear": fit.clear / max(fit.visible, 1), "hands_turn_deg": turn, "hands_shift_mm": 1e3 * shift,
            "hands_mm": float(offset_mm(frame, fit.camera, band).mean()), "hands_camera": fit.camera}


def layout_row(frame: Frame, calibration: SelfCalibration, finder: BinFinder, camera: Camera, scene: Scene, surface: np.ndarray) -> Row:
    """Bins by geometry and the counter height under the band through `camera`, against the truth."""
    depth, file = frame.obs["head_depth"], scene_layout(scene)
    counter = calibration.counter(depth, camera)
    found = finder.find(depth, camera, counter)
    true = {name: truly(frame, np.r_[xy, scene.counter_z][None], name)[0] for name, xy in file.bins.items()}
    half = np.asarray(scene.cfg["bins"]["outer_size"])[::-1] / 2
    rims = [truly(frame, np.r_[xy + half * signs, scene.counter_z + scene.cfg["bins"]["wall_height"]][None], name)[0, 2]
            for name, xy in file.bins.items() for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    rim = finder.rim_z(depth, camera, counter, found)
    return {"rim_mm": None if rim is None else 1e3 * float(rim - max(rims)), "bins_y_mm": [1e3 * float(found[name][1] - true[name][1]) for name in file.bins],
            "bins_mm": [1e3 * float(np.linalg.norm(found[name] - true[name][:2])) for name in file.bins],
            "bins_x_mm": [1e3 * float(found[name][0] - true[name][0]) for name in file.bins],
            "counter_mm": 1e3 * float(counter.z_at(surface[-1, :2]) - truly(frame, surface[-1:])[0, 2])}


def measure(frame: Frame, calibration: SelfCalibration, finder: BinFinder, nominal: Camera, scene: Scene) -> Row:
    """Every calibration stage on one frame as a first look would run it."""
    depth, band = frame.obs["head_depth"], truly(frame, calibration.band)
    surface = np.c_[calibration.band[:, :2], np.full(len(calibration.band), scene.counter_z)]
    plane = calibration.counter_fit(depth, nominal)
    anchor = calibration.scene_anchor(depth, nominal)
    row = {"seed": frame.seed, **plane_row(frame, plane, nominal, surface), "nominal_mm": float(offset_mm(frame, nominal, band).mean()),
           "anchor_ok": anchor is not None, **hands_row(frame, calibration, nominal, plane, band)}
    if anchor is not None:
        row["anchor_mm"] = float(offset_mm(frame, anchor, band).mean())
    arms = row.pop("hands_camera", None)
    chosen = calibration.decide(nominal, anchor, arms if row["hands_ok"] else None)
    row.update(source=chosen.source, chosen_mm=float(offset_mm(frame, chosen.camera, band).mean()))
    if anchor is not None and arms is not None:
        row["apart_mm"] = 1e3 * calibration.apart(anchor, arms)
    if arms is not None:
        row.update(layout_row(frame, calibration, finder, arms, scene, surface))
    return row


def rate(rows: Sequence[Row], key: str) -> str:
    return f"{sum(bool(r.get(key)) for r in rows)}/{len(rows)}"


def stats(rows: Sequence[Row], key: str, absolute: bool = True) -> str:
    """Median / p95 / max of a per-frame value (lists are pooled) over the rows that have it."""
    values = np.array([v for r in rows if r.get(key) is not None for v in np.atleast_1d(r[key])], float)
    s = spread(np.abs(values) if absolute else values)
    return f"{s['median']:5.1f} /{s['p95']:5.1f} /{s['max']:5.1f}  (n {len(values)})"


def share(rows: Sequence[Row], key: str) -> str:
    """Median and minimum of a per-frame share."""
    values = [r[key] for r in rows]
    return f"{np.median(values):.2f} (min {min(values):.2f})" if values else "none"


def render(title: str, rows: Sequence[Row]) -> str:
    """The stage table as plain text; errors are median / p95 / max."""
    fitted = [r for r in rows if r["hands_fit"]]
    checks = sorted(k for k in fitted[0] if k.startswith("check_")) if fitted else []
    sources = {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})}
    return "\n".join([
        title,
        f"nominal camera, band mm            {stats(rows, 'nominal_mm')}",
        f"counter plane fitted {rate(rows, 'plane_ok'):>9}   tilt deg {stats(rows, 'plane_tilt_deg')}   height mm {stats(rows, 'plane_height_mm')}"
        f"   rms mm {stats(rows, 'plane_rms_mm')}",
        f"scene anchor accepted {rate(rows, 'anchor_ok'):>8}   band mm {stats(rows, 'anchor_mm')}",
        f"hand fit made {rate(rows, 'hands_fit'):>8}  believed {rate(rows, 'hands_ok'):>8}   band mm (all fits) {stats(fitted, 'hands_mm')}"
        f"   believed only {stats([r for r in fitted if r['hands_ok']], 'hands_mm')}",
        "  checks passed: " + ", ".join(f"{k[6:]} {rate(fitted, k)}" for k in checks)
        + f"   rms mm {stats(fitted, 'hands_rms_mm')}   coverage {share(fitted, 'hands_coverage')}   known {share(fitted, 'hands_known')}   inliers {share(fitted, 'hands_inliers')}"
        + f"   judged clear of the counter {share(fitted, 'hands_clear')}   turn deg {stats(fitted, 'hands_turn_deg')}   shift mm {stats(fitted, 'hands_shift_mm')}",
        f"bins through the hand camera, centre mm {stats(rows, 'bins_mm')}   x only, signed {stats(rows, 'bins_x_mm', False)}"
        f"   frames with a bin more than 3 cm off {sum(max(r['bins_mm']) > 30 for r in rows if 'bins_mm' in r)}",
        f"counter height under the band through the hand camera, mm {stats(rows, 'counter_mm')}",
        f"highest bin rim through the hand camera, mm, signed {stats(rows, 'rim_mm', False)}   worst low {min((r['rim_mm'] for r in rows if r.get('rim_mm') is not None), default=float('nan')):.1f}",
        f"scene anchor and hand fit apart at the band, mm {stats(rows, 'apart_mm')}",
        f"pose a look with both references would use: {sources}   band mm {stats(rows, 'chosen_mm')}"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--split", choices=SPLITS, default="all")
    ap.add_argument("--depth", choices=LEVELS)
    ap.add_argument("--depth-seed", type=int, default=0)
    ap.add_argument("--shift", choices=sorted(config.load()["perception"]["bench"]["shifts"]))
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "calibration_bench")
    args = ap.parse_args(argv)

    settings = config.load()["perception"]
    head = Calibration.load().cams[CAMERA]
    scene, nominal = Scene.load(), Camera.head()
    frames = [load_frame(p, scene, Pinhole(head.K, head.T_pelvis_optical), settings["bench"]) for p in sorted(args.frames.glob("seed_*.npz"))
              if in_split(int(p.stem.split("_")[1]), args.split, settings["bench"]["heldout_every"])]
    frames = conditioned(frames, scene, settings["bench"], args.shift, args.depth, args.depth_seed)
    calibration, finder = SelfCalibration(scene, settings["calibration"]), BinFinder(scene, settings["bins"])
    rows = [measure(frame, calibration, finder, nominal, scene) for frame in frames]
    label = "_".join(filter(None, [args.frames.name, args.shift, args.depth, args.split]))
    print(render(f"calibration stages on {label}, {len(rows)} frames", rows))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{label}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
