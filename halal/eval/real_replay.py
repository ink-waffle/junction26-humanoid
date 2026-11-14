"""A runner export replayed through the perception chain: per look the camera fit with every guard, the layout, the detections.

    python -m halal.eval.real_replay runs/halal/<run> [--out DIR] [--overlays 6] [--episode 0]
    Reads episode_NNN.npz (with images when the run used --log-images, else head RGB from episode_NNN.mp4) and container.log.
    With depth: the detector as the policy runs it, one per episode, at the still moments. Without: colours and the hand outline only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from halal import config
from halal.perception import ObjectDetection
from halal.perception.color import ColorClassifier, chromas
from halal.perception.detectors.topface import TopFaceDetector
from halal.perception.geometry import MM, Camera
from halal.perception.hands import LINKS
from halal.perception.outline import SIDES

Row = dict[str, Any]
HZ = 30
LOG_LINES = {"reset": "reset task=", "look": " look from ", "arms": " arms: ", "camera": "head camera now fixed by",
             "pick": " pick ", "warning": " WARNING ", "error": " ERROR "}


@dataclass(frozen=True)
class Episode:
    """One recorded episode: the per-tick columns, the images when they were logged, the head video otherwise."""
    name: str
    rows: Mapping[str, np.ndarray]
    video: Path | None
    instruction: str

    @property
    def has_depth(self) -> bool:
        return "head_depth" in self.rows

    def rgb(self, tick: int) -> np.ndarray | None:
        """Head RGB of one tick: from the logged images, else that frame of the video; None when there is neither."""
        if "head_rgb" in self.rows:
            return self.rows["head_rgb"][tick]
        if self.video is None:
            return None
        reader = cv2.VideoCapture(str(self.video))
        reader.set(cv2.CAP_PROP_POS_FRAMES, tick)
        ok, frame = reader.read()
        reader.release()
        return frame[..., ::-1].copy() if ok else None

    def obs(self, tick: int) -> Row:
        """The observation of one tick as the policy got it (head camera only)."""
        r = self.rows
        return {"head_rgb": self.rgb(tick), "head_depth": r["head_depth"][tick] if self.has_depth else None, "arm_q": r["arm_q"][tick],
                "hand_q": r["hand_q"][tick], "arm_dq": r["arm_dq"][tick], "t": float(r["t"][tick]), "instruction": self.instruction}


def load_episode(npz: Path) -> Episode:
    data = np.load(npz)
    summary = [json.loads(line) for line in npz.with_suffix(".jsonl").read_text().splitlines() if '"summary"' in line]
    video = npz.with_suffix(".mp4")
    return Episode(npz.stem, {k: data[k] for k in data.files}, video if video.exists() else None,
                   summary[-1]["summary"]["instruction"] if summary else "")


def look_ticks(episode: Episode, c: Mapping[str, float]) -> list[int]:
    """Ticks at which the arms stand where they stood a moment earlier, no closer together than the look interval."""
    joints = np.c_[episode.rows["arm_q"], episode.rows["hand_q"]]
    back, apart = round(c["still_s"] * HZ), round(c["look_interval_s"] * HZ)
    still = np.flatnonzero(np.abs(joints[back:] - joints[:-back]).max(axis=1) < c["still_tolerance"]) + back
    ticks: list[int] = []
    for tick in still:
        if not ticks or tick - ticks[-1] >= apart:
            ticks.append(int(tick))
    return ticks


def arm_model(detector: TopFaceDetector, obs: Row, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Visible points (N, 3) of the posed arm model through `camera`, and the name of each point's link."""
    hands = detector.calibration.hands
    points, normals = hands.surface.posed(obs["arm_q"], obs["hand_q"], True)
    front = hands.facing(points, normals, camera, obs["head_rgb"].shape[:2])
    return points[front], np.array(LINKS)[hands.surface.link[front]]


def arm_returns(depth: np.ndarray, camera: Camera, points: np.ndarray, links: np.ndarray, c: Mapping[str, Any]) -> Row:
    """Per arm part: visible model points, the share whose pixel returned depth, and the share with depth on the model."""
    uvz = camera.project(points)
    seen = depth[np.round(uvz[:, 1]).astype(int), np.round(uvz[:, 0]).astype(int)] * MM
    out = {}
    for part, endings in c["parts"].items():
        own = np.array([name.endswith(tuple(endings)) for name in links], bool)
        if own.any():
            out[part] = {"points": int(own.sum()), "returned": float(np.mean(seen[own] > 0)),
                         "on_model": float(np.mean(np.abs(seen[own] - uvz[own, 2]) < c["on_model"]))}
    return out


def dark_returns(obs: Row, camera: Camera, points: np.ndarray, pad_px: int, dark_value: float) -> float | None:
    """Share of the dark pixels in the arm model's image box that returned depth: needs no camera fit; None without dark pixels."""
    uv = camera.project(points)[:, :2]
    lo, hi = np.maximum(uv.min(axis=0).astype(int) - pad_px, 0), uv.max(axis=0).astype(int) + pad_px
    box = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
    dark = obs["head_rgb"][box].max(axis=-1) < dark_value
    return float(np.mean(obs["head_depth"][box][dark] > 0)) if dark.any() else None


def calibration_row(detector: TopFaceDetector, obs: Row, c: Mapping[str, Any]) -> Row:
    """Every calibration stage of one look with its guard values, run beside the detector."""
    cal, nominal, depth = detector.calibration, detector.nominal, obs["head_depth"]
    plane, anchor = cal.counter_fit(depth, nominal), cal.scene_anchor(depth, nominal)
    fit = cal.hands.fit(depth, nominal, obs["arm_q"], obs["hand_q"], plane, cal.band[-1])
    camera = nominal if fit is None else fit.camera
    points, links = arm_model(detector, obs, camera)
    row = {"valid_depth": float(np.mean(depth > 0)), "plane_rms_mm": None if plane is None else 1e3 * plane.rms,
           "plane_tilt_deg": None if plane is None else float(np.degrees(np.arccos(min(1.0, plane.normal[2])))),
           "anchor": anchor is not None, "hand_fit": fit is not None, "arm_through": "hand fit" if fit else "nominal camera",
           "arm_parts": arm_returns(depth, camera, points, links, c),
           "dark_returned": dark_returns(obs, camera, points, c["box_pad_px"], detector.outline.cfg["dark_value"])}
    if fit is not None:
        checks, (turn, shift) = cal.hand_checks(fit, nominal), cal.displacement(fit, nominal)
        row.update(believed=all(checks.values()), failed=[k for k, ok in checks.items() if not ok], rms_mm=1e3 * fit.rms,
                   coverage=fit.coverage, known=fit.known, inliers=fit.inliers, spread_mm=1e3 * float(fit.spread.max()),
                   clear=fit.clear, visible=fit.visible, turn_deg=turn, moved_mm=1e3 * shift,
                   apart_mm=None if anchor is None else 1e3 * cal.apart(anchor, fit.camera))
    return row


def detection_row(det: ObjectDetection) -> Row:
    return {"color": det.color, "shape": det.shape, "lying": det.lying, "top": np.round(det.top, 4).tolist(), "visible": round(det.visible, 2),
            "confidence": round(det.confidence, 2), "size": det.size, "yaw_deg": None if det.yaw is None else round(float(np.degrees(det.yaw)), 1)}


def depth_look(detector: TopFaceDetector, obs: Row, c: Mapping[str, Any]) -> tuple[Row, list[ObjectDetection]]:
    """One look with depth: the detector as the policy runs it, then the calibration stages on the same frame."""
    detections = detector.detect(obs)
    layout = detector.layout()
    row = {"source": layout.source, "counter_z": layout.counter_z, "rim_z": layout.rim_z,
           "bins": {name: np.round(xy, 4).tolist() for name, xy in layout.bins.items()},
           "detections": [detection_row(det) for det in detections], **calibration_row(detector, obs, c)}
    return row, detections


def colour_row(rgb: np.ndarray, classifier: ColorClassifier, c: Mapping[str, Any]) -> Row:
    """Colour stage without depth: white balance on pale bright pixels, then the blobs of every family with the colour they would be named."""
    pale = (chromas(rgb) < c["table_chroma"]) & (rgb.max(axis=-1) >= c["table_value"])
    gains = classifier.white_balance(rgb[pale])
    classes = classifier.pixel_classes(rgb.reshape(-1, 3), gains).reshape(rgb.shape[:2])
    blobs = []
    for index, family in enumerate(classifier.cfg["basins"]):
        count, labels, stats, centres = cv2.connectedComponentsWithStats((classes == index).astype(np.uint8))
        for k in range(1, count):
            if c["blob_px"][0] <= stats[k, cv2.CC_STAT_AREA] <= c["blob_px"][1]:
                name, confidence = classifier.name(classifier.sample(rgb[labels == k], gains))
                blobs.append({"family": family, "named": name, "confidence": round(confidence, 2), "px": int(stats[k, cv2.CC_STAT_AREA]),
                              "uv": np.round(centres[k]).astype(int).tolist()})
    return {"table_px": int(pale.sum()), "gains": np.round(gains, 3).tolist(), "blobs": blobs}


def rgb_look(detector: TopFaceDetector, obs: Row, c: Mapping[str, Any]) -> Row:
    """One look without depth: the colour stage, and each black hand's outline through the nominal camera."""
    hands = detector.outline.fit(obs["head_rgb"], detector.nominal, obs["arm_q"], obs["hand_q"])
    return {"colours": colour_row(obs["head_rgb"], detector.colors, c), "outline": {side: asdict(o) for side, o in hands.items()}}


def footprint(centre: np.ndarray, size: np.ndarray, z: float) -> np.ndarray:
    """The four corners (4, 3) of a bin footprint at height z."""
    half = size[::-1] / 2
    return np.array([[*(centre + half * signs), z] for signs in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def overlay(detector: TopFaceDetector, obs: Row, row: Row, detections: Sequence[ObjectDetection]) -> np.ndarray:
    """Side by side: RGB with bins (counter and rim height), detections and arm model; depth with the arm points by what returned."""
    camera, depth = detector.pose.camera, obs["head_depth"]
    left = obs["head_rgb"][..., ::-1].copy()
    right = cv2.applyColorMap(np.clip((depth.astype(float) - 250) / 500 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_BONE)
    right[depth == 0] = (60, 0, 60)
    for xy in row["bins"].values():
        for z, colour in ((row["counter_z"], (255, 255, 0)), (row["rim_z"], (0, 255, 255))):
            if z is not None:
                corners = camera.project(footprint(np.asarray(xy), detector.bins.size, z))[:, :2]
                cv2.polylines(left, [np.round(corners).astype(np.int32)], True, colour, 1)
    for det in detections:
        u, v = np.round(camera.project(det.top[None])[0, :2]).astype(int)
        cv2.drawMarker(left, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(left, f"{det.color} {det.shape or '?'}{' lying' * det.lying} {det.visible:.1f}", (u + 8, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    points, _ = arm_model(detector, obs, camera)
    uvz = camera.project(points)
    for (u, v), z in zip(np.round(uvz[:, :2]).astype(int), uvz[:, 2]):
        seen = depth[v, u] * MM
        colour = (0, 0, 255) if seen == 0 else (0, 255, 0) if abs(seen - z) < 0.008 else (0, 255, 255)
        cv2.circle(right, (u, v), 1, colour, -1)
        cv2.circle(left, (u, v), 1, (0, 255, 0), -1)
    cv2.putText(right, f"camera by {row['source']}; arm model: green on depth, yellow depth elsewhere, red none", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.hstack([left, right])


def log_digest(path: Path) -> Row:
    """What our server wrote in container.log: counts per kind of line, the layout sources of its looks, its hand fit lines."""
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    kinds = {kind: [line for line in lines if marker in line] for kind, marker in LOG_LINES.items()}
    sources = re.findall(r"layout from (\w+)", "\n".join(kinds["look"]))
    return {"lines": len(lines), "counts": {kind: len(found) for kind, found in kinds.items()},
            "look_sources": {s: sources.count(s) for s in sorted(set(sources))},
            "arms": [line.split(" arms: ", 1)[1] for line in kinds["arms"]], "camera": [line.split("topface ", 1)[-1] for line in kinds["camera"]],
            "problems": (kinds["warning"] + kinds["error"])[-10:]}


def median(rows: Sequence[Row], key: str) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def depth_summary(rows: Sequence[Row], c: Mapping[str, Any]) -> list[str]:
    """What the looks with depth say together: sources, guards of the hand fit, and which depth level the arms resemble."""
    fits = [row for row in rows if row["hand_fit"]]
    sources = {s: sum(row["source"] == s for row in rows) for s in sorted({row["source"] for row in rows})}
    failed = sorted({name for row in fits for name in row["failed"]})
    out = [f"looks {len(rows)}  camera by {sources}  scene anchor fitted {sum(row['anchor'] for row in rows)}  hand fit made {len(fits)}, "
           f"believed {sum(row['believed'] for row in fits)}" + (f", checks failed somewhere: {failed}" if failed else ""),
           f"counter plane rms {median(rows, 'plane_rms_mm'):.1f} mm (sim 1.5, model typical 1.2, bad 1.6), tilt against the pelvis "
           f"{median(rows, 'plane_tilt_deg'):.1f} deg; valid depth {median(rows, 'valid_depth'):.2f} of the image",
           f"hand fit medians: rms {median(fits, 'rms_mm'):.1f} mm (sim 1.7, limit 4), covered {median(fits, 'coverage'):.2f} (limit 0.7) of the "
           f"{median(fits, 'known'):.2f} that returned depth, pixels on model {median(fits, 'inliers'):.2f} (limit 0.5), camera turned "
           f"{median(fits, 'turn_deg'):.1f} deg and moved {median(fits, 'moved_mm'):.1f} mm, apart from the scene anchor "
           f"{median(fits, 'apart_mm'):.1f} mm (nan: no look had both)"]
    for part in c["parts"]:
        seen = [row["arm_parts"][part] for row in rows if part in row["arm_parts"]]
        if seen:
            out.append(f"  {part:<8} visible points {np.median([s['points'] for s in seen]):.0f}  returned depth "
                       f"{np.median([s['returned'] for s in seen]):.2f}  depth on the model {np.median([s['on_model'] for s in seen]):.2f}")
    dark = median(rows, "dark_returned")
    level = "no dark pixels" if np.isnan(dark) else next(name for name, least in c["levels"].items() if dark >= least)
    out.append(f"dark pixels around the arms that returned depth: {dark:.2f} -> resembles depth level '{level}' "
               f"(model: black links return 0.8 mild, 0.4 typical, 0 bad)")
    return out


def rgb_summary(rows: Sequence[Row]) -> list[str]:
    """What the looks without depth say together: each black hand's outline, and the colours named."""
    named = [blob["named"] for row in rows for blob in row["colours"]["blobs"] if blob["confidence"] >= 0.5]
    out = [f"looks {len(rows)} (RGB only)  blobs named at confidence >= 0.5: " + str({n: named.count(n) for n in sorted(set(named))})]
    for side in SIDES:
        seen = [row["outline"][side] for row in rows if side in row["outline"]]
        if seen:
            shifts = np.array([o["shift"] for o in seen])
            out.append(f"{side} hand through the nominal camera ({len(seen)} looks): {np.median([o['on_dark'] for o in seen]):.2f} on dark pixels, "
                       f"{np.median([o['residual_px'] for o in seen]):.1f} px away; after the best image shift (median {np.median(shifts[:, 0]):+.0f}, "
                       f"{np.median(shifts[:, 1]):+.0f} px, spread {shifts.std(axis=0).round(1).tolist()}): "
                       f"{np.median([o['shifted_on_dark'] for o in seen]):.2f} on dark, {np.median([o['shifted_px'] for o in seen]):.1f} px")
    return out


def replay(episode: Episode, c: Mapping[str, Any], out: Path, overlays: int) -> Row:
    """Every look of one episode through the chain it has data for; the first few with depth are drawn."""
    detector, deep, flat = TopFaceDetector(), [], []
    for tick in look_ticks(episode, c):
        obs = episode.obs(tick)
        if obs["head_rgb"] is None or not obs["head_rgb"].any():
            continue
        if episode.has_depth and obs["head_depth"].any():
            row, detections = depth_look(detector, obs, c)
            if len(deep) < overlays:
                cv2.imwrite(str(out / f"{episode.name}_tick_{tick:05d}.png"), overlay(detector, obs, row, detections))
            deep.append({"tick": tick, "t": obs["t"], **row})
        else:
            flat.append({"tick": tick, "t": obs["t"], **rgb_look(detector, obs, c)})
    lines = (depth_summary(deep, c) if deep else []) + (rgb_summary(flat) if flat else []) or ["no look with a head image"]
    return {"episode": episode.name, "instruction": episode.instruction, "summary": lines, "looks": deep, "rgb_looks": flat}


def look_line(row: Row) -> str:
    """One look with depth as a line of the report."""
    seen = ", ".join(f"{d['color']}/{d['shape'] or '?'}{'/lying' * d['lying']}({d['top'][0]:+.3f},{d['top'][1]:+.3f};v{d['visible']})" for d in row["detections"])
    fit = (f"hand fit rms {row['rms_mm']:.1f} cov {row['coverage']:.2f} known {row['known']:.2f} (of {row['clear']}/{row['visible']} clear) "
           f"on-model {row['inliers']:.2f} turn {row['turn_deg']:.1f} deg shift {row['moved_mm']:.0f} mm "
           f"{'believed' if row['believed'] else 'REJECTED ' + ','.join(row['failed'])}") if row["hand_fit"] else "no hand fit"
    bins = " ".join(f"{name[0]}({xy[0]:.3f},{xy[1]:+.3f})" for name, xy in row["bins"].items())
    rim = "none" if row["rim_z"] is None else f"{row['rim_z']:.3f}"
    return (f"  t={row['t']:6.1f} by {row['source']:<7} anchor {'fits' if row['anchor'] else 'no  '} {fit}; counter {row['counter_z']:.3f} "
            f"rim {rim} bins {bins}; {seen or 'nothing seen'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a runner run folder: runs/<team>/<run id>")
    ap.add_argument("--out", type=Path, help="where the overlays and report.json go (default: <run>/replay)")
    ap.add_argument("--overlays", type=int, default=6, help="looks drawn per episode")
    ap.add_argument("--episode", type=int, help="only this episode")
    args = ap.parse_args(argv)
    c = config.load()["perception"]["replay"]
    out = args.out or args.run / "replay"
    out.mkdir(parents=True, exist_ok=True)
    digest = log_digest(args.run / "container.log")
    print(f"container.log: {digest['lines']} lines, {digest['counts']}, looks by layout source {digest['look_sources']}")
    print("\n".join(["  server " + line for line in digest["camera"][:6]] + ["  server arms: " + line for line in digest["arms"][:6]]
                    + ["  server problem: " + line[:200] for line in digest["problems"]]))
    results = []
    for npz in sorted(args.run.glob("episode_*.npz")):
        if args.episode is not None and int(npz.stem.split("_")[1]) != args.episode:
            continue
        episode = load_episode(npz)
        result = replay(episode, c, out, args.overlays)
        results.append(result)
        print(f"\n{episode.name}: {len(episode.rows['t'])} ticks, {'images logged' if episode.has_depth else 'no images in the npz'}, "
              f"{'video' if episode.video else 'no video'}, instruction {episode.instruction!r}")
        print("\n".join(result["summary"] + [look_line(row) for row in result["looks"]]))
    (out / "report.json").write_text(json.dumps({"log": digest, "episodes": results}, indent=1))
    print(f"\noverlays and report.json: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
