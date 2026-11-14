"""Draw the nominal scene (scene.yaml) and the FK hand points through a head calibration onto real head frames (B7).

    python -m halal.eval.real.overlay              -> runs/real_frames/overlay/ep*_first_*.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from aaltonomous.baselines.classical.kinematics import SIDES
from aaltonomous.interface.calibration import Calibration
from halal.eval.real import dataset, extract_frames, kin

SCENE_YAML = dataset.REPO / "aaltonomous/scene/scene.yaml"
OUT = dataset.REPO / "runs/real_frames/overlay"
GRID_BGR, BIN_BGR, RIM_BGR, BAND_BGR = (200, 200, 200), (255, 120, 0), (255, 255, 0), (0, 200, 0)
HAND_BGR = {"left": (0, 0, 255), "right": (255, 0, 255)}


def scene_geometry(path: Path = SCENE_YAML) -> dict:
    """Pelvis-frame polylines of the nominal mat grid, bin footprints, bin rims and object band."""
    cfg = yaml.safe_load(path.read_text())
    ox, oy, z = cfg["counter"]["mat_origin_in_pelvis"]
    width, depth = cfg["counter"]["mat_size"]
    step = cfg["counter"]["grid_cm"] / 100.0

    def pelvis(x_mat: float, y_mat: float, height: float = 0.0) -> list[float]:
        return [ox + y_mat, oy + x_mat, z + height]

    grid = [[pelvis(x, 0), pelvis(x, depth)] for x in np.arange(0, width + 1e-9, step)]
    grid += [[pelvis(0, y), pelvis(width, y)] for y in np.arange(0, depth + 1e-9, step)]
    half = np.array(cfg["bins"]["outer_size"]) / 2
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    wall = cfg["bins"]["wall_height"]
    floors, rims = [], []
    for cx, cy in cfg["bins"]["centers"].values():
        floors.append([pelvis(cx + sx * half[0], cy + sy * half[1]) for sx, sy in corners])
        rims.append([pelvis(cx + sx * half[0], cy + sy * half[1], wall) for sx, sy in corners])
    (x0, x1), (y0, y1) = cfg["layout"]["region_x"], cfg["layout"]["region_y"]
    band = [[pelvis(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]]
    return {"grid": grid, "floors": floors, "rims": rims, "band": band}


def polyline(image: np.ndarray, cal: Calibration, points: list, colour: tuple, thickness: int = 1) -> None:
    """Project a pelvis-frame polyline with dense sampling and draw the part in front of the camera."""
    pts = np.asarray(points, float)
    dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(pts[:-1], pts[1:])])
    uvz = cal.project("head", dense)
    ok = uvz[:, 2] > 0.05
    for (a, b), good in zip(zip(uvz[:-1], uvz[1:]), ok[:-1] & ok[1:]):
        if good:
            cv2.line(image, tuple(np.round(a[:2]).astype(int).tolist()), tuple(np.round(b[:2]).astype(int).tolist()),
                     colour, thickness, cv2.LINE_AA)


def draw_scene(image: np.ndarray, cal: Calibration, geometry: dict) -> None:
    for name, colour, thickness in (("grid", GRID_BGR, 1), ("band", BAND_BGR, 1), ("floors", BIN_BGR, 2), ("rims", RIM_BGR, 2)):
        for line in geometry[name]:
            polyline(image, cal, line, colour, thickness)


def draw_hands(image: np.ndarray, cal: Calibration, hands: dict[str, kin.HandTrack], frame: int) -> None:
    """Palm origin (square) and index / middle / thumb tips (circles, tip names I M T) of both hands."""
    for side in SIDES:
        palm = cal.project("head", hands[side].palm[frame])
        cv2.drawMarker(image, tuple(np.round(palm[:2]).astype(int).tolist()), HAND_BGR[side], cv2.MARKER_SQUARE, 8, 1)
        for tip, letter in zip(cal.project("head", hands[side].tips[frame]), "IMT"):
            centre = tuple(np.round(tip[:2]).astype(int).tolist())
            cv2.circle(image, centre, 4, HAND_BGR[side], 1, cv2.LINE_AA)
            cv2.putText(image, letter, (centre[0] + 5, centre[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HAND_BGR[side], 1)


def render(drop: dataset.Drop, cal: Calibration, stems: dict[int, str], out: Path) -> None:
    """Write overlays for the given global frames of the head camera."""
    geometry = scene_geometry()
    hands = kin.track(kin.kinematics(), drop.state[sorted(stems)])
    for k, (index, rgb) in enumerate(dataset.iter_frames(drop.root, "cam_high", stems)):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        draw_scene(image, cal, geometry)
        draw_hands(image, cal, hands, k)
        path = out / f"{stems[index]}_overlay.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)


def main() -> int:
    drop = dataset.load()
    stems = {i: s for i, s in extract_frames.key_frames(drop).items() if "_last_" not in s}
    render(drop, Calibration.load(), stems, OUT)
    print(f"{len(stems)} overlays written under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
