"""Fit the head camera pose to the robot's own hands: FK hand points must fall on the dark hand pixels (question B7).

    python -m halal.eval.real.head_fit             -> prints the correction per episode group, writes overlay PNGs
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from aaltonomous.baselines.classical.kinematics import FINGER_LINKS, SIDES, TIP, ArmKinematics
from aaltonomous.interface.calibration import Calibration
from halal.eval.real import dataset, extract_frames, kin, overlay

GROUPS = {"ep0-9": range(0, 10), "ep10-13": range(10, 14), "ep14-16": range(14, 17), "ep17-19": range(17, 20)}
FRAME_STEP = 12
DARK_VALUE = 70
CHAINS = (("hand_palm_link", "hand_index_0_link", "hand_index_1_link"),
          ("hand_palm_link", "hand_middle_0_link", "hand_middle_1_link"),
          ("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link"))
SAMPLES_PER_SEGMENT = 4
BEYOND_TIP = 0.015
CLIP_PX = 40.0
OUTSIDE_WEIGHT = 2.0
MARGIN_PX = 40
MAX_ROTATION = np.radians(12.0)
MAX_TRANSLATION = 0.08
OUT = dataset.REPO / "runs/real_frames/overlay_fitted"


@dataclass(frozen=True)
class FitData:
    """Per-frame distance maps and pelvis-frame hand points of the frames used for one fit."""
    to_dark: np.ndarray
    to_light: np.ndarray
    inside: np.ndarray
    beyond: np.ndarray
    use_inside: np.ndarray
    use_beyond: np.ndarray


def hand_points(kinematics: ArmKinematics, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-frame points on the finger axes (T,N,3) and just beyond every fingertip (T,6,3) for both hands."""
    inside, beyond = [], []
    weights = np.linspace(0.0, 1.0, SAMPLES_PER_SEGMENT, endpoint=False)
    for row in joints:
        kinematics._fk(row[:dataset.N_ARM], row[dataset.N_ARM:])
        pts, out = [], []
        for side in SIDES:
            tips = kinematics._tips(side)
            for chain, link, tip, tip_local in zip(CHAINS, FINGER_LINKS, tips[[0, 1, 2]], TIP[side]):
                nodes = [kinematics.data.oMf[kinematics.model.getFrameId(f"{side}_{name}")].translation.copy() for name in chain]
                nodes.append(tip)
                for a, b in zip(nodes[:-1], nodes[1:]):
                    pts += [a + w * (b - a) for w in weights]
                pts.append(tip)
                rotation = kinematics.data.oMf[kinematics.model.getFrameId(f"{side}_{link}")].rotation
                direction = rotation @ (np.asarray(tip_local, float) / np.linalg.norm(tip_local))
                out.append(tip + BEYOND_TIP * direction)
        inside.append(pts)
        beyond.append(out)
    return np.array(inside), np.array(beyond)


def distance_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel distance to the nearest dark pixel and to the nearest non-dark pixel, clipped to uint8."""
    dark = (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2] < DARK_VALUE).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    to_dark = cv2.distanceTransform(1 - dark, cv2.DIST_L2, 3)
    to_light = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    return np.clip(to_dark, 0, 255).astype(np.uint8), np.clip(to_light, 0, 255).astype(np.uint8)


def collect(drop: dataset.Drop, kinematics: ArmKinematics, cal: Calibration, episodes: range) -> FitData:
    """Distance maps and hand points for every FRAME_STEP-th frame; points count if nominally well inside the image."""
    indices = [i for ep in episodes for i in range(drop.span(ep).start, drop.span(ep).stop, FRAME_STEP)]
    maps = [distance_maps(rgb) for _, rgb in dataset.iter_frames(drop.root, "cam_high", indices)]
    to_dark, to_light = np.array([m[0] for m in maps]), np.array([m[1] for m in maps])
    inside, beyond = hand_points(kinematics, drop.state[indices])
    return FitData(to_dark, to_light, inside, beyond, in_image(cal.project("head", inside), to_dark.shape, MARGIN_PX),
                   in_image(cal.project("head", beyond), to_dark.shape, MARGIN_PX))


def corrected(cal: Calibration, xi: np.ndarray) -> Calibration:
    """Copy of the calibration with the head pose moved by xi = (rotation vector rad, translation m) in the optical frame."""
    out = Calibration(cal.cfg)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
    delta[:3, 3] = xi[3:]
    out.cams["head"].T_pelvis_optical = cal.cams["head"].T_pelvis_optical @ delta
    return out


def in_image(uvz: np.ndarray, shape: tuple[int, ...], margin: int = 0) -> np.ndarray:
    """Points in front of the camera whose projection lies at least `margin` pixels inside the image."""
    with np.errstate(invalid="ignore"):
        return ((uvz[..., 2] > 0.1) & (uvz[..., 0] >= margin) & (uvz[..., 0] < shape[2] - margin)
                & (uvz[..., 1] >= margin) & (uvz[..., 1] < shape[1] - margin))


def lookup(maps: np.ndarray, uvz: np.ndarray, wanted: np.ndarray, missing: float) -> np.ndarray:
    """Map values at the wanted projected points (T,N,3); points that left the image get `missing`."""
    ok = wanted & in_image(uvz, maps.shape)
    values = np.full(int(wanted.sum()), missing)
    u, v = uvz[ok][:, 0].astype(int), uvz[ok][:, 1].astype(int)
    frame = np.broadcast_to(np.arange(len(maps))[:, None], wanted.shape)[ok]
    values[ok[wanted]] = maps[frame, v, u]
    return values


def cost(xi: np.ndarray, cal: Calibration, data: FitData) -> float:
    """Mean clipped distance of finger-axis points to dark pixels plus a penalty for beyond-tip points on dark pixels."""
    moved = corrected(cal, xi)
    on_hand = lookup(data.to_dark, moved.project("head", data.inside), data.use_inside, CLIP_PX)
    past_tip = lookup(data.to_light, moved.project("head", data.beyond), data.use_beyond, CLIP_PX / 4)
    return float(np.minimum(on_hand, CLIP_PX).mean() + OUTSIDE_WEIGHT * np.minimum(past_tip, CLIP_PX / 4).mean())


def fit(cal: Calibration, data: FitData) -> np.ndarray:
    """Best pose correction xi from a coarse pitch/height grid followed by bounded Powell on all six parameters."""
    starts = [np.array([rx, 0, 0, 0, ty, 0.0]) for rx in np.radians([-8, -4, 0, 4, 8]) for ty in (-0.04, 0.0, 0.04)]
    best = min(starts, key=lambda xi: cost(xi, cal, data))
    bounds = [(-MAX_ROTATION, MAX_ROTATION)] * 3 + [(-MAX_TRANSLATION, MAX_TRANSLATION)] * 3
    result = minimize(cost, best, args=(cal, data), method="Powell", bounds=bounds,
                      options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 4000})
    return result.x


def describe(xi: np.ndarray) -> str:
    angles = np.degrees(xi[:3])
    return (f"rotation about optical x/y/z {angles[0]:+.1f} / {angles[1]:+.1f} / {angles[2]:+.1f} deg, "
            f"translation x/y/z {100 * xi[3]:+.1f} / {100 * xi[4]:+.1f} / {100 * xi[5]:+.1f} cm")


def pixel_shift(cal: Calibration, xi: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Median image shift (du, dv) of the hand points caused by the correction."""
    before, after = cal.project("head", points), corrected(cal, xi).project("head", points)
    return np.median((after - before)[..., :2].reshape(-1, 2), axis=0)


def main() -> int:
    drop = dataset.load()
    kinematics = kin.kinematics()
    cal = Calibration.load()
    fits = {}
    for name, episodes in {"all": range(drop.n_episodes), **GROUPS}.items():
        data = collect(drop, kinematics, cal, episodes)
        xi = fit(cal, data)
        fits[name] = xi
        shift = pixel_shift(cal, xi, data.inside)
        print(f"{name:8s} frames {len(data.inside):4d} cost {cost(np.zeros(6), cal, data):.2f} -> {cost(xi, cal, data):.2f} px | "
              f"{describe(xi)} | median hand shift du {shift[0]:+.0f} dv {shift[1]:+.0f} px")
    np.save(OUT.parent / "cache/head_fit_all.npy", fits["all"])
    stems = {i: s for i, s in extract_frames.key_frames(drop).items() if "_last_" not in s}
    overlay.render(drop, corrected(cal, fits["all"]), stems, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
