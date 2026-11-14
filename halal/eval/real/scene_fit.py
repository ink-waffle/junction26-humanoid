"""Where the real bins and the table are, from the centre bin's hand-read corners (question B7, scene part).

    python -m halal.eval.real.scene_fit    -> camera pitch against the table (PnP on the rim), bin distance, table height
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml

from aaltonomous.interface.calibration import Calibration
from halal.eval.real import dataset, head_fit, labels, overlay

PIXEL_NOISE = 3.0
NOISE_DRAWS = 300
HEIGHTS = np.linspace(-0.25, 0.20, 901)
HAND_FIT = dataset.REPO / "runs/real_frames/cache/head_fit_all.npy"


def rim_pitch(intrinsics: np.ndarray, corners: np.ndarray, side: float) -> float:
    """Angle (deg) of the optical axis below the plane of a square rim seen at `corners` (BL, BR, FR, FL)."""
    half = side / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], float)
    _, rvec, _ = cv2.solvePnP(square, corners.astype(float), intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    normal = cv2.Rodrigues(rvec)[0][:, 2]
    return 90.0 - float(np.degrees(np.arccos(abs(normal[2]))))


def ray_to_height(cal: Calibration, pixel: tuple[int, int], height: float) -> np.ndarray:
    """Pelvis-frame point where the head ray through `pixel` meets the plane z = height."""
    camera = cal.cams["head"]
    direction = camera.T_pelvis_optical[:3, :3] @ np.array([(pixel[0] - camera.K[0, 2]) / camera.K[0, 0],
                                                            (pixel[1] - camera.K[1, 2]) / camera.K[1, 1], 1.0])
    origin = camera.T_pelvis_optical[:3, 3]
    return origin + (height - origin[2]) / direction[2] * direction


def foot_fit(cal: Calibration, corners: tuple[tuple[int, int], tuple[int, int]], side: float) -> tuple[float, np.ndarray]:
    """Table height at which the two bottom corners are `side` apart, and the mid-point of the near edge there."""
    widths = np.array([np.linalg.norm(ray_to_height(cal, corners[0], z) - ray_to_height(cal, corners[1], z)) for z in HEIGHTS])
    height = float(HEIGHTS[np.argmin(np.abs(widths - side))])
    return height, 0.5 * (ray_to_height(cal, corners[0], height) + ray_to_height(cal, corners[1], height))


def main() -> int:
    scene = yaml.safe_load(overlay.SCENE_YAML.read_text())
    side = float(scene["bins"]["outer_size"][0])
    nominal = Calibration.load()
    cameras = {"nominal calibration": nominal}
    if HAND_FIT.exists():
        cameras["hand-fitted calibration"] = head_fit.corrected(nominal, np.load(HAND_FIT))
    pitch_nominal = float(np.degrees(np.arcsin(-nominal.cams["head"].T_pelvis_optical[2, 2])))
    rng = np.random.default_rng(0)
    intrinsics = nominal.cams["head"].K
    print(f"== optical axis below the table plane, centre bin rim as a {side} m square (nominal camera pitch {pitch_nominal:.1f} deg)")
    for ep, corners in labels.CENTRE_BIN_RIM_PX.items():
        corners = np.array(corners, float)
        noisy = [corners + rng.normal(0, PIXEL_NOISE, corners.shape) for _ in range(NOISE_DRAWS)]
        spread = np.std([rim_pitch(intrinsics, c, side) for c in noisy])
        print(f"   ep{ep:2d}: {rim_pitch(intrinsics, corners, side):.1f} deg (sd {spread:.1f} for {PIXEL_NOISE:.0f} px reading noise)")
    origin = scene["counter"]["mat_origin_in_pelvis"]
    near_x = origin[0] + scene["bins"]["centers"]["center"][1] - side / 2
    print(f"== near edge of the centre bin and table height if the bin is {side} m wide (nominal: x {near_x:.3f}, z {origin[2]:+.3f})")
    for name, cal in cameras.items():
        for ep, corners in labels.CENTRE_BIN_FOOT_PX.items():
            height, mid = foot_fit(cal, corners, side)
            print(f"   {name:24s} ep{ep:2d}: near edge x {mid[0]:.3f} y {mid[1]:+.3f}, table z {height:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
