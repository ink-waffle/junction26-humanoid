"""Frame timing of a drop: repeated camera frames and how far the images lag the joint state (question C9).

    python -m halal.eval.real.timing       -> prints repeat rates and the image-vs-joint lag per camera
"""
from __future__ import annotations

import numpy as np

from aaltonomous.interface.calibration import Calibration
from halal.eval.real import dataset, finger_signal, kin, video_features

REPEAT_CHANGE = 0.15
ARM_MOVING = 0.3
LINEAR_WEIGHT = 2.0


def angular_speed(rotation: np.ndarray) -> np.ndarray:
    """Rotation rate (rad/s) between consecutive frames of a (T,3,3) rotation track."""
    relative = np.einsum("tij,tkj->tik", rotation[1:], rotation[:-1])
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    return np.r_[0.0, angle * dataset.FPS]


def motion_signals(drop: dataset.Drop) -> dict[str, np.ndarray]:
    """Per camera a joint-derived motion signal: projected fingertip speed (head), wrist speed (wrist cameras)."""
    hands = kin.track(kin.kinematics(), drop.state)
    pixels = Calibration.load().project("head", hands["left"].tips[:, 0])[:, :2]
    out = {"cam_high": np.r_[0.0, np.linalg.norm(np.diff(pixels, axis=0), axis=1)]}
    for camera, side in (("cam_left_wrist", "left"), ("cam_right_wrist", "right")):
        linear = np.r_[0.0, np.linalg.norm(np.diff(hands[side].palm, axis=0), axis=1) * dataset.FPS]
        out[camera] = angular_speed(hands[side].rotation) + LINEAR_WEIGHT * linear
    return out


def main() -> int:
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    drop = dataset.load()
    features = video_features.load()
    first = np.zeros(len(drop.state), bool)
    first[drop.starts[:-1]] = True
    steps = np.diff(drop.timestamp)[~first[1:]]
    print(f"timestamps: step min {steps.min():.5f} max {steps.max():.5f} s (nominal {1 / dataset.FPS:.5f})")
    arm_speed = np.abs(finger_signal.speeds(drop.state[:, :14], drop.starts)).max(1)
    signals = motion_signals(drop)
    for camera in dataset.CAMERAS:
        change = features[f"{camera}/change"]
        repeat = (change < REPEAT_CHANGE) & ~first
        moving = (arm_speed > ARM_MOVING) & ~first
        gaps = np.diff(np.flatnonzero(repeat))
        signal = signals[camera].copy()
        signal[first] = 0.0
        lag, curve = finger_signal.lag_frames(signal, change, ~first)
        print(f"{camera}: repeated frames {100 * repeat.mean():.1f} % ({100 * repeat[moving].mean():.1f} % while an arm moves), "
              f"most common spacing {np.bincount(gaps).argmax()} frames; image lags the joints by {lag:.1f} frames "
              f"= {1000 * lag / dataset.FPS:.0f} ms (correlation {curve.max():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
