"""Per-frame image features of a drop (frame-to-frame change, mean colour) cached as npz.

    python -m halal.eval.real.video_features            -> runs/real_frames/cache/features.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from halal.eval.real import dataset

OUT = dataset.REPO / "runs/real_frames/cache/features.npz"

def camera_features(root: Path, camera: str) -> dict[str, np.ndarray]:
    """Arrays over all frames of one camera: mean absolute grey change to the previous frame and mean RGB."""
    change, mean_rgb = [], []
    previous = None
    for _, rgb in dataset.iter_frames(root, camera):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
        change.append(0.0 if previous is None else float(np.abs(gray - previous).mean()))
        previous = gray
        mean_rgb.append(rgb.reshape(-1, 3).mean(0))
    return {f"{camera}/change": np.array(change), f"{camera}/mean_rgb": np.array(mean_rgb)}


def load(path: Path = OUT) -> dict[str, np.ndarray]:
    """Cached features as a dict of arrays indexed by global frame."""
    with np.load(path) as data:
        return dict(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=dataset.DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    features: dict[str, np.ndarray] = {}
    for camera in dataset.CAMERAS:
        features.update(camera_features(args.root, camera))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **features)
    print(f"{args.out}: {len(features)} arrays, {len(next(iter(features.values())))} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
