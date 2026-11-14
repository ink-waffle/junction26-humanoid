"""Sample PNG frames of a drop for perception work: first / grasp / last of every episode from all cameras, plus series.

    python -m halal.eval.real.extract_frames       -> runs/real_frames/ep*_*.png and runs/real_frames/series/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from halal.eval.real import dataset, labels

OUT = dataset.REPO / "runs/real_frames"
SERIES_EPISODES = (4, 11, 17)
SERIES_FRAMES = 10


def grasp_frame(drop: dataset.Drop, ep: int) -> int:
    """Episode-local frame that best shows a grasp: middle of the labelled hold, else of the longest closed command."""
    for windows in (labels.HOLD_AIR, labels.HOLD_COUNTER):
        if ep in windows:
            return sum(windows[ep]) // 2
    distal = np.minimum(drop.hand_cmd(ep)[:, 4], drop.hand_cmd(ep)[:, 6])
    return int(np.argmin(distal))


def key_frames(drop: dataset.Drop) -> dict[int, str]:
    """Global frame index -> file stem for the first, grasp and last frame of every episode."""
    out = {}
    for ep in range(drop.n_episodes):
        span = drop.span(ep)
        picks = {"first": 0, "grasp": grasp_frame(drop, ep), "last": span.stop - span.start - 1}
        for tag, local in picks.items():
            out[span.start + local] = f"ep{ep:02d}_{tag}_f{local:04d}"
    return out


def series_frames(drop: dataset.Drop) -> dict[int, str]:
    """Global frame index -> file stem for evenly spaced frames of the series episodes."""
    out = {}
    for ep in SERIES_EPISODES:
        span = drop.span(ep)
        for local in np.linspace(0, span.stop - span.start - 1, SERIES_FRAMES).astype(int):
            out[span.start + int(local)] = f"series/ep{ep:02d}_f{int(local):04d}"
    return out


def write(root: Path, camera: str, stems: dict[int, str], out: Path) -> int:
    """Decode the wanted frames of one camera and save them as PNG."""
    for index, rgb in dataset.iter_frames(root, camera, stems):
        path = out / f"{stems[index]}_{camera}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return len(stems)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=dataset.DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    drop = dataset.load(args.root)
    count = sum(write(args.root, camera, key_frames(drop), args.out) for camera in dataset.CAMERAS)
    count += write(args.root, "cam_high", series_frames(drop), args.out)
    print(f"{count} frames written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
