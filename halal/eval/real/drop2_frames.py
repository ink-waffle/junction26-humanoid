"""Head-camera PNG samples of drop 2: first / middle / last frame of every episode, plus series of four episodes.

    HALAL_REAL_DROP=runs/drops/g1_junction_drop_2 python -m halal.eval.real.drop2_frames   -> runs/real_frames/drop2/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from halal.eval.real import dataset, extract_frames

ROOT = dataset.REPO / "runs/drops/g1_junction_drop_2"
OUT = dataset.REPO / "runs/real_frames/drop2"
SERIES_EPISODES = (0, 17, 30, 40)
SERIES_FRAMES = 10


def key_frames(drop: dataset.Drop) -> dict[int, str]:
    """Global frame index -> file stem for the first, middle and last frame of every episode."""
    out = {}
    for ep in range(drop.n_episodes):
        span = drop.span(ep)
        last = span.stop - span.start - 1
        for tag, local in (("first", 0), ("middle", last // 2), ("last", last)):
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    drop = dataset.load(args.root)
    count = sum(extract_frames.write(args.root, "cam_high", stems, args.out) for stems in (key_frames(drop), series_frames(drop)))
    print(f"{count} head frames written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
