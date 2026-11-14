"""Colours of the real scene in the head camera and what the baseline HSV thresholds make of them (question B6).

    python -m halal.eval.real.colours      -> prints HSV statistics and the confusion table, writes b6_*.png
"""
from __future__ import annotations

import cv2
import numpy as np
from matplotlib import pyplot as plt

from halal import config
from halal.eval.real import dataset, plotstyle, video_features

MIN_COMPONENT_PX = 300
# OpenCV HSV ranges (lo, hi) of the foam objects in drop 1, tuned on hand-picked patches.
OBJECT_HSV = {
    "purple": ((108, 60, 90), (135, 255, 255)),
    "orange": ((9, 110, 140), (17, 255, 255)),
    "yellow": ((20, 60, 110), (50, 255, 255)),
}
REFERENCE_HSV = {
    **{f"{name} object": bounds for name, bounds in OBJECT_HSV.items()},
    "bin tape": ((0, 90, 60), (8, 255, 255)),
    "bin tape (wrap)": ((172, 90, 60), (180, 255, 255)),
    "hand / dark": ((0, 0, 0), (180, 255, 60)),
    "table": ((0, 0, 110), (180, 50, 255)),
}
MASK_BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255), "blue": (255, 0, 0)}


def large_components(mask: np.ndarray, min_px: int = MIN_COMPONENT_PX) -> np.ndarray:
    """Mask with connected components smaller than `min_px` removed."""
    count, labelled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_px)
    return np.isin(labelled, keep[keep > 0])


def reference_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per reference class; objects and tape only as large blobs."""
    out = {}
    for name, (lo, hi) in REFERENCE_HSV.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        out[name] = large_components(mask) if "object" in name or "tape" in name else mask > 0
    out["bin tape"] |= out.pop("bin tape (wrap)")
    return out


def baseline_masks(hsv: np.ndarray, perception: dict) -> dict[str, np.ndarray]:
    """The baseline's colour masks: HSV ranges, opening, blobs of at least min_area_px."""
    kernel = np.ones((int(perception["open_px"]),) * 2, np.uint8)
    out = {}
    for colour, ranges in perception["hsv"].items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        out[colour] = large_components(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), int(perception["min_area_px"]))
    return out


def first_frames(drop: dataset.Drop) -> dict[int, np.ndarray]:
    """RGB head frame at the start of every episode."""
    starts = [int(s) for s in drop.starts[:-1]]
    frames = dataset.frames(drop.root, "cam_high", starts)
    return {ep: frames[i] for ep, i in enumerate(starts)}


def hsv_table(frames: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Per class the HSV pixels of all first frames, hue unwrapped around red for the tape."""
    pixels: dict[str, list[np.ndarray]] = {}
    for rgb in frames.values():
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for name, mask in reference_masks(hsv).items():
            pixels.setdefault(name, []).append(hsv[mask].astype(float))
    out = {name: np.concatenate(parts) for name, parts in pixels.items()}
    tape = out["bin tape"]
    tape[:, 0] = np.where(tape[:, 0] > 90, tape[:, 0] - 180, tape[:, 0])
    return out


def confusion(frames: dict[int, np.ndarray], perception: dict) -> dict[str, dict[str, float]]:
    """Mean pixels per frame that each baseline colour mask takes from each reference class, plus recall of the class."""
    taken: dict[str, dict[str, float]] = {}
    size: dict[str, float] = {}
    for rgb in frames.values():
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        reference = reference_masks(hsv)
        reference["other"] = ~np.any(list(reference.values()), axis=0)
        for colour, mask in baseline_masks(hsv, perception).items():
            for name, ref in reference.items():
                taken.setdefault(colour, {}).setdefault(name, 0.0)
                taken[colour][name] += float((mask & ref).sum()) / len(frames)
        for name, ref in reference.items():
            size[name] = size.get(name, 0.0) + float(ref.sum()) / len(frames)
    taken["class size"] = size
    return taken


def print_statistics(pixels: dict[str, np.ndarray]) -> None:
    print("== HSV per class over the first frame of all episodes (OpenCV H 0-180; p5 / p50 / p95)")
    for name, values in pixels.items():
        q = np.percentile(values, [5, 50, 95], axis=0)
        columns = "   ".join(f"{channel} {q[0, k]:4.0f} {q[1, k]:4.0f} {q[2, k]:4.0f}" for k, channel in enumerate("HSV"))
        print(f"  {name:14s} n={len(values):7d}  {columns}")


def print_confusion(table: dict[str, dict[str, float]]) -> None:
    print("== baseline masks: mean pixels per frame taken from each class (class size in the last row)")
    classes = list(table["class size"])
    print("  " + " " * 12 + "".join(f"{c[:13]:>15s}" for c in classes))
    for colour, row in table.items():
        print(f"  {colour:12s}" + "".join(f"{row[c]:15.0f}" for c in classes))


def exposure_variation(drop: dataset.Drop, features: dict[str, np.ndarray]) -> None:
    """Print per-episode mean brightness and colour balance of the head camera, and the drift inside episodes."""
    rgb = features["cam_high/mean_rgb"]
    print("== head camera exposure / white balance per episode: brightness, R/G, B/G, brightness range inside the episode")
    rows = []
    for ep in range(drop.n_episodes):
        part = rgb[drop.span(ep)]
        gray = part.mean(1)
        red, green, blue = part.mean(0)
        rows.append((gray.mean(), red / green, blue / green, gray.min(), gray.max()))
        print("  ep{:2d} brightness {:5.1f}  R/G {:.3f}  B/G {:.3f}  range {:5.1f}-{:5.1f}".format(ep, *rows[-1]))
    low, high = np.min(rows, axis=0), np.max(rows, axis=0)
    print(f"  across episodes: brightness {low[0]:.0f}-{high[0]:.0f}, R/G {low[1]:.3f}-{high[1]:.3f}, B/G {low[2]:.3f}-{high[2]:.3f}")


def plot_hue_saturation(pixels: dict[str, np.ndarray], perception: dict) -> None:
    """Hue-saturation scatter of the classes with the baseline hue ranges as bands."""
    fig, ax = plt.subplots(figsize=(9, 4))
    rng = np.random.default_rng(0)
    for (name, values), colour in zip(pixels.items(), plotstyle.SERIES):
        sample = values[rng.choice(len(values), min(len(values), 1500), replace=False)]
        ax.scatter(sample[:, 0], sample[:, 1], s=3, alpha=0.35, color=colour, label=name, linewidths=0)
    for colour, ranges in perception["hsv"].items():
        for lo, hi in ranges:
            ax.axvspan(lo[0], hi[0], ymin=lo[1] / 255, color=plotstyle.MUTED, alpha=0.12, linewidth=0)
            ax.annotate(f"baseline {colour}", ((lo[0] + hi[0]) / 2, 250), ha="center", va="top", fontsize=7, color="#52514e")
    ax.set_xlim(-10, 180)
    ax.set_ylim(0, 255)
    ax.set_xlabel("hue (OpenCV 0-180; bin tape unwrapped below 0)")
    ax.set_ylabel("saturation")
    ax.set_title("Real head camera: classes in hue-saturation, grey bands = baseline thresholds")
    ax.legend(markerscale=4, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plotstyle.save(fig, "b6_hue_saturation.png")


def save_mask_examples(frames: dict[int, np.ndarray], perception: dict, episodes: tuple[int, ...] = (0, 5, 11, 17)) -> None:
    """Head frames next to the baseline masks painted in their colour."""
    tiles = []
    for ep in episodes:
        bgr = cv2.cvtColor(frames[ep], cv2.COLOR_RGB2BGR)
        painted = (bgr * 0.35).astype(np.uint8)
        for colour, mask in baseline_masks(cv2.cvtColor(frames[ep], cv2.COLOR_RGB2HSV), perception).items():
            painted[mask] = MASK_BGR[colour]
        tiles.append(np.hstack([bgr, painted]))
    plotstyle.PLOTS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(plotstyle.PLOTS / "b6_baseline_masks.png"), np.vstack(tiles))


def main() -> int:
    plotstyle.apply()
    drop = dataset.load()
    perception = config.baseline_config()["perception"]
    frames = first_frames(drop)
    pixels = hsv_table(frames)
    print_statistics(pixels)
    print_confusion(confusion(frames, perception))
    exposure_variation(drop, video_features.load())
    plot_hue_saturation(pixels, perception)
    save_mask_examples(frames, perception)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
