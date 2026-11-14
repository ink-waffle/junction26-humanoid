"""Can a colour blob in the left wrist image confirm a grasp? Object-colour fraction in a fixed ROI, held vs not (B8).

    python -m halal.eval.real.wrist_blob       -> prints the separation, writes b8_wrist_*.png
"""
from __future__ import annotations

import cv2
import numpy as np
from matplotlib import pyplot as plt

from halal.eval.real import colours, dataset, finger_signal, kin, labels, plotstyle

CAMERA = "cam_left_wrist"
HELD_COLOUR = "purple"
ROI = (slice(200, 480), slice(220, 480))
AIR_HEIGHT = 0.12


def object_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels in the HSV range of the held object's colour."""
    lo, hi = colours.OBJECT_HSV[HELD_COLOUR]
    return cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), np.array(lo), np.array(hi)) > 0


def scan(drop: dataset.Drop, hold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROI colour fraction of every frame and the mean object mask over the held frames."""
    fraction = np.zeros(len(hold))
    heat = np.zeros((480, 640))
    for index, rgb in dataset.iter_frames(drop.root, CAMERA):
        mask = object_mask(rgb)
        fraction[index] = mask[ROI].mean()
        if hold[index]:
            heat += mask
    return fraction, heat / max(int(hold.sum()), 1)


def main() -> int:
    plotstyle.apply()
    drop = dataset.load()
    hold = finger_signal.label_mask(drop, labels.HOLD_AIR)
    near_hold = finger_signal.label_mask(drop, labels.HOLD_AIR, 30) | finger_signal.label_mask(drop, labels.HOLD_COUNTER, 30)
    tips_z = kin.track(kin.kinematics(), drop.state)["left"].tips[:, :, 2].min(1)
    fraction, heat = scan(drop, hold)
    groups = {"holding, in the air": hold, "empty hand in the air": (tips_z > AIR_HEIGHT) & ~near_hold,
              "empty hand near the counter": (tips_z <= AIR_HEIGHT) & ~near_hold}
    for name, sel in groups.items():
        q = np.percentile(fraction[sel], [1, 5, 50, 95, 99])
        print(f"{name:28s} n={sel.sum():5d} ROI {HELD_COLOUR} fraction p1/p5/p50/p95/p99: {q.round(3)}")
    threshold = 0.5 * (np.percentile(fraction[groups["empty hand in the air"]], 99) + np.percentile(fraction[hold], 1))
    print(f"threshold {threshold:.3f}: holding above {100 * np.mean(fraction[hold] > threshold):.1f} %, "
          f"empty in the air above {100 * np.mean(fraction[groups['empty hand in the air']] > threshold):.1f} %, "
          f"empty near the counter above {100 * np.mean(fraction[groups['empty hand near the counter']] > threshold):.1f} %")
    for ep, (first, last) in labels.HOLD_AIR.items():
        part = fraction[drop.starts[ep] + first:drop.starts[ep] + last + 1]
        print(f"  ep{ep:2d} hold: ROI fraction min {part.min():.3f} median {np.median(part):.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.linspace(0, 0.8, 81)
    for (name, sel), colour in zip(groups.items(), plotstyle.SERIES):
        ax.hist(fraction[sel], bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=colour,
                label=f"{name} (n={sel.sum()})")
    ax.axvline(threshold, color=plotstyle.INK, linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"fraction of {HELD_COLOUR} pixels in the wrist-image ROI (u 220-480, v 200-480)")
    ax.set_ylabel("density")
    ax.set_title("Left wrist camera: object colour between the fingers")
    ax.legend()
    plotstyle.save(fig, "b8_wrist_roi_fraction.png")
    heat_image = cv2.applyColorMap((255 * heat).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(plotstyle.PLOTS / "b8_wrist_held_object_heatmap.png"), heat_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
