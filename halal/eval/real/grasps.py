"""Which grasps the operator uses: hand postures while holding, fingertip gaps and palm pose at the grasp (question A2).

    python -m halal.eval.real.grasps       -> prints one row per successful pick and the baseline postures for comparison
"""
from __future__ import annotations

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, mirror_left_to_right
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from halal import config
from halal.eval.real import dataset, kin, labels

DOWN = np.array([0.0, 0.0, -1.0])
CLOSED_DISTAL = -0.6
PRESHAPE_FRAMES = (25, 10)
SETTLE_FRAMES = 12
THUMB_SWEEP = (-1.0, -0.37, 0.0, 0.37)
OPERATOR_THUMB = (0.92, 0.2)


def closing_frame(hand_cmd: np.ndarray, hold_start: int) -> int:
    """Last frame before the labelled hold at which the distal finger joints are not yet commanded closed."""
    frame = hold_start
    while frame > 0 and min(hand_cmd[frame, 4], hand_cmd[frame, 6]) < CLOSED_DISTAL:
        frame -= 1
    return frame


def gaps_mm(tips: np.ndarray) -> tuple[float, float, float]:
    """Thumb-index, thumb-middle and index-middle fingertip distances in mm for tips (index, middle, thumb)."""
    return tuple(1000 * float(np.linalg.norm(tips[a] - tips[b])) for a, b in ((2, 0), (2, 1), (0, 1)))


def left_tips(kinematics: ArmKinematics, left7: np.ndarray) -> np.ndarray:
    """Fingertips in the left palm frame for a left-order posture."""
    return kinematics.tips_in_palm("left", np.r_[left7, mirror_left_to_right(left7)])


def palm_angles(rotation: np.ndarray, approach_axis: np.ndarray) -> tuple[float, float, float]:
    """Tilt of the approach axis from straight down, finger pitch above horizontal and finger yaw, in degrees."""
    approach, finger = rotation @ approach_axis, rotation[:, 0]
    tilt = np.degrees(np.arccos(np.clip(approach @ DOWN, -1.0, 1.0)))
    return float(tilt), float(np.degrees(np.arcsin(finger[2]))), float(np.degrees(np.arctan2(finger[1], finger[0])))


def describe_pick(drop: dataset.Drop, kinematics: ArmKinematics, hand: kin.HandTrack, ep: int, window: tuple[int, int]) -> dict:
    """Postures, fingertip geometry and palm pose of one labelled pick."""
    start = int(drop.starts[ep])
    hold = slice(start + window[0], start + window[1] + 1)
    close = closing_frame(drop.hand_cmd(ep), window[0])
    grasp = start + close + SETTLE_FRAMES
    before = slice(start + close - PRESHAPE_FRAMES[0], start + close - PRESHAPE_FRAMES[1])
    tips = np.median(kin.tips_in_palm(hand)[hold], axis=0)
    left = slice(dataset.N_ARM, dataset.N_ARM + 7)
    return {"ep": ep, "close": close, "preshape": np.median(drop.state[before, left], axis=0),
            "commanded": np.median(drop.action[hold, left], axis=0),
            "measured": np.median(drop.state[hold, left], axis=0), "gaps": gaps_mm(tips), "thumb_z": tips[2, 2],
            "palm": hand.palm[grasp], "tips_z": hand.tips[grasp][:, 2],
            "angles": palm_angles(hand.rotation[grasp], kinematics.axis["left"]),
            "release_palm": hand.palm[hold][-1], "release_tips": hand.tips[hold][-1].mean(0), "carry_top": hand.palm[hold][:, 2].max()}


def print_baseline(kinematics: ArmKinematics) -> None:
    postures = hand_postures(config.baseline_config())
    print("== baseline postures (left order thumb0,1,2, middle0,1, index0,1) and their fingertip gaps")
    for name in ("open", "closed", "release"):
        gaps = gaps_mm(left_tips(kinematics, postures[name][:7]))
        print(f"   {name:8s} {postures[name][:7]}  thumb-index {gaps[0]:.0f} thumb-middle {gaps[1]:.0f} index-middle {gaps[2]:.0f} mm")
    print("== thumb tip z in the palm frame (mm) against thumb_0, baseline thumb vs operator thumb (thumb_1 0.92, thumb_2 0.2)")
    for thumb_0 in THUMB_SWEEP:
        base = left_tips(kinematics, np.r_[thumb_0, postures["closed"][1:7]])[2, 2]
        operator = left_tips(kinematics, np.r_[thumb_0, OPERATOR_THUMB, postures["closed"][3:7]])[2, 2]
        print(f"   thumb_0 {thumb_0:+.2f}: baseline {1000 * base:+.0f}, operator {1000 * operator:+.0f}  (index tip +29, middle tip -29)")


def print_mirror_check(drop: dataset.Drop, kinematics: ArmKinematics) -> None:
    """Thumb tip z of the right hand for a mirrored left posture, with thumb_0 negated (organiser mirror) and kept."""
    left = np.r_[THUMB_SWEEP[1], OPERATOR_THUMB, -0.4, 0.0, -0.4, 0.0]
    negated = mirror_left_to_right(left)
    kept = np.r_[left[0], negated[1:]]
    print("== right-hand mirror of a left 'thumb opposes index' posture: thumb tip z in the right palm frame (index tip at +29 mm)")
    print(f"   left hand: {1000 * left_tips(kinematics, left)[2, 2]:+.0f} mm")
    for name, right in (("thumb_0 negated (mirror_left_to_right)", negated), ("thumb_0 kept", kept)):
        print(f"   {name}: {1000 * kinematics.tips_in_palm('right', np.r_[left, right])[2, 2]:+.0f} mm")
    thumbs = np.median(drop.action[:, [dataset.N_ARM, dataset.N_ARM + 7]], axis=0)
    print(f"   median commanded thumb_0 in the drop: left {thumbs[0]:+.2f}, right {thumbs[1]:+.2f} (the operator's hands rest mirrored)")


def main() -> int:
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    drop = dataset.load()
    kinematics = kin.kinematics()
    hand = kin.track(kinematics, drop.state)["left"]
    picks = [describe_pick(drop, kinematics, hand, ep, window) for ep, window in sorted(labels.HOLD_AIR.items())]
    print_baseline(kinematics)
    print_mirror_check(drop, kinematics)
    print("== operator picks: preshape (measured), then commanded and measured posture while holding")
    for p in picks:
        shape = labels.HELD_SHAPE.get(p["ep"], "cube")
        print(f"   ep{p['ep']:2d} {shape:8s} pre {p['preshape']} cmd {p['commanded']} meas {p['measured']}")
    print("== fingertip gaps while holding (mm) and thumb tip z in the palm frame (mm)")
    for p in picks:
        thumb_index, thumb_middle, index_middle = p["gaps"]
        print(f"   ep{p['ep']:2d} thumb-index {thumb_index:3.0f} thumb-middle {thumb_middle:3.0f} index-middle {index_middle:3.0f} "
              f"thumb z {1000 * p['thumb_z']:+.0f}")
    print("== palm at the grasp (pelvis frame): position, fingertip z (index, middle, thumb), approach tilt, finger pitch, finger yaw")
    for p in picks:
        tilt, pitch, yaw = p["angles"]
        print(f"   ep{p['ep']:2d} palm {np.round(p['palm'], 3)} tips z {np.round(p['tips_z'], 3)} "
              f"tilt {tilt:4.0f} pitch {pitch:+4.0f} yaw {yaw:+4.0f} deg")
    print("== release over the left bin: palm, mean fingertip, highest palm z during the carry")
    for p in picks:
        palm, tips = np.round(p["release_palm"], 3), np.round(p["release_tips"], 3)
        print(f"   ep{p['ep']:2d} palm {palm} tips {tips} carry top {p['carry_top']:.3f}")
    for key in ("preshape", "commanded", "measured", "release_palm", "release_tips"):
        print(f"   median {key}: {np.round(np.median([p[key] for p in picks], axis=0), 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
