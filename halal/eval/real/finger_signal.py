"""Finger tracking error on the real robot: free, closed on nothing and holding; lag; a contact score (question A1).

    python -m halal.eval.real.finger_signal        -> prints the tables, writes plots a1_*.png
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from halal.eval.real import dataset, kin, labels, plotstyle

LEFT = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
RIGHT = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
CLOSE_SIGN_LEFT = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
STEADY_SPEED = 0.05
STEADY_FRAMES = 10
MOVING_SPEED = 0.5
AIR_HEIGHT = 0.10
LABEL_PAD = 20
MAX_LAG = 8
STEP_FRAMES = 6
STEP_WINDOW = 28
STEP_MIN = 0.5
SCORE_THRESHOLD = 0.3


def speeds(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-frame rate of change in rad/s, zero on the first frame of every episode."""
    out = np.zeros_like(values)
    out[1:] = (values[1:] - values[:-1]) * dataset.FPS
    out[starts[:-1]] = 0.0
    return out


def steady(cmd_speed: np.ndarray) -> np.ndarray:
    """Frames where every joint's command has been still for STEADY_FRAMES."""
    still = np.abs(cmd_speed).max(1) < STEADY_SPEED
    run = np.convolve(still, np.ones(STEADY_FRAMES), mode="full")[:len(still)]
    return run >= STEADY_FRAMES


def label_mask(drop: dataset.Drop, windows: dict[int, tuple[int, int]], pad: int = 0) -> np.ndarray:
    """Global boolean mask of a labelled window set."""
    return np.concatenate([labels.mask(windows, ep, drop.starts[ep + 1] - drop.starts[ep], pad)
                           for ep in range(drop.n_episodes)])


def left_free(drop: dataset.Drop, tips_z: np.ndarray) -> np.ndarray:
    """Left hand clear of the counter and of every labelled contact window."""
    contact = np.zeros(len(tips_z), bool)
    for windows in (labels.HOLD_AIR, labels.HOLD_COUNTER, labels.CLOSED_EMPTY):
        contact |= label_mask(drop, windows, LABEL_PAD)
    return (tips_z.min(1) > AIR_HEIGHT) & ~contact


def blocked(error: np.ndarray) -> np.ndarray:
    """Left-hand tracking error turned so that positive means 'stopped short of the commanded closure'."""
    return -error * CLOSE_SIGN_LEFT


def describe(name: str, values: np.ndarray) -> str:
    q = np.percentile(values, [5, 50, 95])
    return f"{name:10s} n={len(values):5d} mean {values.mean():+.3f} sd {values.std():.3f} p5/50/95 {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}"


def lag_frames(cmd: np.ndarray, meas: np.ndarray, valid: np.ndarray) -> tuple[float, np.ndarray]:
    """Lag of measured speed behind commanded speed in frames (parabolic peak of the cross-correlation) and the curve."""
    lags = np.arange(-2, MAX_LAG + 1)
    curve = []
    for lag in lags:
        a, b = (cmd[:len(cmd) - lag], meas[lag:]) if lag >= 0 else (cmd[-lag:], meas[:lag])
        ok = (valid[:len(valid) - lag] & valid[lag:]) if lag >= 0 else (valid[-lag:] & valid[:lag])
        curve.append(np.corrcoef(a[ok], b[ok])[0, 1])
    curve = np.array(curve)
    k = int(np.clip(np.argmax(curve), 1, len(curve) - 2))
    denom = curve[k - 1] - 2 * curve[k] + curve[k + 1]
    return float(lags[k] + 0.5 * (curve[k - 1] - curve[k + 1]) / denom), curve


def crossing(frames: np.ndarray, progress: np.ndarray, level: float) -> float:
    """Frame (fractional) at which `progress` first reaches `level`."""
    k = max(int(np.argmax(progress >= level)), 1)
    return float(frames[k - 1] + (level - progress[k - 1]) / (progress[k] - progress[k - 1]))


def step_responses(cmd: np.ndarray, meas: np.ndarray) -> np.ndarray:
    """Rows (cmd step rad, t10, t50, t90 ms after the command's half-way point, peak speed rad/s) of fast command steps."""
    rows, f = [], STEP_FRAMES
    while f < len(cmd) - STEP_WINDOW - 2:
        step = cmd[f + STEP_FRAMES] - cmd[f]
        settled = np.abs(cmd[f + STEP_FRAMES:f + STEP_WINDOW] - cmd[f + STEP_FRAMES]).max() < 0.15
        travel = np.median(meas[f + STEP_WINDOW - 6:f + STEP_WINDOW]) - meas[f]
        if abs(step) > STEP_MIN and settled and np.abs(cmd[f - 5:f + 1] - cmd[f]).max() < 0.1 and abs(travel) > 0.3:
            frames = np.arange(f, f + STEP_WINDOW, dtype=float)
            progress = (meas[f:f + STEP_WINDOW] - meas[f]) / travel
            half = np.interp(0.5, np.maximum.accumulate((cmd[f:f + STEP_FRAMES + 1] - cmd[f]) / step), frames[:STEP_FRAMES + 1])
            times = [1000 * (crossing(frames, progress, level) - half) / dataset.FPS for level in (0.1, 0.5, 0.9)]
            rows.append([step, *times, np.abs(np.diff(meas[f:f + STEP_WINDOW])).max() * dataset.FPS])
            f += STEP_WINDOW
        else:
            f += 1
    return np.array(rows).reshape(-1, 5)


def contact_score(error_left: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Sum over the seven left joints of the blocked error above the free bias, negative parts dropped."""
    return np.clip(blocked(error_left) - bias, 0.0, None).sum(1)


def plot_distributions(groups: dict[str, np.ndarray]) -> None:
    """Small multiples: per left joint the blocked-error distribution of every group."""
    fig, axes = plt.subplots(1, 7, figsize=(15, 3.2), sharey=True, layout="constrained")
    for j, ax in enumerate(axes):
        parts = ax.boxplot([g[:, j] for g in groups.values()], orientation="horizontal", widths=0.55, showfliers=False,
                           patch_artist=True, medianprops={"color": plotstyle.INK}, whis=(5, 95))
        for patch, colour in zip(parts["boxes"], plotstyle.SERIES):
            patch.set(facecolor=colour, edgecolor=colour, alpha=0.85)
        ax.axvline(0.0, color=plotstyle.MUTED, linewidth=0.8)
        ax.set_title(LEFT[j])
        ax.set_xlim(-0.25, 0.62)
    axes[0].set_yticks(range(1, len(groups) + 1), list(groups))
    axes[0].invert_yaxis()
    fig.supxlabel("blocked error = how far the joint stays short of the commanded closure, rad (box 25-75 %, whiskers 5-95 %)")
    fig.suptitle("Left Dex3 on the real robot: tracking error by situation (drop 1)", fontweight="bold")
    plotstyle.save(fig, "a1_finger_error_by_situation.png")


def plot_score(scores: dict[str, np.ndarray], threshold: float) -> None:
    """Histogram of the contact score per group with the proposed threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.linspace(0.0, 1.6, 65)
    for (name, values), colour in zip(scores.items(), plotstyle.SERIES):
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=colour, label=f"{name} (n={len(values)})")
    ax.axvline(threshold, color=plotstyle.INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"threshold {threshold:.2f} rad", (threshold, ax.get_ylim()[1] * 0.9), xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("contact score: sum over 7 joints of blocked error above the free bias, rad")
    ax.set_ylabel("density")
    ax.set_title("Is something in the left hand? Contact score on the real robot")
    ax.legend()
    plotstyle.save(fig, "a1_contact_score.png")


def plot_episode(drop: dataset.Drop, ep: int) -> None:
    """Commanded and measured left-hand joints over one episode with the labelled hold shaded."""
    t = np.arange(drop.starts[ep + 1] - drop.starts[ep]) / dataset.FPS
    fig, axes = plt.subplots(7, 1, figsize=(10, 9.5), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, drop.hand_cmd(ep)[:, j], color=plotstyle.SERIES[1], label="commanded")
        ax.plot(t, drop.hand_q(ep)[:, j], color=plotstyle.SERIES[0], label="measured")
        if ep in labels.HOLD_AIR:
            first, last = labels.HOLD_AIR[ep]
            ax.axvspan(first / dataset.FPS, last / dataset.FPS, color=plotstyle.SERIES[2], alpha=0.15, linewidth=0)
        ax.set_ylabel(LEFT[j])
    axes[0].legend(ncols=2, loc="upper right")
    axes[0].set_title(f"Episode {ep}: left hand command and measurement (rad); shaded = object held in the air")
    axes[-1].set_xlabel("time, s")
    plotstyle.save(fig, f"a1_left_hand_ep{ep:02d}.png")


def main() -> int:
    plotstyle.apply()
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    drop = dataset.load()
    hand_q, hand_cmd = drop.state[:, dataset.N_ARM:], drop.action[:, dataset.N_ARM:]
    error = hand_q - hand_cmd
    cmd_speed, meas_speed = speeds(hand_cmd, drop.starts), speeds(hand_q, drop.starts)
    tips_z = kin.track(kin.kinematics(), drop.state)["left"].tips[:, :, 2]
    free = left_free(drop, tips_z)
    still_left, still_right = steady(cmd_speed[:, :7]), steady(cmd_speed[:, 7:])
    hold = label_mask(drop, labels.HOLD_AIR)
    hold_counter = label_mask(drop, labels.HOLD_COUNTER)
    empty = label_mask(drop, labels.CLOSED_EMPTY)

    print("== static episode 19: measurement noise (sd, rad) and steady error (rad)")
    last = drop.span(19)
    print("  sd  ", drop.state[last, dataset.N_ARM:].std(0))
    print("  err ", error[last].mean(0))

    print("== left hand, signed error measured - commanded (rad)")
    groups_signed = {"free, steady": free & still_left, "free, moving": free & (np.abs(cmd_speed[:, :7]).max(1) > MOVING_SPEED),
                     "closed empty": empty, "hold counter": hold_counter, "hold air": hold}
    for j, joint in enumerate(LEFT):
        print(f" {joint}")
        for name, sel in groups_signed.items():
            print("   " + describe(name, error[sel, j]))

    print("== right hand (never grasps), steady, signed error (rad)")
    for j, joint in enumerate(RIGHT):
        print("   " + describe(joint, error[still_right, 7 + j]))

    bias = np.median(blocked(error[free & still_left, :7]), axis=0)
    print("== free steady bias in blocked-error convention (left):", bias)
    score = contact_score(error[:, :7], bias)
    scores = {"free, steady": score[free & still_left], "free, moving": score[groups_signed["free, moving"]],
              "closed empty": score[empty], "hold air": score[hold]}
    for name, values in scores.items():
        print("   score " + describe(name, values) + f"  above {SCORE_THRESHOLD}: {100 * np.mean(values > SCORE_THRESHOLD):.1f} %")
    print("== contact score and blocked error above bias per labelled window (median over the window)")
    for name, windows in (("hold air", labels.HOLD_AIR), ("hold counter", labels.HOLD_COUNTER), ("closed empty", labels.CLOSED_EMPTY)):
        for ep, (first, last) in windows.items():
            rows = slice(drop.starts[ep] + first, drop.starts[ep] + last + 1)
            print(f"   {name:12s} ep{ep:2d} score {np.median(score[rows]):.2f} min {score[rows].min():.2f} | "
                  f"{np.median(blocked(error[rows, :7]) - bias, axis=0)}")

    print("== lag of measured behind commanded speed (free frames, cross-correlation) and response to fast command steps")
    curves = {}
    for j, joint in enumerate(LEFT):
        lag, curves[joint] = lag_frames(cmd_speed[:, j], meas_speed[:, j], free)
        steps = np.concatenate([step_responses(drop.hand_cmd(ep)[:, j], drop.hand_q(ep)[:, j]) for ep in range(drop.n_episodes)])
        timing = "no fast steps" if not len(steps) else (
            "{} steps of {:.2f} rad: t10 {:.0f} t50 {:.0f} t90 {:.0f} ms, peak {:.1f} rad/s".format(
                len(steps), *np.median(np.abs(steps), axis=0)))
        fastest = np.abs(cmd_speed[:, j])
        print(f"   {joint:9s} lag {1000 * lag / dataset.FPS:.0f} ms; {timing}; speed p99 cmd {np.percentile(fastest, 99):.1f} "
              f"max cmd {fastest.max():.1f} max meas {np.abs(meas_speed[:, j]).max():.1f} rad/s")

    plot_distributions({name: blocked(error[sel, :7]) for name, sel in groups_signed.items()})
    plot_score(scores, SCORE_THRESHOLD)
    for ep in (4, 7, 17):
        plot_episode(drop, ep)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for joint, curve in curves.items():
        ax.plot(np.arange(-2, MAX_LAG + 1) * 1000 / dataset.FPS, curve, marker="o", markersize=3, label=joint)
    ax.set_xlabel("lag of measured speed behind commanded speed, ms")
    ax.set_ylabel("correlation")
    ax.set_title("Left finger lag (free motion)")
    ax.legend(ncols=2)
    plotstyle.save(fig, "a1_finger_lag.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
