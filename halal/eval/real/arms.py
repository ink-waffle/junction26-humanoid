"""Arms on the real robot: start poses vs ARM_HOME, speeds vs the filter cap, tracking error and lag, limits, reach (A3, A4).

    python -m halal.eval.real.arms         -> prints the tables, writes plots a3_*.png
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.sim.aalto_task.layout import ARM_HOME
from halal.eval.real import dataset, finger_signal, kin, plotstyle

STILL_SPEED = 0.02
STILL_FRAMES = 15
MOVING_SPEED = 0.2


def short(name: str) -> str:
    return name.replace("_joint", "").replace("shoulder", "sh").replace("wrist", "wr")


def still(cmd_speed: np.ndarray) -> np.ndarray:
    """Frames where no arm command has moved for STILL_FRAMES."""
    quiet = np.abs(cmd_speed).max(1) < STILL_SPEED
    return np.convolve(quiet, np.ones(STILL_FRAMES), mode="full")[:len(quiet)] >= STILL_FRAMES


def print_start_poses(drop: dataset.Drop, spec: JointSpec, hands: dict[str, kin.HandTrack]) -> np.ndarray:
    home = np.array([ARM_HOME[name] for name in spec.arm_names])
    starts = np.array([drop.arm_q(ep)[0] for ep in range(drop.n_episodes)])
    print("== start pose of every episode against ARM_HOME (rad)")
    print("   home  ", home)
    print("   median", np.median(starts, axis=0))
    print("   min   ", starts.min(0))
    print("   max   ", starts.max(0))
    print("   largest joint deviation per episode", np.abs(starts - home).max(1))
    for side in ("left", "right"):
        palms = hands[side].palm[drop.starts[:-1]]
        print(f"   {side} palm at start: min {palms.min(0)} median {np.median(palms, axis=0)} max {palms.max(0)}")
    return starts


def print_joint_table(spec: JointSpec, cfg: SafetyConfig, arm_q: np.ndarray, arm_cmd: np.ndarray, cmd_speed: np.ndarray,
                      meas_speed: np.ndarray, valid: np.ndarray) -> None:
    lower, upper = spec.arm_lower + cfg.arm_pos_margin, spec.arm_upper - cfg.arm_pos_margin
    error = arm_q - arm_cmd
    rest = still(cmd_speed)
    print("== per arm joint: commanded speed p95/p99/max, measured max (rad/s), share of frames above the cap, lag, "
          "error per unit speed, steady error mean/sd, measured range, filter range")
    for j, name in enumerate(spec.arm_names):
        speed = np.abs(cmd_speed[:, j])
        moving = speed > MOVING_SPEED
        lag, _ = finger_signal.lag_frames(cmd_speed[:, j], meas_speed[:, j], valid)
        slope = np.polyfit(cmd_speed[moving, j], error[moving, j], 1)[0]
        both = np.r_[arm_q[:, j], arm_cmd[:, j]]
        beyond = "  BEYOND" if both.min() < lower[j] or both.max() > upper[j] else ""
        print(f"   {short(name):16s} cmd {np.percentile(speed, 95):.2f} {np.percentile(speed, 99):.2f} {speed.max():5.1f} "
              f"meas {np.abs(meas_speed[:, j]).max():.1f} | >{cfg.arm_max_vel:.0f} rad/s {100 * np.mean(speed > cfg.arm_max_vel):.1f} % "
              f"| lag {1000 * lag / dataset.FPS:.0f} ms, {1000 * slope:+.0f} ms x speed "
              f"| steady {error[rest, j].mean():+.3f} sd {error[rest, j].std():.3f} "
              f"| meas [{arm_q[:, j].min():+.2f}, {arm_q[:, j].max():+.2f}] filter [{lower[j]:+.2f}, {upper[j]:+.2f}]{beyond}")
    fast = (np.abs(cmd_speed) > cfg.arm_max_vel).any(1)
    outside = ((arm_cmd < lower) | (arm_cmd > upper)).any(1)
    print(f"   frames with any arm command faster than the cap: {fast.sum()} ({100 * fast.mean():.1f} %); "
          f"commands inside the filter margin or beyond: {outside.sum()}; measured beyond URDF: "
          f"{((arm_q < spec.arm_lower) | (arm_q > spec.arm_upper)).any(1).sum()}")


def print_hand_limits(spec: JointSpec, hand_q: np.ndarray, hand_cmd: np.ndarray) -> None:
    print("== hand joints against the URDF limits: measured range, share of frames outside, largest excess (rad)")
    for j, name in enumerate(spec.hand_names):
        excess = np.maximum(spec.hand_lower[j] - hand_q[:, j], hand_q[:, j] - spec.hand_upper[j])
        cmd_excess = np.maximum(spec.hand_lower[j] - hand_cmd[:, j], hand_cmd[:, j] - spec.hand_upper[j])
        print(f"   {name.replace('_joint', ''):22s} meas [{hand_q[:, j].min():+.3f}, {hand_q[:, j].max():+.3f}] "
              f"URDF [{spec.hand_lower[j]:+.3f}, {spec.hand_upper[j]:+.3f}] "
              f"outside {100 * np.mean(excess > 0):.1f} % by {max(excess.max(), 0):.3f}; "
              f"commands outside {100 * np.mean(cmd_excess > 1e-6):.1f} % by {max(cmd_excess.max(), 0):.4f}")


def print_reach(hands: dict[str, kin.HandTrack]) -> None:
    print("== reach in the pelvis frame (m): palm origin and fingertips, min / p1 / p99 / max")
    for side, hand in hands.items():
        for label, points in (("palm", hand.palm), ("tips", hand.tips.reshape(-1, 3))):
            q = np.percentile(points, [0, 1, 99, 100], axis=0)
            print(f"   {side:5s} {label}: x {q[:, 0].round(3)} y {q[:, 1].round(3)} z {q[:, 2].round(3)}")


def plot_start_poses(spec: JointSpec, starts: np.ndarray) -> None:
    """Start pose of every episode per joint with ARM_HOME as a reference tick."""
    home = np.array([ARM_HOME[name] for name in spec.arm_names])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rows = np.arange(dataset.N_ARM)
    for j in rows:
        ax.scatter(starts[:, j], np.full(len(starts), j), s=14, color=plotstyle.SERIES[0], alpha=0.7, linewidths=0,
                   label="episode start (measured)" if j == 0 else None)
    ax.scatter(home, rows, marker="|", s=260, color=plotstyle.SERIES[1], linewidths=2, label="ARM_HOME (sim)")
    ax.set_yticks(rows, [short(n) for n in spec.arm_names])
    ax.invert_yaxis()
    ax.set_xlabel("joint position at the first frame, rad")
    ax.set_title("Real episodes do not start at the sim home pose")
    ax.legend(loc="lower right")
    plotstyle.save(fig, "a3_arm_start_poses.png")


def plot_speeds(spec: JointSpec, cfg: SafetyConfig, cmd_speed: np.ndarray) -> None:
    """Share of frames per commanded-speed bin for the left arm joints, with the filter cap."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.linspace(0, 3, 61)
    for j in range(7):
        counts, _ = np.histogram(np.abs(cmd_speed[:, j]), bins=bins)
        ax.plot(0.5 * (bins[:-1] + bins[1:]), np.maximum(counts / len(cmd_speed), 1e-5), label=short(spec.arm_names[j]))
    ax.axvline(cfg.arm_max_vel, color=plotstyle.INK, linestyle="--", linewidth=1.2)
    ax.annotate("filter cap", (cfg.arm_max_vel, 0.2), xytext=(5, 0), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("commanded joint speed, rad/s")
    ax.set_ylabel("share of frames (log)")
    ax.set_title("Left arm: operator commands against the 1 rad/s cap")
    ax.legend(ncols=2)
    plotstyle.save(fig, "a3_arm_command_speed.png")


def main() -> int:
    plotstyle.apply()
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    drop = dataset.load()
    spec, cfg = JointSpec(), SafetyConfig()
    hands = kin.track(kin.kinematics(), drop.state)
    arm_q, arm_cmd = drop.state[:, :dataset.N_ARM], drop.action[:, :dataset.N_ARM]
    cmd_speed, meas_speed = finger_signal.speeds(arm_cmd, drop.starts), finger_signal.speeds(arm_q, drop.starts)
    valid = np.ones(len(arm_q), bool)
    valid[drop.starts[:-1]] = False
    starts = print_start_poses(drop, spec, hands)
    print_joint_table(spec, cfg, arm_q, arm_cmd, cmd_speed, meas_speed, valid)
    print_hand_limits(spec, drop.state[:, dataset.N_ARM:], drop.action[:, dataset.N_ARM:])
    print_reach(hands)
    repeats = (np.abs(np.diff(drop.state, axis=0)).max(1) == 0).sum()
    nans = int(np.isnan(drop.state).sum() + np.isnan(drop.action).sum())
    print(f"== data quality: identical consecutive state rows {repeats}, NaN {nans}")
    plot_start_poses(spec, starts)
    plot_speeds(spec, cfg, cmd_speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
