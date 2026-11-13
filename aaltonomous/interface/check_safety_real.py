"""M6: real-robot validation of the safety filter at low speed. Prints PASS/FAIL, writes a JSON report.

    python -m aaltonomous.interface.check_safety_real --backend real --network-interface enx00e04cd30dba \
        --raise-arms -0.3

Each test commands something the filter must NOT pass through, and checks what the robot actually did
(measured joints / FK), not just what the filter returned. Limits that would mean a large motion are tested as
VIRTUAL limits placed a few cm / a tenth of a rad from the current pose, so every motion stays small:
  1. velocity   -- a 0.25 rad step target on R wrist yaw: command rate-limited, measured peak speed <= limit
  2. position   -- virtual upper limit 0.08 rad above R wrist yaw; target 0.25 above: arm stops at the limit
  3. invalid    -- NaN / wrong-shape actions: rejected, arm holds
  4. watchdog   -- caller goes silent: stale (hold) after hold_after_s, safe_stop after stop_after_s, arm holds
  5. workspace  -- virtual wall 4 cm in front of the left palm; L shoulder pitch -0.3: palm stops at the wall
Human at the e-stop, robot on the gantry / seated. Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .check_interface import Abort, Report, run_to
from .env import AaltoEnv, HZ
from .safety import N, JointSpec, SafetyConfig

SP = JointSpec()
RWY = SP.arm_names.index("right_wrist_yaw_joint")
LSP = SP.arm_names.index("left_shoulder_pitch_joint")


def steps(env, arm, hand, secs):
    """Send the SAME raw target every tick for secs (no ramp); return measured arm_q per tick + infos."""
    qs, infos = [], []
    for _ in range(max(1, int(secs * HZ))):
        obs, info = env.step({"arm_q_target": arm, "hand_q_target": hand})
        qs.append(obs["arm_q"].astype(float)); infos.append(info)
    return np.array(qs), infos


def back_to(env, arm, secs=1.5):
    hand = env.filter.hand_cmd.copy()
    obs = run_to(env, env._obs(), arm, hand, secs, "return")
    return run_to(env, obs, arm, hand, 0.6, "return")


def t_velocity(env, rep, home):
    hand = env.filter.hand_cmd.copy()
    tgt = home.copy(); tgt[RWY] += 0.25
    qs, infos = steps(env, tgt, hand, 1.2)
    vmax = env.filter.cfg.arm_max_vel
    v = np.abs(np.diff(qs[:, RWY])) * HZ
    clipped = sum("arm_velocity" in i["clip_reasons"] for i in infos)
    moved = qs[-1, RWY] - home[RWY]
    ok = clipped >= 3 and v.max() <= 1.5 * vmax and moved > 0.15
    rep.add("velocity limit", "PASS" if ok else "FAIL",
            f"{clipped} ticks rate-limited; measured peak {v.max():.2f} rad/s (limit {vmax}); moved {moved:+.3f}/0.25")
    env.filter.clipped_steps = 0
    back_to(env, home)


def t_position(env, rep, home):
    f, hand = env.filter, env.filter.hand_cmd.copy()
    saved = f.arm_hi[RWY], f.spec.arm_upper[RWY]
    lim = home[RWY] + 0.08
    f.arm_hi[RWY] = f.spec.arm_upper[RWY] = lim
    try:
        tgt = home.copy(); tgt[RWY] += 0.25
        qs, infos = steps(env, tgt, hand, 1.5)
    finally:
        f.arm_hi[RWY], f.spec.arm_upper[RWY] = saved
    cmd_max = f.arm_cmd[RWY]
    over = qs[:, RWY].max() - lim
    clipped = sum("arm_position" in i["clip_reasons"] for i in infos)
    ok = clipped > 0 and cmd_max <= lim + 1e-6 and over < 0.03
    rep.add("position limit (virtual)", "PASS" if ok else "FAIL",
            f"limit {lim:+.3f}; command max {cmd_max:+.3f}; measured max {qs[:, RWY].max():+.3f} "
            f"(over by {over:+.3f}); {clipped} ticks clipped")
    f.clipped_steps = 0
    back_to(env, home)


def t_invalid(env, rep, home):
    hand = env.filter.hand_cmd.copy()
    bad = [{"arm_q_target": np.full(N, np.nan), "hand_q_target": hand},
           {"arm_q_target": np.zeros(5), "hand_q_target": hand},
           {"hand_q_target": hand},
           "garbage"]
    drift, reasons = 0.0, []
    m0 = env._obs()["arm_q"].astype(float)  # measured, not commanded: the real arm sags ~0.05 rad under gravity
    for a in bad * 8:
        obs, info = env.step(a)
        drift = max(drift, float(np.abs(obs["arm_q"] - m0).max()))
        reasons += info["clip_reasons"]
    held = np.abs(env.filter.arm_cmd - home).max() < 1e-9
    ok = held and drift < 0.03 and all(r.startswith("invalid") for r in reasons) and reasons
    rep.add("invalid actions", "PASS" if ok else "FAIL",
            f"{len(bad) * 8} bad actions; command unchanged={held}; measured drift {drift:.3f} rad; "
            f"reasons {sorted(set(reasons))}")
    env.filter.clipped_steps = 0


def t_watchdog(env, rep, home):
    c = env.filter.cfg
    env.step({"arm_q_target": home, "hand_q_target": env.filter.hand_cmd.copy()})
    m0 = env._obs()["arm_q"].astype(float)
    t0, modes, drift = env._now(), [], 0.0  # env clock: wall time on real, virtual on mock
    while env._now() - t0 < c.stop_after_s + 0.6:
        modes.append((env._now() - t0, env.hold()))
        drift = max(drift, float(np.abs(env._obs()["arm_q"] - m0).max()))
    first = {m: round(t, 2) for t, m in reversed(modes)}
    ok = ("stale" in first and "safe_stop" in first and c.hold_after_s - 0.1 <= first["stale"] <= c.hold_after_s + 0.15
          and c.stop_after_s - 0.1 <= first["safe_stop"] <= c.stop_after_s + 0.15 and drift < 0.03)
    rep.add("watchdog", "PASS" if ok else "FAIL",
            f"stale at {first.get('stale')} s (cfg {c.hold_after_s}), safe_stop at {first.get('safe_stop')} s "
            f"(cfg {c.stop_after_s}, safe_stop={c.safe_stop}); measured drift {drift:.3f} rad")
    # safe_stop latches: re-arm from the current command
    env.filter.reset(env.filter.arm_cmd, env.filter.hand_cmd, env._now())
    env.episode_over = False


def t_workspace(env, rep, home):
    f, hand = env.filter, env.filter.hand_cmd.copy()
    ws = f.ws
    tgt = home.copy(); tgt[LSP] -= 0.3
    p0 = ws.positions(env._obs()["arm_q"].astype(float))[1]  # left_hand_palm_link, measured
    d = ws.positions(tgt)[1] - p0
    ax = int(np.argmax(np.abs(d)))  # wall across the palm's main direction of travel, 40 % of the way
    sgn = float(np.sign(d[ax]))
    wall = p0[ax] + 0.4 * d[ax]
    lo, hi = p0 - np.array([1.0, 0.1, 1.0]), p0 + np.array([1.0, 0.1, 1.0])  # +-0.1 in y: left hand only
    lo[ax], hi[ax] = (wall, wall + 1.0) if sgn > 0 else (wall - 1.0, wall)
    saved = ws.ko
    ws.ko = (lo, hi)
    try:
        qs, infos = [], []
        a0 = f.arm_cmd.copy()
        for k in range(1, int(1.5 * HZ) + 1):  # ramp, 0.2 rad/s
            obs, info = env.step({"arm_q_target": a0 + min(1.0, k / (1.5 * HZ)) * (tgt - a0), "hand_q_target": hand})
            qs.append(obs["arm_q"].astype(float)); infos.append(info)
        q_more, _ = steps(env, tgt, hand, 0.8)
        qs = np.vstack([qs, q_more])
    finally:
        ws.ko = saved
    xs = sgn * np.array([ws.positions(q)[1][ax] for q in qs])
    clipped = sum("workspace" in i["clip_reasons"] for i in infos)
    over = xs.max() - sgn * wall
    ok = clipped > 0 and over < 0.02
    rep.add("workspace wall (virtual)", "PASS" if ok else "FAIL",
            f"wall on {'xyz'[ax]}={wall:.3f} ({100 * abs(0.4 * d[ax]):.1f} cm from palm, free travel would be "
            f"{100 * abs(d[ax]):.1f} cm); palm got {100 * over:+.1f} cm past the wall; {clipped} ticks clipped")
    f.clipped_steps = 0
    back_to(env, home, 2.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "sim", "real"], required=True)
    ap.add_argument("--network-interface", default=None)
    ap.add_argument("--img-server-ip", default="192.168.123.164")
    ap.add_argument("--raise-arms", type=float, default=0.0, metavar="RAD")
    ap.add_argument("--only", default=None, help="comma list: velocity,position,invalid,watchdog,workspace")
    ap.add_argument("--yes", action="store_true", help="skip the YES prompt (tests only; never on the real robot)")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if a.backend != "mock" and not a.yes:
        print(__doc__)
        if input("Type YES to move (human at the e-stop): ").strip() != "YES":
            return 2
    kw = {} if a.backend == "mock" else {"network_interface": a.network_interface, "img_server_ip": a.img_server_ip,
                                          "cameras": False}
    # no counter in front of the robot during this check; virtual walls are added per test
    env = AaltoEnv(backend=a.backend, time_limit=1e9, safety=SafetyConfig(dt=1.0 / HZ, max_clipped_steps=10**6, counter_keepout_min=None,
                                                          counter_keepout_max=None), **kw)
    rep = Report()
    tests = {"velocity": t_velocity, "position": t_position, "invalid": t_invalid, "watchdog": t_watchdog,
             "workspace": t_workspace}
    sel = a.only.split(",") if a.only else list(tests)
    try:
        obs = env.reset()
        start = env.filter.arm_cmd.copy()
        home = start.copy()
        try:
            if a.raise_arms:
                home[0] += a.raise_arms; home[7] += a.raise_arms
                obs = run_to(env, obs, home, env.filter.hand_cmd.copy(), abs(a.raise_arms) / 0.25, "raise")
                obs = run_to(env, obs, home, env.filter.hand_cmd.copy(), 1.0, "raise")
            for name in sel:
                tests[name](env, rep, home)
            if a.raise_arms:
                back_to(env, start, abs(a.raise_arms) / 0.25)
        except Abort as e:
            rep.add("sequence", "FAIL", f"ABORTED: {e} (holding)")
    finally:
        env.close()
    path = Path(a.report or f"logs/check_safety_{a.backend}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(r["status"] == s for r in rep.rows) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    path.write_text(json.dumps({"backend": a.backend, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "counts": counts,
                                "checks": rep.rows}, indent=2))
    print(f"\n{counts}  report: {path}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
