"""M5: sim-vs-real interface verification. Prints a PASS/FAIL/WARN/SKIP table, writes a JSON report.

    python -m aaltonomous.interface.check_interface --backend mock
    python -m aaltonomous.interface.check_interface --backend real --network-interface enx00e04cd30dba
    add --no-motion to run only the passive checks (the arms are still *held* at their current pose).

Motion probes go through AaltoEnv.step, i.e. through the safety filter. Run them only with a human at the
e-stop, robot on the gantry. Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .env import AaltoEnv, HZ
from .safety import N, JointSpec, SafetyConfig

SPEC = {"head_rgb": ((480, 640, 3), np.uint8), "head_depth": ((480, 640), np.uint16),
        "left_wrist_rgb": ((480, 640, 3), np.uint8), "right_wrist_rgb": ((480, 640, 3), np.uint8),
        "arm_q": ((N,), np.float32), "arm_dq": ((N,), np.float32), "hand_q": ((N,), np.float32)}
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"  [{status:4}] {name}: {detail}", flush=True)

    @property
    def failed(self):
        return any(r["status"] == "FAIL" for r in self.rows)


class Abort(Exception):
    pass


def hold_cmd(env):
    """Hold = repeat the last command (see run_to)."""
    return {"arm_q_target": env.filter.arm_cmd.copy(), "hand_q_target": env.filter.hand_cmd.copy()}


def hold_action(obs):
    return {"arm_q_target": obs["arm_q"].astype(float), "hand_q_target": obs["hand_q"].astype(float)}


def passive(env, obs, rep, secs):
    # 1. keys, shapes, dtypes
    bad = []
    for k, (shape, dt) in SPEC.items():
        v = obs.get(k)
        if v is None or v.shape != shape or v.dtype != dt:
            bad.append(f"{k}: {None if v is None else (v.shape, v.dtype)} != {(shape, np.dtype(dt))}")
    if not isinstance(obs.get("instruction"), str):
        bad.append("instruction not str")
    if not isinstance(obs.get("t"), float):
        bad.append("t not float")
    rep.add("obs keys/shapes/dtypes", "FAIL" if bad else "PASS", "; ".join(bad) or "matches spec 4.2")

    # 2. joint limits
    sp = JointSpec()
    probs = []
    for key, lo, hi, names in (("arm_q", sp.arm_lower, sp.arm_upper, sp.arm_names),
                               ("hand_q", sp.hand_lower, sp.hand_upper, sp.hand_names)):
        q = obs[key].astype(float)
        if not np.isfinite(q).all():
            probs.append(f"{key} not finite")
            continue
        for i in np.flatnonzero((q < lo - 0.05) | (q > hi + 0.05)):
            probs.append(f"{names[i]}={q[i]:.3f} outside [{lo[i]:.3f},{hi[i]:.3f}]")
    if not np.isfinite(obs["arm_dq"]).all():
        probs.append("arm_dq not finite")
    rep.add("joint states finite and in limits", "FAIL" if probs else "PASS", "; ".join(probs[:6]) or "ok (0.05 rad slack)")

    # 3. run a hold loop and collect stats
    n = max(2, int(secs * HZ))
    wall, arms, hands, frames, ts, clips = [], [], [], {k: [] for k in CAMS}, [], 0
    depth_nz = 0.0
    for i in range(n):
        t0 = time.monotonic()
        obs, info = env.step(hold_cmd(env))
        wall.append(time.monotonic() - t0)
        arms.append(obs["arm_q"].copy()); hands.append(obs["hand_q"].copy()); ts.append(obs["t"])
        clips += int(info["clipped"])
        if i % 5 == 0:
            for k in CAMS:
                frames[k].append(obs[k].copy())
            depth_nz = max(depth_nz, float((obs["head_depth"] > 0).mean()))
    dur = sum(wall)
    if env.realtime:
        rate = (n - 1) / max(1e-9, sum(wall[1:]))
        jit = np.std(wall[1:]) * 1000
        note = ""
        if getattr(env, "clock", None) is not None:   # sim paced on the sim clock: 30 Hz of SIM time
            wall_rate = rate
            rate = (n - 1) / max(1e-9, ts[-1] - ts[0])
            note = f", sim clock (wall {wall_rate:.1f} Hz -> real-time factor {wall_rate / rate:.2f})"
        ok = abs(rate - HZ) < 0.1 * HZ
        rep.add("step() loop rate", "PASS" if ok else "FAIL",
                f"{rate:.1f} Hz (target {HZ:.0f}), jitter std {jit:.2f} ms, max {max(wall) * 1000:.1f} ms{note}")
        a = np.array(arms)
        distinct = (1 + int(np.any(np.diff(a, axis=0) != 0, axis=1).sum())) / dur
        rep.add("joint state update rate", "PASS" if distinct >= 10 else "FAIL",
                f"{distinct:.1f} distinct arm_q readings/s over {dur:.1f} s (a static mock would show 0)")
    else:
        rep.add("step() loop rate", "SKIP", "virtual clock (mock, realtime=False)")
        rep.add("joint state update rate", "SKIP", "virtual clock")
    rep.add("obs time t monotonic", "PASS" if np.all(np.diff(ts) > 0) else "FAIL", f"t {ts[0]:.3f}..{ts[-1]:.3f}")
    rep.add("hold clipped steps", "PASS" if clips == 0 else "WARN", f"{clips}/{n} ticks clipped while holding")

    # 4. cameras
    if env.backend.name == "mock":
        for k in CAMS:
            rep.add(f"camera {k}", "SKIP", "mock backend returns blank images")
        rep.add("head_depth nonzero", "SKIP", "mock backend")
    else:
        for k in CAMS:
            f = frames[k]
            blank = float(np.std(f[-1])) < 1.0
            changing = max(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) for a, b in zip(f, f[1:])) > 0.05
            rep.add(f"camera {k}", "FAIL" if blank else ("PASS" if changing else "WARN"),
                    "blank" if blank else ("live, changing" if changing else "static (frozen stream or very still scene)"))
        rep.add("head_depth nonzero", "PASS" if depth_nz > 0.05 else "WARN",
                f"max nonzero fraction {depth_nz:.3f}" + ("" if depth_nz > 0.05 else " (depth server off?)"))
    # 5. latency
    rep.add("obs latency", "SKIP", "obs carries no sensor timestamps yet")
    return obs


def run_to(env, obs, arm_tgt, hand_tgt, secs, tag):
    """Step with a linear ramp from the current command to the target; abort on any clip.
    Ramps from the filter's LAST COMMAND, not the measured pose: under gravity the measured arm sags a few
    hundredths of a rad below the command, and starting from it looks like a velocity jump to the filter."""
    a0, h0 = env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy()
    n = max(1, int(secs * HZ))
    for i in range(1, n + 1):
        s = i / n
        obs, info = env.step({"arm_q_target": a0 + s * (arm_tgt - a0), "hand_q_target": h0 + s * (hand_tgt - h0)})
        if info["clipped"] or info["safety_mode"] != "ok":
            raise Abort(f"{tag}: safety filter {info['safety_mode']} {info['clip_reasons']}")
    return obs


def settle(env, obs, arm_tgt, hand_tgt, secs, tag):
    return run_to(env, obs, arm_tgt, hand_tgt, secs, tag) if secs > 0 else obs


def probe_arm(env, obs, rep, i, name, amp, speed):
    q0, h0 = obs["arm_q"].astype(float), obs["hand_q"].astype(float)
    rampt = amp / speed
    result = []
    for sgn in (+1, -1):
        tgt = q0.copy(); tgt[i] += sgn * amp
        obs = run_to(env, obs, tgt, h0, rampt, name)
        obs = run_to(env, obs, tgt, h0, 0.6, name)  # hold so the measurement settles
        moved = float(obs["arm_q"][i] - q0[i])
        others = np.delete(np.abs(obs["arm_q"] - q0), i).max()
        result.append((sgn, moved, others))
        obs = run_to(env, obs, q0, h0, rampt, name)
        obs = run_to(env, obs, q0, h0, 0.6, name)
    # Judge sign + mapping by the SPAN between the +amp and -amp holds: on the real arm (kp 80, gearbox
    # friction) each single offset is skewed by a few hundredths of a rad of hysteresis, the span is not.
    span = result[0][1] - result[1][1]
    wrong = span < 0.5 * 2 * amp or any(m * sg <= 0 for sg, m, _ in result)
    back = abs(float(obs["arm_q"][i] - q0[i]))
    detail = (", ".join(f"{'+' if s > 0 else '-'}{amp}: moved {m:+.3f} (max other {o:.3f})" for s, m, o in result)
              + f"; span {span:.3f}/{2 * amp:.1f}")
    if wrong:
        rep.add(f"arm probe {name}", "FAIL", f"wrong/no direction: {detail}")
        raise Abort(f"{name} moved the wrong way")
    status = "PASS" if back < 0.04 else "WARN" if back < 0.1 else "FAIL"
    rep.add(f"arm probe {name}", status, f"{detail}; return error {back:.3f}")
    if status == "FAIL":
        raise Abort(f"{name} did not return")
    return obs


def probe_hand(env, obs, rep, i, name, lo, hi, speed):
    q0, a0 = obs["hand_q"].astype(float), obs["arm_q"].astype(float)
    up, down = hi - q0[i], q0[i] - lo
    sgn = 1.0 if up >= down else -1.0     # close toward the more distant limit
    amp = min(0.4, 0.6 * max(up, down))
    if amp < 0.05:
        rep.add(f"hand probe {name}", "SKIP", "no room to move")
        return obs
    tgt = q0.copy(); tgt[i] += sgn * amp
    obs = run_to(env, obs, a0, tgt, amp / speed, name)
    obs = run_to(env, obs, a0, tgt, 0.8, name)
    moved = float(obs["hand_q"][i] - q0[i])
    obs = run_to(env, obs, a0, q0, amp / speed, name)
    obs = run_to(env, obs, a0, q0, 0.8, name)
    back = abs(float(obs["hand_q"][i] - q0[i]))
    detail = f"cmd {sgn * amp:+.2f}, moved {moved:+.3f}, return error {back:.3f}"
    if moved * sgn < 0.25 * amp:
        rep.add(f"hand probe {name}", "FAIL", "wrong/no direction: " + detail + " (Dex3 sign mirroring? spec 11)")
        raise Abort(f"{name} moved the wrong way")
    rep.add(f"hand probe {name}", "PASS" if back < 0.1 else "FAIL", detail)
    return obs


def active(env, obs, rep, arm_amp=0.1, arm_speed=0.25, hand_speed=1.0, arms=True, open_hands=False):
    sp = JointSpec()
    cfg = SafetyConfig()
    lo_h, hi_h = sp.hand_lower + cfg.hand_pos_margin, sp.hand_upper - cfg.hand_pos_margin
    try:
        if arms:
            for i, name in enumerate(sp.arm_names):
                obs = probe_arm(env, obs, rep, i, name, arm_amp, arm_speed)
        if open_hands:  # a clenched hand blocks the thumb probes: go to the open pose (zeros, inside limits)
            h0 = env.filter.hand_cmd.copy()
            tgt = np.clip(np.zeros(N), lo_h, hi_h)
            secs = float(np.abs(tgt - h0).max()) / hand_speed
            obs = run_to(env, obs, env.filter.arm_cmd.copy(), tgt, secs, "open hands")
            obs = run_to(env, obs, env.filter.arm_cmd.copy(), tgt, 1.0, "open hands")
        for i, name in enumerate(sp.hand_names):
            obs = probe_hand(env, obs, rep, i, name, lo_h[i], hi_h[i], hand_speed)
    except Abort as e:
        rep.add("motion probes", "FAIL", f"ABORTED: {e}")
        try:  # hold where we are
            env.step(hold_action(env._obs()))
        except Exception:
            pass
    return obs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "sim", "real"], required=True)
    ap.add_argument("--network-interface", default=None)
    ap.add_argument("--img-server-ip", default="192.168.123.164")
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--no-head-rgbd", action="store_true",
                    help="real: take head colour from teleimager (no depth) instead of pc2/head_rgbd_server.py")
    ap.add_argument("--no-counter-keepout", action="store_true",
                    help="drop the counter keep-out box (only when no counter is in front of the robot)")
    ap.add_argument("--raise-arms", type=float, default=0.0, metavar="RAD",
                    help="add this to both shoulder pitches (e.g. -0.3 = hands forward/up) before the probes, "
                         "undo it after; for a seated robot whose hands rest on its thighs")
    ap.add_argument("--hands-only", action="store_true", help="skip the 14 arm probes")
    ap.add_argument("--open-hands", action="store_true", help="open both Dex3 (zeros) before the hand probes")
    ap.add_argument("--yes", action="store_true", help="skip the YES prompt (tests only; never on the real robot)")
    ap.add_argument("--passive-secs", type=float, default=3.0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--head-rgbd", action="store_true",
                    help="head RGB-D from :55560 (sim: Aalto-* tasks publish it; real has it on by default)")
    a = ap.parse_args(argv)

    kw = {} if a.backend == "mock" else {"network_interface": a.network_interface, "img_server_ip": a.img_server_ip}
    if a.head_rgbd or (a.backend == "real" and not a.no_head_rgbd):
        kw["head_rgbd"] = True
    go = not a.no_motion
    if go and a.backend != "mock" and not a.yes:
        print("Motion probes: each arm joint +-0.1 rad, each Dex3 joint a small close/open, through the safety filter.")
        if input("Type YES to move (human at the e-stop, robot on the gantry): ").strip() != "YES":
            print("not confirmed -> passive checks only")
            go = False
    rep = Report()
    safety = SafetyConfig(dt=1.0 / HZ, counter_keepout_min=None, counter_keepout_max=None) if a.no_counter_keepout else None
    env = AaltoEnv(backend=a.backend, safety=safety, **kw)
    try:
        obs = env.reset()
        print(f"== check_interface backend={a.backend} ==")
        obs = passive(env, obs, rep, a.passive_secs)
        if go:
            start = env.filter.arm_cmd.copy()
            try:
                if a.raise_arms:  # lift the hands off the lap/thighs so the probes are not blocked
                    tgt = start.copy(); tgt[0] += a.raise_arms; tgt[7] += a.raise_arms  # shoulder pitches
                    obs = run_to(env, obs, tgt, env.filter.hand_cmd.copy(), abs(a.raise_arms) / 0.25, "raise")
                    obs = run_to(env, obs, tgt, env.filter.hand_cmd.copy(), 1.0, "raise")
                obs = active(env, obs, rep, arms=not a.hands_only, open_hands=a.open_hands)
                if a.raise_arms:
                    obs = run_to(env, obs, start, env.filter.hand_cmd.copy(), abs(a.raise_arms) / 0.25, "lower")
                    obs = run_to(env, obs, start, env.filter.hand_cmd.copy(), 0.6, "lower")
            except Abort as e:
                rep.add("raise/lower", "FAIL", f"ABORTED: {e} (holding)")
        else:
            rep.add("motion probes", "SKIP", "--no-motion or not confirmed")
    finally:
        env.close()
    path = Path(a.report or f"logs/check_interface_{a.backend}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(r["status"] == s for r in rep.rows) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    path.write_text(json.dumps({"backend": a.backend, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "motion": go, "counts": counts, "checks": rep.rows}, indent=2))
    print(f"\n{counts}  report: {path}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
