"""Pick-only test on the real robot (M8 bring-up): no bins, no home pose.

Looks once from wherever the arms are, picks the nearest reachable object (any configured colour), lifts it
`--lift` m, holds it `--hold` s, puts it back where it was, opens, retreats up and returns the arm to its start
pose. Every command goes through AaltoEnv's safety filter. Prints the plan and waits for Enter before moving.

    python -m aaltonomous.baselines.classical.pick_test --network-interface enx6c6e071d1d7c \\
        --config aaltonomous/baselines/classical/config_lab.yaml --scene aaltonomous/scene/scene_lab.yaml
    add --dry-run to only look + plan (no DDS, no motion).

Needs PC2 head_rgbd_server.py (scripts/deploy_rgbd_pc2.sh), robot on the gantry, spotter on the e-stop.
Head frames are saved to logs/classical/<date>/ (…_pick.png/.npz).
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

from .kinematics import ArmKinematics
from .planner import Plan, ramp, time_scale
from .policy import ClassicalPolicy
from .run import apply_scene_keepout, log_dir, save_frame, stdin_reader

DT = 1.0 / 30.0


def plan_pick(pl, det, side, arm_q, hand_q, lift, hold_s):
    """pregrasp -> descend -> close -> lift -> hold -> lower -> open -> retreat -> start pose (one arm)."""
    g, mo = pl.cfg["grasp"], pl.cfg["motion"]
    zc, H, post = pl.scene.counter_z, pl.height, pl.post
    plan = Plan(False, side=side, color=det.color, target_bin="put back")
    sl = ArmKinematics.sl(side)
    hand_pre = hand_q.copy(); hand_pre[sl] = post["preshape"][sl]
    hand_cls = hand_q.copy(); hand_cls[sl] = post["closed"][sl]
    hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]

    xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
    z = zc + g["height_frac"] * H  # grasp height, raised until the preshaped fingertips clear the counter
    for _ in range(4):
        qg, ok = pl._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
        if not ok:
            plan.reason = f"IK grasp {side}"
            return plan
        c = pl.clearance(side, qg, hand_pre, fingers_only=True)
        if c >= g["fingertip_clearance"] - 1e-3:
            break
        z += g["fingertip_clearance"] - c
    else:
        plan.reason = "fingers hit the counter"
        return plan
    if z > zc + H - 0.008:
        plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top"
        return plan

    P = {"pregrasp": np.r_[xy, z + g["approach_height"]], "grasp": np.r_[xy, z], "lift": np.r_[xy, z + lift],
         "release": np.r_[xy, z + 0.02]}  # opening swings the thumb down: release 2 cm up (cube drops 2 cm)
    q_pre, ok = pl._ik(side, P["pregrasp"], arm_q, "pregrasp", plan)
    if not ok:
        plan.reason = f"IK pregrasp {side}"
        return plan
    down = pl._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
    up = down and pl._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
    back = up and pl._cartesian(side, P["lift"], P["release"], up[-1], "lower", plan)
    away = back and pl._cartesian(side, P["release"], P["pregrasp"], back[-1], "retreat", plan)
    if not away:
        plan.reason = f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
        return plan
    plan.waypoints["grasp"] = P["grasp"]

    vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
    parts_a, parts_h, phases = [], [], []
    cur = {"a": arm_q.copy(), "h": hand_q.copy()}

    def add(name, a, h):
        if len(a):
            s = sum(len(x) for x in parts_a)
            parts_a.append(a); parts_h.append(h); phases.append((name, s, s + len(a)))
            cur["a"] = a[-1].copy()

    def move(name, Qs, hand_to=None):
        a = time_scale([cur["a"]] + list(Qs), vmax, amax, DT)
        h = np.repeat(cur["h"][None], len(a), 0)
        if hand_to is not None:
            hr = ramp(cur["h"], hand_to, hv, DT)
            if len(hr) > len(a):
                a = np.vstack([a, np.repeat((a[-1] if len(a) else cur["a"])[None], len(hr) - len(a), 0)])
                h = np.repeat(cur["h"][None], len(a), 0)
            h[:len(hr)] = hr
            h[len(hr):] = hand_to
            cur["h"] = np.asarray(hand_to, float).copy()
        add(name, a, h)

    def hand(name, to, hold):
        h = ramp(cur["h"], to, hv, DT)
        h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold / DT)), 0)])
        cur["h"] = np.asarray(to, float).copy()
        add(name, np.repeat(cur["a"][None], len(h), 0), h)

    # leave the start pose upwards first: shoulder pitch forward lifts the hand ~11 cm, monotonically, from a
    # low resting pose (a direct joint move to pregrasp dips first). Returns the same way.
    q_up = arm_q.copy(); i = sl.start
    q_up[i] = max(arm_q[i] - 0.4, pl.spec.arm_lower[i] + 0.1)
    move("raise", [q_up])
    move("to_pregrasp", [q_pre], hand_to=hand_pre)
    move("descend", down)
    hand("close", hand_cls, mo["close_hold_s"])
    move("lift", up)
    hand("hold", hand_cls, hold_s)
    move("lower", back)
    hand("open", hand_open, mo["open_hold_s"])
    move("retreat", away)
    move("to_up", [q_up], hand_to=hand_q)
    move("to_start", [arm_q])
    plan.arm, plan.hand, plan.phases = np.vstack(parts_a), np.vstack(parts_h), phases

    near = np.zeros(len(plan.arm), bool)  # fingers may be close to the counter only around the object
    for name, s, e in phases:
        if name in ("descend", "close", "lift", "hold", "lower", "open", "retreat"):
            near[s:e] = True
    c0 = pl.clearance(side, arm_q, hand_q)  # arms may start low (e.g. resting near the table)
    for k in range(0, len(plan.arm), 3):
        name = next(n for n, s, e in phases if s <= k < e)
        need = 0.003 if name in ("close", "open") else g["fingertip_clearance"] if near[k] else mo["counter_clearance"]
        if name in ("raise", "to_start"):
            need = min(need, c0)  # leaving / returning to the start pose: never lower than where it started
        c = pl.clearance(side, plan.arm[k], plan.hand[k])
        if c < need - 2e-3:
            plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
            return plan
    plan.safety_clips = pl.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
    if plan.safety_clips:
        plan.reason = f"{plan.safety_clips} safety clips in dry run"
        return plan
    plan.ok = True
    return plan


def choose(pol, obs, arm_q, hand_q, a):
    dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
    if a.color:
        dets = [d for d in dets if d.color == a.color]
    dets.sort(key=lambda d: np.linalg.norm(d.top[:2] - np.array([0.35, 0.0])))  # centre of the reach band first
    print(f"{len(dets)} object(s): " + "; ".join(f"{d.color} ({d.top[0]:+.2f},{d.top[1]:+.2f})" for d in dets))
    for d in dets:
        sides = [a.side] if a.side else (["left", "right"] if d.top[1] > 0 else ["right", "left"])
        why = []
        for side in sides:
            p = plan_pick(pol.planner, d, side, arm_q, hand_q, a.lift, a.hold)
            if p.ok:
                return p
            why.append(f"{side}: {p.reason}")
        print(f"  skip {d.color} ({d.top[0]:+.2f},{d.top[1]:+.2f}): {'; '.join(why)}")
    return None


def describe(p):
    g = p.waypoints["grasp"]
    return (f"{p.color} with the {p.side} arm at ({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}) m pelvis, "
            f"{p.duration:.1f} s, safety_clips={p.safety_clips}")


def head_frame(a):
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    cli = HeadRGBDClient(a.img_server_ip, a.head_rgbd_port)
    f, t0 = None, time.time()
    while f is None and time.time() - t0 < 5.0:
        f = cli.get()
        time.sleep(0.05)
    cli.close()
    if f is None:
        sys.exit("no head RGB-D frame (is head_rgbd_server.py running on PC2?)")
    return {"head_rgb": f[0], "head_depth": f[1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network-interface", default="enx6c6e071d1d7c")
    ap.add_argument("--img-server-ip", default="192.168.123.164")
    ap.add_argument("--head-rgbd-port", type=int, default=55560)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--color", default=None, help="only this colour class (red/yellow/blue in the config)")
    ap.add_argument("--side", choices=["left", "right"], default=None, help="force one arm")
    ap.add_argument("--lift", type=float, default=0.10, help="m lifted above the grasp height")
    ap.add_argument("--hold", type=float, default=2.0, help="s held in the air")
    ap.add_argument("--arm-vel", type=float, default=0.3, help="rad/s per joint")
    ap.add_argument("--dry-run", action="store_true", help="look + plan from the measured arm pose, no motion")
    ap.add_argument("--log-root", default="logs/classical")
    a = ap.parse_args(argv)

    pol = ClassicalPolicy(config=a.config, scene=a.scene, overrides={"motion.arm_max_vel": a.arm_vel})
    pol.scene.cfg["bins"]["centers"] = {}  # no bins in this test: don't drop objects lying where bins would be
    d = log_dir(a)

    if a.dry_run:  # arms assumed where they are now: read them read-only over DDS
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        import yaml
        from types import SimpleNamespace
        from aaltonomous.interface.safety import DEFAULT_SPEC
        ChannelFactoryInitialize(0, a.network_interface)
        box = {}
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(lambda m: box.__setitem__("s", m), 10)
        t0 = time.time()
        while "s" not in box and time.time() - t0 < 5:
            time.sleep(0.05)
        spec = yaml.safe_load(open(DEFAULT_SPEC))
        arm_q = np.array([box["s"].motor_state[j["dds_index"]].q for j in spec["arm"]])
        arm_q = np.clip(arm_q, pol.spec.arm_lower, pol.spec.arm_upper)
        apply_scene_keepout(SimpleNamespace(filter=SimpleNamespace(cfg=pol.safety, ws=None, spec=None), scene=pol.scene))
        print(f"measured arm_q {np.round(arm_q, 2)}")
        obs = head_frame(a)
        save_frame(pol, obs, d, "pickdry")
        p = choose(pol, obs, arm_q, pol.post["open"].copy(), a)
        print("PLAN: " + describe(p) if p else "no pickable object")
        return

    from aaltonomous.interface.env import AaltoEnv
    env = AaltoEnv(backend="real", task="sort_fixed", time_limit=1e9, network_interface=a.network_interface,
                   img_server_ip=a.img_server_ip, head_rgbd=True, head_rgbd_port=a.head_rgbd_port,
                   safety=pol.safety)
    env.scene = pol.scene
    apply_scene_keepout(env)
    keys = queue.Queue()
    try:
        obs = env.reset()  # prints a layout card + waits for Enter: ignore the card, just press Enter
        threading.Thread(target=stdin_reader, args=(keys,), daemon=True).start()  # after reset's own input()
        arm0 = np.clip(np.asarray(obs["arm_q"], float), pol.spec.arm_lower, pol.spec.arm_upper)
        hand0 = np.clip(np.asarray(obs["hand_q"], float), pol.spec.hand_lower, pol.spec.hand_upper)
        hold = {"arm_q_target": arm0, "hand_q_target": hand0}
        while not keys.empty():
            keys.get_nowait()

        job = {"plan": None, "done": False}

        def work(o=obs):
            save_frame(pol, o, d, "pick")
            job["plan"] = choose(pol, o, arm0, hand0, a)
            job["done"] = True
        threading.Thread(target=work, daemon=True).start()
        print("looking + planning (holding the arms still)...")
        while not job["done"]:
            obs, info = env.step(hold)  # keep the safety watchdog fed while planning
            if info["episode_over"]:
                sys.exit(f"stopped: {info['end_reason']}")
        p = job["plan"]
        if p is None:
            print("no pickable object: move one into reach (~0.26-0.40 m in front of the waist) and rerun")
            return
        print(f"\n>>> PICK {describe(p)}\n>>> phases: " + ", ".join(f"{n} {(e - s) / 30:.1f}s" for n, s, e in p.phases))
        print(">>> Enter = go, q = quit")
        while True:
            obs, info = env.step(hold)
            if info["episode_over"]:
                sys.exit(f"stopped: {info['end_reason']}")
            try:
                k = keys.get_nowait()
            except queue.Empty:
                continue
            if k == "q":
                print("quit, nothing moved")
                return
            if k == "":
                break
        print("executing (q + Enter = stop and hold)")
        for i in range(0, len(p.arm), 10):
            if not keys.empty() and keys.get_nowait() == "q":
                print("operator stop: holding")
                break
            obs, info = env.step({"arm_q_target": p.arm[i:i + 10], "hand_q_target": p.hand[i:i + 10]})
            if info["clipped"]:
                print(f"[safety] t={info['t']:.2f} clipped: {info['clip_reasons']}")
            if info["episode_over"]:
                print(f"stopped: {info['end_reason']}")
                break
        print(f"done. safety={env.filter.summary()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
