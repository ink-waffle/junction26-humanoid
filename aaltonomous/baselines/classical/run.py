"""Run the classical rung-1 sorter through AaltoEnv (mock now, real at the venue).

    python -m aaltonomous.baselines.classical.run --backend mock [--seed 3] [--scene PATH]
    python -m aaltonomous.baselines.classical.run --backend mock --dry-run
    python -m aaltonomous.baselines.classical.run --backend real --network-interface enx00e04cd30dba --dry-run
    python -m aaltonomous.baselines.classical.run --backend real --network-interface enx00e04cd30dba

--dry-run: one head frame -> detections, grasp/drop targets, IK + safety pre-check per object. NO motion: on real it
           only opens the PC2 head RGB-D stream (no DDS, no arm controller) and assumes the arms at home_arm_q.
real:      needs PC2 `head_rgbd_server.py` running (teleimager head camera OFF), robot in debug mode on the gantry.
           Before the first move (to home) and before EACH object it prints the plan and waits:
           Enter = go, s+Enter = skip object, q+Enter = stop.
           The arms hold their target while waiting. Human e-stop stays with the spotter.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np


def build_policy(a, **kw):
    from .policy import ClassicalPolicy
    ov = {}
    if a.arm_vel is not None:
        ov["motion.arm_max_vel"] = a.arm_vel
    from .config import load_config
    profile = a.profile if a.profile is not None else ("none" if a.backend == "real" else None)
    return ClassicalPolicy(config=load_config(a.config, ov, profile), scene=a.scene, **kw)


def print_plan_table(pol, obs, header=""):
    dets, rej = pol.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True)
    print(f"\n=== {header} {len(dets)} object(s) detected, {len(rej)} blob(s) rejected ===")
    for d in dets:
        print("  det ", d, " mat(cm)=", np.round(pol.scene.pelvis_to_mat(d.top)[:2] * 100, 1))
    for d, why in rej:
        print("  rej ", d, "->", why)
    arm = np.asarray(obs["arm_q"], float)
    hand = np.asarray(obs["hand_q"], float)
    print("  plans (each from the given start pose):")
    n_ok = 0
    for d in dets:
        p = pol.planner.plan(d, arm, hand)
        n_ok += p.ok
        print("   ", p.summary())
    print(f"  {n_ok}/{len(dets)} objects have a valid plan")
    return dets


def dry_run(a):
    if a.backend == "mock":
        from .mock_world import make_mock_env
        env = make_mock_env(a.scene, seed=a.seed)
        obs = env.reset()
        pol = build_policy(a)
        obs = dict(obs, arm_q=pol.home, hand_q=pol.post["open"])  # plans from home, as after the first move
        print(f"layout: {env.layout}")
    else:
        from aaltonomous.interface.head_rgbd import HeadRGBDClient
        pol = build_policy(a)
        cli = HeadRGBDClient(a.img_server_ip, a.head_rgbd_port)
        t0 = time.time()
        f = None
        while f is None and time.time() - t0 < 5.0:
            f = cli.get()
            time.sleep(0.05)
        cli.close()
        if f is None:
            sys.exit(f"no head RGB-D frame from {a.img_server_ip}:{a.head_rgbd_port} (is head_rgbd_server.py running on PC2?)")
        rgb, depth = f[0], f[1]
        print(f"frame {f[3]}: depth valid {100 * (depth > 0).mean():.0f}%")
        obs = {"head_rgb": rgb, "head_depth": depth, "arm_q": pol.home, "hand_q": pol.post["open"]}
    print_plan_table(pol, obs, "DRY RUN (no motion):")
    if a.save:
        save_debug(pol, obs, a.save)
    if a.backend == "real":
        save_frame(pol, obs, log_dir(a), "dry")


def apply_scene_keepout(env):
    """Counter keep-out follows env.scene's table: top = counter surface + 2 cm, near edge = mat edge."""
    from aaltonomous.interface.safety import WorkspaceChecker
    c = env.filter.cfg
    if c.counter_keepout_min is None:
        return
    c.counter_keepout_min = (float(env.scene.origin[0]),) + tuple(c.counter_keepout_min[1:])
    c.counter_keepout_max = tuple(c.counter_keepout_max[:2]) + (env.scene.counter_z + 0.02,)
    if env.filter.ws is not None:
        env.filter.ws = WorkspaceChecker(env.filter.spec, c)
    print(f"counter keep-out from scene: {c.counter_keepout_min} .. {c.counter_keepout_max}")


def log_dir(a):
    """logs/classical/<YYYY-MM-DD>/ (gitignored): every real head frame is kept for later review."""
    import os
    d = os.path.join(a.log_root, time.strftime("%Y-%m-%d"))
    os.makedirs(d, exist_ok=True)
    return d


def save_frame(pol, obs, d, tag):
    """Annotated PNG + raw RGB-D (.npz) of one head frame, named <HHMMSS>_<tag>."""
    import os
    stem = os.path.join(d, time.strftime("%H%M%S") + f"_{tag}")
    try:
        save_debug(pol, obs, stem + ".png")
        np.savez_compressed(stem + ".npz", rgb=obs["head_rgb"], depth=obs["head_depth"])
    except Exception as e:  # logging must never stop the robot loop
        print(f"[log] could not save {stem}: {e!r}")


def save_debug(pol, obs, path):
    import cv2
    img = cv2.cvtColor(np.ascontiguousarray(obs["head_rgb"]), cv2.COLOR_RGB2BGR)
    for c, m in pol.perception.masks(obs["head_rgb"]).items():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)
    for d in pol.perception.detect(obs["head_rgb"], obs["head_depth"]):
        u, v, _ = pol.cal.project("head", d.top)
        cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, d.color, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(path, img)
    print(f"debug image -> {path}")


def stdin_reader(q):
    for line in sys.stdin:
        q.put(line.strip().lower())


def run(a):
    from aaltonomous.interface.env import AaltoEnv
    real = a.backend == "real"
    if a.backend == "mock":
        from .mock_world import make_mock_env
        env = make_mock_env(a.scene, seed=a.seed, realtime=a.realtime)
    else:
        env = AaltoEnv(backend=a.backend, task="sort_fixed", seed=a.seed, time_limit=a.time_limit,
                       network_interface=a.network_interface, img_server_ip=a.img_server_ip,
                       head_rgbd=True, head_rgbd_port=a.head_rgbd_port)
        if a.scene:
            from aaltonomous.scene.scene import Scene
            env.scene = Scene.load(a.scene)
            apply_scene_keepout(env)
    env.time_limit = a.time_limit
    confirm = real and not a.no_confirm
    pol = build_policy(a, safety=env.filter.cfg, require_confirm=confirm, threaded=(a.backend != "mock"),
                       time_limit=a.time_limit)
    if real:  # keep every head frame the policy plans from (annotated + raw), see log_dir()
        d, look, n = log_dir(a), pol._look, [0]

        def logged_look(obs, _look=look):
            res = _look(obs)
            n[0] += 1
            save_frame(pol, obs, d, f"look{n[0]:02d}_{res[0]}")
            return res
        pol._look = logged_look
        print(f"head frames -> {d}/")
    try:
        obs = env.reset()  # real: prints the layout card and waits for Enter
        pol.reset({"instruction": obs["instruction"]})
        keys = queue.Queue()
        if confirm:
            threading.Thread(target=stdin_reader, args=(keys,), daemon=True).start()
        prompted = False
        info = {}
        while True:
            act = pol.act(obs)
            if pol.waiting and not prompted:
                print(f"\n>>> NEXT: {pol.pending_text}\n>>> Enter = go, s = skip, q = stop")
                prompted = True
            if pol.waiting:
                try:
                    k = keys.get_nowait()
                    {"": pol.confirm, "s": pol.skip, "q": pol.stop}.get(k, lambda: print("Enter / s / q"))()
                    prompted = False
                except queue.Empty:
                    pass
            obs, info = env.step(act)
            if info["clipped"]:
                print(f"[safety] t={info['t']:.2f} clipped: {info['clip_reasons']}")
            if info["episode_over"]:
                break
        print(f"\nepisode over: {info['end_reason']}  t={info['t']:.1f}s  safety={info.get('safety_summary')}")
        if a.backend == "mock":
            outc = env.backend.outcomes(pol.mapping)
            print(f"mock outcomes: {outc}  score={env.scene.score('sort_fixed', outc)}")
        return info
    except KeyboardInterrupt:
        print("\ninterrupted: holding last target, closing")
    finally:
        env.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("mock", "sim", "real"), default="mock")
    ap.add_argument("--network-interface", default=None)
    ap.add_argument("--img-server-ip", default="192.168.123.164")
    ap.add_argument("--head-rgbd-port", type=int, default=55560)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None, help="config yaml (default classical/config.yaml)")
    ap.add_argument("--scene", default=None, help="scene yaml (default scene/scene.yaml)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--profile", default=None, help="config profile (default: `none` = robot values for --backend "
                    "real, config.yaml `profile:` (sim) otherwise)")
    ap.add_argument("--arm-vel", type=float, default=None, help="override motion.arm_max_vel (rad/s)")
    ap.add_argument("--no-confirm", action="store_true", help="real: do NOT ask before each object (not advised)")
    ap.add_argument("--realtime", action="store_true", help="mock: run at wall-clock 30 Hz")
    ap.add_argument("--save", default=None, help="dry run: write an annotated head image here")
    ap.add_argument("--log-root", default="logs/classical", help="real: every head frame used is saved here")
    a = ap.parse_args(argv)
    if a.backend == "real" and not a.network_interface and not a.dry_run:
        ap.error("--backend real needs --network-interface")
    return dry_run(a) if a.dry_run else run(a)


if __name__ == "__main__":
    main()
