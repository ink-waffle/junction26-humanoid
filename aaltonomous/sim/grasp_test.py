"""Single-object grasp sweep for the oracle in the Aalto sim: one object per episode (shape x yaw x position), one
pick attempt (short time limit), outcome from the scene checker. Finds which shapes / poses slip.

    python -m aaltonomous.sim.grasp_test --shapes cube,triangle --yaws -45,-20,0,20,45 [--colors red] [--out f.json]
(env `isaac`, PYTHONPATH=$REPO; AALTO_INSTANCE env vars for the 2nd sim as in evaluate.py)
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading

import numpy as np

from aaltonomous.sim import evaluate as ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="cube,cylinder,triangle")
    ap.add_argument("--colors", default="red,yellow,blue")
    ap.add_argument("--yaws", default="-45,0,30", help="deg, mat frame")
    ap.add_argument("--xs", default=None, help="mat x (m); default 4 points across layout.region_x")
    ap.add_argument("--ys", default=None, help="mat y (m); default the middle of layout.region_y")
    ap.add_argument("--time-limit", type=float, default=22.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print the object pose + plan phase every 0.25 s")
    a = ap.parse_args(argv)

    import tempfile
    from pathlib import Path
    from aaltonomous.deploy.runner import Recorder, run_episode
    tmp = Path(tempfile.mkdtemp(prefix="grasp_test_"))
    from aaltonomous.interface import env as env_mod
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.oracle import OraclePolicy
    from aaltonomous.sim.scene_client import SceneClient

    ev.apply_instance_env()
    import os
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, realtime=True,
                   img_server_ip=a.host, head_rgbd=False, cameras=False,
                   head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    L = env.scene.cfg["layout"]
    xs = [float(v) for v in a.xs.split(",")] if a.xs else list(np.linspace(*L["region_x"], 4))
    ys = [float(v) for v in a.ys.split(",")] if a.ys else [float(np.mean(L["region_y"]))]
    pol = OraclePolicy(SceneClient(a.host), safety=env.filter.cfg, time_limit=a.time_limit,
                       log=(print if a.verbose else None))
    client = ev.LocalClient(pol)
    rows, stop = [], threading.Event()
    cases = list(itertools.product(a.shapes.split(","), [float(y) for y in a.yaws.split(",")], xs, ys))
    colors = itertools.cycle(a.colors.split(","))
    try:
        for i, (shape, yaw, x, y) in enumerate(cases):
            color = next(colors)
            lay = [{"color": color, "shape": shape, "x": x, "y": y, "yaw": float(np.radians(yaw))}]
            env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
            ev.home_arms(env, sc)
            tr_stop = threading.Event()
            if a.trace:
                threading.Thread(target=_trace, args=(SceneClient(a.host), pol, tr_stop), daemon=True).start()
            summ, _ = run_episode(env, client, i, i, Recorder(tmp / f"ep{i:03d}"), stop, "sort_fixed")
            tr_stop.set()
            ev.settle(sc, 3.0, float(env.scene.cfg["success"]["at_rest_speed"]))
            out = sc.evaluate("sort_fixed", env.mapping)["outcomes"][0]["outcome"]
            fin = next(o for o in sc.state()["objects"] if o["active"])
            r = {"shape": shape, "color": color, "yaw": yaw, "x": round(x, 3), "y": round(y, 3), "outcome": out,
                 "end": summ.get("end_reason"), "final": [round(v, 3) for v in fin["pos"]],
                 "events": [e for e in pol.events[-6:]]}
            rows.append(r)
            print(f"{i:3d} {shape:8s} {color:6s} yaw={yaw:+4.0f} x={x:.2f} y={y:.2f} -> {out:8s} "
                  f"final=({fin['pos'][0]:+.2f},{fin['pos'][1]:+.2f},{fin['pos'][2]:+.3f})", flush=True)
    finally:
        client.close()
        env.close()
    by = {}
    for r in rows:
        by.setdefault(r["shape"], []).append(r["outcome"] == "correct")
    for s, v in by.items():
        print(f"{s:8s} {sum(v)}/{len(v)} correct")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1, default=str)


def _trace(sc, pol, stop):
    import time
    t0 = time.monotonic()
    while not stop.is_set():
        o = next((o for o in sc.state()["objects"] if o["active"]), None)
        if o is None:
            time.sleep(0.1)
            continue
        p, q = o["pos"], o["quat_xyzw"]
        yaw = np.degrees(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        tilt = np.degrees(2 * np.arcsin(min(1.0, float(np.hypot(q[0], q[1])))))
        print(f"  t={time.monotonic() - t0:5.2f} pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) yaw={yaw:+5.0f} tilt={tilt:3.0f}",
              flush=True)
        time.sleep(0.25)


if __name__ == "__main__":
    main()
