"""Run the classical rung-1 sorter on the Aalto Isaac sim over several random layouts and score each episode.

    # twinflow, env `isaac`, with `TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh` running
    python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2 [--time-limit 120] [--arm-vel 0.8]

Each episode: AaltoEnv.reset() places a sample_layout (scene service :55580) -> policy runs through the safety
filter at 30 Hz -> SceneClient.evaluate("sort_fixed") scores the final scene (Scene checker, spec §3).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm-vel", type=float, default=None, help="override motion.arm_max_vel (rad/s; filter allows 1.0)")
    ap.add_argument("--profile", default=None, help="config profile (default: config.yaml `profile:` = sim)")
    ap.add_argument("--set", default=None, help='extra overrides as json, e.g. \'{"motion.arm_max_vel": 0.8}\'')
    ap.add_argument("--log", default=None, help="append one json line per episode here")
    a = ap.parse_args(argv)

    import ssl  # noqa: F401  before DDS (see sim/evaluate.py)
    from aaltonomous.sim.evaluate import apply_instance_env
    apply_instance_env()  # 2nd sim instance: AALTO_DDS_DOMAIN / AALTO_CAM_PORT_BASE
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from .policy import ClassicalPolicy

    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, img_server_ip=a.host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))  # 2nd sim instance
    sc = SceneClient(a.host)
    from .config import load_config
    ov = json.loads(a.set) if a.set else {}
    if a.arm_vel is not None:
        ov["motion.arm_max_vel"] = a.arm_vel
    pol = ClassicalPolicy(config=load_config(a.config, ov, a.profile), safety=env.filter.cfg, threaded=True, time_limit=a.time_limit)
    rows = []
    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()  # the policy goes home + settles before its first look (fresh frame)
            pol.reset({"instruction": obs["instruction"]})
            clips, info = [], {}
            while True:
                obs, info = env.step(pol.act(obs))
                if info["clipped"]:
                    clips.append((round(info["t"], 2), info["clip_reasons"]))
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            ev = sc.evaluate("sort_fixed")
            n_ok = sum(1 for o in ev["outcomes"] if o.get("outcome") == "correct")
            row = {"seed": seed, "n_objects": len(env.layout), "correct": n_ok, "score": ev["score"],
                   "end": info["end_reason"], "t": round(info["t"], 1), "clip_ticks": len(clips),
                   "safety": info.get("safety_summary"), "outcomes": ev["outcomes"],
                   "events": pol.events}
            rows.append(row)
            print(f"\n### seed {seed}: {n_ok}/{len(env.layout)} correct, score {ev['score']}, "
                  f"end={info['end_reason']} t={info['t']:.0f}s clip_ticks={len(clips)} {clips[:5]}")
            for o in ev["outcomes"]:
                print("   ", o)
            if a.log:
                with open(a.log, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    finally:
        env.close()
    tot = sum(r["n_objects"] for r in rows)
    print(f"\nTOTAL {sum(r['correct'] for r in rows)}/{tot} correct over seeds {a.seeds}, "
          f"clip ticks {sum(r['clip_ticks'] for r in rows)}")
    return rows


if __name__ == "__main__":
    main()
