"""Replay recorded demo actions in the Aalto Isaac sim through AaltoEnv (safety filter on) and compare the motion.

    # source = a raw episode dir (record_demos.py) or an npz exported from the LeRobot dataset:
    #   (env lerobot) python aaltonomous/data/to_lerobot.py --verify ROOT --export-episode 0 --export-out ep0.npz
    # (env isaac, PYTHONPATH=$REPO; 2nd sim instance: same AALTO_* env vars as sim/evaluate.py)
    python -m aaltonomous.data.replay_sim ep0.npz [--host 127.0.0.1] [--json out.json]

Steps: park objects + arms to the sim home pose (sim/evaluate.home_arms), place the episode's layout, ramp to the
first action, then step every recorded action at 30 Hz and log the measured joints. Reports RMS / max error between
the replayed and the recorded joint trajectories, and the sim's scoring of the final object positions.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl  # noqa: F401  before DDS (see sim/evaluate.py)
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load(src):
    src = Path(src)
    if src.is_dir():  # raw episode
        low, meta = np.load(src / "lowdim.npz"), json.loads((src / "meta.json").read_text())
        return {"action": np.c_[low["action_arm"], low["action_hand"]], "state": np.c_[low["arm_q"], low["hand_q"]],
                "meta": meta}
    d = np.load(src, allow_pickle=False)
    return {"action": d["action"], "state": d["state"], "meta": json.loads(str(d["meta"]))}


def compare(rec_state, rep_state):
    e = rep_state - rec_state
    arm, hand = e[:, :14], e[:, 14:]
    return {"arm_rms": round(float(np.sqrt((arm ** 2).mean())), 4), "arm_max": round(float(np.abs(arm).max()), 4),
            "arm_rms_per_joint": np.round(np.sqrt((arm ** 2).mean(0)), 3).tolist(),
            "hand_rms": round(float(np.sqrt((hand ** 2).mean())), 4), "hand_max": round(float(np.abs(hand).max()), 4),
            "arm_range": round(float(np.ptp(rec_state[:, :14], 0).max()), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-clips", type=int, default=None, help="filter abort threshold (default: the filter's own, "
                    "30). Ticks over the 1 rad/s cap are clipped either way; a large value replays the whole episode")
    a = ap.parse_args(argv)
    from aaltonomous.baselines.classical.planner import ramp
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.evaluate import apply_instance_env, home_arms, settle
    from aaltonomous.sim.scene_client import SceneClient

    ep = load(a.src)
    A, S, meta = ep["action"].astype(float), ep["state"].astype(float), ep["meta"]
    apply_instance_env()
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task=meta["task"], realtime=True, img_server_ip=a.host, cameras=False,
                   time_limit=len(A) / 30.0 + 60.0, instruction=meta["instruction"])
    if a.max_clips is not None:
        env.filter.cfg.max_clipped_steps = a.max_clips
    try:
        home_arms(env, sc)
        orig = env.backend.reset_scene
        env.backend.reset_scene = lambda layout, instruction: orig(meta["layout"], instruction)
        env.reset()
        env.backend.reset_scene = orig
        a0, h0 = env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy()
        RA, RH = ramp(a0, A[0, :14], 0.6, env.dt), ramp(h0, A[0, 14:], 1.0, env.dt)
        n = max(len(RA), len(RH), 1) + 15
        pad = lambda R, x: np.vstack([R, np.repeat(x[None], n - len(R), 0)])
        for ar, hr in zip(pad(RA, A[0, :14]), pad(RH, A[0, 14:])):
            env.step({"arm_q_target": ar, "hand_q_target": hr})
        rep, clips, t0 = [], 0, time.monotonic()
        for act in A:
            arm_q, _, hand_q = env.backend.joints()
            rep.append(np.r_[arm_q, hand_q])
            _, info = env.step({"arm_q_target": act[:14], "hand_q_target": act[14:]})
            clips += info["clipped"]
            if info["episode_over"]:
                print(f"episode ended early: {info['end_reason']}")
                break
        wall = time.monotonic() - t0
        rep = np.array(rep)
        settle(sc, a.settle, float(env.scene.cfg["success"]["at_rest_speed"]))
        mapping = meta.get("mapping")
        ev = sc.evaluate(meta["task"], mapping)
    finally:
        env.close()
    res = {"src": str(a.src), "frames": len(A), "replayed": len(rep), "rate_hz": round(len(rep) / wall, 2),
           "clipped_ticks": int(clips), **compare(S[:len(rep)], rep),
           "replay_outcomes": [o["outcome"] for o in ev["outcomes"]], "replay_all_correct": ev["all_correct"],
           "recorded_success": meta.get("success")}
    print(json.dumps({k: v for k, v in res.items() if k != "arm_rms_per_joint"}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
