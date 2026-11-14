"""Perception dev set from a running sim: per seed one observation plus privileged object poses and DR parameters.

    python -m halal.eval.collect_frames --name dr_train --seeds 2000 2200     (sim env vars select the instance)
    python -m halal.eval.collect_frames --name lookpose --seeds 2300 2315 --arm-q 0.2 0.7 ... (14 values: arms held there)
"""
from __future__ import annotations

import argparse
import json
import os
import ssl  # noqa: F401  must load before DDS, as in the organisers' evaluator
import sys
from pathlib import Path

import numpy as np

from aaltonomous.baselines.classical.planner import ramp
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.sim import evaluate
from aaltonomous.sim.scene_client import SceneClient

REPO = Path(__file__).resolve().parents[2]
IMAGE_KEYS = ("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb")
SETTLE_TICKS = 25
ARM_SPEED = 0.6


def hold(env: AaltoEnv, ticks: int) -> dict:
    """Keep the commanded pose for `ticks` and return the last observation."""
    action = {"arm_q_target": env.filter.arm_cmd.copy(), "hand_q_target": env.filter.hand_cmd.copy()}
    for _ in range(ticks):
        obs, _ = env.step(action)
    return obs


def move_arms(env: AaltoEnv, target: np.ndarray) -> None:
    """Ramp the arm command to `target` at ARM_SPEED rad/s per joint, hands as they are."""
    hand = env.filter.hand_cmd.copy()
    for arm in ramp(env.filter.arm_cmd.copy(), target, ARM_SPEED, env.dt):
        env.step({"arm_q_target": arm, "hand_q_target": hand})


def collect_seed(env: AaltoEnv, scene: SceneClient, seed: int, task: str, arm_q: np.ndarray | None) -> dict:
    evaluate.home_arms(env, scene)
    env.rng = np.random.default_rng(seed)
    dr = scene.randomize(seed).get("params")
    env.task = task
    env.reset()
    if arm_q is not None:
        move_arms(env, arm_q)
    obs = hold(env, SETTLE_TICKS)
    objects = [o for o in scene.state()["objects"] if o["active"]]
    meta = {"seed": seed, "task": task, "instruction": env.instruction, "mapping": env.mapping, "layout": env.layout,
            "objects": objects, "dr": dr}
    return {**{k: obs[k] for k in IMAGE_KEYS}, "arm_q": obs["arm_q"], "hand_q": obs["hand_q"],
            "meta": np.array(json.dumps(meta))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, metavar=("FIRST", "LAST"), required=True)
    ap.add_argument("--task", default="sort_language")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--arm-q", type=float, nargs=14, default=None, help="arm pose to look from (default: the sim home pose)")
    args = ap.parse_args(argv)

    evaluate.apply_instance_env()
    out = REPO / "runs" / "frames" / args.name
    out.mkdir(parents=True, exist_ok=True)
    scene = SceneClient(args.host)
    env = AaltoEnv(backend="sim", task=args.task, seed=0, time_limit=1e9, realtime=True, img_server_ip=args.host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    try:
        for seed in range(args.seeds[0], args.seeds[1]):
            arm_q = None if args.arm_q is None else np.asarray(args.arm_q)
            np.savez_compressed(out / f"seed_{seed:05d}.npz", **collect_seed(env, scene, seed, args.task, arm_q))
            print(f"seed {seed} saved", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
