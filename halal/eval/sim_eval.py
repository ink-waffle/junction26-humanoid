"""The organisers' sim evaluator with our instruction source and an optional recorder of perception frames.

    --instructions unseen                        swaps training templates for held-out paraphrases
    --record-frames NAME [--record-every N]      saves every N-th observation with the scene state to runs/frames/NAME
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from aaltonomous.deploy import runner
from aaltonomous.interface import env as env_module
from aaltonomous.interface import tasks
from aaltonomous.sim import evaluate

from . import paraphrases

REPO = Path(__file__).resolve().parents[2]
UNSEEN_CATEGORIES = ("color_full", "bin_synonym", "color_respectively")
FRAME_KEYS = ("head_rgb", "head_depth", "arm_q", "hand_q")


def unseen_instruction(task: str, rng: np.random.Generator) -> tuple[str, dict[str, str]]:
    """Same rng draws and mapping as the training sampler (layouts stay paired across legs), reworded by a paraphrase."""
    text, mapping = tasks.make_instruction(task, rng)
    if task != "sort_language":
        return text, mapping
    pool = paraphrases.load(split="test", categories=UNSEEN_CATEGORIES)
    pick = pool[np.random.default_rng(zlib.crc32(text.encode())).integers(len(pool))]
    return paraphrases.render(pick, list(mapping), list(mapping.values()))


class FrameRecorder:
    """Wraps the runner's episode loop: every n-th env step is saved with the privileged object poses of that moment."""

    def __init__(self, name: str, every: int) -> None:
        from aaltonomous.sim.scene_client import SceneClient  # needs zmq, present only where a sim runs
        self.out = REPO / "runs" / "frames" / name
        self.every = every
        self.scene = SceneClient()
        self.run_episode = runner.run_episode
        self.out.mkdir(parents=True, exist_ok=True)

    def episode(self, env: Any, client: Any, idx: int, seed: int, *args: Any, **kwargs: Any) -> Any:
        """One evaluator episode with `env.step` recording; same result as the wrapped loop."""
        dr = self.scene.randomize(seed).get("params")
        step, tick = env.step, 0

        def recording_step(action: dict) -> tuple[dict, dict]:
            nonlocal tick
            obs, info = step(action)
            tick += 1
            if tick % self.every == 0:
                self.save(obs, {"seed": seed, "tick": tick, "t": float(obs["t"]), "mapping": env.mapping, "dr": dr})
            return obs, info

        env.step = recording_step
        try:
            return self.run_episode(env, client, idx, seed, *args, **kwargs)
        finally:
            env.step = step

    def save(self, obs: dict, meta: dict) -> None:
        objects = [o for o in self.scene.state()["objects"] if o["active"]]
        meta = {**meta, "instruction": obs["instruction"], "objects": objects}
        path = self.out / f"seed_{meta['seed']:05d}_tick_{meta['tick']:05d}.npz"
        np.savez_compressed(path, **{k: obs[k] for k in FRAME_KEYS}, meta=np.array(json.dumps(meta)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instructions", choices=("train", "unseen"), default="train")
    ap.add_argument("--record-frames", metavar="NAME")
    ap.add_argument("--record-every", type=int, default=20)
    args, rest = ap.parse_known_args(argv)
    if args.instructions == "unseen":
        env_module.make_instruction = unseen_instruction
    if args.record_frames:
        runner.run_episode = FrameRecorder(args.record_frames, args.record_every).episode
    return evaluate.main(rest)


if __name__ == "__main__":
    sys.exit(main())
