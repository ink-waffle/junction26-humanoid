"""Dev-only episode film: head and wrist cameras of every tick with a caption strip, from the evaluator's own observations."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aaltonomous.deploy import runner

REPO = Path(__file__).resolve().parents[2]
FPS = 30                      # one frame per control tick: real-time playback of sim time
HEAD = (640, 480)             # width, height of the head view; the wrists are stacked beside it at half that size
CAPTION_HEIGHT = 66
CAPTION_CHARS = 118           # instruction wrap width at the caption font
FONT, FONT_SCALE, LINE = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 20
SCORE_EVERY = 15              # ticks between two score requests to the scene service


class VideoRecorder:
    """Wraps the runner's episode loop: every observation becomes one frame of runs/videos/<name>/seed_<seed>.mp4."""

    def __init__(self, name: str) -> None:
        from aaltonomous.sim.scene_client import SceneClient  # needs zmq, present only where a sim runs
        self.out = REPO / "runs" / "videos" / name
        self.scene = SceneClient()
        self.run_episode = runner.run_episode
        self.out.mkdir(parents=True, exist_ok=True)

    def episode(self, env: Any, client: Any, idx: int, seed: int, rec: Any, stop: Any, task: str = "sort_fixed", **kwargs: Any) -> Any:
        """One evaluator episode with `env._obs` filming; same result as the wrapped loop."""
        size = (HEAD[0] * 3 // 2, HEAD[1] + CAPTION_HEIGHT)
        writer = cv2.VideoWriter(str(self.out / f"seed_{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        observe, tick, status = env._obs, 0, ""

        def filming_obs() -> dict:
            nonlocal tick, status
            obs = observe()
            if tick % SCORE_EVERY == 0:
                status = self.status(task, env.mapping)
            writer.write(frame(obs, status))
            tick += 1
            return obs

        env._obs = filming_obs
        try:
            return self.run_episode(env, client, idx, seed, rec, stop, task, **kwargs)
        finally:
            env._obs = observe
            writer.release()

    def status(self, task: str, mapping: dict | None) -> str:
        """Running outcome counts from the scene service's own checker."""
        outcomes = [o["outcome"] for o in self.scene.evaluate(task, mapping)["outcomes"]]
        return f"sorted {outcomes.count('correct')}/{len(outcomes)}   wrong bin {outcomes.count('wrong')}   knocked off {outcomes.count('off')}"


def frame(obs: dict, status: str) -> np.ndarray:
    """Head view, both wrist views and the caption strip as one BGR image."""
    half = (HEAD[0] // 2, HEAD[1] // 2)
    wrists = np.vstack([cv2.resize(obs[key], half) for key in ("left_wrist_rgb", "right_wrist_rgb")])
    views = cv2.cvtColor(np.hstack([cv2.resize(obs["head_rgb"], HEAD), wrists]), cv2.COLOR_RGB2BGR)
    caption = np.zeros((CAPTION_HEIGHT, views.shape[1], 3), np.uint8)
    lines = textwrap.wrap(obs["instruction"], CAPTION_CHARS)[:2] + [f"t = {obs['t']:5.1f} s   {status}"]
    for row, text in enumerate(lines):
        cv2.putText(caption, text, (8, 16 + LINE * row), FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([views, caption])
