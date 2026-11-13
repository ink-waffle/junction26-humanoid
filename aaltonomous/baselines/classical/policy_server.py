"""Reference submission: the classical rung-1 sorter (ClassicalPolicy) behind the policy-server protocol.

    python -m aaltonomous.baselines.classical.policy_server            # :8000, what the Docker image runs
    docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .

Same planner/perception as run.py / sim_eval.py (no forked logic). It runs threaded: while it looks and plans
(IK, ~0.1-few s) act() returns 1-tick holds at once, so it never blocks the runner (command rate stays ~30 Hz);
motion comes back in `motion.chunk`-tick chunks. It returns "done" when nothing sortable is left.
Uses only head_rgb, head_depth, arm_q, hand_q, t from obs; camera intrinsics/extrinsics come from the
calibration.yaml baked into the image (spec §11), so rebuild after a calibration change.
Rung 1 only: the colour -> bin mapping is scene.yaml's, whatever the instruction says.
Env: AALTO_TIME_LIMIT (s, default 120; must match the runner's --time-limit), AALTO_ARM_VEL (rad/s override),
AALTO_PROFILE (config.yaml profile; default `sim` = sim-tuned, `none` = robot base values).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .policy import ClassicalPolicy

log = logging.getLogger("aaltonomous.baseline")


class ClassicalServerPolicy(Policy):
    name = "classical-rung1"

    def __init__(self, time_limit=None, arm_vel=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        ov = {}
        av = arm_vel or os.environ.get("AALTO_ARM_VEL")
        if av:
            ov["motion.arm_max_vel"] = float(av)
        self.pol = ClassicalPolicy(threaded=True, require_confirm=False, time_limit=tl, overrides=ov,
                                   log=lambda m: log.info(m), **kw)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])

    def reset(self, instruction, task=None, seed=None):
        # An episode can end mid-look (time limit): let that planning thread finish before the next one starts,
        # the kinematics/planner state is not thread-safe.
        job, t0 = self.pol._job, time.monotonic()
        while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
            time.sleep(0.02)
        if task not in (None, "sort_fixed"):
            log.warning("task %r: the classical baseline only does rung 1 (colour = bin from scene.yaml)", task)
        self.pol.reset({"instruction": instruction})
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = ClassicalServerPolicy()
    log.info("classical baseline loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
