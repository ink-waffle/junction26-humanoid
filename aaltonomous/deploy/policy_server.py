"""Participant-facing policy server template. Subclass Policy, then serve() it on port 8000.

    class MyPolicy(Policy):
        def reset(self, instruction, task=None, seed=None): ...
        def act(self, obs): return {"arm_q_target": ..., "hand_q_target": ...}   # (14,) or (k,14), absolute rad

Run the example:  python -m aaltonomous.deploy.policy_server
obs keys: head_rgb, head_depth, left_wrist_rgb, right_wrist_rgb, arm_q, arm_dq, hand_q, instruction, t (spec §4.2).
Return a chunk (k,14) to be executed at 30 Hz; the runner asks again when it is used up and holds the last
target while you compute. Return "done": True to end the episode early. Everything is filtered for safety.
"""
from __future__ import annotations

import logging
import traceback

import numpy as np

from . import protocol as P

log = logging.getLogger("aaltonomous.policy_server")


class Policy:
    name = "policy"
    action_horizon = None  # chunk length you return (informational)

    def reset(self, instruction: str, task: str | None = None, seed: int | None = None) -> None:
        pass

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


class HoldPolicy(Policy):
    """Example: keep the arms and hands where they were at the start of the episode."""
    name = "hold"
    action_horizon = 1

    def reset(self, instruction, task=None, seed=None):
        self.home = None

    def act(self, obs):
        if self.home is None:
            self.home = (np.array(obs["arm_q"], float), np.array(obs["hand_q"], float))
        return {"arm_q_target": self.home[0], "hand_q_target": self.home[1]}


def handle(policy: Policy, msg: dict) -> dict:
    t = msg.get("type") if isinstance(msg, dict) else None
    if t == "reset":
        policy.reset(msg.get("instruction", ""), task=msg.get("task"), seed=msg.get("seed"))
        return {"type": "reset_ok"}
    if t == "act":
        return {"type": "action", "action": policy.act(msg["obs"])}
    return {"type": "error", "message": f"unknown message type {t!r}"}


def serve(policy: Policy, port: int = P.PORT, host: str = "0.0.0.0", ready=None):
    """Blocking. `ready` (threading.Event) is set once the socket is listening."""
    from websockets.sync.server import serve as ws_serve

    meta = {"type": "metadata", "version": P.VERSION, "name": policy.name, "action_horizon": policy.action_horizon}

    def conn(ws):
        ws.send(P.pack(meta))
        for raw in ws:
            try:
                out = handle(policy, P.unpack(raw))
            except Exception as e:  # report to the runner, keep serving
                log.error("policy error: %s", traceback.format_exc())
                out = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            ws.send(P.pack(out))

    with ws_serve(conn, host, port, max_size=None, compression=None) as server:
        log.info("policy server '%s' listening on %s:%d", policy.name, host, port)
        if ready is not None:
            ready.set()
            ready.server = server
        server.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Serve HoldPolicy (holds the start pose) on a websocket; template for submissions.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=P.PORT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(HoldPolicy(), port=a.port, host=a.host)
