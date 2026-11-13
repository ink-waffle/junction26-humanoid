"""Client for the Aalto-* sim scene service (aalto_task/hooks.py, tcp :55580). Plain pyzmq + json; any env.

    from aaltonomous.sim.scene_client import SceneClient
    sc = SceneClient("127.0.0.1")
    sc.reset(layout)                  # tasks.sample_layout() format, mat coords
    sc.reset_random(seed=3)           # -> {"layout": [...], "active": [names]}
    sc.state()["objects"]             # pelvis-frame poses + speeds
    sc.evaluate("sort_fixed")         # -> {"outcomes": [...], "score": int, "all_correct": bool}

CLI: python -m aaltonomous.sim.scene_client [--host H] {ping,state,random [seed],eval [task]}
"""
from __future__ import annotations

import json
import os
import sys

import zmq


class SceneClient:
    def __init__(self, host="127.0.0.1", port=None, timeout_ms=5000):
        port = int(os.environ.get("AALTO_SCENE_PORT", "55580")) if port is None else port   # 2nd sim instance
        self.addr, self.timeout = f"tcp://{host}:{port}", timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.sock.connect(self.addr)

    def call(self, **req):
        self.sock.send(json.dumps(req).encode())
        try:
            r = json.loads(self.sock.recv())
        except zmq.Again:            # REQ is stuck after a lost reply: recreate it
            self.sock.close(0)
            self._connect()
            raise TimeoutError(f"scene service at {self.addr} did not answer (is an Aalto-* sim running?)")
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return r

    def ping(self):
        return self.call(cmd="ping")

    def reset(self, layout):
        return self.call(cmd="reset", layout=layout)

    def reset_random(self, seed=None):
        return self.call(cmd="reset_random", seed=seed)

    def state(self):
        return self.call(cmd="state")

    def info(self):
        return self.call(cmd="info")

    def randomize(self, seed=None):
        """DR: sample + apply one parameter set (no-op, params None, when the sim runs without AALTO_DR)."""
        return self.call(cmd="randomize", seed=seed)

    def evaluate(self, task="sort_fixed", mapping=None):
        """mapping: colour/shape -> bin; (colour, shape) tuple keys are sent as 'colour/shape'."""
        m = None if mapping is None else {("/".join(k) if isinstance(k, tuple) else k): v for k, v in mapping.items()}
        return self.call(cmd="evaluate", task=task, mapping=m)

    def close(self):
        self.sock.close(0)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="default $AALTO_SCENE_PORT or 55580")
    ap.add_argument("cmd", choices=["ping", "state", "random", "eval"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args(argv)
    c = SceneClient(a.host, a.port)
    if a.cmd == "ping":
        r = c.ping()
    elif a.cmd == "state":
        r = c.state()
    elif a.cmd == "random":
        r = c.reset_random(None if a.arg is None else int(a.arg))
    else:
        r = c.evaluate(a.arg or "sort_fixed")
    json.dump(r, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
