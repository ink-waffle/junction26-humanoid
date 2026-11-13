"""Dry-run: replay a recorded log's observations to a policy server / container and check it keeps up.

    python -m aaltonomous.deploy.replay runs/team/RUN/episode_000.npz --policy ws://localhost:8000 [--container IMG]
    python -m aaltonomous.deploy.replay --synthetic 100 --policy ...     # no log: blank images, zero joints

Checks per act(): reply shapes valid ((14,) or (k,14), finite, hand matches arm), and the command rate: ticks of
targets the policy delivered (chunks count k, holds YOU return count) / total wall time, the same metric and
threshold as the sim gate (evaluate.py gate_ok: >= 27 Hz). With --container also the startup time (limit 120 s).
Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import protocol as P
from .runner import Container, PolicyClient

MIN_COMMAND_HZ = 27.0  # = evaluate.py --gate-min-hz


def load_obs(path, n=None, synthetic=False):
    from ..interface.backends import blank_images
    if synthetic:
        steps = n or 100
        return [{**blank_images(), "arm_q": np.zeros(14, np.float32), "arm_dq": np.zeros(14, np.float32),
                 "hand_q": np.zeros(14, np.float32), "instruction": "Sort each cube into the bin of the same color.",
                 "t": i / 30.0} for i in range(steps)]
    d = np.load(path, allow_pickle=False)
    T = len(d["arm_q"])
    blank = blank_images()
    out = []
    for i in range(T):
        o = {k: (d[k][i] if k in d.files else v) for k, v in blank.items()}
        o.update(arm_q=d["arm_q"][i].astype(np.float32), arm_dq=d["arm_dq"][i].astype(np.float32),
                 hand_q=d["hand_q"][i].astype(np.float32), instruction="Sort each cube into the bin of the same color.",
                 t=float(d["t"][i]))
        out.append(o)
    return out[:n] if n else out


def validate(action):
    if not isinstance(action, dict):
        return "action is not a dict"
    shapes = {}
    for k in ("arm_q_target", "hand_q_target"):
        v = action.get(k)
        if v is None:
            return f"missing {k}"
        v = np.asarray(v, float)
        if v.ndim not in (1, 2) or v.shape[-1] != 14:
            return f"{k} shape {v.shape}, want (14,) or (k,14)"
        if not np.isfinite(v).all():
            return f"{k} not finite"
        shapes[k] = v.shape[0] if v.ndim == 2 else 1
    if shapes["arm_q_target"] != shapes["hand_q_target"] and 1 not in shapes.values():
        return "arm/hand chunk lengths differ"
    return None


def replay(url, obs_list, min_hz=MIN_COMMAND_HZ, startup_timeout=120.0, alive=None):
    client = PolicyClient(url, startup_timeout, alive=alive)
    errors, ticks, lat = [], 0, []
    try:
        client.reset("sort_fixed", obs_list[0]["instruction"], 0)
        t0 = time.monotonic()
        for i, obs in enumerate(obs_list):
            t = time.monotonic()
            try:
                a = client.act(obs)
            except Exception as e:
                errors.append(f"step {i}: {e}")
                continue
            lat.append(time.monotonic() - t)
            err = validate(a)
            if err:
                errors.append(f"step {i}: {err}")
                continue
            arm = np.asarray(a["arm_q_target"])
            ticks += arm.shape[0] if arm.ndim == 2 else 1
        wall = time.monotonic() - t0
    finally:
        client.close()
    hz = ticks / max(wall, 1e-9)
    res = {"calls": len(obs_list), "ticks": ticks, "command_hz": round(hz, 1), "min_hz": min_hz, "latency_ms_p50": None,
           "latency_ms_max": None, "errors": errors[:10], "n_errors": len(errors), "pass": hz >= min_hz and not errors}
    res["connect_s"] = round(client.connect_s, 2)
    if lat:
        res["latency_ms_p50"] = round(1e3 * float(np.median(lat)), 2)
        res["latency_ms_max"] = round(1e3 * float(np.max(lat)), 2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="episode_NNN.npz from the runner")
    ap.add_argument("--synthetic", type=int, metavar="N")
    ap.add_argument("--policy", default=f"ws://localhost:{P.PORT}")
    ap.add_argument("--container")
    ap.add_argument("--host-port", type=int, default=P.PORT)
    ap.add_argument("--min-hz", type=float, default=MIN_COMMAND_HZ, help="command-rate gate (sim gate: 27)")
    ap.add_argument("--gpus", default="all", help='docker --gpus value; "none" for a CPU-only host')
    ap.add_argument("--steps", type=int)
    a = ap.parse_args(argv)
    if not a.log and not a.synthetic:
        ap.error("give a log or --synthetic N")
    obs = load_obs(None, a.synthetic, True) if a.synthetic else load_obs(a.log, a.steps)
    cm = Container(a.container, a.host_port, gpus=None if a.gpus.lower() == "none" else a.gpus) if a.container else None
    try:
        url = a.policy
        if cm:
            cm.prepare().__enter__()
            url = f"ws://localhost:{a.host_port}"
        try:
            res = replay(url, obs, a.min_hz, alive=cm.alive if cm else None)
        except ConnectionError as e:
            print(cm.logs(tail=40) if cm else "", file=sys.stderr)
            res = {"pass": False, "error": str(e)}
        if cm:
            res["startup_s"] = res.get("connect_s")  # docker run -> websocket handshake done
            res["pass"] = res["pass"] and res["startup_s"] is not None and res["startup_s"] <= 120
    finally:
        if cm:
            cm.__exit__()
    print(res)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
