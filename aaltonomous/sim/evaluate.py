"""Seeded sim evaluation (spec §3 scoring, §6 sim gate; TODO M12) on a running Aalto-* sim.

    # scripted oracle (privileged poses), rung 1 / rung 2
    python -m aaltonomous.sim.evaluate --policy oracle --task sort_fixed --trials 10
    python -m aaltonomous.sim.evaluate --policy oracle --task sort_language --trials 10
    # a submission: policy server already listening (websocket + msgpack, aaltonomous/deploy/protocol.py)
    python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --team demo --trials 20
    # a submission image (runner.Container: docker run --gpus all -p 8000:8000)
    python -m aaltonomous.sim.evaluate --container team/sub:latest --team demo --trials 20

Trial i uses seed = --seed + i for the layout + instruction (AaltoEnv rng) and, if the sim runs with a DR config
(AALTO_DR, see aalto_task/dr.py), for the per-episode randomization (scene service `randomize`). Same seeds +
same sim variant = same episodes. Episodes run through deploy.runner.run_episode: AaltoEnv(backend="sim") +
SafetyFilter at 30 Hz, chunks, holds while the policy computes, watchdog. After the episode the objects settle
(at rest or --settle s) and the scene service scores them with scene.py (the one checker).

Output: per-trial JSON (layout, instruction, outcomes, score, safety summary, rate) + a short table; logs per
episode (npz/jsonl, deploy.runner.Recorder) next to the JSON. Gate (spec §6 "sim gate before real-robot time"):
every trial ends without safety abort / watchdog stop / crash / lost connection, clipped steps <= --gate-max-clips,
command rate >= --gate-min-hz (sim time when the sim publishes its clock) AND policy_hz_rt >= --gate-min-hz (what the
policy's own wall latency allows, so a slow policy can't pass just because the sim runs slower than real time). Exit code 0 = gate passed, 2 = failed.

Second sim instance: AALTO_DDS_DOMAIN / AALTO_SCENE_PORT / AALTO_RGBD_PORT / AALTO_CAM_PORT_BASE must match the
values that sim was started with (run_sim.sh); defaults = the standard sim (domain 1, 55580, 55560, 55555).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import ssl  # noqa: F401  before any DDS import: cyclonedds loads the system libcrypto, then _ssl (websockets) fails in env isaac
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BAD_ENDS = ("aborted", "watchdog", "safe_stop", "policy_connection_lost", "policy_timeout", "crash", "interrupted")


def apply_instance_env():
    """Point the DDS sim backend at a non-default sim instance (no edits to interface/backends.py)."""
    dom = os.environ.get("AALTO_DDS_DOMAIN")
    if dom and int(dom) != 1:
        import unitree_sdk2py.core.channel as ch
        orig = ch.ChannelFactoryInitialize

        def init(id=0, networkInterface=None):
            return orig(int(dom) if id == 1 else id, networkInterface)
        ch.ChannelFactoryInitialize = init
    base = os.environ.get("AALTO_CAM_PORT_BASE")
    if base:
        from aaltonomous.interface import backends
        b = int(base)
        backends._V2ImageClient.PORTS = (("head_camera", b), ("left_wrist_camera", b + 1), ("right_wrist_camera", b + 2))


class LocalClient:
    """In-process policy with the runner's PolicyClient API (reset / act_async / close)."""

    def __init__(self, policy, name="oracle"):
        self.policy, self.pool = policy, ThreadPoolExecutor(1)
        self.metadata = {"type": "metadata", "name": name, "local": True}

    def reset(self, task, instruction, seed):
        self.policy.reset({"instruction": instruction})

    def act(self, obs):
        return self.policy.act(obs)

    def act_async(self, obs):
        return self.pool.submit(self.policy.act, obs)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def home_arms(env, sc, vmax=0.6, hold_s=0.5):
    """Spec §3 "arms start in a fixed home pose": park every object, then move arms to the sim home pose
    (layout.ARM_HOME) and hands to 0 through the safety filter, so the next layout is placed with the arms clear.
    Uses (and ends) a throw-away AaltoEnv episode; call before seeding env.rng for the trial."""
    from aaltonomous.baselines.classical.planner import ramp
    from aaltonomous.sim.aalto_task.layout import ARM_HOME
    spec = env.filter.spec
    home = np.array([ARM_HOME[n] for n in spec.arm_names])
    hand = np.clip(np.zeros(len(spec.hand_names)), spec.hand_lower + 0.05, spec.hand_upper - 0.05)
    sc.reset([])
    orig = env.backend.reset_scene
    env.backend.reset_scene = lambda layout, instruction: None
    try:
        env.reset()
    finally:
        env.backend.reset_scene = orig
    a0, h0 = env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy()
    A, H = ramp(a0, home, vmax, env.dt), ramp(h0, hand, 1.0, env.dt)
    n = max(len(A), len(H)) + int(hold_s / env.dt)
    A = np.vstack([A, np.repeat(home[None], n - len(A), 0)]) if len(A) < n else A
    H = np.vstack([H, np.repeat(hand[None], n - len(H), 0)]) if len(H) < n else H
    for a, h in zip(A, H):
        _, info = env.step({"arm_q_target": a, "hand_q_target": h})
        if info["episode_over"]:
            break
    err = float(np.abs(env.backend.joints()[0] - home).max())
    if err > 0.1:
        print(f"[evaluate] warning: arms {err:.2f} rad from home after homing")
    return err


def settle(sc, timeout, rest_speed):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        objs = [o for o in sc.state()["objects"] if o["active"]]
        if all(o["speed"] < rest_speed for o in objs):
            return time.monotonic() - t0
        time.sleep(0.1)
    return timeout


def gate_ok(trial, max_clips, min_hz):
    s = trial.get("safety") or {}
    why = []
    if trial["end_reason"] in BAD_ENDS:
        why.append(trial["end_reason"])
    if (s.get("clipped_steps") or 0) > max_clips:
        why.append(f"clipped {s.get('clipped_steps')} > {max_clips}")
    hz = trial.get("command_hz", trial.get("effective_hz")) or 0   # spec §8: "effective control at 30 Hz (counting
    if hz < min_hz:                                                  # chunks)" = ticks with a policy target, not holds
        why.append(f"command rate {hz} Hz < {min_hz}")               # (sim clock: per SIM second)
    rt = trial.get("policy_hz_rt")   # the rate the policy's own WALL latency allows (load-independent): in a sim that
    if rt is not None and rt < min_hz:   # runs slower than real time a slow policy still fills every sim tick, but
        why.append(f"policy real-time rate {rt} Hz < {min_hz}")   # on the robot (rtf 1) it would fall back to holds
    return not why, why


def evaluate(a):
    from aaltonomous.deploy.runner import PolicyClient, Recorder, Container, run_episode
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient

    apply_instance_env()
    sc = SceneClient(a.host)
    try:
        sim_info = sc.call(cmd="info")
    except RuntimeError:                     # older sim without `info`: no DR
        sim_info = {"dr": None}
    sim_info.pop("ok", None)
    variant = a.variant or (sim_info.get("dr") or {}).get("name") or "nominal"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = a.team or ("oracle" if a.policy == "oracle" else "policy")
    out_dir = Path(a.out) / f"{name}_{a.task}_{variant}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = AaltoEnv(backend="sim", task=a.task, seed=a.seed, time_limit=a.time_limit, realtime=True,
                   img_server_ip=a.host, head_rgbd=not a.no_head_rgbd,
                   head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=not a.no_cameras)
    cm, client = None, None
    trials = []
    stop = threading.Event()
    try:
        if a.policy == "oracle":
            from aaltonomous.sim.oracle import OraclePolicy
            pol = OraclePolicy(SceneClient(a.host), safety=env.filter.cfg, time_limit=a.time_limit,
                               privileged_mapping=lambda: env.mapping, log=(print if a.verbose else None))
            client = LocalClient(pol)
        else:
            url = a.policy
            if a.container:
                cm = Container(a.container, a.host_port, gpus=None if a.gpus == "none" else a.gpus).__enter__()
                url = f"ws://localhost:{a.host_port}"
            client = PolicyClient(url, a.startup_timeout)
        print(f"policy: {client.metadata}  sim: {sim_info}  -> {out_dir}")
        rest = float(env.scene.cfg["success"]["at_rest_speed"])
        for i in range(a.trials):
            seed = a.seed + i
            home_err = home_arms(env, sc)
            env.rng = np.random.default_rng(seed)
            dr = None
            if sim_info.get("dr"):
                dr = sc.call(cmd="randomize", seed=seed).get("params")
            rec = Recorder(out_dir / f"episode_{i:03d}")
            t0 = time.monotonic()
            try:
                summary, events = run_episode(env, client, i, seed, rec, stop, a.task)
            except KeyboardInterrupt:
                stop.set()
                summary, events = {"episode": i, "seed": seed, "end_reason": "interrupted"}, list(env.filter.events)
            except Exception as e:
                traceback.print_exc()
                summary, events = {"episode": i, "seed": seed, "end_reason": "crash", "error": repr(e),
                                   "safety": env.filter.summary()}, list(env.filter.events)
            rtf = summary.get("rtf") or 1.0     # a.settle is in sim seconds; the sim may run slower than real time
            settled = settle(sc, a.settle / max(min(rtf, 1.0), 0.2), rest)
            ev = sc.evaluate(a.task, env.mapping)
            final = [{"name": o["name"], "pos": o["pos"], "speed": o["speed"]} for o in sc.state()["objects"] if o["active"]]
            outc = [o["outcome"] for o in ev["outcomes"]]
            good = "inserted" if a.task == "insert_shapes" else "correct"
            per_ok = env.scene.cfg["scoring"][a.task][good]
            trial = {**summary, "seed": seed, "task": a.task, "variant": variant, "layout": env.layout,
                     "mapping": env.mapping, "outcomes": ev["outcomes"], "score": ev["score"],
                     "max_score": per_ok * len(outc), "n_objects": len(outc), "n_correct": outc.count(good),
                     "all_correct": ev["all_correct"], "settle_s": round(settled, 2),
                     "episode_wall_s": round(time.monotonic() - t0, 1), "home_err": round(home_err, 3), "final_objects": final,
                     "dr_params": dr}
            if a.policy == "oracle":
                trial["mapping_source"] = pol.mapping_source
                trial["oracle_events"] = pol.events[-40:]
            trial["gate_ok"], trial["gate_why"] = gate_ok(trial, a.gate_max_clips, a.gate_min_hz)
            if events or summary.get("end_reason") != "done":
                trial["safety_events"] = events[:20]
            rec.save(events, {k: v for k, v in trial.items() if k not in ("oracle_events",)})
            trials.append(trial)
            print(f"trial {i:3d} seed {seed:4d}: {trial['n_correct']}/{trial['n_objects']} correct  score "
                  f"{trial['score']:+d}/{trial['max_score']}  end={trial['end_reason']}  t={trial.get('wall_s')}s  "
                  f"clips={(trial.get('safety') or {}).get('clipped_steps')}  hz={trial.get('command_hz')}  "
                  f"rt_hz={trial.get('policy_hz_rt')}  rtf={trial.get('rtf')}  "
                  f"gate={'ok' if trial['gate_ok'] else 'FAIL ' + ','.join(trial['gate_why'])}", flush=True)
            if stop.is_set():
                break
    finally:
        if client:
            client.close()
        env.close()
        if cm:
            cm.__exit__()

    res = summarize(trials, a, variant, sim_info)
    (out_dir / "eval.json").write_text(json.dumps({"summary": res, "trials": trials}, indent=1, default=_js))
    print_table(res)
    print(f"-> {out_dir / 'eval.json'}")
    return res


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def summarize(trials, a, variant, sim_info):
    n = len(trials)
    if not n:
        return {"trials": 0}
    sc = [t["score"] for t in trials]
    obj = sum(t["n_objects"] for t in trials)
    cor = sum(t["n_correct"] for t in trials)
    counts = {}
    for t in trials:
        for o in t["outcomes"]:
            counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
    return {
        "policy": a.container or a.policy, "team": a.team, "task": a.task, "variant": variant, "dr": sim_info.get("dr"),
        "trials": n, "seeds": [t["seed"] for t in trials],
        "score_mean": round(float(np.mean(sc)), 3), "score_std": round(float(np.std(sc)), 3),
        "score_frac": round(sum(sc) / max(1, sum(t["max_score"] for t in trials)), 3),
        "objects": obj, "objects_correct": cor, "object_success": round(cor / max(1, obj), 3),
        "episodes_all_correct": sum(t["all_correct"] for t in trials), "outcome_counts": counts,
        "end_reasons": {r: sum(t["end_reason"] == r for t in trials) for r in sorted({t["end_reason"] for t in trials})},
        "safety_clipped_steps_total": sum((t.get("safety") or {}).get("clipped_steps") or 0 for t in trials),
        "safety_clipped_steps_max": max((t.get("safety") or {}).get("clipped_steps") or 0 for t in trials),
        "effective_hz_min": min((t.get("effective_hz") or 0) for t in trials),
        "command_hz_min": min((t.get("command_hz", t.get("effective_hz")) or 0) for t in trials),
        "policy_hz_rt_min": min((t["policy_hz_rt"] for t in trials if t.get("policy_hz_rt") is not None), default=None),
        "clock": trials[0].get("clock", "wall"),
        "rtf_min": min((t["rtf"] for t in trials if t.get("rtf")), default=None),
        "rtf_mean": (round(float(np.mean([t["rtf"] for t in trials if t.get("rtf")])), 3)
                     if any(t.get("rtf") for t in trials) else None),
        "gate": {"passed": all(t["gate_ok"] for t in trials), "failed_trials": [t["episode"] for t in trials if not t["gate_ok"]],
                 "max_clips": a.gate_max_clips, "min_hz": a.gate_min_hz},
    }


def print_table(r):
    if not r.get("trials"):
        print("no trials")
        return
    rows = [("task / variant", f"{r['task']} / {r['variant']}"), ("trials", r["trials"]),
            ("score mean ± std", f"{r['score_mean']:+.2f} ± {r['score_std']:.2f}  ({100 * r['score_frac']:.0f} % of max)"),
            ("objects correct", f"{r['objects_correct']}/{r['objects']} ({100 * r['object_success']:.0f} %)"),
            ("episodes all correct", f"{r['episodes_all_correct']}/{r['trials']}"),
            ("outcomes", r["outcome_counts"]), ("end reasons", r["end_reasons"]),
            ("safety clips total / max", f"{r['safety_clipped_steps_total']} / {r['safety_clipped_steps_max']}"),
            ("min tick / command rate", f"{r['effective_hz_min']} / {r['command_hz_min']} Hz ({r.get('clock')} clock)"),
            ("min policy real-time rate", f"{r.get('policy_hz_rt_min')} Hz"),
            ("sim real-time factor", f"mean {r.get('rtf_mean')}  min {r.get('rtf_min')}"),
            ("SIM GATE", "PASS" if r["gate"]["passed"] else f"FAIL (trials {r['gate']['failed_trials']})")]
    w = max(len(k) for k, _ in rows)
    print("\n" + "\n".join(f"  {k:<{w}}  {v}" for k, v in rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="oracle", help="'oracle' or a policy server url ws://host:8000")
    ap.add_argument("--container", help="docker image of a submission (overrides --policy)")
    ap.add_argument("--host-port", type=int, default=8000)
    ap.add_argument("--gpus", default="all", choices=["all", "none"], help="docker --gpus for --container (none: no "
                    "nvidia-container-toolkit)")
    ap.add_argument("--startup-timeout", type=float, default=120.0)
    ap.add_argument("--team", default=None)
    ap.add_argument("--task", default="sort_fixed", choices=["sort_fixed", "sort_language", "insert_shapes"],
                    help="insert_shapes needs the Aalto-Insert sim task (lid); --policy oracle does not do it")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000, help="first trial seed (finals: a secret seed)")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=3.0, help="max s to wait for objects at rest before scoring")
    ap.add_argument("--host", default="127.0.0.1", help="sim host (scene service, cameras)")
    ap.add_argument("--variant", default=None, help="label; default = the sim's DR config name")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--gate-max-clips", type=int, default=0)
    ap.add_argument("--gate-min-hz", type=float, default=27.0)
    ap.add_argument("--no-head-rgbd", action="store_true", help="head_rgb from teleimager instead of :55560")
    ap.add_argument("--no-cameras", action="store_true", help="no teleimager (wrist cams blank); oracle runs")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    r = evaluate(a)
    return 0 if r.get("gate", {}).get("passed") else 2


if __name__ == "__main__":
    sys.exit(main())
