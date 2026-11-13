"""Runner: owns the robot (or sim/mock) connection and the safety filter; the submission is a websocket client's
server (spec §4.5, §8). Every command goes through AaltoEnv -> SafetyFilter.

    python -m aaltonomous.deploy.runner --backend mock --policy ws://localhost:8000 --team demo --episodes 2
    python -m aaltonomous.deploy.runner --backend sim --img-server-ip 127.0.0.1 --container my/submission:latest \
        --team demo --episodes 5 --export

Container (--container IMAGE | path/to/image.tar): docker load (tar) or pull (if not local, or --pull), docker run
-d --gpus all -p PORT:8000, wait for the websocket (startup limit 120 s, fails fast if the container exits), run the
episodes, save `docker logs`, stop + remove. Startup time goes into run.json.

Loop (30 Hz): take the next target from the current action chunk and step the env; when the chunk is used up the
policy is called in a background thread. While that request is in flight the runner sends the last command again
as an explicit hold target every tick (through the safety filter, so the arms stay put and the filter's watchdog sees
a live runner), up to --max-policy-latency s per call (default 30) -> then the episode ends "policy_timeout".
A closed connection (container died) ends it at once ("policy_connection_lost"). After an error reply the runner
holds WITHOUT refreshing, so the filter watchdog (hold 0.2 s, safe stop 1 s -> "watchdog") ends a policy that keeps
failing. Waiting ticks are logged as holds and do not count as commands (command_hz, sim gate). Ctrl-C ends cleanly.
Clock: real = wall. Sim = the sim's clock (AaltoEnv paces on it; the sim runs below real time): tick rates, ticks and
the episode time limit count SIM seconds; wall_s, rtf, command_hz_wall and policy_hz_rt (wall latency) are kept.

Logs: <out>/<team>/<run_id>/episode_NNN.npz (per tick), episode_NNN.jsonl (clip events + summary line),
episode_NNN.mp4 (--video), summary.jsonl (one line per episode), run.json (image, startup time, policy metadata),
container.log; --export also writes <out>/<team>/<run_id>.tar.gz.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import ssl  # noqa: F401  before any DDS import: cyclonedds loads the system libcrypto, then _ssl (websockets) fails in env isaac
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import protocol as P

log = logging.getLogger("aaltonomous.runner")
HZ = 30.0
N = 14
MAX_POLICY_LATENCY = 30.0  # s per act() call before the episode ends "policy_timeout"


class PolicyError(RuntimeError):
    pass


class PolicyClient:
    """Sync websocket client. act_async() and reset() share one worker thread, so requests never overlap."""

    def __init__(self, url, startup_timeout=120.0, recv_timeout=60.0, alive=None):
        self.url, self.recv_timeout, self.alive = url, recv_timeout, alive
        self.pool = ThreadPoolExecutor(1)
        self._connect(startup_timeout)

    def _connect(self, timeout):
        from websockets.sync.client import connect
        t0 = time.monotonic()
        deadline, last = t0 + timeout, None
        while True:
            try:
                self.ws = connect(self.url, max_size=None, compression=None, open_timeout=5)
                self.metadata = P.check_metadata(P.unpack(self.ws.recv(timeout=self.recv_timeout)))
                break
            except ConnectionError as e:  # includes a bad handshake: no point retrying a version mismatch
                if "version mismatch" in str(e) or "metadata" in str(e):
                    raise
                last = e
            except Exception as e:
                last = e
            if self.alive is not None and not self.alive():
                raise ConnectionError(f"policy container exited before {self.url} came up: {last}")
            if time.monotonic() > deadline:
                raise ConnectionError(f"policy server {self.url} not up after {timeout:.0f} s: {last}")
            time.sleep(0.5)
        self.connect_s = time.monotonic() - t0

    def reconnect(self, timeout=10.0):
        """Drop a connection with a request stuck in flight and open a fresh one."""
        try:
            self.ws.close()
        except Exception:
            pass
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool = ThreadPoolExecutor(1)
        self._connect(timeout)

    def _rpc(self, msg):
        self.ws.send(P.pack(msg))
        out = P.unpack(self.ws.recv(timeout=self.recv_timeout))
        if out.get("type") == "error":
            raise PolicyError(out.get("message", "policy error"))
        return out

    def reset(self, task, instruction, seed):
        self.pool.submit(self._rpc, {"type": "reset", "task": task, "instruction": instruction,
                                     "seed": int(seed)}).result()

    def act(self, obs):
        return self._rpc({"type": "act", "obs": obs})["action"]

    def act_async(self, obs):
        return self.pool.submit(self.act, obs)

    def close(self):
        try:
            self.ws.close()
        finally:
            self.pool.shutdown(wait=False, cancel_futures=True)


def _docker(*args, check=True, timeout=None):
    r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {(r.stderr or r.stdout).strip()}")
    return r


class Container:
    """Run a submission image: load/pull it, docker run (GPUs, port mapping), keep its logs, tear it down on exit.

    image: a tag/registry ref, or a path to a `docker save` tar(.gz). gpus: "all" (finals), or None for CPU-only
    hosts. pull: always docker pull first (otherwise only when the image is not local)."""

    def __init__(self, image, host_port=P.PORT, extra_args=(), gpus="all", pull=False):
        self.image, self.host_port, self.extra, self.gpus, self.pull = image, host_port, list(extra_args), gpus, pull
        self.name = f"aalto-sub-{os.getpid()}-{time.time_ns() % 10**9}"
        self.image_id, self.t_start, self.started = None, None, False

    def prepare(self):
        if os.path.isfile(self.image):  # docker save tarball
            out = _docker("load", "-i", self.image, timeout=1800).stdout
            refs = [ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Loaded image")]
            if not refs:
                raise RuntimeError(f"docker load {self.image}: no image in output {out!r}")
            self.image = refs[-1]
        elif self.pull or _docker("image", "inspect", self.image, check=False).returncode:
            log.info("pulling %s", self.image)
            _docker("pull", self.image, timeout=3600)
        self.image_id = _docker("image", "inspect", "-f", "{{.Id}}", self.image).stdout.strip()
        return self

    def __enter__(self):
        if self.image_id is None:
            self.prepare()
        gpu = ["--gpus", self.gpus] if self.gpus else []
        self.t_start = time.monotonic()
        try:
            _docker("run", "-d", *gpu, "--name", self.name, "-p", f"{self.host_port}:{P.PORT}", *self.extra, self.image)
        except RuntimeError:
            _docker("rm", "-f", self.name, check=False)  # a failed run (e.g. no GPU runtime) leaves it "Created"
            raise
        self.started = True
        return self

    def alive(self):
        r = _docker("inspect", "-f", "{{.State.Running}}", self.name, check=False)
        return r.stdout.strip() == "true"

    def logs(self, tail=None):
        r = _docker("logs", *(["--tail", str(tail)] if tail else []), self.name, check=False)
        return r.stdout + r.stderr

    def __exit__(self, *exc):
        if self.started:
            _docker("stop", "-t", "5", self.name, check=False)
        _docker("rm", "-f", self.name, check=False)
        self.started = False


def split_action(action):
    """Normalise a policy reply into per-tick action dicts. Anything malformed is passed through unchanged
    so the safety filter rejects it (holds the last command) and logs the reason."""
    if not isinstance(action, dict):
        return [action]
    try:
        keys = [k for k in ("arm_q_target", "hand_q_target") if action.get(k) is not None]
        arrs = {k: np.asarray(action[k], float) for k in keys}
        ks = {a.shape[0] for a in arrs.values() if a.ndim == 2}
        if not ks or any(a.ndim not in (1, 2) for a in arrs.values()) or len(ks) != 1:
            return [action]
        k = ks.pop()
        ticks = []
        for i in range(k):
            t = {key: (a[i] if a.ndim == 2 else a) for key, a in arrs.items()}
            ticks.append(t)
        if action.get("done"):
            ticks[-1]["done"] = True
        return ticks
    except Exception:
        return [action]


class Recorder:
    def __init__(self, path: Path, images=False, video=False):
        self.path, self.images, self.video = path, images, video
        self.rows, self.imgs, self.vw = [], [], None

    def add(self, row, obs):
        self.rows.append(row)
        if self.images:
            self.imgs.append({k: obs[k] for k in ("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb")})
        if self.video:
            import cv2
            f = obs["head_rgb"]
            if self.vw is None:
                self.vw = cv2.VideoWriter(str(self.path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                          HZ, (f.shape[1], f.shape[0]))
            self.vw.write(np.ascontiguousarray(f[..., ::-1]))

    def save(self, events, summary):
        if self.vw is not None:
            self.vw.release()
        cols = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}
        if self.images and self.imgs:
            cols.update({k: np.stack([i[k] for i in self.imgs]) for k in self.imgs[0]})
        np.savez_compressed(self.path.with_suffix(".npz"), **cols)
        with open(self.path.with_suffix(".jsonl"), "w") as f:
            for ev in events:
                f.write(json.dumps({"event": ev}) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


TASKS = ["sort_fixed", "sort_language", "insert_shapes"]


def run_episode(env, client, idx, seed, rec: Recorder, stop: threading.Event, task="sort_fixed",
                max_latency=MAX_POLICY_LATENCY):
    obs = env.reset()
    client.reset(task, obs["instruction"], seed)
    queue, pending, t_req = [], None, 0.0
    healthy = True  # False after an error reply: holds then do NOT refresh the watchdog
    latencies, ticks, waits, end, wall0 = [], 0, 0, None, time.monotonic()
    info = {}
    while end is None:
        if stop.is_set():
            end = "interrupted"
            break
        if len(queue) <= 1 and pending is None:  # prefetch on the last queued tick, so 1-step policies don't stall
            pending, t_req = client.act_async(obs), time.monotonic()
        if pending is not None and not queue:  # fast policy: wait a bit before spending a tick on a hold
            concurrent.futures.wait([pending], timeout=0.5 * env.dt)
        if pending is not None and pending.done():
            latencies.append(time.monotonic() - t_req)
            try:
                queue += split_action(pending.result())
                healthy = True
            except Exception as e:  # error reply: hold, the watchdog decides; anything else: connection is gone
                log.error("policy failed: %s", e)
                healthy = False
                if not isinstance(e, PolicyError):
                    end = "policy_connection_lost"
            pending = None
        elif pending is not None and time.monotonic() - t_req > max_latency:
            log.error("policy call exceeded %.1f s", max_latency)
            end = "policy_timeout"
        if end is not None:
            break
        hold = not queue
        wait = hold and pending is not None and healthy
        if wait:
            # Request in flight: re-send the last command as an explicit target. The filter checks it like any
            # other target (zero step, inside limits), so the arms hold and the watchdog sees a live runner.
            obs, info = env.step({"arm_q_target": env.filter.arm_cmd.copy(), "hand_q_target": env.filter.hand_cmd.copy()})
            mode, clipped, reasons = info["safety_mode"], info["clipped"], info["clip_reasons"]
            end = info["end_reason"]
            waits += 1
        elif hold:
            mode = env.hold()
            clipped, reasons = False, []
            if mode in ("safe_stop", "aborted"):
                end = "watchdog" if mode == "safe_stop" else "aborted"
            elif env._now() >= env.time_limit:
                end = "time_limit"
            obs = env._obs()
        else:
            obs, info = env.step(queue.pop(0))
            mode, clipped, reasons = info["safety_mode"], info["clipped"], info["clip_reasons"]
            end = info["end_reason"]
        arm, hand = env.filter.arm_cmd, env.filter.hand_cmd
        ticks += 1
        rec.add({"t": float(obs["t"]) if not hold else env._now(), "wall": time.monotonic() - wall0, "hold": hold,
                 "policy_wait": wait, "arm_q": obs["arm_q"], "arm_dq": obs["arm_dq"], "hand_q": obs["hand_q"],
                 "arm_q_cmd": arm.copy(), "hand_q_cmd": hand.copy(), "clipped": bool(clipped),
                 "mode": ("ok", "stale", "safe_stop", "aborted").index(mode)}, obs)
    if pending is not None and not pending.done():  # don't let a stale reply answer the next episode's reset
        concurrent.futures.wait([pending], timeout=max(0.0, min(5.0, max_latency - (time.monotonic() - t_req))))
        if not pending.done() and hasattr(client, "reconnect"):
            try:
                client.reconnect()
            except Exception as e:
                log.error("reconnect after a stuck policy call failed: %s", e)
    wall = time.monotonic() - wall0
    holds = sum(1 for r in rec.rows if r["hold"])
    # Rates count in the env's clock: wall time on the real robot, SIM time when the sim backend paces on the sim
    # clock (env.clock; the sim runs below real time). wall-clock rates kept as *_wall, rtf = sim s / wall s.
    span = env._now() if getattr(env, "clock", None) is not None else wall
    summary = {"episode": idx, "seed": seed, "instruction": obs["instruction"], "end_reason": end, "ticks": ticks,
               "hold_ticks": holds, "wait_ticks": waits, "wall_s": round(wall, 3),
               "clock": "sim" if span is not wall else "wall", "sim_s": round(span, 3) if span is not wall else None,
               "rtf": round(span / max(wall, 1e-9), 3) if span is not wall else None,
               "effective_hz": round(ticks / max(span, 1e-9), 2),
               "command_hz": round((ticks - holds) / max(span, 1e-9), 2),   # policy targets only (spec §8 gate)
               "command_hz_wall": round((ticks - holds) / max(wall, 1e-9), 2),
               "policy_calls": len(latencies),
               "latency_ms_mean": round(1e3 * float(np.mean(latencies)), 2) if latencies else None,
               "latency_ms_max": round(1e3 * float(np.max(latencies)), 2) if latencies else None,
               # Per-tick readiness: fraction of ticks that had a policy target (not a hold). In sim this counts
               # sim ticks, so it is lenient by 1/rtf for slow policies; policy_hz_rt is load-independent: the
               # command rate the policy's own WALL latency allows (targets delivered / wall time spent answering;
               # calls are sequential). At rtf 1 (real robot) a policy with policy_hz_rt < 30 makes holds.
               "ready_frac": round((ticks - holds) / max(ticks, 1), 4),
               "policy_hz_rt": (round((ticks - holds) / max(float(np.sum(latencies)), 1e-9), 2)
                                if latencies else None),
               "safety": env.filter.summary(), "success": info.get("success")}
    return summary, list(env.filter.events)


def export(run_dir: Path) -> Path:
    """Pack one run directory as <run_dir>.tar.gz (per-team log hand-back)."""
    out = run_dir.with_suffix(".tar.gz")
    with tarfile.open(out, "w:gz") as tf:
        tf.add(run_dir, arcname=f"{run_dir.parent.name}_{run_dir.name}")
    return out


def run(backend, policy_url, team, episodes, task="sort_fixed", seed=0, out="runs", video=False, log_images=False,
        container=None, startup_timeout=120.0, time_limit=120.0, host_port=P.PORT, env_kwargs=None, env=None,
        max_latency=MAX_POLICY_LATENCY, gpus="all", pull=False, do_export=False):
    from ..interface.env import AaltoEnv
    run_dir = Path(out) / team / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    summaries = []
    cm = Container(container, host_port, gpus=gpus, pull=pull) if container else None
    own_env = env is None
    client = None
    meta = {"team": team, "backend": backend, "task": task, "seed": seed, "episodes": episodes,
            "time_limit": time_limit, "max_policy_latency_s": max_latency, "policy": container or policy_url}
    try:
        if cm:
            cm.prepare()
            meta.update(image=cm.image, image_id=cm.image_id, gpus=gpus)
            cm.__enter__()
            policy_url = f"ws://localhost:{host_port}"
        try:
            client = PolicyClient(policy_url, startup_timeout, recv_timeout=max_latency + 30.0,
                                  alive=cm.alive if cm else None)
        except ConnectionError:
            if cm:
                log.error("container logs:\n%s", cm.logs(tail=40))
            raise
        if cm:
            meta["startup_s"] = round(time.monotonic() - cm.t_start, 2)
        meta["policy_metadata"] = client.metadata
        log.info("connected: %s (startup %s s)", client.metadata, meta.get("startup_s"))
        if own_env:
            env = AaltoEnv(backend=backend, task=task, seed=seed, time_limit=time_limit, realtime=True,
                           **(env_kwargs or {}))
        for i in range(episodes):
            if stop.is_set():
                break
            rec = Recorder(run_dir / f"episode_{i:03d}", images=log_images, video=video)
            try:
                summary, events = run_episode(env, client, i, seed + i, rec, stop, task, max_latency)
            except KeyboardInterrupt:
                stop.set()
                summary, events = {"episode": i, "end_reason": "interrupted"}, list(env.filter.events)
            except Exception as e:  # e.g. reset() on a connection that died between episodes
                log.error("episode %d failed: %r", i, e)
                summary = {"episode": i, "seed": seed + i, "end_reason": "policy_connection_lost", "error": repr(e)}
                events = list(env.filter.events)
            rec.save(events, summary)
            summaries.append(summary)
            with open(run_dir / "summary.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            log.info("episode %d: %s", i, summary)
            if summary["end_reason"] == "policy_connection_lost":
                if cm and not cm.alive():
                    log.error("container exited; last logs:\n%s", cm.logs(tail=20))
                log.error("policy connection lost: no further episodes in this run")
                break
    except KeyboardInterrupt:
        log.warning("interrupted")
    finally:
        if client:
            client.close()
        if env is not None and own_env:
            env.close()
        if cm:
            if cm.started:
                (run_dir / "container.log").write_text(cm.logs())
            cm.__exit__()
        (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
        if do_export:
            log.info("exported %s", export(run_dir))
    return summaries, run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "sim", "real"])
    ap.add_argument("--policy", default=f"ws://localhost:{P.PORT}")
    ap.add_argument("--container", help="docker image (ref or `docker save` tar) to run; overrides --policy")
    ap.add_argument("--pull", action="store_true", help="docker pull even if the image is local")
    ap.add_argument("--gpus", default="all", help='docker --gpus value; "none" for a CPU-only host')
    ap.add_argument("--host-port", type=int, default=P.PORT)
    ap.add_argument("--startup-timeout", type=float, default=120.0)
    ap.add_argument("--max-policy-latency", type=float, default=MAX_POLICY_LATENCY,
                    help="s one act() call may take (runner holds meanwhile) before the episode ends")
    ap.add_argument("--team", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--task", default="sort_fixed", choices=TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--export", action="store_true", help="also pack the run as <out>/<team>/<run>.tar.gz")
    ap.add_argument("--video", action="store_true", help="write head-cam mp4 per episode")
    ap.add_argument("--log-images", action="store_true", help="store all camera images in the npz (large; for replay)")
    ap.add_argument("--network-interface")
    ap.add_argument("--img-server-ip", help="default: 192.168.123.164 (real), 127.0.0.1 (sim)")
    ap.add_argument("--head-rgbd", action=argparse.BooleanOptionalAction, default=True,
                    help="head RGB + depth from the RGB-D server (sim :55560 / PC2); off = teleimager RGB, depth 0")
    ap.add_argument("--head-rgbd-port", type=int, default=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    kw = {}
    if a.backend == "sim":
        # 2nd+ sim instance (AALTO_INSTANCE): same env vars as sim/evaluate.py; without this the runner would
        # command the default instance on DDS domain 1. Sim only; the real backend is never remapped.
        from aaltonomous.sim.evaluate import apply_instance_env
        apply_instance_env()
    if a.backend != "mock":
        ip = a.img_server_ip or ("127.0.0.1" if a.backend == "sim" else "192.168.123.164")
        kw = {"network_interface": a.network_interface, "img_server_ip": ip, "head_rgbd": a.head_rgbd,
              "head_rgbd_port": a.head_rgbd_port}
    gpus = None if a.gpus.lower() == "none" else a.gpus
    summaries, run_dir = run(a.backend, a.policy, a.team, a.episodes, a.task, a.seed, a.out, a.video, a.log_images,
                             a.container, a.startup_timeout, a.time_limit, a.host_port, kw,
                             max_latency=a.max_policy_latency, gpus=gpus, pull=a.pull, do_export=a.export)
    print(f"logs: {run_dir}")
    for s in summaries:
        print(s.get("episode"), s.get("end_reason"), s.get("command_hz"), s.get("safety", {}).get("clipped_steps"))


if __name__ == "__main__":
    main()
