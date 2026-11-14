"""Finals proxy: score(sort_fixed) + score(sort_language, known) + 1.5 x score(sort_language, unseen), legs spread over the running sims.

    python -m halal.eval.proxy --name v1 --server "python -m halal.server" --trials 8 --sims a,b
    python -m halal.eval.proxy --name v1-image --image halal/policy:v1 --trials 8 --sims a,b
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIMS_SH = REPO / "halal/ops/sims.sh"
RUNS = REPO / "runs"


@dataclass(frozen=True)
class Leg:
    key: str
    task: str
    instructions: str
    weight: float


LEGS = (Leg("fixed", "sort_fixed", "train", 1.0), Leg("known", "sort_language", "train", 1.0),
        Leg("unseen", "sort_language", "unseen", 1.5))


def sim_env(tag: str) -> dict[str, str]:
    """Environment that points evaluator clients at sim `tag`."""
    script = f'eval "$(bash {shlex.quote(str(SIMS_SH))} env {tag})"; env -0'
    out = subprocess.run(["bash", "-c", script], capture_output=True, check=True).stdout
    return dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)


def wait_port(port: int, proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited with {proc.returncode}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise TimeoutError(f"policy server not listening on :{port}")


def run_leg(args: argparse.Namespace, leg: Leg, tag: str, port: int, out_dir: Path) -> dict:
    """One evaluator run against its own policy server; returns the eval.json summary."""
    env = sim_env(tag)
    team = f"{args.name}-{leg.key}"
    cmd = [sys.executable, "-m", "halal.eval.sim_eval", "--instructions", leg.instructions, "--task", leg.task,
           "--team", team, "--trials", str(args.trials), "--seed", str(args.seed), "--out", str(out_dir),
           "--policy", f"ws://localhost:{port}", "--time-limit", str(args.time_limit)]
    server = None
    if args.image:
        cmd += ["--container", args.image, "--host-port", str(port)]
    else:
        log = open(out_dir / f"{team}.server.log", "w")
        server = subprocess.Popen(shlex.split(args.server), cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PORT": str(port), "AALTO_TIME_LIMIT": str(args.time_limit)})
        wait_port(port, server)
    try:
        with open(out_dir / f"{team}.eval.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        if server is not None:
            server.terminate()
    found = sorted(out_dir.glob(f"{team}_{leg.task}_*/eval.json"))
    if not found:
        return {"leg": leg.key, "error": f"evaluator exit {code}, no eval.json"}
    return {"leg": leg.key, "exit": code, "dir": str(found[-1].parent), **json.loads(found[-1].read_text())["summary"]}


def run_sim_queue(args: argparse.Namespace, tag: str, legs: list[tuple[int, Leg]], out_dir: Path, results: dict) -> None:
    for i, leg in legs:
        try:
            results[leg.key] = run_leg(args, leg, tag, args.port + i, out_dir)
        except Exception as e:
            results[leg.key] = {"leg": leg.key, "error": repr(e)}


def report(name: str, results: dict[str, dict]) -> dict:
    rows, proxy, gate = [], 0.0, True
    for leg in LEGS:
        r = results.get(leg.key)
        if r is None:
            continue
        ok = "error" not in r and r["gate"]["passed"]
        gate &= ok
        proxy += leg.weight * r.get("score_mean", 0.0)
        rows.append(f"  {leg.key:<7} {r.get('score_mean', float('nan')):+6.2f} ± {r.get('score_std', 0):.2f}   "
                    f"objects {r.get('objects_correct', '?')}/{r.get('objects', '?')}   outcomes {r.get('outcome_counts')}   "
                    f"ends {r.get('end_reasons')}   clips {r.get('safety_clipped_steps_total')}   "
                    f"hz {r.get('command_hz_min')} rt_hz {r.get('policy_hz_rt_min')}   "
                    f"gate {'PASS' if ok else 'FAIL ' + str(r.get('error', r.get('gate')))}")
    print(f"\n=== {name}: proxy {proxy:+.2f}   gate {'PASS' if gate else 'FAIL'}")
    print("\n".join(rows))
    return {"name": name, "proxy": round(proxy, 3), "gate": gate, "legs": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--server", default=f"{sys.executable} -m halal.server", help="policy server command (PORT env)")
    ap.add_argument("--image", help="docker image to evaluate instead of a local server")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--sims", default="a", help="comma-separated sim tags to spread the legs over")
    ap.add_argument("--legs", default="fixed,known,unseen")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args(argv)

    out_dir = RUNS / "proxy" / f"{args.name}_{time.strftime('%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = args.sims.split(",")
    legs = [(i, leg) for i, leg in enumerate(LEGS) if leg.key in args.legs.split(",")]
    queues = {tag: [x for k, x in enumerate(legs) if tags[k % len(tags)] == tag] for tag in tags}
    results: dict[str, dict] = {}
    threads = [threading.Thread(target=run_sim_queue, args=(args, tag, q, out_dir, results)) for tag, q in queues.items() if q]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = report(args.name, results)
    (out_dir / "proxy.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out_dir / 'proxy.json'}")
    return 0 if summary["gate"] else 2


if __name__ == "__main__":
    sys.exit(main())
