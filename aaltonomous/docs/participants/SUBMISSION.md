# Submission: a Docker policy server

*Draft 2026-11-11.* Your submission is a Docker image that serves your policy over a websocket. Our runner is the
client: it owns the sim or the robot, runs the safety filter and calls your server. You never talk to the robot.

## 1. Contract

| Item | Requirement |
|---|---|
| Image | x86_64 (linux/amd64) Docker image. Base on CUDA **≥ 12.8** (the finals GPU is an RTX 5090, Blackwell). Template: `aaltonomous/docker/submission.Dockerfile` (`nvidia/cuda:12.8.0-base-ubuntu22.04`, ~0.6 GB with the example); torch example: `submission-gpu.Dockerfile` |
| Start | `CMD` starts the policy server on **port 8000** (`0.0.0.0`). We run `docker run -d --gpus all -p <port>:8000 <image>`, no volumes, no extra arguments |
| Startup | Listening **within 120 s** of `docker run` |
| Weights | **Inside the image.** No downloads at startup |
| GPU | Must fit in **32 GB VRAM** (one RTX 5090) |
| Network | Internet access from the container during eval / finals: **TBD** (relevant for API-based planners) |
| Image size limit | **TBD** (intake default 40 GB uncompressed) |

## 2. Protocol (`aaltonomous/deploy/protocol.py`, version 1)

Websocket, binary **msgpack** messages, every message a dict with a `"type"`. numpy arrays are encoded as
`{"__nd__": 1, "dtype": <numpy dtype str>, "shape": [...], "data": <raw bytes>}` (openpi-style; `protocol.pack` /
`unpack` do it for you).

| Direction | Message |
|---|---|
| server → client, on connect | `{"type": "metadata", "version": 1, "name": str, "action_horizon": int or None}` |
| client → server | `{"type": "reset", "task": str, "instruction": str, "seed": int}` |
| server → client | `{"type": "reset_ok"}` |
| client → server | `{"type": "act", "obs": {...}}` — keys and shapes in [INTERFACE.md §2](INTERFACE.md#2-observations-obs) |
| server → client | `{"type": "action", "action": {"arm_q_target": (14,) or (k,14), "hand_q_target": ..., "done": bool?}}` |
| server → client, on failure | `{"type": "error", "message": str}` — the runner holds and the watchdog clock keeps running (1 s of errors ends the episode) |

- One connection per run; `reset` before every episode, then `act` repeatedly. Only one request is in flight at a
  time.
- While an `act()` call is in flight the runner holds the robot at its last target, for up to **30 s per call**
  (`--max-policy-latency`); a longer call ends the episode (`policy_timeout`), a closed connection ends it at once
  and stops the run. Held ticks are not commands (sim gate rate). Details: [INTERFACE.md §3](INTERFACE.md#timing-30-hz-chunks-holds).

## 3. Writing the server

The easy way is to subclass `Policy` from `aaltonomous/deploy/policy_server.py` and call `serve()`:

```python
import numpy as np
from aaltonomous.deploy.policy_server import Policy, serve

class MyPolicy(Policy):
    name = "my-team"
    action_horizon = 10                      # informational

    def reset(self, instruction, task=None, seed=None):
        self.instruction = instruction

    def act(self, obs):
        arm, hand = obs["arm_q"], obs["hand_q"]  # replace with your model
        k = 10
        return {"arm_q_target": np.repeat(arm[None], k, 0), "hand_q_target": np.repeat(hand[None], k, 0)}

if __name__ == "__main__":
    serve(MyPolicy(), port=8000)
```

Slow models (VLM calls, big VLAs): the runner holds the robot safely while a call is in flight (≤ 30 s), but to
pass the sim gate run inference in your own background thread and let `act()` return immediately with "hold"
targets (or the rest of a plan) until the result is ready. Holds that **you** return count as commands;
holds that the runner has to insert because you're late do not (sim gate: ≥ 27 Hz command rate).
`aaltonomous/baselines/classical/policy.py` does exactly this.

You may also implement the protocol yourself in any language, as long as the messages match.

## 4. Build

```bash
# from the repo root; edit the COPY lines for your code and weights
docker build -f aaltonomous/docker/submission.Dockerfile -t myteam/policy:v1 .
docker run --rm --gpus all -p 8000:8000 myteam/policy:v1
```

- GPU example (torch cu128, refuses to start without a GPU, reports the GPU in its metadata name):
  `aaltonomous/docker/submission-gpu.Dockerfile` + `example_gpu_policy.py`.
- `--gpus all` needs the NVIDIA Container Toolkit on your machine (`nvidia-ctk runtime configure --runtime=docker`).
- The build context is the repo root: keep it small. BuildKit (`docker buildx`) uses
  `aaltonomous/docker/<name>.Dockerfile.dockerignore`; the legacy builder only reads `.dockerignore` in the context
  root, so copy one of those files there (`cp aaltonomous/docker/submission.Dockerfile.dockerignore .dockerignore`).
  Both exclude `third_party/` (except the URDF), `runs/`, `data/`, `.git` etc.; add your own large directories.
- A complete worked example (non-trivial dependencies, files from the repo, a threaded planner): the reference
  submission in §7.

## 5. Test locally (in this order)

```bash
# 0. the example server (holds still), to see the plumbing work
python -m aaltonomous.deploy.policy_server

# 1. dry run: blank images, zero joints; checks reply shapes and the command rate (targets you return per second,
#    chunks count per tick; >= 27 Hz, same threshold as the sim gate). Exit 0 = pass
python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8000
python -m aaltonomous.deploy.replay --synthetic 100 --container myteam/policy:v1      # also checks startup (< 120 s)

# 2. full episodes on the mock backend (no GPU, no sim): runner + safety filter + logs in runs/<team>/
python -m aaltonomous.deploy.runner --backend mock --policy ws://localhost:8000 --team myteam --episodes 2
python -m aaltonomous.deploy.runner --backend mock --container myteam/policy:v1 --team myteam --episodes 2
#    (add --gpus none if your machine has no nvidia-container-toolkit)

# 3. scored episodes + the sim gate in Isaac (cloud RTX instance, sim running; see the Quickstart)
python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --team myteam --task sort_fixed --trials 20
python -m aaltonomous.sim.evaluate --container myteam/policy:v1 --team myteam --task sort_language --trials 20
#   exit code 0 = gate passed, 2 = failed; results in runs/eval/<name>_<task>_<variant>_<ts>/eval.json
#   Inside the aalto-sim container there is no docker CLI: start your image on the host
#   (docker run -d --gpus all -p 8000:8000 myteam/policy:v1) and use --policy ws://localhost:8000
#   (the sim container runs with --network host). --container works on the bare-metal setup.

# 4. replay a recorded episode (e.g. your shakedown log) to the server
python -m aaltonomous.deploy.replay runs/<team>/<run>/episode_000.npz --policy ws://localhost:8000
```

- Setting up the sim for step 3 (`run_sim.sh`, the `isaac` environment, domain randomisation `AALTO_DR=train`):
  [Quickstart](../../cloud/QUICKSTART.md).
- Your local `evaluate.py` runs on the **training** variant. The leaderboard runs the same script on the held-out
  variant with secret seeds.
- `evaluate.py` gate defaults: `--gate-max-clips 0`, `--gate-min-hz 27`, `--time-limit 120`. Run with the defaults.
- Runner logs: `runs/<team>/<run>/episode_NNN.npz` (per tick: joints, commands, holds, clips, safety mode),
  `episode_NNN.jsonl` (every clip event with its reason + a summary line), `summary.jsonl`. Read the clip reasons
  before submitting.

## 6. Hand-in

1. Push your image to a registry you control (Docker Hub / GHCR private repo with read access for our account
   **TBD**, or public) and note its digest (`docker push` prints `sha256:...`).
2. Hand in a `submit.yaml` (where: **TBD**, announced at kickoff):
   ```yaml
   team: rocket                          # your team id from kickoff (lowercase, letters/digits/-)
   image: ghcr.io/rocket/policy@sha256:...   # pin the digest, not just a tag
   tasks: [sort_fixed, sort_language]    # optional, default both
   contact: "@you on the event chat"     # organisers only, never shown
   ```
   No registry? `docker save myteam/policy:v1 | gzip > rocket-v1.tar.gz`, upload it next to the yaml and write
   `image_tar: rocket-v1.tar.gz` instead of `image:` (upload location **TBD**; slower for big images).
3. We first run a quick pre-check (image pulls, linux/amd64, size limit **TBD**, server up within 120 s,
   `replay --synthetic` at >= 27 Hz: run the same check yourself with step 1 of §5). A failure comes back with the
   reason within minutes and costs no sim time. Then the held-out sim evaluation runs; the leaderboard updates when
   it finishes (about an hour per submission, queue permitting).
4. Status: a public queue page shows each submission's state (queued, validating, evaluating, done, failed + reason)
   and, when done, the mean score and the sim-gate result with its reasons. Per-episode held-out logs are not
   shared.

- Leaderboard submissions: one evaluation per team at a time; a newer submission replaces your older one that is
  still waiting; at most one evaluation start per team per **TBD (default 60)** minutes.
- Final submission deadline Sunday, time **TBD**. The image you hand in last before the deadline is the one that
  runs in the finals; shakedown changes must be re-submitted the same way. After a shakedown or finals run you
  get your own run logs back (`runs/<team>/<run>/`, incl. your container's log).

## 7. Reference submission: the classical baseline

The classical rung-1 baseline ([README](../../baselines/classical/README.md)) packaged exactly like a team
submission. Copy it as a starting point; it is also what we run through the runner in the finals dry run.

| File | What |
|---|---|
| `aaltonomous/baselines/classical/policy_server.py` | `Policy` subclass wrapping `ClassicalPolicy` (reset/act); `serve()` on :8000 |
| `aaltonomous/docker/baseline.Dockerfile` | `nvidia/cuda:12.8.0-base-ubuntu22.04` + numpy, scipy, opencv-headless, Pinocchio (`pin`), the URDF, calibration, scene + config files |

```bash
docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .
python -m aaltonomous.deploy.replay --synthetic 100 --container aalto/baseline-classical:v1   # + --gpus none without a GPU runtime
python -m aaltonomous.deploy.runner --backend sim --container aalto/baseline-classical:v1 --team baseline --episodes 5
```

- Measured on twinflow (2026-11-11, CPU only): image 1.46 GB (307 MB compressed), build context < 1 MB, up in
  0.5–0.7 s; replay 300+ Hz with p50 latency ~4 ms. Through the runner: rendered mock world 9/9 sorted; Isaac sim
  (5 seeds) 29.8 Hz command rate, 0 clips, but **0/24 sorted** (grasps slip). It is a plumbing and code reference,
  not a bar to beat.
- **How it stays above the 27 Hz gate:** looking + planning (IK, 0.1 to a few s) runs in a background thread;
  meanwhile `act()` returns a 1-tick hold at once, and motion comes back in 10-tick chunks. Its own holds count
  as commands, so the rate stays ~30 Hz although planning is slow.
- Needs no GPU. Uses only `head_rgb`, `head_depth`, `arm_q`, `hand_q`, `t`; camera intrinsics/extrinsics come from
  `interface/calibration.yaml` **baked into the image**, so rebuild it when the calibration is updated.
- The Pinocchio wheel on PyPI (`pin` 3.3.1) needs numpy ≥ 2.2 and `libgomp1`; see the Dockerfile.

