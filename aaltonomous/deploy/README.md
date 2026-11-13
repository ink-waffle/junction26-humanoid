# aaltonomous.deploy

Policy-server template, runner, dry-run replay. Protocol in `protocol.py` (websocket + msgpack, port 8000, version 1).

**Participants:** subclass `Policy` (`policy_server.py`), return `{"arm_q_target", "hand_q_target"}` ((14,) or (k,14) chunk),
`serve(policy)`. Package as a Docker image with `docker/submission.Dockerfile` (CUDA 12.8 base, port 8000, up in < 120 s);
torch/GPU example: `docker/submission-gpu.Dockerfile` + `docker/example_gpu_policy.py`.

```bash
python -m aaltonomous.deploy.policy_server                      # example hold policy on :8000
python -m aaltonomous.deploy.replay --synthetic 100             # dry-run: valid shapes, command rate >= 27 Hz?
python -m aaltonomous.deploy.replay --synthetic 100 --container myteam/policy:v1   # + startup time (< 120 s)
python -m aaltonomous.deploy.runner --backend mock --policy ws://localhost:8000 --team demo --episodes 2
python -m aaltonomous.deploy.runner --backend sim --container myteam/policy:v1 --team myteam --episodes 5 --video --export
python -m unittest aaltonomous/deploy/test_deploy.py            # + AALTO_TEST_IMAGE=<image> for the docker test
```

Container path (`--container IMAGE` or a `docker save` tar): load/pull, `docker run -d --gpus all -p PORT:8000`, wait
for the websocket (fails fast if the container exits), episodes, `container.log`, stop + rm. `--gpus none` on a host
without nvidia-container-toolkit. `run.json` records image id, startup time and the policy's metadata.

Runner behaviour: 30 Hz, chunks executed tick by tick, every command goes through the safety filter.
- **Policy computing** (request in flight): the runner re-sends the last command as an explicit hold target every
  tick (zero step, checked by the filter like any target), so slow policies (VLM, 10–20 s/call) are held, not
  stopped. Cap: `--max-policy-latency` (default 30 s) per call -> episode ends `policy_timeout`.
- **Connection closed** (container died): ends at once, `policy_connection_lost`; the run stops.
- **Error reply**: hold without refresh, so the filter watchdog (hold 0.2 s, safe stop 1 s) ends it as `watchdog`
  if the policy keeps failing. The watchdog also catches a stalled runner loop.
- Holds the runner inserts (`hold`/`policy_wait` columns) never count as commands: `command_hz` (sim gate >= 27 Hz)
  counts policy targets only. Holds the policy returns itself do count.
- Ctrl-C stops cleanly. Sim/real: head RGB-D on by default (`--no-head-rgbd` = teleimager RGB, depth 0);
  `--img-server-ip` defaults to 127.0.0.1 (sim) / 192.168.123.164 (real).

Logs go to `runs/<team>/<run>/episode_NNN.{npz,jsonl,mp4}` + `summary.jsonl`, `run.json`, `container.log`;
`--export` packs `runs/<team>/<run>.tar.gz`.
