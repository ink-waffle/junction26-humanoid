# aaltonomous/sim — Isaac Lab tasks (WP3)

Base: `third_party/unitree_sim_isaaclab` (Isaac Sim 6.2 / Isaac Lab 3.0, twinflow RTX 5090). Install: `docs/ISAAC_TWINFLOW.md`.
`third_party/` is never edited: `aalto_sim_main.py` runs upstream `sim_main.py` with our tasks registered and two
runtime patches (see its comments).

## Run (twinflow, env `isaac`)

```bash
TASK=Aalto-Sort-G129-Dex3-Joint   bash aaltonomous/sim/run_sim.sh     # rungs 1-2   (ready in ~2 min)
TASK=Aalto-Insert-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh     # + rung-3 lid on the lid bin
python aaltonomous/sim/check_scene.py [--lid]                          # 8 PASS / 0 FAIL (2026-11-10)
python aaltonomous/sim/check_sim_shim.py --backend sim --img-server-ip 127.0.0.1 --yes --head-rgbd   # 38 / 0
python -m aaltonomous.sim.scene_client {ping,state,random 3,eval sort_fixed}
```
Stop with SIGINT to the python pid (`pgrep -f "^python .*aalto_sim_main"`), not `pkill -f`.

## What is in the task (`aalto_task/`)

| Piece | Source of truth | Notes |
|---|---|---|
| Robot | upstream `g1_29dof_dex3_base_fix` | Pelvis at world (0, 0, 0.79), facing +x: world = pelvis frame + 0.79 z. Arm home pose `layout.ARM_HOME` (arm zero = hands on the counter). **Absolute** joint targets; uncommanded joints (legs, waist) held at the default pose. |
| Counter, mat | `scene.yaml` `counter`, `assets/mat_texture.png` | Static 4 cm slab + textured quad (the printed mat). |
| Bins | `scene.yaml` `bins` | Static floor + 4 walls each; one colour for all (`bins.rgb`, orange). |
| Objects (18) | `scene.yaml` `objects`, `assets/object_*.stl` | The print meshes, convex-hull colliders, 50 g, friction 1.2/1.0 (`scene_cfg.py` constants). Names `obj_<colour>_<shape>_<k>`; unused ones parked on the floor behind the robot (never scored). |
| Lid (Insert task) | `scene.yaml` `lid`, `assets/lid_shapes.stl` | Static exact-mesh collider; every shape drops through its own hole. |
| Head camera | `interface/calibration.yaml` `head` | ONE 640x480 pinhole at `T_pelvis_optical` (on `pelvis`, waist fixed). RGB on teleimager :55555 (mono), RGB-D on :55560 in the `pc2/head_rgbd_server.py` format, so `AaltoEnv(..., head_rgbd=True)` reads sim and real the same way. ~17 fps. |
| Wrist cameras | upstream presets | Placeholder poses until M7 measures the mounts (then write a `--camera_calib` json). |
| Layouts / success | `interface/tasks.py`, `scene/scene.py` | Scene service on tcp :55580 (`hooks.py`, client `scene_client.py`): `reset` (layout, optional per-object `z`), `reset_random`, `state`, `evaluate`. `DDSBackend.reset_scene` uses it, so `AaltoEnv.reset()` places its layout. DDS `rt/reset_pose/cmd` 1/2 = random layout. |
| Episode | — | No auto-reset / time-out inside Isaac; AaltoEnv / the eval own episodes. |

Pure-numpy parts (`layout.py`) are unit-tested offline: `python -m unittest aaltonomous/sim/aalto_task/test_layout.py`.

## Oracle, evaluation, DR (M4 rest + M12)

```bash
python -m aaltonomous.sim.evaluate --policy oracle --task sort_fixed    --trials 10 --no-cameras   # rung 1
python -m aaltonomous.sim.evaluate --policy oracle --task sort_language --trials 10 --no-cameras   # rung 2
python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --team demo --trials 20             # policy server
python -m aaltonomous.sim.evaluate --container team/sub:latest --team demo --trials 20               # docker image
python -m unittest aaltonomous/sim/test_sim_offline.py                                              # offline
```
(env `isaac` + `PYTHONPATH=$REPO`; `msgpack` was pip-installed into `isaac` for the deploy protocol.)

| Piece | What |
|---|---|
| `oracle.py` | `OraclePolicy`: the classical baseline's planner/IK (imported) on **privileged** object poses from the scene service; per-object bin via `scene.resolve_bin`; rung-2 mapping parsed from the instruction (all training templates; falls back to the privileged `env.mapping`); relay hand-over for cross-body objects; plans the next object while executing. Home = `layout.ARM_HOME` (see note). |
| `evaluate.py` | N seeded trials (seed = `--seed`+i -> layout, instruction, DR draw). Per trial: park objects, arms to `ARM_HOME` through the filter, `deploy.runner.run_episode` (30 Hz, chunks, holds, watchdog), settle, score with the scene service (scene.py checker). Writes `runs/eval/<name>_<task>_<variant>_<ts>/eval.json` + per-episode npz/jsonl and prints a table. **Sim gate**: no abort / watchdog / crash / lost connection, clipped steps <= `--gate-max-clips` (0), rate >= `--gate-min-hz` (27); exit code 2 if it fails. |
| `aalto_task/dr.py`, `dr/*.yaml` | DR knobs (lights intensity/colour/position, mat brightness/tint, counter + bin + object colours (HSV), object mass + friction, layout xy/yaw jitter, head camera pose jitter, RGB gain/noise, depth noise/holes). Seeded, per episode via scene service `randomize`; **off unless `AALTO_DR` is set** (`train`, `eval_heldout` or a yaml path). |
| `check_dr.py` | On a DR sim: randomize seeds, check every group applied + head image changes (`--save DIR` writes PNGs). |
| 2nd sim | `AALTO_INSTANCE=b AALTO_DR=train TASK=... bash aaltonomous/sim/run_sim.sh`: own DDS domain (7), shm names, ports (cams 56555-7, rgbd 56560, scene 56580, sim clock 56581, teleimager responder 61000; stub in `aalto_sim_main.py`). Clients: same env vars (`AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560`). |


Status, held-out details and organiser test fixtures: `ORGANISER_NOTES.md` (organiser-only, not in the bundle).

## Timing (sim clock, 2026-11-11)

The sim runs **below real time**: real-time factor (RTF = sim s / wall s) 0.70 alone, ~0.59 with 2 sims, 0.50-0.53 with 3 on twinflow
(each sim uses 3-6 CPU cores, physics on CPU). Per 10 ms sim step (~19 ms wall, 3 sims):
physics 2x1.9 ms, RTX render 6 ms (21 ms per frame at 28.6 Hz), camera readout + JPEG 4.8 ms, head RGB-D 1.2 ms,
rest ~2 ms. Rendering is ~2/3 of the cost, so 3 cameras at 30 fps can't reach 1.0x here.
Fix: the sim publishes its step clock (zmq PUB :scene port + 1, e.g. 55581; `aalto_task/hooks.py`) and
`AaltoEnv(backend="sim")` paces its 30 Hz ticks on it (`interface/backends.py` `SimClock`): the physics sees
commands at 30 Hz of sim time whatever the load. `obs["t"]`, time limits, the filter watchdog and the runner's
`command_hz` count sim seconds; `wall_s`, `rtf`, `command_hz_wall`, `policy_hz_rt` (rate the policy's wall latency
allows) are logged too. Real robot: unchanged (wall). A sim started before this has no clock -> the backend warns
and falls back to wall pacing (restart the sim). `AALTO_SIM_CLOCK=0` forces wall pacing; the scene service `ping`/
`info` return `sim_t`, `rtf`; the sim log warns every 10 s while RTF < 0.95. `AALTO_PROFILE=1` prints a per-step
time breakdown. Episodes take 1/RTF longer in wall time (120 s sim ~ 220 s wall).

Notes: the classical `home_arm_q` (wrist pitch 1.0, arms out) collides hand-vs-body in the sim: wrist pitch stalls
at ~0.6 and wrist yaw is pushed ~0.7 rad off, so every grasp from it misses. Camera noise applies to the :55560
head RGB-D only (not the teleimager mono / wrist streams).

## Still open (M4 rest)

- Oracle ~84 % on rung 1, needs >90 %: triangle grasps (thumb vs one finger spins the prism). Earlier tuning ran
  ~1.8x too fast in physics time (wall pacing at RTF ~0.55); re-check grasp parameters on the sim clock.
- Wrist camera poses from M7; head pose re-check against the real overlay (M7).
- Real arm home pose (spec §3) to replace `ARM_HOME`; real Dex3 friction / PD fit (N1).
- Upstream `rt/sim_state` is disabled for Aalto tasks (its 4 KB slot overflows with 18 objects); use the scene service.
