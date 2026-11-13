# Data card: Aaltonomous Humanoid teleop demos (Drop <N>)

> Template. Replace each `<...>`, and delete the lines that don't apply. Counts come from `meta/info.json` and
> `meta/aalto_episodes.jsonl`.

## What
Real-robot VR teleoperation demos of the Junction "Aaltonomous Humanoid" task. Väiski is a Unitree G1 29-DoF with
Dex3-1 hands, seated behind an 80 cm table with a stiff lower body (debug mode, no locomotion). It sorts 3 to 5
coloured 45 to 50 mm objects (cube, cylinder, triangle) into 3 bins. The setup is the same as the challenge's
real-robot runs (spec §2, layout v1.1).

| | rung 1 `sort_fixed` | rung 2 `sort_language` |
|---|---|---|
| episodes (success / converted) | <n> / <n> | <n> / <n> |
| frames (30 fps) | <n> | <n> |
| deliberately messy (regrasp / recovery) | <n> | <n> |
| layout cards (seeds) | <1000-10xx> | <2000-20xx> |

Recorded <date> at <place> by <operators>, with teleop `scripts/teleop_safe.py --ee=dex3 <args>` and lighting <...>.

## Format
Two copies of the same episodes and frames:
- **LeRobot v3.0** (`<name>/`), written by `lerobot==0.7.0` (`LeRobotDataset("aalto/<name>", root=...)`), with head
  depth. Described below.
- **GR00T** (`<name>_groot/`): LeRobot v2.1 + `meta/modality.json`, the layout Isaac-GR00T fine-tunes on. Same
  `observation.state`, `action`, `observation.arm_dq` and the 3 RGB videos (H.264), **no depth** (GR00T's video input
  is 8-bit RGB). The instruction is `task_index` → `meta/tasks.jsonl` (`annotation.human.task_description`).
  Modality groups `left_arm`, `right_arm`, `left_hand`, `right_hand` (7 each). Train with
  `--embodiment-tag NEW_EMBODIMENT --modality-config-path <name>_groot/new_embodiment_config.py` (arm actions
  relative, hands absolute; your policy server must still send absolute targets).

LeRobot v3.0 details: Keys follow the challenge interface (spec §4). Every vector is
[left 7, right 7] in `joint_spec.yaml` order (arm = DDS motor 15..28; each Dex3 in its own DDS order, so left is
thumb, middle, index and right is thumb, index, middle; left/right limits mirrored).

| Feature | Shape | Meaning |
|---|---|---|
| `observation.images.head_rgb` | 480×640×3 video | Head RealSense D435i colour |
| `observation.images.head_depth` | 480×640×1 depth video | Aligned depth, **mm**; 0 = invalid. Stored as LeRobot depth video (HEVC 12-bit lossless, **linear** quantisation 0 to 4.095 m = 1 mm/code), so it decodes bit-exact up to 4095 mm and farther pixels clip to 4095. If your policy only takes RGB, drop it from `input_features`. |
| `observation.images.{left,right}_wrist_rgb` | 480×640×3 video | Wrist UVC cams |
| `observation.state` | 28 | `arm_q` (14) ++ `hand_q` (14), measured, rad |
| `observation.arm_dq` | 14 | Measured arm velocity, rad/s |
| `action` | 28 | `arm_q_target` (14) ++ `hand_q_target` (14): the **commanded** absolute joint targets |
| `task` | str | The episode's instruction (rung 2: language-defined bin mapping) |

To use it with AaltoEnv or a policy server: `arm_q_target = action[:14]` and `hand_q_target = action[14:]`.

Per-episode extras are in `meta/aalto_episodes.jsonl` (both copies), keyed by `episode_index`: success label, task,
instruction, mapping, layout (object colour, shape and pose in the mat frame), card seed, operator, notes, safety
filter check, DDS rates, warnings, and head depth health (`head_depth`, `depth.stale_ticks`, `depth.blank_frames`,
`depth.valid_frac_mean`).

## How it was made
- Teleop publishes `rt/lowcmd` (debug mode, arm targets after its 250 Hz rate limiter) and `rt/dex3/*/cmd`. A
  separate read-only recorder (`aaltonomous/data/record_demos.py`) logs every command message and samples
  observations at 30 Hz. `action[t]` is the time-average of the command over ±1/60 s around the tick of
  `obs[t]` (an anti-aliased resample to 30 Hz).
- Cameras: head RGB-D from `head_rgbd_server.py` on PC2 and wrists from teleimager. Each frame is the newest image
  at the tick. A camera slower than 30 fps repeats frames.
- **Raw drop:** successful, failed and messy episodes are all included. Use `success`, `notes` and `warnings` in
  `meta/aalto_episodes.jsonl` to choose what to train on; judging data quality is part of the challenge.
- Calibration (intrinsics and extrinsics for the 3 cameras), `joint_spec.yaml`, and the scene geometry and mat
  texture ship in the same drop.

## Known issues
- Demos were made at teleop speed caps of <0.9> rad/s for arms and <3> rad/s for fingers. `filter_check` in the
  sidecar shows how many ticks our safety filter (1 rad/s arms) would clip: <...>.
- Operator style: <one or two hands, typical grasp, ...>. No obs/action latency compensation (not measured).
- <camera frame drops / stale DDS ticks, see `warnings`>
- <head depth: holes on the dark mat / near the bin walls>

## License
<CC BY 4.0 / for Junction 2026 participants only>. Contact: <...>.
