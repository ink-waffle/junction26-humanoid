# Starter paths

*Draft 2026-11-11. Pointers, not solutions.* Any method is allowed and hybrids are probably where the best results
come from. There is deliberately **no skill API**: you get the low-level interface ([INTERFACE.md](INTERFACE.md)) and
a classical baseline to read and beat.

## What everyone has

| Thing | Where |
|---|---|
| Interface, joint spec, safety filter | `aaltonomous/interface/` (`env.py`, `joint_spec.yaml`, `safety.py`) |
| Camera calibration (provisional) | `aaltonomous/interface/calibration.yaml` + `calibration.py` |
| Scene geometry, success checker, scoring | `aaltonomous/scene/scene.yaml`, `scene.py` |
| Isaac Lab task + DR | `aaltonomous/sim/` (`Aalto-Sort-G129-Dex3-Joint`, `Aalto-Insert-G129-Dex3-Joint`; `AALTO_DR=train`, knobs in `sim/dr/train.yaml`) |
| Robot model (URDF, for IK/FK) | `third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf` (G1 29-DoF + Dex3) |
| Classical baseline | `aaltonomous/baselines/classical/` ([README](../../baselines/classical/README.md)) |
| Real demos (LeRobot format) | [Data card](../../data/DATA_CARD.md) |
| Policy-server template, runner, eval | `aaltonomous/deploy/`, `aaltonomous/sim/evaluate.py` ([SUBMISSION.md](SUBMISSION.md)) |

## Route 1: classical / modular

Depth + detection → object pose → grasp → IK → scripted Dex3 grasp.

- Start from the baseline: `perception.py` (HSV blobs + head depth → pelvis frame), `kinematics.py` (Pinocchio
  damped-least-squares IK), `planner.py` (Cartesian waypoints → 30 Hz trajectories pre-checked against the safety
  filter), `policy.py` (state machine, hold while planning). Run it in sim:
  `python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2`. Overview: [baseline README](../../baselines/classical/README.md);
  packaged as a submission image (copy it): [SUBMISSION.md §7](SUBMISSION.md#7-reference-submission-the-classical-baseline).
- Where it fails today: grasps slip (triangles, cylinders), objects near the centre line need the arm on the wrong
  side, ~10–20 s per pick against a 120 s budget, and it is rung 1 only (no instruction parsing, no shape).
- Ideas to look at: yaw-aware grasps from the depth cloud, open-vocabulary detection/segmentation
  ([Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [SAM 2](https://github.com/facebookresearch/sam2)),
  GPU motion planning ([cuRobo](https://github.com/NVlabs/curobo)), [Pinocchio](https://github.com/stack-of-tasks/pinocchio),
  both arms at once.
- Main real-robot risk: calibration error and depth holes. Drop 1 has real depth noise on our objects.

## Route 2: VLM planner + your own skills

A VLM/LLM reads the instruction and the image and calls skills **you** write (pick, place, look, ...).

- Fits rung 2 (mapping by language, bins named by position).
- **Latency budget:** frontier VLMs take ~10–20 s per call. The runner holds the robot still while an `act()`
  call is in flight (up to 30 s per call), but held ticks don't count toward the sim gate's 27 Hz command rate,
  so: call the model in a background thread, keep returning hold targets from `act()`, and execute skills at
  30 Hz in between. Plan once per episode (or per object), not per
  tick. Budget: with 120 s and 3–5 objects you can afford a few calls, not dozens.
- The planner is only as good as the skills under it: the precision (grasp, place) still has to come from route 1, 3
  or 4.
- External APIs are allowed at your own risk; we provide no network or credits (see [FAQ](FAQ.md)). Plan a fallback
  (local model, or a deterministic parser for the training templates in `tasks.py`).

## Route 3: imitation learning / VLA

Real demos (+ synthetic demos) → policy that maps `obs` → joint targets.

- Our demos are in LeRobot format with the same keys as the interface; actions are the commanded targets at 30 Hz
  ([Data card](../../data/DATA_CARD.md)). About 20 per rung — scarce on purpose. Two copies of the same episodes:
  LeRobot **v3.0** (with head depth) for LeRobot policies, and a **GR00T** copy (LeRobot v2.1 + `meta/modality.json`,
  RGB only) that Isaac-GR00T fine-tunes on directly.
- Small policies: [LeRobot](https://github.com/huggingface/lerobot) ACT and Diffusion Policy; SmolVLA (in LeRobot).
- VLAs: [openpi](https://github.com/Physical-Intelligence/openpi) (π0 / π0.5; its websocket + msgpack server
  pattern is what our protocol is modelled on), [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T).
  Fine-tuning needs 80 GB GPUs (get them from Verda yourself); inference fits the 32 GB finals GPU.
- More data: generate synthetic demos in our Isaac task with
  Isaac Lab Mimic (see the imitation-learning section of the [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/))
  from the real demos replayed in sim (no ready-made example from us), or roll out your own scripted
  policy with domain randomisation on.
- Return action chunks (k, 14) to meet the 27 Hz command rate with a slow model.
- Main risk: the sim-to-real visual gap. Use DR and Drop 1 (venue images).

### Public data for pretraining

No public G1 colour + shape sorting dataset exists (checked 2026-11-07); our demos are the only in-domain data.
Related G1 data on Hugging Face (check each card for its action space and rate before mixing it with ours):

- Unitree's LeRobot datasets, incl. G1 Dex3 ones: `unitreerobotics/G1_Stack_Block`, `G1_Dex3_PickGum_Dataset`
  (browse <https://huggingface.co/unitreerobotics>)
- Community: `datamentorshf/G1_Dex3_BlockStacking_Dataset` (30 Hz, colour-ordered stacking)

## Route 4: RL

Train one skill (e.g. grasp + lift) in Isaac Lab with privileged state, then distil it into a vision policy, or
use it as a residual on top of BC or a classical planner.

- Our task builds on [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) (Isaac Lab,
  G1 29-DoF + Dex3, same DDS topics as the robot). Privileged object poses come from the scene service
  (`aaltonomous/sim/scene_client.py`) — fine for training, **not available in a submission**.
- The scorer `aaltonomous/scene/scene.py` gives you the success signal; shape the reward yourself.
- Main risk: the contact-physics gap (Dex3 friction and PD gains are not yet fitted to the real robot).
  Respect the 1 rad/s velocity cap in the policy itself, not just in the filter.

## Route 5: real2sim / data

Make the sim match reality and every other route gets better: re-texture from Drop 1 captures, Gaussian splats of
the counter, Dex3 friction / PD fitting from the real demos, better DR ranges.

## Difficulty knobs

Current values are what the code does today. Finals values are not announced.

| Knob | Easier | Harder | Now (code) |
|---|---|---|---|
| Objects per episode | 3 | 5 | 3–5 |
| Instruction | colour only | colour + shape, negation, held-out paraphrases | colour only, 8 training templates; held-out templates exist |
| Layout | spread out, yaw ±15° | min spacing, yaw ±45°, distractor object | yaw ±45°, ≥ 8 cm spacing, no distractors |
| Time limit | 180 s | 90 s | 120 s |
| Hole clearance (rung 3) | +15 mm | +3 mm | +10 mm |
| Shapes with holes (rung 3) | circle only | all three | all three |
