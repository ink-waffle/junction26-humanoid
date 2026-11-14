# halal: Halal Team's policy for the Aaltonomous Humanoid challenge

Working notes for everyone (humans and agents) who touches this package. Organiser docs stay the source of truth:
`aaltonomous/docs/participants/` (README, RULES_AND_SCORING, INTERFACE, SUBMISSION, STARTER_PATHS, FAQ),
`aaltonomous/sim/README.md`, `aaltonomous/baselines/classical/README.md`, our Russian summary `OVERVIEW.md`.
Instance access (private, not in git): `team/SETUP.md` in the main checkout.

## What we optimise

Finals proxy = `score(sort_fixed) + score(sort_language, training template) + 1.5 × score(sort_language, unseen paraphrase)`,
mean episode score per leg, same seeds for every version, sim started with `AALTO_DR=train`. A version only counts
if the sim gate passes on every episode: 0 clipped steps, command rate ≥ 27 Hz, `policy_hz_rt` ≥ 27 Hz, no
abort/timeout/crash. Per object: correct bin +1 (`sort_fixed`) / +2 (`sort_language`), wrong bin −1, knocked off the
counter −1, left on the counter 0. So: never put an object in a bin we are not sure about, never sweep objects off.

The real robot decides the final ranking, so nothing may depend on sim-only facts: no privileged state in the
policy, wrist camera poses are uncalibrated, head calibration is ±1 cm / ±1°, lighting and depth noise differ,
Dex3 friction is not fitted. Prefer mechanisms that calibrate themselves from what the robot can observe.

## Facts established so far (14.11 night)

Interface and timing
- One `act()` round trip costs ~80 ms wall (4 images over msgpack), ~160 ms when the instance CPU is saturated.
  `policy_hz_rt = executed policy ticks / total wall time spent answering`, so return chunks of ~10 ticks; many
  1–3 tick replies fail the gate. Holds we return ourselves count as commands; runner holds do not.
- Slow work (planning, perception) runs in a background thread while `act()` returns hold chunks.
- The safety filter caps the change of the command: 1 rad/s arms, 3 rad/s hands, per tick, plus joint limits and
  the workspace / counter keep-out boxes for wrist and palm frames. Plans are dry-run through `SafetyFilter`
  before execution (baseline `Planner.dry_run_safety`); a plan with a clip is never sent.
- REAL hand joints are position controlled with kp 1.5, kd 0.2 (fixed by the runner): grip torque is
  `1.5 × (target − measured)`, so holding needs a closure commanded 0.15–0.4 rad past contact (what the teleop
  operator does). The SIM ignores those gains: its hand actuators are implicit PD with stiffness 100, damping 10
  (`third_party/unitree_sim_isaaclab/robots/unitree.py`), ~67× stiffer. In sim a 0.05 rad error is already ~50 N,
  friction and mass barely matter for holding and failures are geometric (spin, squirt, wedge). A grasp must be
  judged for both regimes.
- In sim the right hand's DDS slots 3,4 drive the MIDDLE finger and 5,6 the INDEX finger, the opposite of
  `INTERFACE.md` and the real robot. Keep index and middle commands identical, and never read contact from one
  finger only.
- The real fingers lag: 90 % of a step after 300–380 ms (thumb_0 ~450 ms); free fingers carry a 0.02–0.04 rad
  bias. Contact test that works on real data: `Σ_joints max(0, s·(e − e0)) > 0.3`, read ≥ 0.4 s after the hand
  command settled, with e0 from closing the empty hand once per episode.

Sim
- `evaluate.py` homes the arms to `ARM_HOME`, calls `randomize(seed)`, resets the layout, runs the episode on the
  sim clock, lets objects settle, scores with the scene service. Seeds 1000+ are our fixed eval seeds.
- Real-time factor on our instance: ~0.45 with one sim, ~0.27 each with two (CPU bound, 30 vCPU). A 120 s
  episode is 4–7 min of wall time, so sim time is the scarce resource: do cheap loops offline first.
- `evaluate.py --container X` without `--policy ws://...` silently scores the oracle. Always pass `--policy`.
- Dropping the wrist cameras from the sim (`--camera_include front_camera`) crashes the task. Do not try again.
- Sim start is ~15 s when warm. Black frames after a start: stop and start again.

Scene and view (head camera, 640×480, pitched 48° down)
- Objects lie in pelvis x 0.26–0.33, y ±0.17; the near ones are at the very bottom edge of the head image and can be
  cut off. At `ARM_HOME` both hands are inside the head image and can hide objects near y ±0.17.
- The counter height relative to the pelvis varied by 1–5 cm between real recording sessions: the planner takes it
  from the detection (`det.top[2]` minus the object height, `motion.counter_from_detection`), so perception's top
  z must be good to ~5 mm.
- Bins: near edge x 0.40, centres (0.505, +0.26 / 0 / −0.26), 6 cm walls. The baseline releases 5 cm inside the near wall.
- The mat has a printed 5 cm grid (`scene/assets/mat_texture.png`, same print on the real counter).
- The sim head camera does not render with `calibration.yaml`'s principal point (325.0, 250.9): Omniverse ignores
  aperture offsets, so the rendered image has it at the centre (319.5, 239.5), f = 604.8. Deprojecting with the
  yaml puts objects ~6 mm too high and 4–5 mm off in xy even without DR (like 1.0° pitch + 0.5° yaw of camera
  error). On the real robot the yaml is presumably right, so we do not hardcode the sim value: per-episode
  self-calibration has to absorb it.
- Organiser detector on 120 DR frames (472 objects): recall 0.99, colour 1.00, but precision 0.69 (phantom "yellow"
  objects on washed-out bin rims) and top-centre xy error median 11.6 mm, p95 24 mm, almost all of it camera
  model error. A third of the objects are partly hidden by the hands at `ARM_HOME` (thumbs over |y| > 0.1).

Hand (left palm frame: x = finger direction, fingers curl towards −y, thumb in the z = 0 plane)
- Index and middle fingertips sit at z = ±29 mm (57 mm apart); the thumb opposes them along x at z = 0.
- A centred 45 mm object slips between the two fingers and wedges, so the baseline shifts the grasp point 15 mm
  in z: thumb against ONE finger. That pinch has a 29 mm lever arm and spins triangles (the baseline's main miss).
- `thumb_0` swings the thumb sideways (about palm y); every baseline posture leaves it at 0.

Contact signal (sim, from runner logs of baseline and oracle episodes)
- Object held: steady `|hand_q − hand_q_target|` of 0.03–0.08 rad on thumb_1 and the pinching finger's joint 0,
  constant through the carry. Hand empty: 0.00 on every joint. No noise in sim; the real robot will have some.

Baseline and oracle (sort_fixed)
- Classical baseline, nominal sim: 6/8 objects (2 seeds), 76–82 % in the organisers' runs; ~10 s per object;
  misses are triangles and the 120 s limit. It ignores the instruction, so it scores about −1 per object whenever
  the `sort_language` mapping differs from the fixed one.
- Oracle (privileged poses, face-aligned grasps, joint-space carries, planning ahead): 12/12 nominal on 3 seeds
  without triangles; the organisers quote ~84 % overall. Its planner tricks are in `aaltonomous/sim/oracle.py`
  (`_OraclePlanner`): reading them is encouraged, importing privileged perception into the policy is forbidden.
- The left arm can pick objects as far right as y = −0.17 and still reach the left bin (and vice versa), with tilt.

Organiser statements (challenge channel, Sat night)
- Finals instructions: colour comes first; shape-based instructions may be tested if several teams handle colour;
  no negation.
- Real runner hand gains are kp 1.5 / kd 0.2, fixed for every team. Real pad friction is unmeasured and expected to
  be lower than the sim's 1.2 / 1.0: prefer grasps that hold by geometry, not by friction alone.
- Under training DR the organiser baseline sorts ~50 % of objects on our seeds (nominal: ~75 %).

Real data
- `zupermiez/g1_pick_cube` on Hugging Face (on the instance: `/data/drops/g1_pick_cube`): 20 teleop episodes of
  "pick up cube", 30 fps, three RGB cameras, 28-dim state and action in our joint order, no depth, no labels.
  Start poses vary, some action jumps exceed the safety limits. Use it for appearance, postures and finger signals.

## Package layout and ownership

| Path | What | Owner (workstream) |
|---|---|---|
| `halal/language/` | instruction → mapping (rule parser; adapters behind one interface) | language |
| `halal/eval/paraphrases.{yaml,py}` | held-out instruction benchmark (dev / test split) | language |
| `halal/eval/sim_eval.py`, `proxy.py`, `collect_frames.py` | sim evaluation harness, proxy score, frame collection | orchestrator |
| `halal/perception/` | obs → detected objects (colour, shape, yaw, position), calibration refinement | perception |
| `halal/motion/` | grasp strategies, pick-place planning, trajectories | grasp + policy |
| `halal/policy/`, `halal/server.py`, `halal/docker/` | state machine, policy server, submission image | policy |
| `halal/ops/` | instance tooling (`sims.sh`, `env.sh`, `push.sh`) | orchestrator |

Organiser code under `aaltonomous/` is never edited; import it or subclass it.

## Working on it

Local (Mac, no sim): `.venv` in the main checkout (Python 3.12, numpy 2.3, scipy 1.17, pinocchio 3.9, OpenCV 5: the
same versions as the instance's `isaac` env). From any checkout:
`PYTHONPATH=$PWD /Users/ink_waffle/OverengineeringLife/Junction2026/humanoid/.venv/bin/python -m ...`.
The URDF lives in the main checkout's `third_party/` (gitignored): symlink it into a worktree.
Mock episodes: `python -m aaltonomous.baselines.classical.run --backend mock --seed 3` (kinematic world with a
rendered head RGB-D; no physics) and `python -m aaltonomous.deploy.runner --backend mock --policy ws://...`.

Instance (one RTX PRO 6000, `ssh` line in `team/SETUP.md`):
- `bash halal/ops/push.sh` rsyncs the current checkout: `main` → `/data/halal`, branch `ws/<name>` → `/data/wt/<name>`.
- On the instance: `source <tree>/halal/ops/env.sh <sim tag>` activates the `isaac` env, puts that tree on
  `PYTHONPATH` and points evaluator clients at sim `a`, `b` or `c`.
- Sims: `bash /data/halal/halal/ops/sims.sh status | start <tag> | stop <tag>` (DR `train` by default). Each
  workstream uses only the sim it was assigned; long jobs run in `tmux` (session `aalto`, one window per job).
- Proxy score: `python -m halal.eval.proxy --name <version> --server "python -m halal.server" --trials 8 --sims a --port 81xx`
  (three legs, results under `runs/proxy/`). Single leg: `--legs fixed`. One leg of 8 seeds is 15–25 min.
- Frames with ground truth for perception work: `python -m halal.eval.collect_frames --name <set> --seeds 2000 2120`
  → `runs/frames/<set>/seed_*.npz` (`/data/halal/runs/frames/dr_train` holds the first 120).

## Rules of the house

- Measure before and after every idea on the same seeds and write the numbers down, failures included.
- Code style: type hints, small single-purpose functions, comments only where the code cannot explain itself and
  never longer than one line, docstrings exactly one line, no dead code. Sources of data with several possible
  providers go behind an adapter interface (see `halal/language`). Tests live outside the repo unless they are
  evals of something non-deterministic.
- Everything tunable goes into config, not into literals spread over the code.
- The repo is public: no secrets, no instance addresses, no Discord excerpts in tracked files.
