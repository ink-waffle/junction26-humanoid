# Night log, Aaltonomous Humanoid (Halal Team)

Newest status first; experiments below, grouped by idea. All sim numbers are with training domain randomisation
(`AALTO_DR=train`) on seeds 1000–1007 unless stated. Proxy = `fixed + known + 1.5 × unseen` (mean episode score per leg).
Per-workstream detail: `halal/logs/*.md`. How things are organised: `halal/README.md`.

## Status

### Sat 14.11, 10:30
- **Best measured version: v2, proxy +19.50, sim gate PASS on all 24 episodes** (0 clips, 30 Hz, policy
  real-time rate ≥ 167 Hz). Fixed +3.38 (27/31), known template +6.75 (27/28), unseen paraphrase +6.25 (25/28).
  The organiser baseline scores +1.62 on the fixed leg and cannot follow instructions.
- No object went to a wrong bin and none was knocked off in any episode; all 8 lost objects stayed on the counter
  after failed grasps (cylinders and triangles with the organiser pinch).
- **Hand-in images on the instance:** `halal/policy:v2` (the version of this table, 1.48 GB, starts in 0.5 s,
  passes the synthetic pre-check) and `halal/policy:v3` (v2 + cage grasps + perception fix), which is running the
  three legs through the container right now.
- **Hand-in path proven (10:35):** `ghcr.io/ink-waffle/halal-policy@sha256:ee26f64beef67349d06cd046a9aee0602ca9360bfd79bf6d7b0194f50fb9ee94`
  is v2, public, and pulls anonymously by digest. Later versions: `halal/ops/publish.sh <version>`.
- Organisers (Discord 10:06): still no deadlines, no hand-in form, no new data drop.

### Sat 14.11, 10:00
- **Outage:** the Mac lost its network from about 04:15 to 10:00; every agent stopped and the instance sat idle.
  Nothing committed was lost; agents are resumed.
- **Best version so far (fixed task only, 8 seeds, DR, gate PASS on every episode):** our state machine with the
  `topface` detector: 27/31 objects (87 %), score +3.38 ± 0.70, against the organiser baseline's 13/31 (+1.62).
  Language legs and the full proxy are running now on all three sims; no current hand-in image yet (the `v0`
  image predates the detector).
- **Now:** full three-leg proxy + image of that version (policy); grasp per shape for cylinders and triangles
  (motion); hand-fit false-acceptance fix and look sequence (perception); fresh blind instruction set; Discord.
- **Credits:** ~10 h of instance time used, about €173 of €200 left. Verda console state unknown since 02:30.

### Sat 14.11, 02:40
- **Best version:** none of ours yet. Floor = organiser baseline (rung 1 only), numbers below.
- **Now:** five workstreams in parallel, each in its own git worktree (`ws/*` branches) and on its own sim:
  policy core (state machine, verification, retries, hand-in image), motion (grasp research per shape, faster
  planner), perception (benchmark on 120 DR frames with ground truth, camera self-calibration, shape + yaw),
  real-data analysis (organisers' first drop), language (rule parser + held-out paraphrase benchmark).
- **Instance:** two sims with DR at real-time factor ~0.27 each (one alone: 0.45). Credits: ~€199 at launch
  (Fri 22:10 UTC), ~€2.6/h, so ~€187 left; the Verda console is down (their domain), the instance is unaffected.
- **Organisers (Discord, read 02:30):** no hand-in form, no deadlines, no sim image yet. First real data drop is
  out (20 teleop episodes "pick up cube", RGB only). Finals instructions: colour first, shape-based ones possible,
  no negation. Real hand pads grip less than the sim's.

## Floor: organiser baseline and oracle

| Policy | Sim | Seeds | Objects | Score | Gate |
|---|---|---|---|---|---|
| classical baseline (image) | nominal | 1000–1001 | 6/8 | +3.00 | pass |
| oracle (privileged poses) | nominal | 1000–1002 | 12/12 | +4.00 | pass |
| **classical baseline (server)** | **DR train** | **1000–1007** | **13/31 (42 %)** | **+1.62 ± 0.86** | **pass** |

Per seed under DR: 2/3, 2/5, 2/4, 3/3, 0/5, 2/4, 1/4, 1/3 (the first two without DR: 3/3, 3/5). 100–200 s of sim
time per episode. It never reads the instruction, so on `sort_language` it scores about −1 per object whenever the
mapping differs from the fixed one.

## Sim results of our policy (fixed task, seeds 1000–1007, DR train)

| Version | Objects | Score | Ends | Gate |
|---|---|---|---|---|
| organiser baseline (floor) | 13/31 (42 %) | +1.62 ± 0.86 | 8 done | pass |
| v0: our state machine, organiser detector | 12/31 (39 %) | +1.50 ± 1.23 | 5 done, 3 time limit | pass |
| v0 + `topface` detector | 27/31 (87 %) | +3.38 ± 0.70 | 8 done | pass |
| v1: + grasp verification | 27/31 (87 %) | +3.38 ± 0.99 | 8 done | pass |
| v1 + look pose | 24/31 (77 %) | +3.00 ± 1.12 | 8 done | pass |

Full proxy (three legs, 8 seeds each, DR train):

| Version | Fixed | Known template | Unseen paraphrase | Proxy | Gate |
|---|---|---|---|---|---|
| v2: topface, verification, retries, organiser pinch | +3.38 (27/31) | +6.75 (27/28) | +6.25 (25/28) | **+19.50** | pass |

v2 per pick (232 picks): cubes held 92 %, cylinders 58 %, triangles 26 %; first attempt 64 %, second 38 %, third
27 %: a pinch that failed once mostly fails again and pushes the object towards the robot. Instructions: 16/16
parsed to the true mapping. v2 ran with a perception bug (hand-based camera fit overriding the scene anchor,
+11–13 mm in x) on two of the three legs.

The detector is the jump: same policy, 39 % → 87 %. Verification is neutral on score here; the look-pose variant is
3 objects lower, not yet explained (the sim varies by 1–2 objects run to run).

## Ideas

### 1. Read the instruction (rule-based parser), 02:55
- **What:** `halal.language`: tagger + chunker + pairing search over a closed vocabulary, deterministic, 0.4 ms,
  returns a mapping (colour, shape or colour/shape keys, "everything else") with a confidence.
- **Why:** rung 2 pays +2 per object and the finals weigh the unseen instruction ×1.5; the baseline ignores the text.
- **Measured on** a held-out paraphrase benchmark written blind by a separate agent (181 templates × colour/bin
  permutations, dev/test split; the parser author only ever sees dev failures):

| Split | Colour-only paraphrases | All categories | Wrong with confidence ≥ 0.5 |
|---|---|---|---|
| dev (2781 cases), v1 → v2 | 99.1 → 100 % exact | 95.0 → 98.7 % | 3.0 → 1.3 % |
| test (2853 cases), v1 → v2 | 100 → 100 % | 93.5 → 93.5 % | 4.0 → 1.5 % |

- Final (v6, 10:25) on text it had never seen: a fresh blind set by a second author (136 templates, 4087 cases)
  97.9 % exact, 2.1 % wrong with confidence; 60 instructions in a non-native researcher's voice 55/60 with no
  object sent to a wrong bin. After fixing the mechanisms behind those misses: 100 % on all three sets. Every
  round found a new family (ordered lists anchored to one side; "right" meaning "correct"; "round" as in "this
  round"), so the next unseen wording may too.
- v2 (03:10): a stated counting side governs ordinals, rules are resolved over the nine object types by
  specificity, explicit pairs beat enumerations. Remaining test misses: ad-hoc bin labels ("call the bins A, B,
  C"), a list ordered by direction of travel, self-corrections, one colour+shape form, elimination puzzles.
  Still being improved. Sim effect: measured with the first proxy run of our policy.

### 2. Perception that survives randomisation (`topface` detector), 03:05
- **What:** depth-only self-calibration of the head camera per episode (counter plane → roll, pitch, height; front
  wall of the bin row → yaw, x, y), objects found as blobs at object-top height, the three known outlines fitted
  to a metric top view (gives centre, shape, yaw; hidden cells cost evidence, not accuracy), colour from the most
  saturated pixels.
- **Why:** the organiser detector was benchmarked first on 120 DR frames with ground truth: 31 % phantom
  detections (washed-out bin rims read as yellow) and 11.6 mm median position error, almost all camera-model
  error. The sim camera also renders with its principal point at the image centre, not at the calibration
  file's value (worth 6 mm on its own), and a third of the objects are partly hidden by the hands at home.

| 120 DR frames, 472 objects | organiser detector | `topface` (30 held-out frames) |
|---|---|---|
| recall / precision | 0.989 / 0.692 | 1.000 / 1.000 |
| top-centre xy error, median / max | 11.6 / 29.9 mm | 1.6 / 2.7 mm |
| top z error, median / p95 | 5.4 / 15.2 mm | 0.0 / 0.2 mm |
| shape decided / correct when decided | not produced | 92 % / 100 % |
| yaw error, median / p95 | not produced | 0.5° / 1.6° |
| runtime | 21 ms | 40 ms |

- Sim effect on sorting: being measured against the organiser detector (policy A/B legs).

### 2b. Camera pose from the robot's own arms, perceived scene layout, 03:30
- **What:** depth pixels on both forearms and hands are registered against the surface posed by forward kinematics
  of the measured joints (6-DOF, 28 ms, once per episode): that ties the camera to the frame the arm moves in.
  The scene-anchored pose (plane + bin row) is used only when the hands confirm it within 10 mm; otherwise the
  hand pose decides and the bins and counter plane are reported where they are seen (`Detector.layout()`).
- **Why:** on the real recordings the bins and table sit centimetres and degrees away from the scene file
  relative to the robot, and the nominal head calibration is 30–90 px off; anchoring the camera to the bins there
  would move every object by centimetres. Only the arm is rigidly tied to the arm frame.

| 120 DR frames | xy error median / p95 / max | z error median / p95 / max |
|---|---|---|
| nominal calibration | 11.1 / 22.4 / 29.1 mm | 6.2 / 17.9 / 22.2 mm |
| hands only | 2.2 / 2.9 / 3.8 mm | 3.7 / 4.2 / 4.4 mm |
| scene anchor (plane + bins), also the fusion in sim | 1.7 / 2.5 / 2.7 mm | 0.0 / 0.3 / 0.3 mm |

- Bins found by depth geometry alone (anchor off): centres within 4.9 mm median, 7.5 mm max.
- Caveat: sim has exact kinematics, exact meshes and clean depth. Real wrists carry a camera mount and cables,
  and glossy black fingers are hard for stereo depth; unmeasured until a drop with depth arrives.
- Colour is now three hue basins; hues between them (orange) are not returned.

### 3. Sim throughput, 02:50
- **What:** start every sim with `OMP_NUM_THREADS=1`.
- **Why:** `perf` showed 81 % of a sim's CPU in OpenMP spin-waits: torch on CPU opens a 30-thread pool for
  single-environment tensors.
- **Effect:** real-time factor of a third sim under load 0.29 → 0.44, CPU per sim 5.5 → 2.4 cores; an oracle
  episode on the tuned sim behaves normally (3/3, gate pass). All three sims restarted with it by 03:10 and
  running jobs: 0.40 + 0.36 + 0.49 = 1.25 sim-seconds per second, against 0.54 with two untuned sims.
- **Final settings (03:25, now the `sims.sh` defaults):** plus four pinned cores per sim, Kit task pool 4, PhysX 2
  threads: 0.58 idle and ~0.45 under an evaluator on the test sim (like for like: 0.44 → 0.585 idle, 0.37 → 0.445
  with a client). Pinning alone changed nothing. Not yet measured with all three sims on the final settings.
- **Found on the way:** the sim is not deterministic run to run (same seed, same policy, a grasp fails in some
  runs); the DR camera corruption in organiser code costs 5 ms per step once a client has called `randomize`.
- **Failed:** dropping the wrist cameras from the sim crashes the task.

### 4. Motion planning, 03:00
- **Grasp per shape (10:15):** one centred three-finger grasp ("cage": object between index and middle, thumb
  behind) with a thumb depth per shape replaces the organiser pinch. Single object, hard DR corner, exact pose:

| | cube | cylinder | triangle |
|---|---|---|---|
| organiser pinch | 12/12 | 8/12 | 1/12 |
| cage, per-shape thumb depth | 30/30 | 18/18 | 12/12 |

  Why the pinch fails: the fingertips are 40 mm apart, a cylinder nests 10 mm and a triangle 34 mm between
  them, and the organiser thumb stops short (finger error 0.001 rad: never pinched, not spun out). Unknown shape
  (one depth for all): 17/18. Six trials per cell; a 24-per-shape confirmation with position error is running.
- **What:** joint-space carries first, Cartesian segments shortcut to one joint move when the grasp point stays
  on the line, cached reach check, bin margin 0; options on `plan_pick` for a release offset inside the bin,
  ending above the bin instead of at home, and placing on the counter; counter height from the detection.
- **Effect (offline, 30 layouts, 393 object–bin pairs):** 100 % plannable by at least one arm; planning time mean
  0.19 s, max 2.1 s (cross-body failures used to take 10–30 s). Pick duration unchanged so far (mean 9.1 s).
- **Why the release offset:** in the floor run a second object for the same bin landed on the first, above the
  rim, and scored 0.
- **Grasp sweep, first numbers** (single object, hard DR corner, exact pose, organiser pinch): cube 12/12,
  cylinder 9/12. Triangles and face-aligned grasps running.

### 5. What the real robot data says (organisers' first drop), 02:55
- 20 teleop episodes, left hand only, foam objects on bare wood: not the final scene, but the hands are real.
- The finger contact signal exists but free fingers carry a 0.02–0.04 rad bias and lag 300–380 ms; it needs a
  closure commanded ≥ 0.2 rad past contact and a per-episode empty-hand reference. A sum over all finger joints
  relative to that reference separates all 12 labelled holds from empty hands (threshold 0.3).
- The operator grasps differently from the baseline: thumb stretched forward and swung sideways to oppose one
  finger directly, fingers hooked around the object, palm turned across the body.
- The sim's finger actuators are ~67× stiffer than the real ones (found in the sim code): holding in sim proves
  geometry, not grip force. Plan: identify the actuator during the empty-hand close and choose the squeeze from it.
- The counter height relative to the robot varied by 1–5 cm between sessions.
