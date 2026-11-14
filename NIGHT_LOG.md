# Night log, Aaltonomous Humanoid (Halal Team)

Newest status first; experiments below, grouped by idea. All sim numbers are with training domain randomisation
(`AALTO_DR=train`) on seeds 1000–1007 unless stated. Proxy = `fixed + known + 1.5 × unseen` (mean episode score per leg).
Per-workstream detail: `halal/logs/*.md`. How things are organised: `halal/README.md`.

## Status

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

- Open: the counter height is anchored to the scene file (exact in sim, unknown on the robot); a reference from
  the robot's own hands is being built. Sim effect on sorting: to be measured against the organiser detector.

### 3. Sim throughput, 02:50
- **What:** start every sim with `OMP_NUM_THREADS=1`.
- **Why:** `perf` showed 81 % of a sim's CPU in OpenMP spin-waits: torch on CPU opens a 30-thread pool for
  single-environment tensors.
- **Effect:** real-time factor of a third sim under load 0.29 → 0.44, CPU per sim 5.5 → 2.4 cores; an oracle
  episode on the tuned sim behaves normally (3/3, gate pass). All three sims restarted with it by 03:10 and
  running jobs: 0.40 + 0.36 + 0.49 = 1.25 sim-seconds per second, against 0.54 with two untuned sims.
- **Failed:** dropping the wrist cameras from the sim crashes the task.

### 4. Motion planning, 03:00
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
