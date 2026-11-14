# Policy workstream log

State machine, policy server and submission image (`halal/policy/`, `halal/server.py`, `halal/docker/`).
All sim numbers: sim `a`, `AALTO_DR=train`, seeds 1000–1007, 120 s limit, unless stated. Newest entries at the bottom
of each section.

## Floor

| Policy | Leg | Score | Objects | Outcomes | Gate |
|---|---|---|---|---|---|
| organiser baseline (server) | fixed | +1.62 ± 0.86 | 13/31 | 13 correct, 18 counter | pass (30.0 Hz, rt 60–83 Hz) |

Where the floor loses objects (from its runner logs, `runs/eval/baseline_sort_fixed_train_20261114_002421`):
- Finger signal while the hand command is at `closed`: 23 of 41 grasps over six episodes read 0.000 rad one second
  after closing (nothing in the hand) and were still carried to the bin and back: ~6 s lost each. Seed 1004: 9
  grasps, all empty, 0/5; seed 1005: 7 of 10 empty.
- A held object reads 0.041–0.18 rad (sum over the hand's joints of the shortfall behind the closed posture),
  constant through the carry; 0.25 s after closing an empty hand still reads 0.05–0.11 (lag), at 0.5 s 0.002–0.016
  (0.06–0.07 when the object is slipping out), at 1.0 s exactly 0.000.
- Seed 1002: the second blue object came to rest on the first one in the right bin at z = 0.080 (rim 0.070) and
  scored 0: both were released at the same point.
- Failed grasps shove objects up to 15 cm; the organiser detector then sees them elsewhere and the attempt counter
  (by position) starts again.

## Results, fixed leg (8 seeds, sim `a`, night of 14.11; all gate PASS, 0 clips, command rate 30.0 Hz)

| Run | Policy | Detector | Score | Objects | Picks / read empty | Mean episode | `policy_hz_rt` min |
|---|---|---|---|---|---|---|---|
| floor | organiser baseline | organiser | +1.62 ± 0.86 | 13/31 | 41+ / – | – | 60 |
| A | v0 | organiser | +1.50 ± 1.23 | 12/31 | 73 / – | 84 s | 128 |
| B | v0 | topface | +3.38 ± 0.70 | 27/31 | 60 / – | 54 s | 126 |
| C | v1 = v0 + grasp verification | topface | +3.38 ± 0.99 | 27/31 | 58 / 15 | 48 s | 117 |
| D | v1, looks from `motion.look_arm_q` | topface | +3.00 ± 1.12 | 24/31 | 63 / 15 | 54 s | 124 |

- A vs floor: our state machine with the organisers' detector is no better than their policy (12 vs 13): with a
  detector that is 1–3 cm off under DR the grasps miss whoever plans them (seeds 1001, 1004, 1005: 2 of 14 objects,
  three time limits). The target filter threw out 118 phantom detections in the 8 episodes; none was picked.
- B vs A, the detector: +15 objects, 39 % → 87 %; every episode ends with `done`, in 54 s on average.
- C vs B, verification: the same 27/31. It read 15 of 58 picks as empty after the lift and returned at once, which
  shortened the episodes by 11 %, but with the cap of 2 picks per spot the freed time was not used: all 4 lost
  objects (2 cylinders, 1 cube, 1 triangle) were left alone after two empty picks with 50–75 s still on the clock.
  Held objects read 0.043–0.154 at the first check, empty hands 0.000–0.002; no pick that read "held" was lost and
  no object that read "empty" was found in a bin: the signal made no wrong call in 58 picks.
- D vs C, look pose: 3 objects fewer, and in 3 of 8 episodes the first three picks were all empty (the check then
  distrusts itself and switches off). Plans that start from `look_arm_q` grasp worse than plans from the sim home
  pose; not adopted. With topface the hands at `ARM_HOME` cost no accuracy (B: first-look error within ±5 mm of
  the layout card), so there is nothing for a look pose to win in sim.
- Stacking: 16 bins received two objects in B–D; one object came to rest above the rim (B). Slots 7 cm apart help
  but do not exclude it.
- Rate gate: in a sim that is slower than real time the measured latency of every reply is one tick of wall time
  (the runner looks at the reply once per tick), so `policy_hz_rt` = ticks per reply × rtf × 30: 10-tick replies at
  rtf 0.44 give 128. It would fail at rtf < 0.09 with 10-tick replies; replies are 15 ticks (10 while planning)
  from v2 on. The true round trip is ~5 ms (replay through the container: p50 5.0 ms).

## Versions (three legs, 8 seeds each, DR train, 120 s; images on the instance as `halal/policy:<version>`)

| Version | Image id | Built from | Fixed | Known | Unseen | Proxy | Gate | Notes |
|---|---|---|---|---|---|---|---|---|
| floor | – | organiser baseline | +1.62 ± 0.86, 13/31 | – | – | – | PASS | ignores the instruction |
| v0 | 62ae624ee6ea | 3085846 | – | – | – | – | pre-check only | organiser detector; never run through the image |
| v1 | d35c253d992a | d27fbad | +3.38 ± 0.99, 27/31 | – | – | – | PASS | run C of the night; local server |
| v2 | ee26f64beef6 | 78b1e61 | +3.38 ± 0.70, 27/31 | +6.75 ± 1.71, 27/28 | +6.25 ± 1.85, 25/28 | **+19.50** | PASS (0 clips, 30.0 Hz, rt ≥ 167) | local server on sims a / b / c; pinch grasps; camera fixed by the hands on a and b (+1 cm in x) |
| v3 | c6a50633cd73 | e7934c6 | +3.62 ± 0.48, 29/31 | – | – | – | PASS (through the image) | cage grasps; camera calibrated on a pre-reset frame; language legs skipped in favour of v4 |
| v4 | 994a14f7b199 | e3f9687 | +3.62 ± 0.70, 29/31 | +6.38 ± 1.87, 26/28 (1 wrong bin) | running | – | PASS (through the image; exact hand-in path `evaluate --container`: 8/8 in 2 episodes, rt 176) | start settle, visibility rule, clear looks before `done`; 3 of its 4 lost objects stayed wedged in the hand at the release |
| v5 | 5a4d3187fcf0 | fe03f4c | – | – | – | – | pre-check only | superseded by v6 before it ran |
| v6 | ec29a3c0ad75 | 58573f1 | running | running | queued | – | pre-check pass | v4 + perception 47e63f3 (anchor refit at every look, two still looks at the start, colour confidence 0.5, one pick for an unclaimed blob) + release check + motion 44bc3a4 (soft squeeze = grasp depth + 0.45, grip score on thumb_1 and finger_0) + retries keep the shape's depth |

v2 in detail:
- Language: 16/16 instructions parsed to the true mapping (15 at confidence 1.00, one at 0.95); no object in a
  wrong bin and none knocked off in any leg. Every lost object stayed on the counter after failed grasps.
- By shape over the three legs: cubes 29/29, cylinders 32/35, triangles 18/23.
- Per pick (155 picks with a grip verdict): cubes 90 % held, cylinders 64 %, triangles 27 %; first attempt 70 %,
  second 28 %, third 23 %: a pinch that failed once mostly fails again, and each failure shoves the object
  towards the robot (held 78 % at x ≥ 0.31, 27 % at x < 0.28; seed 1002: a red cylinder pushed from x 0.274 to
  0.208 in four tries, then out of the camera's view). Cylinders picked across the body (object on the other side
  of the arm): 4/15 held, on the arm's own side 16/17.
- Retries (v2 vs v1/C on the fixed leg): 27/31 both. More attempts alone do not pay with this grasp.
- The verification called 76 of 155 picks empty; no "held" pick was lost on the way and no "empty" one was found
  in a bin.

## What the policy is (v4)

`SortPolicy` (own state machine). Episode start: hold for the settle time (the first frame still shows the counter
before the reset), look from where the arms are with the hands untouched (the camera fit wants them in view), and
close both hands on nothing while that frame is processed; the first plan waits for the hand class. Then per look:
targets = detections that the instruction maps to a bin, on the counter and clear of the perceived bins, colour
confidence ≥ 0.3, picks left, top face seen ≥ 50 % (each object keeps the shape of the clearest look at it) →
least-tried target first, then nearest its bin; arm on the bin's side first; a retry first forgets the detected
shape, the one after turns to the next arm and grasp; next free release slot → execute in 15-tick replies with the
grip check after the lift and through the carry → home → look again. When only poorly seen targets are left, and
before `done`, the arms swing to `motion.look_arm_q` and the look is repeated from there; `done` needs two such
clear looks 2 s apart without a sortable object. Perception and planning run in one low-priority background
thread; `act()` never waits (worst measured 5 ms) and answers 10-tick holds meanwhile.
`set_layout(detector.layout())` after every look; if the perceived layout leaves no pick (or no safe move) the
scene file's layout is used for the rest of the episode. A look that fails or rejects every target pauses 1 s and
looks again; an exception in a step drops what was in progress and starts over from a look. A pick starts only if
its release fits before the time limit; if the episode outlasts the assumed limit the policy carries on.
Instruction confidence below 0.5: nothing is moved. Hand class from the calibration close (steady bias below
0.01 rad = stiff, the sim; else soft, the real hand) selects the grip threshold (0.02 / 0.3) and the squeeze passed
to `plan_pick` (none / 1.0).

v3 = the same without the start settle, the visibility rule, the clear-view looks and the shape memory (it looks
from home only and needs two empty looks from there for `done`).

Failure injection (scratch scripts outside the repo, kinematic mock world with a finger contact signal, runner-like
observation lag): blank head frames for the first 40 calls, first frames showing the empty counter, no depth at
all, detector raising on two looks, planner raising once / always, reset during a 1 s plan followed directly by
the next episode, assumed limit 20 s with a real limit of 60 s, start pose far from home with closed hands, an
object lost during the carry, a mock without any finger signal, a perceived layout 10 cm off, every hand classed
soft: no exception reaches the runner, 0 clipped steps, no early `done`.
Found by it and fixed: a calibration close from an already closed hand divided by zero and froze the state
machine (now: open first, and any step failure starts over); a missing depth frame ended the episode with `done`
after 3 s; an observation two ticks newer than assumed read the reopening hand as a steady bias; the detector's
hand-based camera fit accepted frames without any arm in them (reported, fixed in perception).

## Ideas, one at a time

### Geometric target filter (v0)
Why: the organiser detector reports washed-out bin rims and things behind the bins as yellow objects. Rule: a
target lies on the counter top (2 cm inside its edge) and not in a bin or within 3 cm of a bin footprint (bins as
perceived this episode); reach is left to the planner. Run A: 118 detections rejected in 8 episodes, none picked.
With topface: 0 rejected in B–D. Kept as a guard.

### Release slots (v0)
Why: floor seed 1002, an object at rest on another one above the rim scores 0. Two release points per bin, 7 cm
apart, used in turn (`policy.release_shifts`, via `plan_pick(drop_shift=...)`). Reach checked offline: sideways
shifts of 6–9 cm plan with every arm that reaches the bin (+0.5 s); +x shifts from 6 cm on do not. With the
motion insets of 03:20 (side 7 cm) both slots keep ≥ 6.4 cm to every inner side wall and 5 cm to the near wall.
Verdict: kept; 1 of 16 doubly filled bins still had an object above the rim.

### Hand calibration close + grasp verification (v1)
Why: the floor carried air to the bin in 23 of 41 grasps (~6 s each) and never learned that a grasp failed.
What: once per episode both hands close on nothing; one function turns the trace into `HandResponse` per hand:
empty-hand error e0 per joint, steady bias, lag, class (stiff when the bias is below 0.01 rad; sim: bias 0.0001,
lag 0.10 s; real hand per the teleop data: bias 0.02–0.04, lag ~0.2 s). After the lift and through the carry
`score = Σ_j max(0, s_j (e_j − e0_j))` over the hand's seven joints against the class threshold
(`policy.grip_threshold`); an empty hand goes straight back to the look pose (checked joint move). Three "empty"
verdicts with no "held" verdict in the episode switch the check off. The class also selects the squeeze depth
passed to `plan_pick` (`policy.squeeze`, sim 0.0).
Verdict (run C): correct on all 58 picks, −11 % episode time, score unchanged until retries use the time.

### Look pose `motion.look_arm_q` (run D)
Verdict: rejected for now, see the table. `policy.look_pose` stays `home`; the first look of an episode is taken
where the arms start in any case (the camera fit wants them in view).

### Retries and time use (v2)
Why: C lost 4 objects to the cap of 2 picks per spot with half the episode unused.
What: up to 5 picks per spot, least-tried target first, the next candidate grasp (and the other arm where both
reach the bin) on a retry; a look without a plan for a target no longer writes it off.
Verdict: no gain on the fixed leg (27/31 as before) with pinch grasps: second and third attempts hold 28 % and
23 %. Kept (it costs nothing and the cage grasps change the per-attempt odds), but retries are not where the
objects are.

### First frame of an episode (found in v3, fixed in v4)
The observation of the first `act()` was rendered before the episode's reset and randomisation: the counter is
still empty in it and the camera stands where the previous episode's DR put it. v3 looks at that frame (first look
before the calibration close): it finds 0 objects and, worse, the detector fixes its camera pose on it for the
whole episode. First-look detections against the layout card, mean per episode (v3 fixed leg): dx / dy =
+16/−16, −6/+14, −7/−18, +4/+8, −10/+23, −15/−14, +9/−12, −4/+14 mm (v1, which looked at t ≈ 2 s: all within
±4.4 mm). v3 still sorted 29/31 because the cage grasp tolerates about a centimetre; seed 1004 burnt 5 empty
picks on it. v4 holds 0.5 s before the first look and, if the counter is still empty then, looks again from the
same pose before moving the arms.

### Cage grasps (motion, in v3)
Per pick on the v3 fixed leg (41 verdicts, with the calibration error above): 76 % held (pinch in v2: 51 %);
triangles 6/6, cylinders 16/22, cubes 7/10; across the body 9/12, own side 13/15.

### Hands that do not let go (found in v4, guarded in v6)
Why: v4's grasps hold (2 empty picks in 16 episodes) and 3 of its 4 lost objects were released over the right
bin without leaving the hand: a yellow triangle (thumb depth 0.80) rode back in the right hand and fell into the
right bin at that hand's next release (our first −1; the next lift with that hand read 0.237 instead of
0.10–0.13); a red triangle rode home and dropped on the counter at the next pick, toppled; a red cube retried
with the unknown-shape depth 0.75 ended the episode in the hand (z = 0.355). The fingers show nothing after
opening: the residual behind the released posture was 0.132–0.136 in all 60 releases.
What (v6): after a grasp deeper than 0.6 rad the hand closes once more at the retreat pose above the bin, the
grip score is read and the hand reopens: an open-close-open cycle with a verdict. Still holding → the cycle is
repeated (3 closes at most), then the arm is not used again in the episode. A retry keeps the detected shape's
depth (the unknown-shape depth only for `shape=None`) and varies arm, grasp and a 5 mm offset instead.
Verdict: pending v6.
