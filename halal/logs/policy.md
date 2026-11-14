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
| v4 | 994a14f7b199 | e3f9687 | +3.62 ± 0.70, 29/31 | +6.38 ± 1.87, 26/28 (1 wrong bin) | +6.50 ± 1.94, 26/28 | **+19.75** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 166; exact hand-in path `evaluate --container`: 8/8 in 2 episodes, rt 176) | start settle, visibility rule, clear looks before `done`; 16/16 instructions parsed; 4 of its 6 lost objects stayed in a hand that did not let go, 2 triangles toppled by a failed grasp |
| v5 | 5a4d3187fcf0 | fe03f4c | – | – | – | – | pre-check only | superseded by v6 before it ran |
| v6 | ec29a3c0ad75 | 58573f1 | – | 6/6, 6/6 then stopped | – | – | pre-check pass | v4 + perception 47e63f3 + release check after grasps deeper than 0.6 + motion 44bc3a4; replaced after 5 min: v4's unseen leg had a cube wedge at depth 0.30 |
| v7 | 5a8d69474d2e | 88aacba | 3/3, 5/5 then stopped | 2/3, 2/3 then stopped | – | – | pre-check pass | release check after every grasp; the check found a triangle still in the hand in both known episodes (0.096, like its carry score) and three plain reopenings did not free it: arm retired with the object in it |
| v8 | 550b70a8b003 | 2a1155c | +3.88 ± 0.78, 31/31 | +6.75 ± 1.71, 27/28 | +6.50 ± 0.87, 26/28 (1 time limit) | **+20.38** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 166) | v7 + hand-aware grasps (`candidate_grasps(hand=...)`) + a late-finger opening at the end of the check; nothing in a wrong bin or off the counter; the unseen leg is its second run (`v8-image-a2`), the first was killed by a container name clash after 2 episodes |
| v9 | b06b10e1cb9a | 7dee1dd | +3.88 ± 0.78, 31/31 | +7.00 ± 1.41, 28/28 | +6.50 ± 1.94, 26/28 (1 time limit) | **+20.63** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 162) | v8 + release check started during the retreat + three picks for a toppled object; 3 wedge incidents, 3 arms retired, 1 object lost to that; 1 triangle lost after 4 empty picks |
| v10 | 7e65414e990c | 513e266 | +3.88 ± 0.78, 31/31 | +6.75 ± 1.71, 27/28 | +7.00 ± 1.41, 28/28 | **+21.13** | PASS (through the image, 0 clips, ≥ 29.99 Hz, rt ≥ 186) | v9 + motion's staggered release (main 2120e4f) + finger-by-finger reopening before every re-check + last check back home before an arm is retired + hand opens on the way home + release slots sideways only; 86 release checks, 0 positive; lost: the seed-1001 triangle, pushed out of view by its first pick |
| v11 | 13ae623395f5 | 8428e32 | soft sim only: +3.88 ± 0.78, 31/31 | – | – | – | PASS (soft sim) | v10 + main b28f322 + soft hands told by a 0.002 rad bias + release check only after unknown-shape picks + known shapes first + the check back home at the calibration posture; not run on the stock sims (v12 contains it) |
| v12 | d2cadf30d4c2 | 8382cb2 | +3.88 ± 0.78, 31/31 | +6.75 ± 1.71, 27/28 | +7.00 ± 1.41, 28/28 | **+21.13** | PASS (through the image, 0 clips, ≥ 29.99 Hz, rt ≥ 151) | v11 + main 1f02750 (vertex-aligned triangle grasp, soft carry at 4 rad/s²) + lag as a second vote for a soft hand + slots 65 / 60 mm apart + attempts follow an object pushed in small steps; 48 of 48 hands classed stiff from a completed calibration; lost: a triangle knocked over by the grasp of its neighbour 7.5 cm away |
| v13 | – | 8701a4e | – | – | – | – | mock suites pass | v12 + main f91db01 and 66b64dc (obstacle-aware plans, real colour words) + other detections passed as obstacles, a blocked target waits for its neighbour + free-standing objects first + soft grasps for a hand without a completed calibration + calibration at home when the start pose has no room |

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
v7 (first two episodes of the known leg): the close above the bin read 0.096, 0.096, 0.097 for a triangle whose
carry score was 0.095, twice in two episodes that v4 had sorted 3/3: after a plain opening the triangle still
sits in the cage 1.3 s later; v4 let the move home shake it out (mostly into the bin), v7's check re-closed on it,
could not release it with three plain reopenings and retired the arm with the object in it (0 instead of +2, but
no −1). So the check sees the problem and plain reopening is no remedy.
v8: the reopening after such a close starts one finger 0.25 s late (`policy.reopen_stagger_s`, the motion
engineer's finding: staggered opening took wedged-in-hand from 16/24 to 0/24 in his sweep).
Verdict (v8, three legs + seeds 1010–1015 of the known leg, 30 episodes, ~110 releases): the check found an object
still in the hand 4 times (triangle at depth 0.80 twice, a cube at 0.30 whose lift score was 0.079 instead of the
usual 0.16–0.19, a cube picked without a shape at 0.75). The score at the check equals the carry score (0.095 /
0.097) or decays slowly (0.079 → 0.066 → 0.051 → 0.047). All 4 objects ended in the correct bin, no −1. But:
- v8 never opened late-finger-first before a re-check: after a positive read the hand opened with a plain ramp,
  closed and was read again; the staggered opening came only after the last read, with the arm already retired.
  Three positive reads per incident are three plain openings; whether the one staggered opening or the move home
  freed the object the logs cannot tell.
- Retiring the arm is what cost points: unseen seed 1002 lost two red objects that only the retired left arm
  reached (retired at t = 17 s, "no plan" for the remaining 100 s), seed 1008 a toppled triangle. 3 objects lost
  to retirement, none to the wedge.
- Cost of the check: 57 ticks (1.9 s) per pick in v8, 43 (1.45 s) when it starts during the retreat (v9).

v10: (1) motion's staggered release at the first opening (`motion.release_stagger_s` 0.25: 118/118 out of the
hand within the open phase in the grasp bench); (2) a positive read is followed by thumb and one finger opening
fully, a rest (`policy.reopen_pause_s`), then the last finger, an open hold, a close and the next read: 3 reads
above the bin; (3) after the third positive read the hand opens the same way, the arm goes home and the hand is
read once more there: only a positive read at home retires the arm; (4) after an empty read the hand opens during
the move home if that keeps the plan's own clearance (always, in the mock: 5–12 cm), so the check costs ~21
ticks (0.7 s) per pick; (5) every release logs its verdict. Mock (object kept for the first n openings): n = 1
freed at the first re-check; n = 3 freed before the way home, arm kept, 3/3; n ≥ 4 arm retired. 0 clips.
Verdict (v10, 24 episodes): 86 releases read, 0 positive, highest empty read 0.004 (threshold 0.02); the check
cost 18–22 ticks per pick. v9, same check without the stagger: 3 positive in 82. With the staggered release the
check finds nothing, so from v11 on it runs only after picks of objects without a shape
(`policy.release_check_after: [unknown]`; motion's bench: 3 of 24 stuck at the unknown-shape depth with a plain
release). From v11 the read back home closes to the calibration posture (the one posture with a measured
empty-hand reading), two positive reads above the bin that it does not confirm switch the check off for the
episode (`policy.release_check_patience`), and a hand without a calibration close is never checked.

### Release slots with the staggered release (v10)
The staggered release lands objects −27..+118 mm ahead of the release point and −48..+63 mm outward (motion's
bench, n = 119): two slots cannot be separated forward any more. Slots now differ sideways only: side bins 0 and
4 cm outward, centre bin ±2 cm (±3.5 cm would leave 4 mm to the side wall for an outward throw).
v12, with motion's release points 8.5 cm inside the side walls (landing box x −28..+77 mm, outward −40..+48 mm):
side bins 2 cm inward and 4.5 cm outward (65 mm apart), centre bin ±3 cm, so two objects cannot rest on each other.

### Two evaluators on one sim (14.11, 09:28 UTC, my mistake)
`halal.eval.proxy --port P` serves leg i on P + i and `serve_image.sh` names its container `halal-eval-<port>`
and removes that name on exit: a second run whose ports overlap a running one removes the running leg's
container, and a second evaluator on a sim makes both episodes garbage (a calibration close read both hands as
soft, bias 0.34 / 0.56 rad). Runs now start through `/data/scratch/policy/run_legs.sh`, which refuses a sim that
has a client, a port in use and an existing container name. Discount: `v8-image-a_1114_090854` unseen leg (2 clean
episodes, 6 crashed), `v8-seeds1008-c` seeds 1008 and 1009.

### Soft-hand sim rehearsal (14.11, sim `c` with `HAND_GAINS="1.5,0.2,0.0075"`, fixed leg, seeds 1000–1007)
Caveat: gains fitted to the real finger's free step response; no friction, dead zone or backlash in the sim finger.

| Image | Hand class (16 closes) | Picks held at first try | Lift score held / empty | Objects | Gate |
|---|---|---|---|---|---|
| v8 (2 episodes, stopped) | stiff 4/4: bias 0.0055 rad, lag 0.137–0.149 s | 6/19 | 0.166–0.287 / 0.004–0.005 (threshold 0.02) | 7/8 | pass, 0 clips |
| v11 | soft 16/16: bias 0.0055, lag 0.126–0.139 | 31/31 (cube 9, cylinder 17, triangle 5) | 0.313–0.567 / – (threshold 0.15) | 31/31, +3.88 ± 0.78 | PASS, 0 clips |
| v12 | soft 16/16: bias 0.0055, lag 0.130–0.141 | 31/32, the retry sorted the object | 0.314–0.563 / 0.012 (threshold 0.15) | 31/31, +3.88 ± 0.78 | PASS, 0 clips |
| v11 tree, release check after every shape (4 seeds) | soft | 12/12 in three episodes, then see below | empty hand at the soft grasp posture 0.009–0.013 (13 reads), no positive | 13/15 | pass, 0 clips |

What broke, by cost:
1. Identification (v8–v10): `stiff_hand_bias` 0.01 sat between the soft sim's 0.0055 and the real hand's 0.02–0.04; the
   stiff grasps hold 1 in 3 on a soft hand. v11: 0.002 (stiff sim 0.0001). v12 adds the lag as a second vote
   (`stiff_hand_lag_s` 0.125; stiff sim 0.09–0.107, soft sim 0.126–0.149: thin, hence only an OR).
2. A cube lost in the carry (lift score 0.384, 0.012 a second later; right arm across the body, 8 rad/s²): motion's
   soft grasps carry at 4 rad/s² from main 6933099 (v12).
3. Retries: motion's second cube candidate, the organiser `pinch`, has no soft closure (7 picks, 7 empty), and the
   policy repeated it seven times as "try 2", each push 2–3 cm, walking the cube from x 0.331 to 0.227: the
   position memory (3 cm radius) saw every new spot next to exactly one earlier pick. v12: after an empty pick the
   nearest detection of that colour within `policy.displaced_radius` inherits at least the attempts of the missed
   spot, so tries count up, arm and grasp rotate and `max_attempts` ends it.
Not exercised: stiction-driven effects, a failed soft cage grasp on a clean scene, the soft hand's release of a
wedged object.

### Toppled or pushed objects (v9–v12)
Three blob picks (v9) are worth nothing: the seed-1001 yellow triangle (layout yaw −36°, right arm, cage) came up
empty at the first pick in 5 of 5 runs of v8–v10; pushed 10 cm and read as a blob it was missed three more times
(v9), in v10 it left the camera's view. The remedy is the grasp: motion's vertex-aligned `cage_apex` (v12).

### Vertex-aligned triangle grasp and neighbours (v12, v13)
`cage_apex` (motion) holds the seed-1001 triangle at the first pick in both language legs of v12 (lift score
0.095; 5 of 5 empty with `cage`). In the known leg that pick knocked over the blue triangle 7.5 cm away, which
then took three empty picks and was lost; in the unseen leg it did not. v13 passes every detection of the look to
`plan_pick(obstacles=...)`: a target whose options fail with "hand would touch" is not a failed attempt while
another target can go; if none can, the first blocked target is planned without its nearest neighbour (logged).
Targets without another detection within `policy.neighbour_radius` (9 cm) are picked first.

### Which class for a hand nobody calibrated (v13)
Motion's table: a soft hand with stiff grasps holds 2 of 18, a stiff hand with soft grasps loses only triangles
(6/12). So only a completed calibration with a small bias and a small lag gives the stiff grasps; everything
else is soft. The contact threshold of an uncalibrated hand stays 0.02 without an empty-hand error: an empty
real hand then reads as held (one empty trip, ~6 s), a full hand is never opened at lift height. If the start pose
leaves the hands no room to close (`calibration_clearance`), the arms go home and calibrate there before the
first pick (mock: calibrated at home, 3/3).

### Time limit (decision, not implemented)
The limit is unknown (90–180 s) and the image assumes 120 s only to decide whether a pick still fits. Preferring
a pick that ends before 90 s over the first in order would matter in a 2 s window of episodes that reach 82 s
with two targets of different length left: about one object in a hundred episodes. Not built. Order of picks is
least tried, known shape, free-standing, nearest its bin; a pick cut off by the limit keeps the object in the hand.

### Finals robustness in the mock (v13 tree; kinematic mock world with a finger contact signal and runner-like observation lag)

| Case | Objects | End | Clips |
|---|---|---|---|
| start pose far from home, hands closed | 3/3 | done, 57 s | 0 |
| blank head frames for the first 40 calls | 4/4 | done, 91 s | 0 |
| no depth at all | 0/3, nothing moved | time limit | 0 |
| detector raises on looks 1 and 3 | 5/5 | done | 0 |
| planner raises on its 2nd call / on every call | 3/3 / 0/3 | done / time limit | 0 |
| reset during a 1 s plan, next episode at once | 5/5 in the next episode | done | 0 |
| assumed limit 20 s, real limit 60 s | 3/3 | time limit at 60 s | 0 |
| no room for the calibration close at the start pose | 3/3, calibrated at home | done | 0 |
| instruction for one colour/shape ("Put the purple cube into the center box.") | 1 pick, the rest untouched | done, 31–34 s | 0 |
| object stays in the hand for the first n openings (release check on) | n ≤ 3: 3/3, arm kept; n ≥ 4: arm retired | done / time limit | 0 |

Worst `act()` 2–5 ms in every case. Not covered by the mock: physics of a failed grasp, real camera frames, a
detector that never returns (the look would wait for it for the rest of the episode).
