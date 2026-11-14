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

## What the policy is (v2, committed 78b1e61)

`SortPolicy` (own state machine): calibration close of both hands where the arms start → look → targets
(instruction mapping, on the counter and clear of the perceived bins, detection confidence ≥ 0.3, picks left) →
least-tried target first, then nearest its bin; arm on the bin's side first; candidate grasps rotated on a retry;
next free release slot → execute in 15-tick replies with the grip check after the lift and through the carry →
look pose → look again. Perception and planning run in one low-priority background thread; `act()` never waits
(worst measured 5 ms) and answers 10-tick holds meanwhile. `set_layout(detector.layout())` after every look; if
the perceived layout leaves no pick (or no safe way to the look pose) the scene file's layout is used for the rest
of the episode. `done` only after two looks spanning 2 s without a sortable object; a look that fails or rejects
every target pauses 1 s and looks again; an exception in a step drops what was in progress and starts over from a
look. A pick starts only if its release fits before the time limit; if the episode outlasts the assumed limit the
policy carries on. Instruction confidence below 0.5: nothing is moved.

Failure injection (scratch scripts outside the repo, kinematic mock world with a finger contact signal, runner-like
observation lag): blank head frames for the first 40 calls, no depth at all, detector raising on two looks, planner
raising once / always, reset during a 1 s plan followed directly by the next episode, assumed limit 20 s with a
real limit of 60 s, start pose far from home with closed hands, an object lost during the carry, a mock without any
finger signal, a perceived layout 10 cm off: no exception reaches the runner, 0 clipped steps, no early `done`.
Found by it and fixed: a calibration close from an already closed hand divided by zero and froze the state
machine (now: open first, and any step failure starts over); a missing depth frame ended the episode with `done`
after 3 s; an observation two ticks newer than assumed read the reopening hand as a steady bias.

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
Verdict: pending the v2 proxy.
