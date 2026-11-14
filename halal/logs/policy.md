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
| v13 | 3063629595a9 | d3a28be | +3.75 ± 0.66, 30/31 | +6.75 ± 1.71, 27/28 | +7.00 ± 1.41, 28/28 | **+21.00** | PASS (through the image, 0 clips, 29.99 Hz, rt ≥ 186) | v12 + main 7cfd3f1 (obstacle-aware plans, colour families orange → red and purple → blue, real colour words) + other detections passed as obstacles, a blocked target waits for its neighbour + free-standing objects first + soft grasps for a hand without a completed calibration + calibration at home when the start pose has no room + release slots 30 / 40 mm apart; lost: a cube stacked on its twin (slots), a triangle picked with plain `cage` after a neighbour ruled out `cage_apex`; soft sim 29/31, both lost objects stacked; not published |
| v14 | e061f54d6437 | 7e94c20 | +3.75 ± 0.66, 30/31 | +7.00 ± 1.41, 28/28 | +7.00 ± 1.41, 28/28 | **+21.25** | PASS (through the image, 0 clips, 29.99 Hz, rt ≥ 192) | v13 + v12's release slots + a target whose plan would touch a neighbour waits, the last resort plans as if alone + detector deadline 5 s + one target's failure no longer fails the look; lost: the same stacked cube as v13 (pick order); soft sim 30/31: a cube slipped out in the carry and lay 3 cm in front of a bin where no plan reaches |
| v15 | 455b3c06f0f1 | 7d017e4 | +3.88 ± 0.78, 31/31 | +7.00 ± 1.41, 28/28 | +7.00 ± 1.41, 28/28 | **+21.38** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 153) | v14 + main 0a9cc58 (orange / yellow boundary from the hue gap of the look, measured object size, lying prisms) + v12's pick order + most room for the fingers when no target is clear + `done` after 10 s of empty looks + lying prisms grasped as unknown objects; 87 of 87 objects, no empty pick; soft sim 31/31, all picks held at the first try |
| v16 | ae811fa244d1 | f58d8a4 | +3.88 ± 0.78, 31/31 | +7.00 ± 1.41, 28/28 | +7.00 ± 1.41, 28/28 | **+21.38** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 149) | v15 + main 99e1976: grasps keyed as motion keys them (`lying` passed through, strategy `cage_lying` = the unknown-shape hold for now); no change of behaviour intended: a repeat of v15's result, 87 of 87, no empty pick; soft sim 31/31, all picks held at the first try |
| v17 | 2332657ddf0e | 3ebbf19 | +3.88 ± 0.78, 31/31 | 7 episodes: 25/25, stopped | – | – | PASS on the 15 episodes run | v16 + main a8d0b63 (`min_wall_height` 0.07, `cage_side` last candidate) + `Grasp.last_resort` kept out of the retry rotation; stopped for v18 (robot slot), contained in it |
| v18 | b99fd5dd4a9f | e981865 | +3.88 ± 0.78, 31/31 | +7.00 ± 1.41, 28/28 | +7.00 ± 1.41, 28/28 | **+21.38** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 134 with four sims running) | v17 + main 4d93acb (perception hardening, bin row finder) + perception's layout is the only layout (no scene-file re-plan, no latch); known leg on sim `d`; soft sim 31/31, 0 empty picks |
| v19 | ede4203ebfb1 | 6216ebb | +3.88 ± 0.78, 31/31 | +7.00 ± 1.41, 28/28 | +7.00 ± 1.41, 28/28 | **+21.38** | PASS (through the image, 0 clips, 30.0 Hz, rt ≥ 115 at rtf 0.33–0.38) | v18 + main c56b420 (fail-fast reach, `rim_z`, landing box, stiff lying list, black / colourless detections, `agree_within` 12 mm, `reaches_bin` fix) + planner keeps a tried plan's reason + arm choice for last-resort-only arms + `release_rise` list + a log line for a target no arm reaches |

v13 per leg (through the image, 14.11, 11:39–12:08 UTC; seeds 1000–1007):

| Leg | Sim | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Hands (16 closes) | Empty picks | "hand would touch" / last resort | Seed 1001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | a | +3.75 ± 0.66 | 30/31, 1 counter | done 8 | 0 | 29.99 | 186 | PASS | stiff 16 | 0 | 12 / 1 | 4/5: second red cube stacked on the first |
| known | c | +6.75 ± 1.71 | 27/28, 1 counter | done 8 | 0 | 29.99 | 200 | PASS | stiff 16 | 3 | 2 / 0 | 2/3: yellow triangle, `cage` after `cage_apex` was refused, 3 empty picks |
| unseen | a | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 29.99 | 195 | PASS | stiff 16 | 0 | 2 / 0 | 3/3: the same fall-through to `cage` held (0.101) |
| fixed, soft-hand sim | c | +3.62 ± 0.70 | 29/31, 2 counter | done 8 | 0 | 30.0 | 168 | PASS | soft 16 (bias 0.0055, lag 0.126–0.142) | 0 | 12 / 1 | 4/5: second red cube stacked (seed 1002: a blue cube stacked) |

Pre-check `replay --synthetic 100 --container`: pass, p50 6.2 ms. All 31 soft picks held at the first try (lift
score 0.307–0.561, threshold 0.15). Every lost object but the triangle came to rest on another one in its bin.

v14 per leg (through the image, 14.11, 12:08–12:39 UTC; seeds 1000–1007):

| Leg | Sim | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Hands (16 closes) | Empty picks | "hand would touch" / as if alone | Seed 1001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | a | +3.75 ± 0.66 | 30/31, 1 counter | done 8 | 0 | 29.99 | 194 | PASS | stiff 16 | 0 | 2 / 1 | 4/5: second red cube stacked on the first (order) |
| known | c | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 29.99 | 197 | PASS | stiff 16 | 2 | 6 / 3 | 3/3 after two empty picks: the triangles block each other |
| unseen | a | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 29.99 | 192 | PASS | stiff 16 | 1 | 5 / 2 | 3/3 after one empty pick |
| fixed, soft-hand sim | c | +3.75 ± 0.97 | 30/31, 1 counter | done 7, time limit 1 | 0 | 29.99 | 151 | PASS | soft 16 (bias 0.0055, lag 0.126–0.139) | 1 | 3 / 1 | 5/5 (the pair rested 113 mm apart) |

Pre-check: pass, p50 5.3 ms. The detector deadline never fired; no error lines. Same-bin pairs: 25 over the
four legs, one stacked. Soft leg, seed 1003: a red cube (left arm, `cage`, lift score 0.428) read 0.020 half a
second later: it slipped out in the carry and came to rest at (0.368, +0.023), 3 cm in front of the centre bin.
Every plan for it was then refused ("hand −0.1 / −0.4 / −0.6 cm above a bin rim during descend", 47 times until
the time limit). Reported to motion: an object within ~3 cm of a bin's near wall is out of the low sequence's
reach. Soft lift scores 0.315–0.558.

v15 per leg (through the image, 14.11, 12:36–13:15 UTC; seeds 1000–1007):

| Leg | Sim | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Hands (16 closes) | Empty picks | "hand would touch" / as if alone | Seed 1001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | a | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 182 | PASS | stiff 16 | 0 | 2 / 1 | 5/5: the red cubes rest 70 mm apart (v12's order) |
| known | c | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 153 | PASS | stiff 16 | 0 | 6 / 2 | 3/3 at the first try: the blue triangle goes first, fingers 35 mm from the yellow one |
| unseen | a | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 181 | PASS | stiff 16 | 0 | 7 / 2 | 3/3 at the first try, same choice |
| fixed, soft-hand sim | c | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 175 | PASS | soft 16 (bias 0.0055, lag 0.126–0.140) | 0 | 2 / 1 | 5/5 |

Pre-check: pass, p50 5.5 ms. Same-bin pairs on the stock legs: 17, smallest gap 50 mm, none stacked or leaning;
soft leg: 7, smallest gap 45 mm, all flat; soft lift scores 0.313–0.559 (threshold 0.15).
No detector deadline, no error line, no lying detection. The 10 s `done` span makes a leg 15.7 min instead of 14.

v16 per leg (through the image, 14.11, 13:07–13:50 UTC; seeds 1000–1007):

| Leg | Sim | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Hands (16 closes) | Empty picks | "hand would touch" / as if alone | Seed 1001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | a | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 180 | PASS | stiff 16 | 0 | 3 / 1 | 5/5 |
| known | c | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 149 | PASS | stiff 16 | 0 | 6 / 2 | 3/3 at the first try |
| unseen | a | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 182 | PASS | stiff 16 | 0 | 6 / 2 | 3/3 at the first try |
| fixed, soft-hand sim | c | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 189 | PASS | soft 16 (bias 0.0055) | 0 | 2 / 1 | 5/5 |

Pre-check: pass, p50 5.3 ms. Same-bin pairs on the stock legs: 16, smallest gap 49 mm, all flat; soft leg: 7,
smallest gap 51 mm, all flat; soft lift scores 0.310–0.567. No detector deadline, no error line, no lying
detection. v15 and v16 together: 174 of 174 objects on the stock legs, 62 of 62 on the soft-hand sim.

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

Since v4 (state of the v16 tree; each item has its section below):
- Hands: a calibration close at the first look classes each hand stiff or soft (bias and lag); soft or unknown
  hands get the soft grasps, slower carries and their own grip threshold. No room to close at the start pose:
  home first, calibrate there.
- Picks: order = least tried, kinds with a grasp of their own (`policy.last_kinds` go last), nearest its bin.
  Every detection of the look goes to the planner as an obstacle. A target waits when the plan it would get
  alone would touch a neighbour.
  When no target is clear, the least tried one whose plan keeps its fingers furthest from the others goes,
  planned without obstacles. Obstacles change the order, never the grasp.
- After an empty pick the nearest detection of that colour inherits the attempts (pushed objects); 5 picks per
  object, 3 for a toppled one (lying prism or blob).
- Release: two slots per bin, sideways only, 65 mm apart; the release check (one more close above the bin) runs
  only after picks of `policy.release_check_after` kinds and only for a calibrated hand.
- Looks: the detector call has a deadline (`policy.detect_deadline_s`, doubling after a miss); anything one
  target breaks counts as no plan for that target only. `done` needs clear looks without a sortable object
  spanning 10 s.

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
v13 tried motion's narrower slots (side 0 and 3 cm outward, centre ±2 cm; his landing box in the grippy-and-light
corner is x −35..+79 mm, outward −59..+69 mm around the release point, so the first slot at 0 keeps every wall
clear). Objects that share a bin, centre to centre at rest (objects are 45 mm wide, the rim is at z 0.070):

| Slots | Legs | Pairs | Smallest gap | Not flat | Lost |
|---|---|---|---|---|---|
| v10: side 0 / +4 cm, centre ±2 cm | 3 | 15 | 32 mm | 3 leaning, z 0.051–0.067 | 0 |
| v12: side −2 / +4.5 cm, centre ±3 cm | 4 (one on the soft sim) | 22 | 52 mm | 0 | 0 |
| v13: side 0 / +3 cm, centre ±2 cm | 4 (one on the soft sim) | 22 | 2 mm | 3 stacked at z 0.077–0.081, 2 leaning at z 0.043–0.048 | 3 |

v13, fixed seed 1001: two red cubes released at y 0.243 and 0.273 rested 2 mm apart, one on the other, above the
rim: scored as left on the counter; the soft-hand sim stacked the same pair and one more (gaps 14 and 33 mm). Two
more pairs landed on each other and slid apart (gap 44–46 mm along x). No object of any version came to rest
outside its bin. v14 is back on v12's slots: stacking has cost an object in
measured episodes, the walls have not. Open: the grippy-and-light corner is not in our eval seeds.
v14 stacked the fixed seed-1001 pair again, 65 mm slots or not (gap 14 mm, z 0.081): the pick ORDER decides it.
v13 put objects without a neighbour within 9 cm first, so the red cube at (0.27, −0.14) went before the one at
(0.33, −0.07). Picked across the body by the left arm, it comes to rest at y 0.258–0.263 whatever its slot
(released at 0.223 in v14, at 0.243 in v13: thrown 20–35 mm outward); the second cube, released at the outer
slot, falls 12–16 mm inward and lands on it. Stacked in 3 of 3 runs with that order (v13, v13 soft, v14), 0 of 3
with v12's order (v10, v12, v12 soft: the near cube takes the inner slot and falls inward, gaps 57–77 mm). v15
drops the neighbour count from the order (v12's order again: least tried, known shape, nearest its bin); what it
was for is done by the waiting rule below. A slot is still chosen blind to where the first object came to rest.

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
v13 through its image: 28 "hand would touch" refusals in 32 episodes, the last resort twice. Known seed 1001: the refusal took `cage_apex`
away from the yellow triangle (the blue one 7.5 cm away), the next option was plain `cage`, the planner passed it,
the pick came up empty and toppled the triangle: lost after two more picks (the same fall-through held it in the
unseen leg, lift score 0.101). So v13 traded a risk for the neighbour against a worse grasp for the target.
v14: a target waits as soon as the plan it would get alone on the counter would touch a neighbour, and no other
arm or grasp is tried for it in that look; clear targets go first. When no target is clear (objects that touch,
neighbours that are not to be sorted) the first one in order is planned without obstacles, as v12 did (logged):
obstacles change the order of picks, never the grasp, and can not keep a target from being picked. Mock, real
planner, two cubes 50–70 mm apart and a free cylinder: cylinder first, then the nearer cube as if alone, then
the other one clear, 3/3; at 85 mm no refusal. This run is also the check that the planner's reason text still
reaches the chooser (the policy recognises a neighbour by the words "hand would touch").
v14 through its image, known seed 1001: the two triangles block each other, the yellow one goes as if alone with
`cage_apex`, comes up empty and topples the blue one; both are sorted on the retries (3/3; v12 lost the blue one
there, v13 the yellow one). Nothing else was tried for them because nothing else exists: one arm reaches a side
bin, the vertex-aligned grasp has one reachable heading, and the only other candidate is the plain `cage` that
v13 fell through to. v15: when no target is clear, the least tried one whose as-if-alone plan keeps its fingers
furthest from the centre of any other object goes (motion's `digit_gap`; logged with the distance). Mock, real
planner: cubes 50 / 60 / 70 mm apart are picked with the fingers 23 / 33 / 42 mm from the neighbour; with one
cube turned 30° the other cube is chosen (28 mm). Fake planner: clear first; all blocked: most room; a target
already tried once waits for an untried one.
One object behind another as seen from the robot (checked after v15, nothing changed). Motion's sweep 19: a
cube with a neighbour 8 cm behind it (where the open thumb comes down), planned blind: 0 of 6 held, neighbour hit;
planned with the obstacles: 6 of 6, neighbour untouched (26 of 26 over all his cases). Mock, real planner, pairs
6–8 cm apart in x: at 6 cm both objects are refused and the most-room rule takes the NEAR one (4 of 4, 28 mm);
at 7–8 cm the far one usually goes first (7 of 8), with a plan the planner made with the obstacles and found
clear, which is the case the sweep measured as safe. A rule "wait while another detection lies up to 10 cm
nearer the robot and within 4 cm sideways" made the mock near-first in 12 of 12 and was taken out again: it can
not fire in a sampled layout (the object band is 7 cm deep and objects are 8 cm apart, so a pair is never within
4 cm sideways: 0 pairs in our 24 eval layouts), and "behind" is along the hand's heading, which only the planner
knows. Ordering everything by x instead would bring back the order that stacks fixed seed 1001.

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
| detector call never returns on look 2 / on looks 1, 3, 4 (v14) | 3/3 / 4/4 | done | 0 |
| detector never returns at all, then a normal episode (v14) | 0/3, then 5/5 | time limit, then done | 0 |
| every detector call takes 1.5 × the deadline (v14) | 3/3 after one abandoned look | done | 0 |
| a detection with an unknown shape word in every look, ahead of a real target in the order (v14) | 3/3 | time limit | 0 |
| a detection with NaN position, NaN yaw, NaN confidence, unknown colour, 30 cm above or below the counter | 3/3 | done / time limit | 0 |

Worst `act()` 2–5 ms in every case. Not covered by the mock: physics of a failed grasp, real camera frames.

### A detector call that never returns (v14)
Before: the look waited for it for the rest of the episode, and because jobs share one lock the next episodes
waited too (mock: 0/3, then 0/5 in the following episode). v14: the detector call runs in a thread of its own and
is left behind after `policy.detect_deadline_s` (5 s; a call takes 0.03–0.1 s, a whole look 0.8 s of wall time
under three sims): the look fails, the policy pauses 1 s and looks at a fresh frame. Every miss doubles the
deadline for the calls after it, so a detector that is only slow costs one deadline, not every look. The
abandoned call never reaches the planner. Not given a deadline: the planner (its loops are bounded, and a second
call on the same planner while one is stuck would share its state).

### One target that breaks planning (v14)
The options of a target (arms, grasps, slots) were built outside the guard around the planner call: a detection
whose shape is not one of ours raised in `candidate_grasps`, the whole look failed, and when that target was
ahead of others in the order nothing behind it was ever picked (mock: 2/3 with 17 failed looks). v14 counts
anything that fails for one target as no plan for that target: 3/3. A phantom that can never be planned still
keeps the episode from ending with `done` (it runs to the time limit; nothing is lost by that).

### Real colours in the mock (v13 and v14 trees; rendered mock world, object colours orange 25°, pale yellow 55°, purple 220°)
The detector's `red` family covers orange and `blue` covers purple (main 7cfd3f1); the parser maps the words.

| Instruction | Objects | What happened | Clips |
|---|---|---|---|
| "Put the purple objects in the left bin, the yellow objects in the center bin and the orange objects in the right bin." | real colours, 3 seeds | {blue: left, yellow: center, red: right}, conf 1.00; 3/3, 4/4, 5/5, done in 61–95 s | 0 |
| "Put the orange cube into the right box." | real colours, 4 seeds | {red/cube: right}; every orange cube moved, nothing else touched, done in 35–62 s | 0 |
| the sim's training sentence with permuted bins (red / yellow / blue words) | real colours, 2 seeds | orange sorted as red, purple as blue: 3/3, 4/4 | 0 |
| the fixed-task sentence | real colours | 5/5 | 0 |
| "Sort the blocks into the orange boxes: purple left, yellow middle, orange right." | real colours | boxes not taken for a rule; 4/4 | 0 |
| the real three-colour sentence, the one-cube sentence | sim colours | 3/3; only red cubes moved | 0 |

The mock draws every prism as a box, so "cube" there includes its triangles. One colour moved at a time, the real
three-colour sentence: orange read at 15–34° correct; at 36°, 40°, 45° it is detected as yellow with full
confidence and goes to the yellow bin (the one way to a wrong bin found here); purple at 200–284° correct, at
295° not detected (stays); yellow at saturation 0.30 correct, at 0.20 not detected (stays). Reported to
perception, who are moving the orange / yellow boundary into the hue gap the scene shows; this sweep is the
acceptance test for that.

The same sweep before and after perception's per-episode boundary (main 0b3f737: the boundary moves into the gap
between two warm groups; without that evidence a warm object loses its confidence near the fixed 36° boundary):

| Orange object read at | v14 tree | v15 tree |
|---|---|---|
| 15–34°, a yellow object on the counter | correct | correct |
| 36–41°, a yellow object on the counter | yellow bin | correct |
| 42–45°, a yellow object on the counter | yellow bin | yellow bin |
| 30°, no yellow object | correct | correct |
| 34°, no yellow object | correct | left on the counter |
| 36–38°, no yellow object | yellow bin | left on the counter |
| 40–45°, no yellow object | yellow bin | yellow bin |
| every hue turned by −10°, +5°, +10° | correct | correct |
| every hue turned by +15° | yellow bin | correct |
| every hue turned by +20° | yellow bin | yellow bin |

Purple (200–284° correct, 295° stays) and pale yellow (saturation 0.30 correct, 0.20 stays) are unchanged.

### `done` (v15)
`done` earns nothing and a wrong one loses whatever is still sortable. With looks 0.6 s apart and a 2 s span, a
detector that saw nothing for ~4 s of looks ended the episode (mock, blind for 15 s from t = 15: `done` at
t = 25 with 3 of 4 objects left; blind for 9 s: survived only because a pick was running). v15:
`policy.empty_span_before_done_s` 10 s: blind for 15 s from t = 15: 4/4. Cost: 8 s of sim time at the end of
every episode, about 9 min more wall time for a four-leg run. Not changed: an object dropped within
`policy.bin_clearance` of a bin is outside the object area and is not picked again.

### Prisms lying on a side (v15, temporary)
Perception now reports a toppled prism as `shape="triangle"`, `lying=True`, `yaw` = ridge direction, `visible`
0.67–1.0 instead of a `visible=0` blob (main 0a9cc58). The bin follows the shape (shape-keyed instructions work
for lying prisms). Motion's grasp for them is not on main yet, so `picking.grasp_hints` tells the grasp nothing
about a lying prism (unknown shape and yaw: plain `cage` at the unknown-shape depth, as the blob was picked;
`cage_apex` would aim at a vertex computed from a ridge direction). Everything keyed on what the grasp knows
follows that one function: grasps, order (after the known shapes) and the release check. A lying prism keeps
the blob's 3 picks (`policy.blob_attempts`). When `candidate_grasps(..., lying=...)` exists: pass `det.lying`
there, return the detection's shape and yaw from `grasp_hints`, and give lying prisms `max_attempts`.
After v15 (main 99e1976 has `candidate_grasps(shape, yaw, side, hand, lying)` and the strategy `cage_lying`, for
now the unknown-shape hold): `grasp_hints` is gone; the policy passes the detection's shape, yaw and `lying`, and
names an object's grasp kind the way motion does (`picking.grasp_kind`: lying, else the shape, else unknown).
`policy.last_kinds` (picked after the others) and `policy.release_check_after` both list `unknown` and `lying`:
when motion's measured lying grasp lands, take `lying` out of either list in config as its numbers allow; the
3-pick limit of a lying prism is the remaining line (`looking._targets`).
Scratch check (real detector, one object's detections rewritten): "Put the orange triangular prism into the
right box." → {red/triangle: right}; reported lying: picked with `cage`, grasp asked for (None, None), release
check run, right bin, nothing else touched; reported upright: `cage_apex` for ('triangle', yaw); reported as a
cube: not touched. Under the three-colour sentence the lying prism goes last, to its colour's bin.

### What the finals may vary, in the mock (14.11 afternoon, v16 tree unless stated)
Organisers (15:07): colours, count and placement vary at random; cubes, prisms and cylinders; prisms stand or lie; 120 s;
real `sort_fixed` is yellow → left, purple → center, orange → right. Scratch script `crowd.py`: objects placed at random
in the object band (cubes and cylinders, the mock draws prisms as boxes), 6.5 cm apart at least, real colours, the real
sentence unless stated. The fast mock runs ~10× real time and charges planning at wall time, which made a 5-object
episode look 98 s long; every number below is from runs paced at real time (planning as fast as the Mac plans).

| Case | Result | Clips |
|---|---|---|
| 6 objects, 4 layouts | 24/24, last release 73–99 s, done | 0 |
| 7 objects, 4 layouts | 28/28, last release 81–118 s (two only just) | 0 |
| 8 objects, 4 layouts, 4 runs at a time | 31/32, last release 109–114 s, time limit | 0 |
| one colour only, 5 and 8 objects, each colour | 15/15 and 24/24 (8: last release 107–116 s) | 0 |
| sentence names one colour / two colours, 6 / 7 objects | 4/4 and 5/5 sorted, the other 2 + 2 untouched | 0 |
| "Put all (the) cylinders in / to the left box." | every cylinder left, cubes untouched | 0 |
| "cylinders left, cubes right", "cubes left, cylinders center, prisms right" | 6/6, 6/6 | 0 |
| "yellow cylinders left, purple cubes right", "the orange cube into the center box" | 4/4 + 2 untouched, 2/2 + 4 untouched | 0 |
| an object in a colour nobody names: green, cyan, teal, pink, magenta, white, grey, black | not detected, untouched | 0 |
| the same, brown (orange hue, dark) and lime (85°) | sorted as orange / as yellow: wrong bin (perception's hue basins; told the orchestrator) | 0 |

Nothing went to a wrong bin or was moved without being asked for, except the two foreign colours.
Time: sim v16 needs 11–12 s per object (5 objects: done at ~70 s including the 10 s `done` span); 8 objects is the
ceiling at 120 s, and only with objects that hold at the first pick. In a crowd (neighbours closer than 8.5 cm) nearly
every look finds every target blocked: each is planned with the obstacles, refused, then the least tried are planned as
if alone. A refused plan costs 0.45–1.1 s on the Mac (the planner runs the whole IK chain for each heading before it
gives up; a clear plan is 0.14–0.3 s): 6–8 s per look with 7 targets, 19–24 refusals per 8-object episode.
Tried and not adopted: a memory of refusals (a target refused for a neighbour is not planned with obstacles again while
no object within 15 cm of it came, went or moved, same number of tries, same hands): refusals 22–24 → 15–20 per episode,
32/32 instead of 31/32, last release 108–118 s: inside the noise of four runs. With planning twice as slow both sort 5–6
of 8. The lever is a cheaper refusal in the planner (reported). Patch kept outside the repo.
Not measurable in the mock: a third object in one bin takes the first object's release slot again (two slots per bin);
the mock has no stacking and the sim never spawns more than five objects.

### `cage_side` and the retry order (v17)
Main a8d0b63 (motion ec8dcf9): `motion.min_wall_height` 0.07 (release 10 mm higher; the policy's transit checks go
through the planner and see the same wall, nothing in the policy assumed 60 mm) and `cage_side` as the last candidate
for cube, cylinder and unknown shape. The policy starts a retry with the NEXT candidate (`_rotated`), so the plain merge
would have retried a cylinder, an unknown shape or a soft-hand cube missed once in the open with `cage_side` first
(motion's verdict: only when nothing else plans; soft hand not measured). v17: a strategy can be flagged
`last_resort: true` in `motion.grasps` (`Grasp.last_resort`); the policy rotates the others and keeps those at the end.
Order tried, stiff cylinder, tries 1–4: before `cage | side side cage | side cage side | cage side side`, now always
`cage side side`; stiff cube: `cage pinch side side`, `pinch cage side side`, alternating.
Mock, real planner, a cube and two cylinders 3.5–4 cm before the bins' near walls and one free cylinder: v16 tree 1/4
(the three never get a plan, time limit), v17 tree 4/4, the three with `cage_side`, 0 clips. All suites on the v17
tree: 36 cases as before, neighbours_e2e (same orders; "hand would touch" reaches the chooser; the logged finger room
is now measured to the neighbour's outline, 30 mm less than before), behind_pair, lying_prism, harden, real colours,
blind_spell: unchanged, 0 clips.

### v16 on fresh seeds (14.11, 14:24–15:11 UTC, through the image, sim `a`, seeds 1008–1015)

| Leg | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Empty picks | "hand would touch" / as if alone |
|---|---|---|---|---|---|---|---|---|---|
| fixed | +4.12 ± 0.78 | 33/33 | done 8 | 0 | 30.0 | 185 | PASS | 0 | 8 / 0 |
| known | +7.25 ± 1.39 | 29/29 | done 8 | 0 | 30.0 | 179 | PASS | 1 (seed 1014, sorted on the retry) | 7 / 1 |
| unseen | +7.25 ± 1.39 | 29/29 | done 8 | 0 | 30.0 | 174 | PASS | 0 | 7 / 1 |

Proxy +22.25 of 22.25. No error line, no detector deadline; 10 same-bin pairs on the fixed leg, smallest gap 51 mm, all
flat. With seeds 1000–1007: 178 of 178 objects for v16 on stock legs, so there is no rare failure to explain at this
sample size; seeds 1016–1023 were dropped in favour of v18's legs. (In the eval log `t=` is wall time; `sim_s` in
`eval.json` is the episode: 46–73 s, 11–12 s per object.)

### Perception's layout is the only layout (v18)
Found by the orchestrator in the real-look sim (boxes 6 cm farther than the scene file, v16, fixed seed 1004): the
detector's layout was `hands` all episode; at t = 48.5 the one remaining target (a lying prism) had no plan, the look
re-planned it with the scene file's layout, got a plan, and `use_file_layout` stayed set: every later release went to the
file's slot (x 0.463, 6 cm short) and an object landed on the counter in front of its box. The fallback dated from
when a perceived layout could be garbage; since the detector reports `scene` or `nominal` (= the file's bins) unless
the hands fit, it only ever replaced a measured layout.
v18: the look plans with `detector.layout()` and nothing else; no re-plan, no latch, and a view move without a safe way
is replaced by "look from here", never by the file's layout.
Mock (`far_bins.py`: world with the boxes 6 cm farther, detector layout `hands` + 6 cm, real planner, one colour
refused whenever the planner holds the perceived layout, real-time pace):

| Case | v17 tree | v18 tree |
|---|---|---|
| every target has a plan | 5/5, releases at x 0.493 / 0.503 | the same |
| blue has no plan in the perceived layout | source turns to `scene`, both blue objects released at x 0.463 (near wall of the real boxes at 0.46) | blue stays on the counter, 3/3 others, releases at 0.493 / 0.503 only |
| red has no plan | all releases at 0.463, a yellow cube on the counter in front of its box (x 0.459) | red stays, 3/3 others |
| a view move finds no safe way | the same latch, the same yellow cube | 4/5 at the limit, no file layout |

Cost: a measured layout that is grossly wrong (bins 10 cm closer and counter 6 cm higher, source `hands`) was saved by
the fallback (3/3) and now loses the episode (0/3, `done` at 34 s: every detection lies within a perceived bin's
clearance). Seen on the way, for motion: with the boxes 6 cm farther many plans fail on IK before one passes and a
failing plan is slow; 5 objects need the whole 120 s in the mock, and the real-look log shows 8.8 s between a release
and the next pick.
Mock timing caveat: other agents' benches load the Mac (load 39 at times); the fast mock charges planning at wall time,
so its episode lengths stretch under load (`robust` looked regressed until rerun alone).

### v18 per leg (through the image, 14.11, 15:43–16:45 UTC; seeds 1000–1007; four sims ran, rtf 0.38 on the stock legs)

| Leg | Sim | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Hands | Empty picks |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | a | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 143 | PASS | stiff | 0 |
| known | d | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 134 | PASS | stiff | 0 |
| unseen | a | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 142 | PASS | stiff | 0 |
| fixed, soft-hand sim | a | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 177 | PASS | soft 16/16 | 0 |

No error line; soft lift scores 0.315–0.566; 7 same-bin pairs on the soft leg, smallest gap 53 mm, all flat. Equal to
v16 on every leg. v17's 15 episodes before it was stopped: 56/56.
Void runs of this hour (my mistake, kept under `VOID*` and `runs/void/`): the frozen tree's `sims.sh` did not know
sim tag `d`, so the "d" evaluator fell back to the default sim and two evaluators drove sim `a` at once: objects swapped
mid-episode, 1/3 and 0/3, and sim `a`'s robot was left with its arms below the counter (clips from step 9 of the next
episode) until the sim was restarted. Check `AALTO_INSTANCE` in `/proc/<sim_eval pid>/environ` after every start.

### Main f6d3f15: the landing-box reason hid "hand would touch" (found by `neighbours_e2e.py`, fixed in v19)
The planner loops over the near insets; the insets whose landing box crosses the near wall come last and their text
("release would land on a bin wall") replaced the reason of a plan that had been tried. The chooser then counted a
target with a neighbour as unplannable: it neither waits nor goes as if alone. Mock, real planner, two cubes 50 / 60 /
55 mm apart and a free cylinder: 1 of 3 sorted (2 of 3 once), time limit, 0 "hand would touch" lines, 48 landing lines;
at 70 mm the cube fell through to `pinch`. Fix (117aee7, in motion's planner): a plan that was tried keeps its reason.
After it: 3/3 in all five cases, orders and finger rooms as v18. Motion now exports the reason prefixes
(`halal.motion.TOUCHING`, `LANDING`); the policy imports `TOUCHING`.

### Lying prisms and the arm (v19)
Stiff hands have a measured grasp for a lying prism (motion: `ridge_opposed` 18/23, `ridge_pinch` 15/21, `cage_lying`
5/19 and `last_resort`), but an arm can turn its fingers along the ridge only within ±45° of its natural heading. The
chooser now puts an arm whose candidates are all last resorts after the other arm. First option by ridge direction
(12 directions, 15° apart): centre bin 0 of 12 start with `cage_lying` (2 of 12 before), side bins 2 of 12 (only
their own arm reaches). `lying_prism.py`: lying prisms are picked with `ridge_opposed`, release check run, right bin.
`policy.last_kinds`, `release_check_after` and the 3-pick limit still list `lying`: the soft list is `cage_lying`
only and no number says the release check can go.

### Black and colourless detections (v19, main 2ac315e)
Detector: `black` is a colour family; a colourless blob comes back with `color=None`, confidence < 0.05.
Policy: nothing to change. Mock: colours-only sentence with a black cube between two targets 6 cm away and a black
cylinder: 3/3 coloured sorted, black untouched (it blocks: two as-if-alone picks); "black left, yellow right": both
black objects left; "orange right, everything else center": black to the centre; "all cylinders left": the black
cylinder too (literal, by the orchestrator's decision). A `color=None` detection injected next to a target and in the
open: `bin_for` was called with red / yellow / blue only, every plan call received all detections as obstacles, 3/3.

### `release_rise` and the silent target (v19)
Real-look sim, v18, boxes 6 cm farther: purple → centre objects never got a plan and nothing in the log said why:
`reaches_bin` was false for both arms, so the target had no option and the per-option "no plan for ..." line never
ran (silent since v16). v19 logs "no plan for <target>: no arm in use reaches the <bin> bin with a grasp";
`motion.release_rise` is `[0.0, 0.03, 0.06, 0.09, 0.12]` (reach engineer) and main has the `reaches_bin` stale-inset
fix. Mock at nominal: release points and pick durations unchanged.

### v19 per leg (through the image, 14.11, 16:45–18:10 UTC; seeds 1000–1007; sim `a`, instance loaded: rtf 0.33–0.38)

| Leg | Score | Objects | Ends | Clips | hz | rt_hz min | Gate | Empty picks | "hand would touch" / as if alone | Layout source per look |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | +3.88 ± 0.78 | 31/31 | done 8 | 0 | 30.0 | 125 | PASS | 0 | 3 / 1 | scene 177 of 177 |
| known | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 118 | PASS | 0 | 7 / 2 | scene 174 of 174 |
| unseen | +7.00 ± 1.41 | 28/28 | done 8 | 0 | 30.0 | 115 | PASS | 0 | 7 / 2 | scene 174 of 174 |

Grasps used: cage 70, cage_apex 42; no `cage_side`, pinch or lying grasp, no landing-box line, no "no arm reaches"
line, no error line: with `release_rise` on and perception's 12 mm `agree_within` the nominal sim behaves as v18.
Mock note for the version after v19 (main with perception's H11: on `nominal` the layout is the bins as seen): in the
mock world with the boxes really 6 cm farther the mock shows no arms, the source is `nominal`; the v19 tree then
releases at the file's x 0.463 (a cube on the counter in front of its box), the newer tree's detector puts the row at
x 0.56 and the releases at 0.508–0.518. `far_bins.py` must not shift the layout itself on such a tree (`SEEN=1`
in `far_one.py` uses the detector's own layout).
