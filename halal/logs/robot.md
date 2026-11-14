# Real robot, first runs (14.11 evening): what happened and why

Analysis only: no code, config or branch was changed. Image v18 (`b99fd5dd4a9f`, tree `e981865`), organisers' laptop, CPU,
`--backend real --task sort_fixed`, 120 s. Data: `runs/robot/export_1957/halal/` (four run folders, two with an episode).
Scripts were scratch (FK through the v18 tree for planner questions, main for perception); every number below can be
recomputed from the npz, the mp4 and `container.log`.

## Verdict

Nothing was sorted, for five separate reasons, in the order they bit:

1. **The episode starts with both hands lying on the table**, not at `ARM_HOME`. The policy cannot leave that pose: every
   plan and every move home is refused by our own clearance and keep-out checks. Run 1: no motion at all. Run 2: the left arm
   never moved, the hands were never calibrated, no look was ever taken from home.
2. **The hand fit was rejected because the hands lay inside its own 6.5 cm floor gate**, so the camera stayed `nominal` in
   every look. The coverage guard counts model surface whose depth pixels the floor gate has removed. v21's wider turn/shift
   guards accept none of the three fits.
3. **With the nominal camera the grasp point is about 3 cm off** (hand outline: 32 px), enough for all four cage grasps of
   run 2 to close beside the cube.
4. **v18 released at the scene file's bins**; the real boxes stand 11 to 13 cm farther (nominal camera). Main (H11) looks for
   the boxes, but then main's planner says no arm reaches them: its limit is a near wall at about x 0.48.
5. **The head colour image was frozen** (organisers' side): both videos are one and the same still picture for their whole
   length, while depth was live. Colours in run 2 were read from a stale picture.

Also found: white balance on the wooden table turns orange foam into "yellow" (seen in run 1, look 2), and the `safe_stop`
was the organisers' link (robot state frozen 1.2 s, then 2.8 s without a tick), not the policy.

## The four folders

| folder | what it is |
|---|---|
| `20261114_191127` | `run.json` only, policy `ws://192.168.0.254:8000`: no episode |
| `20261114_192730` | container started (3 log lines), no reset: no episode |
| `20261114_193813` | **run 1**: 3600 ticks, `time_limit`, 0 clipped steps, 354 policy calls, latency 43 ms mean / 100 max |
| `20261114_194616` | **run 2**: 1512 ticks, `safe_stop` at 53.1 s (`watchdog_stop`, `idle_s` 2.809), 0 clipped steps, latency 34 / 73 ms |

"Missed a cube, carried on, released in front of the box" and "safe_stop at 53 s" are both run 2.

## What the server believed

Common to both runs:
- **Hand class: `unknown` throughout.** No "hand is stiff/soft" line exists: the calibration close never ran (it needs the
  closed hands 8 cm above the counter; they were at -0.6 to +1.3 cm). Grasps were therefore built for `soft`, the grip
  threshold was `unknown` = 0.020.
- **Camera source: `nominal` in all 7 looks** (2 in run 1, 5 in run 2). The scene anchor is logged at DEBUG only; it cannot
  hold on this table (boxes 11 to 13 cm from the scene file's, bare wood, no mat).
- **Counter and bins** (v18, source not `hands`): bins from the scene file (near wall x 0.40, centres x 0.505, y 0 / ±0.26);
  counter height per object from `counter_from_detection` (detected top minus object height) through the nominal camera.

### The three "arms:" fits (v18 limits: rms 4 mm, spread 5 mm, coverage 0.7, inliers 0.5, turn 6 deg, shift 5 cm)

| fit | pose | px | rms | coverage of known | inliers | spread | turn | shift | failed |
|---|---|---|---|---|---|---|---|---|---|
| A run 1, t 1.6 | both hands on the table, left a fist | 1283 | 2.6 ok | **0.49** of 0.79 | **0.50** (printed; under 0.5) | **5.6 mm** | ok | ok | coverage by 0.21, inliers by a hair, spread by 0.6 mm |
| B run 2, t 1.6 | both hands on the table, open | 2182 | 2.6 ok | **0.39** of 0.92 | 0.59 ok | 2.4 ok | ok | ok | coverage by 0.31 only |
| C run 2, t 42.9 | right arm at home, left on the table | 3474 | 3.1 ok | **0.53** of 0.91 | **0.33** | 1.7 ok | **over 6 deg** | ok | coverage by 0.17, inliers by 0.17, turn (size not logged) |

The log prints pass/fail for turn and shift, not their values: the size of C's turn is unknown.

### Run 1 (19:38)

- Start pose (npz tick 0, constant for 120 s): left arm `[-0.18 0.17 -0.54 1.00 -1.19 -1.07 -0.56]`, right
  `[-0.04 -0.08 -0.03 0.69 0.36 -0.80 0.05]`, left hand a closed fist, right hand flat. FK: palm frames at pelvis z 0.009
  (left) and 0.028 (right), lowest hand points 0.003 to 0.009 (left), 0.018 to 0.021 (right). The picture shows the hands
  resting on the table.
- Look 1 (t 34.0, the first to finish planning), 9 detections, 4 outside the object area. Against the picture: yellow cube
  (+0.287,+0.029) right; purple cube (+0.329,-0.113) "blue cube" right; orange cylinder (+0.374,-0.072) "red cylinder" right;
  purple cube (+0.250,-0.056) "blue", shape None; orange cube (+0.373,+0.018) "red cube" (visible 0.36). The rest: box tape
  ("red", x 0.58 to 0.64) and two blue blobs at visible 0.00 on the right hand's side. Every position projects onto its
  object in the picture.
- Look 2 (t 42.4, hands now open, arms unmoved): orange cylinder **"yellow cylinder"** (c 1.00), orange cube **"yellow
  triangle"** (c 0.88), purple cube "blue triangle", yellow cube only a blob (visible 0.00). See "Colours".
- Refusals, all 43 "no plan" lines: 34 x "hand 0.7 to 1.6 cm above counter during to_pregrasp (need 2.0)", 4 x "hand 0.1 cm
  above a bin rim during to_pregrasp", 2 x "hand -0.5 cm above a bin rim during descend", 3 x "hand would touch the object".
  Two "no safe move" warnings (home, clear view). Arms never moved; hands opened at t 34.
- End: `time_limit`.

### Run 2 (19:46, after staff moved the robot)

- Start pose: left `[-0.11 0.04 -0.54 0.96 -1.19 -1.16 -0.52]`, right `[-0.03 0.00 -0.03 0.81 0.36 -0.95 0.05]`; lowest hand
  points z 0.004 (left) and 0.007 (right): still on the table. The left arm command never changed in the whole episode.
- Looks: start (5 detections: "blue cube" (+0.306,-0.052) visible 0.58, three blue blobs at c 0.20 to 0.49, box tape), then
  four from "home" (really: left arm on the table, right arm where its plan ended). The cube after each try:
  (+0.307,-0.056) -> (+0.256,-0.074) -> (+0.244,-0.058) -> (+0.243,-0.053): pushed 5.4 cm towards the robot by try 1,
  2 cm by try 2, 5 mm by try 3.
- Four picks, right arm, grasp `cage`, release at (+0.463,-0.223). Grip scores after the lift: 0.017; 0.020 then 0.017;
  0.014; try 4 not read. All "empty" (below 0.020), each time "no clear way home: finishing the plan", so the empty hand
  went to the bin and opened 5 to 7 cm short of the real box's near wall.
- No plan refusals in run 2; 192 clip lines are our own dry run of six refused moves home (left arm, see below).
- End: at t 49.17 (tick 1475) the measured arm joints stop changing while the command moves on (35 ticks), then one 2.81 s
  gap between ticks (t 50.33 to 53.14) and `watchdog_stop`. Policy latency max 73 ms, 1 hold tick, 0 clips: the stall is on
  the robot-state side, as the organisers say.

## Why the robot could not start (runs 1 and 2)

The real `env.reset()` does not home the arms: the episode begins wherever they are, here resting on the table. Against that
pose (v18 planner, FK on the recorded joints):

| check | needs | start pose gives |
|---|---|---|
| pick plan, `to_pregrasp` (organiser `Planner.plan`, `motion.counter_clearance`) | hand 2.0 cm above the counter on every tick, the first included | 0.7 to 1.6 cm in run 1 (logged) |
| move home / to the clear view (`transit_clearance`) | 2.0 cm | -0.8 to -0.3 cm against the scene file's counter (run 1), -0.6 (run 2) |
| calibration close (`calibration_clearance`) | 8.0 cm | -0.1 / 1.3 cm (run 1), -0.6 / 0.0 (run 2) |
| our dry run through the default `SafetyConfig` (keep-out box up to pelvis z 0.03) | no clip | all four wrist and palm frames start inside the box (z -0.006 to +0.028); a move that deepens the summed violation by 0.1 mm is clipped. Both arms together from either start pose: 0 clips. Left arm alone (right already home): 31 clips, the logged 'workspace' + 'arm_velocity' -> 'abort' |

So in run 1 everything was refused at tick 0 of `to_pregrasp`. In run 2 the right hand happened to clear the believed counter
by 1.8 cm or more, the right arm picked, and from then on the left arm had to move home alone and was clipped by our dry
run every time. The organisers' own filter clipped 0 steps in both runs.

**Where the counter height came from and whether it was wrong** (question 3). From `counter_from_detection` through the
nominal camera, per object. Back-solved from the logged clearances it was about pelvis z -0.007 (yellow cube, left arm) to
+0.004 (purple cube, right arm); the picture gives +0.002 to +0.013 for the same objects (top-face pixel on the ray at the
logged x, 46 mm foam). The scene file says +0.010. These agree within a centimetre with where FK puts the resting hands
(lowest points z 0.003 to 0.021). The height was not the problem: a hand that lies on the table has no 2 cm of clearance
under any camera. "-0.5 cm above a bin rim during descend" is the same class: the scene file's right bin (footprint from
x 0.40, y -0.155) with `min_wall_height` 0.07 against a hand descending on an object at (+0.338,-0.166), where no box stands
in reality. Fix: a planner/transit rule for leaving the start pose, not a better camera.

The organisers' `pick_test.py` already has the rule ("leaving / returning to the start pose: never lower than where it
started", `need = min(need, c0)`); their `planner.py`, which we run (`sequences: [organiser]`), does not. Main's
`halal/policy/{sorter,transit,hands}.py` are unchanged since v18: v21 behaves the same.

## The hand fit on real depth (question 2)

**Depth does return on the arm**: 0.79 to 0.92 of the visible model surface has a depth value at its pixel, black Dex3
links included. The H1 worry (hands that return nothing) did not materialise.

**Why coverage is low: the floor gate, not the arm model.** Evidence pixels must lie 6.5 cm above the counter
(`floor_clearance`); the model points do not have to, and `_returned` counts any depth at their pixel as "known". With the
hands on the table most of the visible model is under the gate, so its own depth pixels were thrown away and it can never
be "covered". Share of the visible model above the gate, recomputed from the recorded joints (nominal camera, counter z
-0.015 / -0.005 / +0.005):

| fit | visible model points | above the gate | ceiling for coverage | logged |
|---|---|---|---|---|
| A | 398 (palms only: left fist bent past `open_curl`, so no fingers for either hand) | 0.43 / 0.15 / 0.00 | at most 0.54 | 0.49 |
| B | 964 | 0.39 / 0.27 / 0.16 | at most 0.42 | 0.39 |
| C | 987 (right arm at home: all of it above; left hand 0.54 / 0.41 / 0.28) | 0.72 / 0.63 / 0.55 | at most 0.79 | 0.53 |

A and B sit on their ceilings: in those poses no fit, however good, reaches 0.7. Nothing in A and B points at a wrong
model or a wrong pose: rms 2.6 mm, turn and shift inside the old guards. The visible model there is hands only (no wrist or
forearm point is in the image), which is why A has 1283 px and a 5.6 mm spread.

**C is different and unexplained.** With the right arm at home the ceiling is about 0.7 to 0.8, yet coverage is 0.53, two
thirds of the depth around the arms is off the model (inliers 0.33) and the fit turned more than 6 deg. The hand outline
(below) puts the head error near 3 deg, so a turn above 6 deg is more likely a wrong minimum than the truth. Candidates the
log cannot separate: wrist camera mount and cables, the two arms disagreeing (one on the table, one at home), an FK offset
of the right arm. One run with `--log-images` decides it.

**v21's guards** (turn 10 deg, shift 7 cm): accept none. A fails coverage, inliers and spread; B fails coverage; C fails
coverage and inliers whatever its turn. H13 changes nothing for these runs.

**Justified change**: judge coverage (and `known`) only on model points that lie above the floor gate, the same test the
evidence pixels pass. In sim at `ARM_HOME` the hands are 18 cm up, the gate cuts nothing and every number stays as it is;
`lowhands` (hands 9 cm up) should be re-run. On this data: B would then pass every guard (about 350 covered of about 380
eligible points); A still fails spread and inliers; C still fails. That makes B-like fits (fingertips of resting hands, a
3.5 cm slab) believable, which is thin: the better order is to leave the table first and fit at home, and to log turn (deg)
and shift (mm) so the next log says how far the fit moved. Not justified by this evidence: lowering `min_coverage` or
`min_inliers` globally (C shows a bad fit that only those two guards stopped once turn goes to 10 deg).

## The video (question 4)

`real_replay` ran on both folders (RGB only). Its numbers need a correction first:

**Both mp4s are one frozen picture.** Mean change between consecutive frames: median 0.001 grey levels over 3600 and 1512
frames; frame 0 of run 1 and frame 0 of run 2 are bit-identical after decoding, 8 minutes apart, with the robot moved in
between. The picture shows run 1's scene and the left hand as a fist (run 1's start); it never shows the hands opening at
t 34 of run 1, nor the right arm's four picks of run 2. The runner writes the mp4 from the same `obs["head_rgb"]` the policy
receives (`Recorder.add`), and the head RGB-D client hands out the newest message without an age check, so the policy saw
this picture too. Depth was live: run 2's detections differ from run 1's and follow the pushed cube, and run 2's first
"blue cube" at (+0.306,-0.052) projects onto bare table in the picture. So colour and depth came apart on the organisers'
side (colour stream of the D435 stalled, depth kept running): report it to them.

Consequences:
- **Run 2's colours are not evidence.** "blue cube" was named from pixels of the stale picture around the cube's place (the
  two purple cubes of run 1 lie there). What colour the cube was is unknown.
- **`real_replay`'s outline numbers are valid only for run 1 before the hands opened** (ticks 0 to about 1000): residual
  4.5 px, 0.75 on dark, best shift (+4..8, -20) -> 3.8 px. Later rows and all of run 2 compare moving joints with a still.
- Its blob count (`blue 480, yellow 481, red 1`) is not usable for orange: after its white balance the wooden table itself
  is in the red basin (hue 28, chroma 0.32) and swallows the orange objects into one oversized blob.

**Head calibration error at the hands** (my own pass, per hand, 2 px steps, run 1 tick 0): left hand as projected 0.91 on
dark, best shift **(-2, -32) px** -> 0.96, and the zoomed overlay shows the model sitting 30 px below the real fist. That is
3.0 deg, 2.4 cm at the hand (0.45 m), about 3 cm along the table at the objects; the real hand appears farther than the
nominal camera predicts, so objects are really nearer the robot than believed. It matches realdata B5 to B8 (30 to 90 px).
The right hand needs (+60, -4) or more (search limit, hand cut by the image edge) and does not fit the same shift: either
the right arm stood elsewhere when the picture froze, or its FK is off. One stale frame cannot say. Caveat for the left hand
too: the capture time of the picture is unknown.

**Colours** (classifier of main on the picture's objects, families red = orange, blue = purple):

| white balance | yellow cube | orange cube | orange cylinder | purple cubes | box tape |
|---|---|---|---|---|---|
| none | 52.8 yellow | 25.8 red | 26.4 red | 217 to 219 blue | 14 to 16 red |
| on the bare counter, as the detector does (wood: RGB 154/132/104, hue 34, chroma 0.32; gains 0.84 / 0.99 / 1.25) | 64.4 yellow | 30.4 red | **43.1 yellow** | 222 blue | 15 to 18 red |

Purple and yellow are named right with 20 deg or more to spare. Orange is right on the raw picture (7 deg inside the red
basin) and wrong after white balance on the wood, which is not grey: the gains push orange 5 to 17 deg towards yellow, across
the 33 / 39 boundary. That is run 1's look 2 ("yellow cylinder", "yellow triangle" at c 1.00 and 0.88, the same pixels that
look 1 called red). The `warm_split` did not save it: the gap test passes with zero turn, so the boundary stays at 36.
Under a red/yellow instruction this is a wrong bin, -1 per orange object.

## Why the grasps missed and the release fell short (run 2)

- Camera `nominal`, error about 3 cm at the objects (above): the cage closes with the cube outside it. Try 1: one finger
  joint stopped 0.27 rad short at the end of the close (it met the cube), 0.00 to 0.02 after the lift; try 2: 0.12 rad on
  one joint, then nothing; try 3: nothing at all. The pushes (5.4 cm towards the robot, then 2 cm, then 5 mm) fit a hand
  that comes down too far forward.
- The hand also came down about 1 cm lower than commanded: lowest hand point commanded z +0.004 to +0.005, measured -0.003
  to -0.006 after the close; the measured palm was 9 to 11 mm under its command just before the close of tries 1 and 2 (the
  arm was still moving at 0.5 rad/s then, so part of this is lag). Fingertips at or under table level.
- Real empty-hand reading for the `cage` closure, three closes on nothing: 0.014 to 0.026 summed over joints [1, 3, 5],
  0.047 to 0.066 over all seven, per joint 0.006 to 0.018 rad. The `unknown` threshold 0.020 sits inside that range (try 2
  read 0.020, "held", before dropping to 0.017). The config comment expected 0.06 to 0.12 for a real empty hand.
- Release at (+0.463,-0.223): the scene file's right bin. In the picture, through the nominal camera, the near walls stand
  at x 0.51 to 0.53 (counter z 0), rim about 7 cm, the centre box spans y +0.09 to -0.13 (22 cm, gaps about 1.3 cm). Even
  with 3 cm taken off for the camera error the walls are at 0.48 to 0.50.
- **Reach, main's planner** (cube at x 0.30, y ±0.06, counter z 0, rim 0.07, from home): near wall at 0.46: plans for
  left->left, both->centre, right->right. 0.48: side bins yes, centre "release would land on a bin wall". 0.49 and beyond:
  "cannot reach" or "would land on a bin wall" for every arm and bin. So with boxes where the picture has them, main finds
  them (if the bin finder works on real depth, untested) and then refuses every pick.

## Fixes for tomorrow, ranked

1. **Leave the table first.** At episode start, if the hands are lower than the transit clearance, the first motion is a
   move of BOTH arms to home, judged by "never lower than where it started" (`min(needed, clearance at tick 0)`, a few mm
   of tolerance) instead of an absolute 2 cm; the same rule for tick 0 of `to_pregrasp`. Evidence: 34 refusals of run 1,
   every "no safe move" of both runs; both recorded start poses reach home with 0 dry-run clips when the arms move together
   (lowest point within 1 mm of the start). Unlocks the calibration close, the home view, the left arm and a hand fit with
   the arms above the floor gate. Sim: none at `ARM_HOME` (18 cm up, the rule reduces to today's). Risk on the robot: a
   fist opening against the table on the way up; the move passes near x 0.2 to 0.33, the objects' near edge.
2. **Get the boxes within reach, and ask where they will stand.** Reach ends at a near wall of about x 0.48; tonight's
   stood at 0.48 to 0.53. Either the staff place the robot 3 to 5 cm closer (ask; it is their set-up step), or motion finds
   more reach (higher release, tilt, a near-wall inset under 3 cm). Evidence: the short release of run 2, the reach scan.
   Cost: reach work is sim-testable with `--shift` scenes; without it v21 sorts nothing even after fix 1.
3. **Ask the organisers for a live colour stream and one `--log-images` run.** The frozen picture makes colour meaningless
   and hid everything the arms did. On our side, cheap: log an ERROR when `head_rgb` is byte-identical to the previous
   look's while the arms have moved (log only; a static sim scene without DR can repeat frames, so act on nothing).
4. **No white balance on a counter that is not grey**: gains of ones when the counter's median chroma is above about 0.15
   (wood 0.32). Evidence: orange cylinder 26.4 -> 43.1 deg, run 1's look 2. Sim: check the mat's chroma first; a grey mat
   keeps today's gains. Without it orange goes to the yellow bin.
5. **Hand fit: coverage and `known` over model points above the floor gate; log turn and shift values.** Evidence: A and B
   sit on ceilings of 0.54 and 0.42. Sim: no change at home; re-run `lowhands` and the calibration bench. Do it together
   with fix 1, so the fit normally happens at home. It will not rescue C: if fits at home still fail tomorrow, the
   `--log-images` run says why.
6. **A camera correction that needs no depth guard to pass**: the hand outline in RGB gave 32 px at the left hand from one
   frame. The parked "camera pose from the arm outline" idea, or at least a pitch-only correction from it, is the fallback
   when the hand fit is rejected. Cost: new code the night before; sim cannot show the real error, only that it does no harm
   (expect about 0 px there).
7. **Grip threshold without calibration**: a real empty hand reads 0.014 to 0.026; `unknown` 0.020 cuts through it. With
   fix 1 the close runs and the class becomes `soft` (0.15), which is the real answer; if `unknown` stays reachable, raise it
   above 0.03. Sim: `unknown` is not reached there (the close always runs).
8. **Descent height on the real arm**: fingertips ended 1 cm under their command, at table level. After fixes 1 and 5 see
   whether it persists (settle before the close, or 5 to 10 mm more fingertip clearance on a `soft` hand). Lowest priority:
   it did not decide any of tonight's misses by itself.

Not worth doing on this evidence: widening the turn/shift guards further (H13 already did nothing here and C suggests the
turn guard was right), lowering `min_coverage` / `min_inliers`, a counter-height margin in the planner.
