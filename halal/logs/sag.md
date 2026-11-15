# Arm sag on the real robot (15.11 night): measurement, model, compensation

Data: robot run `halal_20261115_023102` (servo5-servo, 120 s, 3600 ticks), the 14.11 runs (`runs/robot/export_1957`, run 2
has a free right arm), drop 3 teleop (15 episodes, `observation.state` vs `action`), and sim episodes (v27 fixed ep 0 and 2,
rs19 soft-hand sim ep 0). Scratch scripts only for the analysis; FK and gravity are pinocchio on our URDF.

## Verdict

Real-arm sim (ARM_GAINS), seeds 1000-1003, objects in their box:

| boxes | v28 | v29dev (static 0.78 / moving 0.82) | v31 (0.84 / 0.88) | v31 + release tilt 20° + reach (tilt20rdev) |
|---|---|---|---|---|
| scene file | 12/13 (+11), releases at rim height | 10/13 (+9), two lying prisms refused | 13/13 (+13), releases +45 mm above the rim | not run: offline its plans equal v31's there, nothing tilted |
| +6 cm | 7/13 (+7), releases 11 mm under the rim, wall hits | 0/7 (2 seeds, all refused) | 3/13 (+3), small-inset releases land on the near wall | 10/13 (+10), 1 time limit, 0 clips (section at the end) |


- The real arm sinks under gravity: the runner drives it with a PD and no feedforward. At home the grasp point sits 28-34 mm
  under its command, at the hover 45-52 mm, at the release 165-215 mm.
- The steady error is `s * e = c * tau_g(q)` with `s = min(1, 0.12 / max|e|)`. Without saturation the model predicts the
  hand sag within 1-3 mm at home and at hovers (table below).
- Compensation (built, `motion.sag`): every arm command is raised by the sag in excess of the sim arm's, scaled per arm by a
  `kappa` fitted on the episode's own still ticks. Replaying tonight's run gives kappa 0.95 (left) and 1.06 (right) after
  8 s. Predicted residual after compensation: 0.8-2.9 mm at home and 2.6-3.6 mm at hovers. Sim episodes fit 0.002-0.003
  (dead band 0.3), so the sim is an exact no-op: mock action streams are bit-identical on seeds 3, 5 and 7.
- **The far release cannot be fixed by compensation.** The runner clips the PD target to 0.12 rad of the measured joint,
  which caps the static torque at 9.6 Nm (shoulder, elbow) and 4.8 Nm (wrist). Damping costs a further 3 Nm at 1 rad/s.
  The shoulder pitch needs 6.5 Nm at x 0.44 and 8.5 Nm at x 0.52 (URDF), times 1.15-1.25 on the real arm. The hand sinks
  during the carry, from x ≈ 0.42-0.44 on, and is at the bottom 1.3-2.3 s before the command reaches the release pose.
  Fix in planning (re-time far segments, refuse releases over the budget) or in the runner (velocity limit, feedforward).

## The runner's arm controller

Our runner arms the upstream `G1_29_ArmController` (third_party xr_teleoperate) and calls `ctrl_dual_arm(q, zeros(14))`:
- PD per joint: kp 80, kd 3 on shoulder and elbow; kp 40, kd 1.5 on the wrists. No gravity torque is sent (the teleop app
  sends `pin.rnea` gravity as `tauff`; our runner sends zeros).
- At 250 Hz the target is clipped to `current_q + delta / max(1, max|delta| / (30 rad/s * 4 ms))`: the WHOLE 14-vector is
  scaled so that its largest joint error is at most 0.12 rad. So one saturated joint shrinks the effective error of every
  other joint, on both arms.
- Evidence for the clip: at t = 46 (right arm at the release, max|e| 0.288) the left arm at home sags 2.2-2.5 times its usual
  error. Fitted from that coupling (home arm, shoulder pitch and elbow, 99 and 177 saturated ticks): 0.111-0.125 rad.

## Fit (still ticks: command constant 0.5 s, hand clear of the counter)

| data | model | shoulder pitch R² | rms residual (rad, 7 joints) |
|---|---|---|---|
| tonight, left / right | raw `e = c * tau_g` | 0.43 / 0.29 | 0.005-0.019 |
| tonight, left / right | `s * e = c * tau_g` | 0.81 / 0.67 | 0.002-0.008 |
| both nights, pooled, unsaturated (5026 ticks) | `s * e = c * tau_g` | 0.46 (pooled over different poses) | 0.003-0.006 |

`alpha = c * kp` (real torque over URDF gravity), pooled: shoulder pitch 1.01, roll 1.53, yaw 1.39, elbow 1.33, wrist pitch
1.12, wrist yaw 1.04 (wrist roll has no gravity torque to fit). Per hold, shoulder pitch: home 0.91-1.17 (friction, the
arm comes home from different sides), hovers 1.26-1.38, saturated releases at least 1.15-1.23. A constant offset is not
needed: the home error of -0.05 rad is gravity (tau/kp = -0.040).

Drop 3 teleop: still errors 5-14 mrad, R² against `tau_g` at most 0.17. The teleop app sends gravity feedforward, which
confirms both that the sag is gravity and that the URDF gravity is good to about 1 Nm.

Sim: arm actuators have stiffness 300 / 400 / 400 (shoulder / elbow / wrist, `unitree_sim_isaaclab/robots/unitree.py`),
alpha_sim 1.05-1.12, R² 0.8-0.99. **The sim hand sags too**: 6 mm at home and 15-20 mm at releases (x 0.47-0.53).

## Validation (a): the committed model on the robot holds

Grasp-point sag (mm, z) of the measured pose, from the effective error. "After" is the predicted distance from the sim-like
target once compensated: `q + (measured - predicted)` effective sag. kappa (0.95, 1.06).

| hold | x | measured z | predicted z | sim arm z | after compensation |
|---|---|---|---|---|---|
| home, 19 holds over both arms | 0.30 | 28.4-34.0 | 29.3-32.4 | 6.2-6.3 | 0.8-2.9 mm |
| hover, left t 17.1 | 0.37 | 51.7 | 48.8 | 11.6 | 3.6 mm |
| hover, right t 38.8 | 0.33 | 44.8 | 45.1 | 9.5 | 3.0 mm |
| low hover, t 61.5 / 95.4 | 0.23-0.27 | 25.1-30.3 | 23.5-27.7 | 4.8-5.0 | 2.6-2.7 mm |
| bottom of a descent, t 41.8 / 99.4 | 0.23-0.33 | 12.7-20.2 | 28.0-44.2 | | contact (hand on the object): excluded from the fit |
| release (saturated), t 24.1 / 46.4 / 105 | 0.45-0.53 | | | 16-18 | not holdable (next section) |

## The release: torque cap and sink during the carry

Shoulder pitch gravity torque (URDF, top-down IK at y ±0.24) and the share of the cap it uses with alpha 1.2:
x 0.36: 4.2-4.8 Nm (0.6); 0.44: 6.4-6.8 (0.8); 0.48: 7.5-7.7 (0.95); 0.50: 8.1-8.2 (1.02); 0.52: 8.3-8.5 (1.06).
The centre box needs about 0.3 Nm more, and the height (z 0.12-0.28) changes the torque by ±0.3 Nm only.

Tonight's three carries. The measured hand rises to its peak, sinks while the command moves on, and the command arrives later:

| carry | peak of measured hand | -2 cm | -5 cm | -10 cm | when the command reaches the release |
|---|---|---|---|---|---|
| left, t 21.9 | z 0.152 at x 0.40 | +233 ms (x 0.433) | +467 ms (x 0.451) | not reached | t 23.5: hand 171 mm under it, z 0.094 |
| right, t 43.2 | z 0.174 at x 0.34 | +667 ms (x 0.431) | +867 ms (x 0.444) | +1133 ms (x 0.447) | t 45.5: 178 mm under, z 0.031 (the near wall) |
| left, t 102.5 | z 0.130 at x 0.436 | +100 ms | +267 ms | not reached | t 104.0: 165 mm under, z 0.063 |

The measured x stops at 0.44-0.45 while the command goes on to 0.52, and max|e| grows to 0.33-0.41. With dq_des = 0 the
motor law is `kp * s * (q_des - q) - kd * dq`. At 1 rad/s the shoulder spends 3 of its 9.6 Nm on damping and holds about
6.6 Nm of gravity, which runs out at x ≈ 0.42-0.44. An "open on arrival" release would not help: the hand is already down.

## Design (built: 7a, 7b)

- `halal/motion/sag.py`, `Sag`:
  - Per arm, `kappa` = least squares of the excess effective error over the reference (sim) compliance, regressed on the
    excess compliance (`real - reference`) times `tau_g`.
  - A sample is one observation per reply: the arm's sent command unchanged for 18 ticks, the measured joints still since
    the last sample (2 mrad; the sim's `arm_dq` is noise at 0.05-0.09 rad/s, so position is used, not velocity), and the
    hand 4 cm above the counter and 3 cm in front of the boxes' near inner wall.
  - Prior: 0 until 4 samples are in. A fit under the dead band 0.3 counts as 0; the fit is clamped to 1.5.
  - The estimate is committed at every look start, so a look's plans and the commands that run them use the same kappa.
  - Reset every episode.
- `CommandStream` is the one place: plan-space rows go in, sent rows `plan + offset` come out.
  - The offset moves toward `kappa * (c_real - c_ref) * tau_g(plan)`, at most 0.5 rad/s and only as far as the plan's own
    step leaves room under 0.98 rad/s per joint. No sent row is faster than 0.98 rad/s.
  - Rows are clamped into the filter's joint margins.
  - Holds repeat the last sent row. `arm_cmd` stays plan-space, for planning and for the servo's tracking offset. The hand
    fit reads measured joints and is unaffected.
- `PickPlanner.dry_run_safety` runs the organiser's dry run on the same transform from the stream's current offset.
  Mock, soft-arm mode, seeds 3, 5 and 7: every moving trajectory the stream loaded matched a dry run's sent rows (44 loads,
  one off by 3.5e-7 rad: `_assembled` starts at the first row). The only loads without a dry run were `_settle` holds.
- Keep-out boxes: the filter now checks a command above the real hand (that was already so). Our own clearance checks run
  in plan space, which is where the compensated hand will be, give or take the sim's own sag.

## Validation (b, c): mock

Deterministic in-process mock (render mock world, jobs inline, 120 s), sha256 of the action stream:

| seed | stiff mock, main 98003f4 | stiff mock, ws/sag | mock arm that sags like the real one (`c_real`), compensation on | the same, compensation off |
|---|---|---|---|---|
| 3 | 94e5e360a70638e7, 5/5, 0 clips | identical | kappa 1.00 / 1.00, 5/5, 0 clips | 5/5, 0 clips |
| 5 | 0a5ced74f39885be, 5/5, 0 clips | identical | kappa 1.00 / 1.00, 5/5, 0 clips | 5/5, 0 clips |
| 7 | 264852e6442d1e67, 4/5, 0 clips | identical | kappa 1.00 / 1.00, 4/5, 0 clips | 4/5, 0 clips |

Held grasp point against the sim-like target, seed 3, soft mock: 0.4 mm median with compensation (max 30 mm: the holds
before the first commit) against 27.5 mm median and 54 mm max without. The mock grasps attach within 4 cm, so its
outcomes cannot show the difference.

## Torque budget in planning (built: 7c)

`motion.sag.budget`, in shares of the 0.12 rad clip, at the estimated compliance `c_ref + kappa * (c_real - c_ref)`. At
kappa 0 it does nothing (mock streams bit-identical, seeds 3/5/7).
- **Static** (0.78): the arm must stand still at the grasp and at the release (the arm pose of the `close`, `squeeze` and
  `open` phases) and at the hover. Over the budget, the plan is refused ("arm cannot hold still during open (sag uses
  0.80 of the clip)"). The reach question (`_reaches`, so also the policy's arm filter and the rim fallback) asks the same
  of its IK pose with a closed hand. Tonight's arms stood at the cap at 0.83-0.86 of the clip in these units: right t 46.5
  0.81 × kappa 1.06, left t 24.5 0.87 × 0.95.
- **Moving** (0.82): along every arm move (every phase of a pick plan, and every `go_to`: view moves, hover moves) the
  signed `c * tau_g + kd / kp * dq` must stay under the budget. Moving with gravity frees room. Ticks are stretched
  until they fit (`Sag.timed`, linear resampling at 30 Hz, phases rebuilt and dry-run again). Each slowed plan logs one
  line ("left arm plan slowed: carry +0.6 s, retreat +2.1 s").
- The descent to the grasp at x 0.31 uses 0.58 of the clip at rest and moves with gravity: never slowed. Lift: 0.58, not
  slowed. Carry to a box at the scene file's distance: +0.1-0.6 s. Retreat from it (straight up 8 cm at x 0.47, ending at
  0.81): +0.5-2.1 s.

Reach with the box row moved away (reach bench: spots (0.30, ±0.10) and (0.33, ±0.02), every bin, arm and slot; soft hand):

| boxes | kappa 0 cubes | kappa 1 cubes | kappa 1 cylinders (rim insets) | pick duration kappa 0 -> 1 |
|---|---|---|---|---|
| scene file | 16/16, inset 6 | 16/16, insets 3-6 | 16/16 | 8.58 -> 9.74 s |
| +3 cm | 16/16 | 2/16 (inset 3) | 10/16 (insets 0-3) | 9.22 -> 10.9 s |
| +6 cm | 16/16 | 0/16 | 2/16 (inset 0) | |
| +8 cm | 16/16 | 0/16 | 0/16 | |
| +10 cm | 8/16 | 0/16 | 0/16 | |

Tonight's boxes stood about 6 cm beyond the scene file (releases at x 0.51-0.53 with inset 6 cm). With the budget the
real arm refuses every release there rather than sinking onto the near wall. What can still win them: boxes 3-6 cm nearer
(staff), the runner's `velocity_limit` raised (the clip grows with it) or gravity feedforward, or a static budget closer to
the cap (0.83-0.86) if a 5-10 % deficit at the release is acceptable.

Soft-arm mock (seeds 3/5/7): 0 clips, outcomes unchanged (5/5, 5/5, 4/5), episodes 4-10 s longer. The sent rows of every
moving load match a planner dry run within 6.7e-7 rad: `_assembled` dry-runs from its first row instead of the start.

## Real-arm sim (d): `ARM_GAINS` (dev-only, next to `HAND_GAINS`)

`ARM_GAINS="kp,kd,kp_wrist,kd_wrist[,limit,limit_wrist]"` replaces the arm actuators of the upstream robot config
(`halal/ops/soft_hand_sim.py`, composes with `REAL_SIM`). The legs below use `71,3,40,1.5,8.5,4.8`:
- The sim's USD masses give alpha_sim 1.07 against the real 1.2 (far shoulder). So kp 80 * 1.07 / 1.2 = 71 reproduces
  the real steady errors: home shoulder pitch 3.42 Nm / 71 = 0.048 rad (real 0.047), hover 0.093 (real 0.098).
- The torque limit 0.12 * 71 = 8.5 Nm stands in for the 0.12 rad clip. It is per joint, not scaled over the 14-vector,
  and it also caps the damping force, so the sim arm is somewhat stronger while moving than the real one.
- How well it reproduces the robot (v28, robot6edge24 with the camera error, seed 1000, before the leg was stopped):
  - At home the steady error is [-0.049 0.021 0.017 -0.048 -0.001 -0.029 0.006] rad; the robot's is
    [-0.047 0.03 0.021 -0.049 0.011 -0.014 0.002].
  - In six releases the hand ends 127-161 mm under its command (robot 165-184), its grasp point 21-35 mm under the box
    rim: tonight's failure.
- Sim artefact: when still, the soft-arm sim reports the elbow's `arm_dq` at 0.105 rad/s (stock sim 0.086, robot 0.018 at
  the max joint). That is over the detector's still gate of 0.1, so no hand fit ever runs and the camera stays nominal.
  The A/B legs therefore run without the `camera` feature (`lying,colours,boxes,table,offsets`).

### Legs: boxes at the scene file (robot6edge24 without `camera`, seeds 1000-1003, soft hands, real words)

| | v28 (75cc8a0a8e34) | v29dev (72e14afea99c, ws/sag a004ec7) |
|---|---|---|
| objects | 12/13 (+11), 0 clips | 10/13 (+9), 0 clips |
| release: hand under plan | 67 mm median (max 76) | 22 mm (max 28); the sim arm's own sag is 16-20 mm |
| release: grasp point above the box rim | +0 mm, at rim height | +41 mm median (compensated releases +40..+47) |
| pick plan, mean | 10.5 s | 11.7 s |
| episode, sim time | 107-211 s | 142-315 s |

- kappa 0.94-1.01 per arm. It is committed after the first pick (t 26-33 s): the arms rest on the counter until t 12-17 s,
  so there are no still samples at home before the first look. Each episode's first release is uncompensated.
- Slowed plans: 14 picks. Retreat +20.4 s in total (straight up 8 cm at x ≈ 0.46, about 0.81 of the clip), carry
  +2.1 s, to_pregrasp +3.5 s. 37 plans refused at 0.78-0.85.
- Lost to the budget: two lying prisms. v28 sorted them with `ridge_opposed` (right arm, centre box, x 0.463). v29dev
  refused the ridge grasps there (0.84-0.85) and missed three times each with `cage_lying`. This sim arm holds those
  poses; the robot's stood at its cap at 0.83-0.86.

## After the first A/B (v31 candidates: 7be24ff, ebdcc39, 5b56b76)

- `arm_tau_ff` (7be24ff): every reply carries the URDF gravity torque at its plan-space rows. The model is
  g1_body29_hand14 with the Dex3 hands, the same file teleop's IK uses. A runner that applies it removes the sag, kappa
  falls into the dead band and the angle offset switches itself off.
  - Mock q streams unchanged; the key is in every reply.
  - `deploy.replay` passes on synthetic frames and on the robot npz. Cost 0.025 ms per 15-row reply.
- Budgets (ebdcc39, 5b56b76): static 0.84 (only the clearly impossible are refused), moving 0.88. A moving budget under
  the static one makes the last centimetres to a near-cap release crawl at the floor speed.
  - Reach bench, kappa 1, cubes, boxes +0 / +3 / +6 cm. Plans: 16/16, 16/16, 4/16 (cylinders 16/16, 16/16, 12/16).
  - Mean pick, by moving budget: 0.82 12.0 / 13.0 / 13.4 s; 0.86 9.5 / 10.6 / 11.7 s; 0.88 9.1 / 9.6 / 10.1 s;
    0.92 8.7 / 8.9 / 9.1 s. kappa 0: 8.6 / 9.2 / 9.3 s.
- First pick waits for the estimate (ebdcc39, `policy.sag_wait_s` 3): while the estimate committed at the last look
  start rests on fewer than `min_samples` on either arm, the pick settles and looks again. There is no wait from the
  start view, where resting hands give no samples.
  - Mock: no wait, streams unchanged. Soft-arm mock: one wait of 0.7 s, and the first pick is compensated.

- v31 (official, 5052e5c7b334; static 0.84, moving 0.88), same seeds: **13/13 (+13)**, 0 clips, all in their boxes.
  - Releases: grasp point 24 mm under the plan (max 59), +45 mm above the rim (min +11), +61 mm past the near wall
    (min +34).
  - The first-pick wait triggered twice (7 and 9 samples per arm, under 4 at the commit before): one 0.3 s settle and
    a new look, then kappa 1.01 committed at t 14.7 / 17.0. Later in the episodes kappa drifts to 0.76-0.96.
  - 5 plans refused at 0.84-0.85. Slowed: 22 plans, retreat +15.1 s in total, to_pregrasp +3.9 s, carry +1.0 s, lift
    +0.9 s. Pick plan mean 11.8 s.

### Legs: boxes +6 cm (robot6edge24near6 = robot6edge24 with the box row at [0.06, 0], seeds 1000-1003)

- v28: 7/13 (+7), 2 time limits, 0 clips. In 24 release attempts the grasp point was 168 mm under the plan (max 189),
  -11 mm under the rim (min -34) and only +20 mm past the near wall (min +6). Objects hit the wall and were picked again;
  6 ended on the counter. This is the robot night.
- v29dev (static 0.78): 0/4 and 0/3 on seeds 1000-1001, both time limits. Every release was refused, as the reach bench
  predicts for 0.78 (cubes 0/16). The leg was stopped as moot once the budget went to 0.84 / 0.88.
- v31 (official, 5052e5c7b334 = main fd4d848 = ws/sag 5b56b76; static 0.84, moving 0.88): 3/13 (+3), 2 time limits,
  0 clips, 10 objects left on the counter, nothing in a wrong box or knocked off. (v28: 7/13, 6 on the counter.)
  - kappa committed at t 7.0-11.7 on 16-30 samples per arm, at the first look from home; the first-pick wait never
    triggered.
  - The 8 compensated releases had the grasp point 24-39 mm under the plan and +31..+58 mm above the rim, but only
    +1..+30 mm past the near inner wall (release x 0.463-0.493, wall at 0.46). Of the 8 objects released, 3 landed in
    their box; the rest fell on the near wall and then the counter.
  - 12 plans refused at 0.84-0.89; cubes went unattempted ("no arm in use reaches the left bin").
  - Slowed: retreat +23.6 s over 21 plans, carry +4.4 s, to_pregrasp +2.4 s; pick plan mean 12.0 s (v28 11.7 s).
  - One ridge prism stayed stuck in the hand at x 0.498: the release checks held the arm there for about 10 s near the
    cap, and it sank 24 cm under the plan.

### Landing rule and the forward exit

- The landing box `limit` (objects leave the hand 6-117 mm forward) was measured with the arm stretched to its
  kinematic reach limit (reach.md: forward drift there, median +25..+58 mm; none at +6 / +8 cm away from the limit;
  cause not established).
- With the torque budget inside the reach question, the planner sees 0-2 cm as the furthest inset. `_landing_box` then
  calls those releases `limit` although the compensated arm stands well inside its kinematic reach. There the object
  drops straight down onto the near wall.
- v28 scores 7/13 at +6 cm in this sim only because its arm sinks below the rim to about +20 mm past the wall before it
  lets go. On the robot the arm sank further (x 0.453, short of the wall) and hit the wall.
- Judging `limit` by kinematic reach only would refuse those releases. At +6 cm that turns 3/13 into about 0/13: the
  wall drops score 0 and knocked nothing off, so refusing them only removes the three that went in. Not built (main's
  decision). At +6 cm no planning change buys points with this controller; the levers are box placement and `arm_tau_ff`.

## Design note (not built): a forward exit for a compensated arm at a small inset

Goal: at a small near inset (0-2 cm), the released object should leave the hand 3-6 cm forward, as it does from a
stretched arm, so that its centre clears the near wall.

**A. Tilt the hand forward for the open.** In the last ticks of the carry, rotate the hand about the pelvis y axis
around the grasp point by 20-30° (open side towards the box), then open (thumb and one finger first, as the staggered
release does). The object then slides off the opening side instead of dropping straight.
- Wrist budget: in the top-down release pose the hand's centre of mass lies almost on the wrist axes. FK at x 0.47:
  wrist pitch gravity 0.00-0.13 Nm for wrist offsets of 0-45°, i.e. 0.00-0.03 of the wrist clip (kp 40, cap 4.8 Nm).
  The worst case, the whole hand (≈0.8 kg with the fingers and a foam object) at its full 7 cm lever, is 0.55 Nm: 0.13
  of the wrist clip at alpha 1.1. The wrist is not the limit.
- Shoulder: the hand's centre moves about 5 cm × sin θ forward. At 30° that adds ≈0.2 Nm, +0.02 of the shoulder clip.
  The same FK gives 0.72-0.80 for those poses, so a release at static use 0.82 or under stays under 0.84.
- Joints: a pure world-y rotation of a top-down hand needs wrist pitch together with roll/yaw. Plan it as an IK target
  (grasp point fixed, approach axis tilted θ towards +x) on the release row, then an "open" phase from that pose.
- Risk: a tilted cage can let a cube slide out before the open (soft-hand grip at 0.3-0.45 rad past contact, unknown real
  friction). Tilt only in the last 0.2 s and only for the `limit` small insets.

**B. Open during the last centimetres of forward motion.** The object takes the hand's horizontal speed v. Over a fall of
about 10 cm (0.14 s) it moves v × 0.14 s. A 3 cm exit needs v ≥ 0.2 m/s at the moment the fingers part.
- Near the cap the moving budget allows (0.88 - 0.84) × 0.12 / 0.0375 ≈ 0.13 rad/s at the shoulder, ≈6.5 cm/s at the
  hand: only ≈1 cm of exit.
- The finger lag (90 % after 300-380 ms on the real hand) puts ±1 cm of jitter on it. Not worth it, unless the forward
  motion goes with gravity (a forward-and-down arc frees budget), which only works from a higher carry and costs
  clearance over the near wall.

**Test in the ARM_GAINS realsim.** grasp_bench-style scripted drops (privileged object poses, landing minus release point,
as in the reach sweeps), with the sag offset applied through `Sag.commands` at a fixed kappa from a home measurement:
- boxes +6 cm, insets 0 / 1 / 2 cm, tilt 0 / 20 / 30°, cube / cylinder / triangle, 8 drops each (about 1 h of sim c);
- read: binned share, forward landing offset (median, p2.5) and objects lost before the open;
- the stock-arm sim at the same points tells whether any tilt effect belongs to the soft arm.
Accept when tilt 20-30° binned ≥ 7 of 8 at inset 1 cm on all three shapes, with no loss before the open.

## Forward-exit release (`motion.release_tilt`, ws/tilt): built, measured

**Verdict.** At +6 cm without gravity feed-forward, tilt 20° with `reach` sorts 10/13 in the real-arm realsim, where v31
sorts 3/13 (same seeds 1000-1003). With the runner's feed-forward (`--arm-gravity-ff`) the arm fits kappa ~0 and the tilt
never triggers: with the recommended config the mock streams at kappa 0 are bit-identical to v31 (stiff 3/5/7). So it
is a free fallback for runs without feed-forward.
- Recommended for a hand-in that may run without feed-forward: `release_tilt: {deg: 20, max_inset: 0.04, reach: true}`,
  the rest as committed. Committed default: off (deg 0), which keeps v31 bit-identical.
- 30° loses objects in the turn once the turn is blended into the carry.
- Cost: a reach release at inset 4 cm crawls near the cap (side-box picks +4 s).

What it does (`halal/motion/tilt.py`, `PickPlanner._tilted`):
- Applies when deg > 0, the release inset is at most `max_inset`, and the arm's sag uses at least `min_use` (0.5) of the
  clip there. kappa 0 uses ~0.25, so a stiff or feed-forward arm never tilts.
- The turned pose is an IK target: grasp point fixed, approach axis turned `deg` towards +x about pelvis y. At +6 cm it
  is mostly wrist yaw (0.46 rad at 20°, 0.76 at 30°), and the hand moves back, so the static use drops by 0.05-0.07.
- The turn is blended into the carry's last `turn_s` (1.0 s, phase `tilt`, re-timed at the speed limits). The arm stays
  turned through the open and the retreat and sheds the turn on the way home.
- The landing box (`landing.tilted`), the pose error, counter and rim clearance (tilt, open, retreat, home), the static
  budget at the turned open and the sag-aware dry run all check it. A release that fails any of them stays upright (v31).
- `reach: true`: the reach question and the static budget accept a release whose turned hand holds still. Offline at
  +6 cm, kappa 1, reach bench: cube plans 4/16 -> 16/16 (centre box 0/8 -> 8/8), insets 0-2 -> 3-4 cm. Scene-file boxes:
  plans identical to v31 for every config (16/16 at inset 6 cm, nothing tilted).
- Mock (render mock, 3/5/7): flag off bit-identical (stiff 94e5e360 / 0a5ced74 / 264852e6, soft-arm 4ffc9394 / ef3cff0c /
  7503cfd6). Forced on (30°, every inset, reach): 0 clips, every moving load matches a dry run.

### Sweep: fixed insets, tilt phase standing before the open (99983b7)

Real-arm sim (ARM_GAINS 71,3,40,1.5,8.5,4.8, soft hands, REAL_SIM=boxes,offsets, boxes +6 cm, counter +1 cm), grasp_bench
with `sag: true` (kappa 1.008 measured at home), side boxes, policy candidates, 8 drops per shape and cell. Binned /
carried:

| inset | tilt 0 | tilt 20° | tilt 30° |
|---|---|---|---|
| 0 cm | 2/24 | 2/24 | 1/23 |
| 1 cm | 1/23 | 14/24 (cube 2/8, cylinder 4/8, triangle 8/8) | 16/23 (6/7, 7/8, 3/8) |
| 2 cm | 18/24 (cube 5/8, cylinder 5/8, triangle 8/8) | 24/24 | 23/24 |

- 0 clips, nothing knocked off, nothing in a wrong box, 1 of 215 lost before the open.
- In this sim the held object is 20-30 mm short of the planned release x when the hand opens, so at inset 0 its centre is
  still behind the near wall: every drop there lands ~30 mm short, on the counter, whatever the tilt.
- Landing of binned tilted drops (n 80): forward +5..+47 mm (median +18 at 20°, +21 at 30°), outward -35..+35 mm.
  Untilted: forward +3..+49 mm, median +5. The turn shifts the landing forward by ~1 cm per 10°.
- The design note's bar (≥ 7/8 per shape at inset 1 cm) is not met. At 2 cm the tilt makes the release reliable.

### Round 2: the planner's own insets, turn blended into the carry (9d1422b)

Same sim and bench. Every variant has `min_use` 0.5 and `turn_s` 1.0. `r` = `reach: true` with `max_inset` 0.04, plain
tilt has `max_inset` 0.02. Centre box: one arm at a time, each on its own-side slot (y ±0.03). Binned / cases:

| box | v31 | tilt 20° | tilt 20° + reach | tilt 30° + reach |
|---|---|---|---|---|
| side boxes | 15/24 (inset 2 cm, pick 11.9 s; cylinders 3/8, cubes 4/8) | 23/24 (inset 2 cm, 10.7 s) | 23/24 (inset 4 cm, 15.9 s) | 23/24 (inset 4 cm, 16.6 s), **9 lost before the open** |
| centre box | 1/12 (4 cubes unplannable, the rest at insets 0-1 cm onto the wall) | | 11/12 (inset 3 cm, 11.7 s) | |

- Misses with tilt 20°: two cubes not lifted (grasp misses) and one triangle that fell out of the hand in the carry. 2 of
  the 36 reach cylinders left the hand before the open and still fell into the box. 0 clips, nothing knocked off,
  nothing in a wrong box.
- 30° with the turn in the carry lets cylinders slide out before the open (8 of 8 left the hand early, but they still
  fell into the box at inset 4 cm). Not worth the risk on the real hand: 20°.
- Without reach, tilt 20° is as good in the side boxes and 5 s faster per pick. Only reach plans the centre box at +6 cm.
  The reach releases at inset 4 cm sit near the cap and crawl (side picks +4 s against v31).
- Landing box of a tilted release, 20°, binned drops of the sweep and round 2 (n 97, rest minus planned release point):
  forward -13..+46 mm, outward -25..+35 mm. The low end comes from this sim's compensated arm, which still ends 20-30 mm
  short in x. `landing.tilted` keeps `limit`'s values (+6..+117 mm): the tilted check only ever falls back to the upright
  release at the same point, and that lands worse at every small inset (sweep: 1/23 against 14/24 at 1 cm), so the
  stricter measured box would only cost points.

### Policy legs: boxes +6 cm (robot6edge24near6, real-arm realsim, seeds 1000-1003)

halal/policy:tilt20rdev (3c6d8c0a7dac = 9d1422b with `deg: 20, max_inset: 0.04, reach: true`), same harness and seeds
as the v31 leg above; one sim per seed group (a: 1000-1001, b: 1002, c: 1003).

| seed | v31 | tilt20rdev |
|---|---|---|
| 1000 | 0/4, time limit | 3/4, time limit |
| 1001 | 1/3, time limit | 2/3 |
| 1002 | 2/3 | 2/3 |
| 1003 | 0/3 | 3/3 |
| total | 3/13 (+3) | **10/13 (+10)**, 0 clips, gate PASS |

- Tilted releases planned at insets 2 / 3 / 4 cm: 5 / 21 / 16 times (the plans of every look, not only those executed).
  kappa 1.01 at the first commit, drifting to 0.85-0.91.
- The seed 1002 miss is an upright release at inset 6 cm: a lying prism stuck in the hand (`ridge_opposed`), not a tilt
  case. Nothing knocked off, nothing in a wrong box.
- Scene-file boxes: no leg (main stopped the experiments once the runner's feed-forward was merged). Offline reach
  bench at kappa 1 there: every tilt config plans exactly v31's releases (16/16 per shape at inset 6 cm, nothing tilted).
