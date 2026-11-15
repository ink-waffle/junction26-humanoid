# Arm sag on the real robot (15.11 night): measurement, model, compensation

Data: robot run `halal_20261115_023102` (servo5-servo, 120 s, 3600 ticks), the 14.11 runs (`runs/robot/export_1957`, run 2
has a free right arm), drop 3 teleop (15 episodes, `observation.state` vs `action`), and sim episodes (v27 fixed ep 0 and 2,
rs19 soft-hand sim ep 0). Scratch scripts only for the analysis; FK and gravity are pinocchio on our URDF.

## Verdict

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
