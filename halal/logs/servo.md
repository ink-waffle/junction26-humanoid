# Grasp feedback from a hover look (workstream `servo`, Sat 14.11 night)

Why: on the robot (`robot.md`) the per-episode hand fit was rejected, the nominal head camera was ~3 cm off at the objects and
every cube grasp closed beside its cube. A pick is open-loop: one look from home, then ~9 s blind. Near the hand the head pose
error mostly cancels in the hand -> object vector (3 deg + 2 cm head error: object 23.5 mm off, hand -> object vector 7.4 mm at
10 cm, 3.7 mm at 5 cm; `team/notes/servo/occlusion.py`). The hand hides the top face from above (0-98 % at the 7 cm
pregrasp, 91-98 % at the grasp), so the look is taken from beside the object.

## What it does (`policy.grasp_feedback`, env `HALAL_GRASP_FEEDBACK`)

- `open`: today's pick. Hooks inert; the action stream is bit-identical to main 126bf20 (deterministic mock, jobs inline,
  seeds 3 / 5 / 7: equal hashes).
- `measure`: before the descent the hand goes to a hover pose (grasp point + 7 cm up + 5 cm towards the arm's own side,
  `motion.hover`), holds 0.3 s (longer while arm_dq > 0.1 rad/s), looks once (`perception.hover`: the look's own arm fit if
  believed, a translation-only fit of the picking wrist and hand, the top face re-fitted), logs the correction and grasps the
  open-loop target, planned from the hover.
- `servo`: the same, then grasps at target + correction (object in the arm frame + commanded minus measured grasp point at the
  hover, capped 4 cm / 2 cm in z), with a second look from a re-centred hover when the first step is >= 3 mm. Any refusal
  (guard, cap, no plan onto the corrected target) falls back to the open-loop target from the hover.

## Offline

- Hover (`ws/servo-hover`, `grasp.md`): own-side arm 100 % get a hover, cross-body about 60 % (the move from home dips
  under the over-object height); hidden share max 7.4 %, lowest hand point 8.6 cm (gate 7.5); re-plan from the hover 100 %.
- Measurement (`ws/servo-measure`), realsim_looks still frames (hand ~12 cm from the object), xy error mm median / p95 / max:
  head error 0: open 13.5 / 13.5 / 24.1, hover 2.0 / 4.8 / 5.3; -7.4 deg 4 cm: open 28.3 / 40.0 / 41.4, hover 3.3 / 6.1 / 6.9;
  59 of 60 believed at every error. Translation-only (no arm fit at the hover): 2.7 mm at 0, 11-16 mm under 3-7.4 deg.

## Stock sim, fixed leg, seeds 1000-1003, DR train (images through the proxy)

| Leg | Image | Objects | Score | Clips | rt_hz min | Gate | Hover |
|---|---|---|---|---|---|---|---|
| s2-measure | servo2-measure 9b51270551d6 (d7ec748) | 15/15 | +3.75 | 0 | 140.6 | PASS | 15 hovered, 1 look refused (inliers 0.46), all held |
| s2-servo | servo2-servo 5cab02c49bc1 (d7ec748) | 15/15 | +3.75 | 0 | 157.7 | PASS | 11 corrected and held, 1 refused -> open loop from the hover, held; 1 cross-body no hover |
| s3-measure | servo3-measure 69925dc52e42 (32024af) | 15/15 | +3.75 | 0 | 139.0 | PASS | 14 hovered, 8 believable, 3 refused (rms), all held; 1 no hover |
| s3-servo | servo3-servo 34443d4b8b17 (32024af) | 15/15 | +3.75 | 0 | 169.6 | PASS | 16 looks applied, 3 refused (rms: the cylinder misread among them), 5 on visible; all held; 1 no hover |

- Believable corrections median 12-15 mm, max 21 mm. Of that, the arm hangs under its command at the hover: tracking
  (commanded minus measured grasp point) about (+3, +1, +7) mm; the object in the arm frame differs from the scene camera by
  3-5 mm.
- Two looks misread by 10-18 mm (a cylinder on the left arm): hand-fit rms 3.3-3.6 mm against 1.5-2.1 on every good look;
  servo2 applied it, the cylinder still held. servo3 (32024af) believes a look only below 3 mm rms.
- Second look: step under 2 mm when the top face is seen; 3 of 8 second looks saw too little of it (visible 0.10-0.38).
- Cost per hovered pick: about 2.5 s (measure), 4-5 s (servo with its second look).

## Real-look sim, robot-night variant (`realsim.md`: -7.4 / -1.9 / -0.4 deg, -4 cm; all features, soft hands, resting start), fixed, seeds 1000-1003

Dev-only images from servo3 (32024af): `-nofit` = `perception.topface.arm_attempts: 0` (no home hand fit: nominal camera all
episode, as on the robot), `-trans` = also `perception.hover.arm_fit: false` (translation-only hand shift at the hover).

| Leg | Image | Objects | Score | Ends | Clips | Gate | Servo |
|---|---|---|---|---|---|---|---|
| r3-open-nofit | servo3-open-nofit | 1/15 | +0.25 | 4 time limits | 0 | PASS | - |
| r3-servo-nofit-trans | servo3-servo-nofit-trans | 2/15 | +0.50 | 4 time limits | 0 | PASS | every look refused (rms 3.2-4.1 mm), open loop from the hover |
| r3-servo | servo3-servo (home fit on) | 15/15 | +3.75 | 4 done | 0 | PASS | camera already fixed at home: does not hurt |
| r3-servo-nofit | servo3-servo-nofit (hover arm fit allowed) | 3/15 | +0.75 | 4 time limits | 0 | PASS | first 20 looks: 6 applied, 6 "too large", 6 "visible", 2 rms |

- Translation-only under a 7.4 deg error: hand shifts 3-34 mm, inconsistent between the arms for the same object (e.g. x -31 /
  +11 mm), rms 3.2-4.1 mm. A rotation over the hand patch is no shift; the rms guard refused all, so servo ran open-loop
  (same misses, ~3 s more per pick). Several looks also failed `visible`: a hover planned beside a believed object 3-6 cm
  off can stand over the real one.

- servo-nofit, old caps: the hover's own 6-DOF arm fit is believed in the sim (rms 1.7-2.9 mm, inliers 0.75-0.85) and its turn
  moves the target (-15..-39, +3..+23, -8..-23) mm, the true robot6 offset at the objects. The 4 cm total cap refused 6 of
  20 such looks ("too large"); 6 more missed the top face because the hover, planned beside the believed object, stood over
  the real one. 8f70397: the arm-fit turn is capped on its own (`servo.max_turn_correction` 8 cm). e944ede: a first look that
  only missed the top face, with its arm fit believed, hovers again beside the turned target.
