# Real head depth, drop 3 (15.11 night): does the perception believe the real robot?

Data: organisers' `zupermiez/g1_junction_drop_3` (LeRobot v3), 15 teleop episodes, 4823 frames at 30 fps. It has head RGB, real head
depth, both wrist RGBs, 28-dim state and action in our joint order, and tactile. Every number below comes from the policy's own code
(`SelfCalibration`, `HandReference`, `HandOutline`, `BinFinder`, `TopFaceDetector`, `HoverMeasurer`) fed like the policy
(`head_depth` uint16 mm, `calibration.yaml` intrinsics, measured `arm_q` / `hand_q`). Scratch scripts lived outside the repo.
Code changes on `ws/realdepth`: `4b7a6c9` (rim) and `31ad9b4` (one-arm hand fit). Everything else is analysis.

## Verdict

1. **The depth is lossless, not lossy.** The stream's x265 SEI says `lossless` (rc=cqp; the `crf 30` in `info.json` is ignored).
   The 12-bit value is mm exactly: `gray16le >> 4` equals `gray12le` bit for bit. Its noise is the raw D435's. Valid depth covers
   0.87 to 0.92 of the image.
2. **The hand fit of both arms works only on 35 % of frames on real depth, because the two arms disagree by about 2 cm.**
   - Both arms (what main ran): 173 of 488 frames believed, 40 of 79 still frames. It fails on coverage (median 0.68, limit 0.70)
     and inliers (median 0.61). rms, spread, turn and shift are fine.
   - The **left arm alone** is believed on 80 %. It gives the same correction in all 15 episodes, (-18, +7, 0) mm at the object
     band. The RGB outline confirms it independently: the left hand lands 2 px from its dark pixels under that camera, against
     (+21, -18) px under the nominal camera.
   - The **real right hand sits (-7, -17, -2) mm from its FK** in that frame: 21 mm, steady across episodes. No single camera
     fits both arms. The guards were right to refuse the compromise (robot fit C was the same case).
3. **The table is tilted 14 deg in the pelvis frame in every episode.** The counter normal is (-0.25, 0.00, 0.97). The camera
   relative to the arms is within 2.6 deg of nominal, so the robot leans forward against the table. The robot night was 3 to 6 deg
   in the same frame (back-solved below): drop 3 is a different posture. At 14 deg, the old `rim_z` modelled 17 cm boxes.
4. **Bins and detector work on real depth.**
   - The box row is found in 487 of 488 looks, with the near wall at x 0.46 (fitted camera) to 0.48 (nominal).
   - Colours and named shapes were right in every look I checked. Tops are consistent to under 1 mm (about 49 mm above the fitted
     counter).
   - The only phantoms are box tape between the boxes, outside the object area.
5. **Grip: real held readings exist now.**
   - Operator picks score 0.32 to 0.57 on our `grip_joints` [1, 3, 5] in 13 of 14 holds. One prism pinch held at **0.104**, under
     `soft` 0.15.
   - Empty closed hands read 0.03 to 0.08.
6. **Right thumb_0 keeps its sign** (URDF mirror). But it has a hard stop at about -0.50 rad on the real hand; the left reaches
   -0.97. Tactile: 150 of 216 taxels are dead, and the right hand shows nothing.

## Q1 The hand fit on real depth

### Both arms, every 10th frame (policy `hand_fit`, shifted then turned starts)

| ep | frames | fitted | believed | still (believed) | px | rms mm | spread mm | coverage | inliers | turn deg | shift mm | failed (count) | believed band correction mm (dx, dy, dz) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 50 | 12 | 18 (9) | 9081 | 3.0 | 1.9 | 0.68 | 0.61 | 2.3 | 25 | coverage 38, inliers 16 | -14, +14, -4 |
| 1 | 30 | 26 | 11 | 2 (2) | 4427 | 3.2 | 1.8 | 0.72 | 0.51 | 2.3 | 30 | coverage 8, inliers 13 | -11, +17, -1 |
| 2 | 29 | 24 | 12 | 2 (1) | 6004 | 3.0 | 1.7 | 0.71 | 0.59 | 2.2 | 26 | coverage 7, inliers 6 | -13, +18, -1 |
| 3 | 27 | 24 | 12 | 4 (3) | 5726 | 2.9 | 1.8 | 0.74 | 0.62 | 2.0 | 26 | coverage 7, inliers 7 | -15, +18, -2 |
| 4 | 36 | 30 | 9 | 10 (6) | 6074 | 2.9 | 1.7 | 0.68 | 0.61 | 1.6 | 24 | coverage 16, inliers 6 | -15, +15, -1 |
| 5 | 49 | 48 | 18 | 7 (6) | 6302 | 3.1 | 2.0 | 0.71 | 0.60 | 2.7 | 29 | coverage 23, inliers 18 | -15, +18, -2 |
| 6 | 43 | 43 | 18 | 9 (4) | 7441 | 2.9 | 1.7 | 0.68 | 0.70 | 2.2 | 28 | coverage 24, inliers 4 | -13, +16, -2 |
| 7 | 35 | 35 | 7 | 3 (1) | 8301 | 3.1 | 1.8 | 0.66 | 0.63 | 1.7 | 17 | coverage 27, inliers 4 | -18, +9, -2 |
| 8 | 25 | 25 | 2 | 2 (0) | 8449 | 3.1 | 2.0 | 0.60 | 0.54 | 1.9 | 24 | coverage 23, inliers 6 | -20, +27, -7 |
| 9 | 28 | 28 | 3 | 8 (0) | 7978 | 2.9 | 1.9 | 0.61 | 0.61 | 1.5 | 15 | coverage 25 | -26, +23, -8 |
| 10 | 22 | 22 | 4 | 1 (0) | 7794 | 2.9 | 2.0 | 0.63 | 0.64 | 2.2 | 20 | coverage 16, inliers 5 | -25, +26, -9 |
| 11 | 22 | 22 | 7 | 4 (1) | 6486 | 3.0 | 2.3 | 0.63 | 0.54 | 2.8 | 29 | coverage 15 | -26, +25, -9 |
| 12 | 34 | 34 | 16 | 5 (3) | 7099 | 3.4 | 2.1 | 0.70 | 0.55 | 2.5 | 23 | coverage 17, inliers 6 | -27, +21, -6 |
| 13 | 30 | 30 | 23 | 1 (1) | 9671 | 3.0 | 1.9 | 0.75 | 0.72 | 1.7 | 26 | coverage 7, inliers 5 | -13, +13, -4 |
| 14 | 28 | 28 | 19 | 3 (3) | 8079 | 3.1 | 1.7 | 0.72 | 0.69 | 3.2 | 21 | coverage 8, inliers 7 | -18, +14, -3 |
| all | 488 | 469 | 173 | 79 (40) | 7059 | 3.0 | 1.9 | 0.68 | 0.61 | 2.2 | 25 | coverage 261, inliers 110, turn 12, shift 12 | -15, +16, -3 (sd 9, 6, 6) |

The floor gate is not the issue here: the arms are clear of the counter (visible model share above the gate 0.85 to 1.00).

### One arm at a time (same guards, same frames)

| model | fitted | believed | rms mm | coverage | inliers | turn deg | believed band correction mm | sd mm |
|---|---|---|---|---|---|---|---|---|
| left arm | 335 | 269 (80 %) | 2.2 | 0.83 | 0.73 | 2.6 | -18, +7, 0 | 7.5, 5.5, 7 |
| right arm | 456 | 270 | 3.1 | 0.78 | 0.76 | **7.2** (108 turn rejections) | -1.5, +27, +1 | 14, 6, 6 |
| both | 469 | 173 | 3.0 | 0.68 | 0.61 | 2.2 | -15, +16, -3 | 9, 6, 6 |

- Frames where both one-arm fits are believed: left minus right = (-18, -20, -1) mm at the band (sd 14, 8, 9).
- The left-arm correction is the same in all 15 episodes: x -14 to -22, y +2 to +8 mm.

### Independent check in RGB: each black hand's best outline shift (`HandOutline`, frames with arm speed < 0.3 rad/s)

| camera | left hand: best shift px (median) | own hand within 8 / 10 px | right hand: best shift px | own hand within 8 / 10 px |
|---|---|---|---|---|
| nominal | (+21, -18), residual 4.0 px | 0.03 / 0.04 | (+44, -3), 8.9 px | 0.00 / 0.00 |
| both-arm fit | (-9, +6) | 0.19 / 0.35 | (+24, +18) | 0.01 / 0.01 |
| left-arm fit | **(+2, +2), 2.2 px** | 0.66 / 0.77 | (+38, +18) | |
| right-arm fit | (-25, -12) | | (+16, +11) | 0.11 / 0.16 |

The left-arm camera explains its own hand in RGB; the right-arm camera does not. The same shift was in drop 1, where the left hand
was (+20, -22) px under nominal (realdata.md). Robot night had the same note ("the right hand does not fit the same shift").

### Why the right arm is off

**Per link, under the both-arm camera:** share of clear visible model points with depth on them. "Nearer" and "farther" are the
shares of those points whose depth lies more than 8 mm in front of or behind the model.

| link | covered | nearer | farther | note |
|---|---|---|---|---|
| left palm / index_0 / thumb_1 / thumb_2 | 0.90 / 0.94 / 0.93 / 0.97 | <= 0.06 | <= 0.05 | the left hand fits |
| left wrist roll / pitch / yaw | 0.55 / 0.62 / 0.55 | 0.26 / 0.21 / 0.34 (17-42 mm in front) | 0 | white wrist-camera mount and cables, not in the URDF |
| left thumb_0 | 0.12 | 0.85 (26 mm in front) | 0 | hidden behind the mount |
| right palm | 0.52 | 0.09 | 0.24 | |
| right middle_0 / middle_1 | 0.19 / 0.00 | 0 | 0.21 / 0.22 (40-49 % no depth) | |
| right thumb_2 | **0.00** | 0.14 | **0.67 (119 mm behind)** | the model's right thumb lies where the table is |

**Translation of each hand onto depth** (palm and fingers only, no wrist links, through the left-arm camera; frames with
coverage > 0.6 and rms < 4 mm):

| hand | frames | depth minus FK, mm (median) | sd mm | ep 0-7 | ep 8-14 |
|---|---|---|---|---|---|
| left (control) | 218 | (+0.8, -0.2, +0.2), 1.1 mm | 1.6, 1.0, 0.9 | (0.8, -0.2, 0.1) | (0.7, -0.3, 0.3) |
| right | 105 | **(-7, -17, -2), 21 mm** | 7, 5, 9 | (-6, -21, -1) | (-7, -16, -3) |

- So the real right hand stands about 17 mm towards -y and 7 mm nearer than its FK puts it.
- With a left-consistent camera that is the error of every right-arm grasp. With today's two-arm compromise, both arms are about
  1 to 1.5 cm off.

**Joint offsets do not explain it.** A pooled 7-joint offset fit of the right arm (camera from the left arm, rigid links, 266
frames) converged the same on even and odd frames: shoulder pitch -3.6, roll -4.2, elbow -2.5 deg. It failed validation:
- the right-hand RGB outline got worse, from (+38, +18) to (+40, +35) px;
- the both-arm fit did not improve (148 believed vs 173).

It fitted the wrist mount, not the arm. Do not use it.

### The policy's procedure per episode, before and after P1 (`31ad9b4`)

Still looks are taken as the policy takes them: a primer frame 0.1 s before the look, every joint within 0.02 rad, arm speed
< 0.1 rad/s, at most one per second. Teleop gives only 0 to 5 attempts per episode; at home in the finals there are up to 12. The
error is the distance at the band from the RGB-confirmed left-arm reference (-18, +7, 0) mm.

| ep | attempts | main: camera | main: off the left reference mm | P1: camera (fits used) | P1: off the left reference mm |
|---|---|---|---|---|---|
| 0 | 3 | hands (both) | 10.0 | hands (left, both, both): both-arm medoid | 10.0 |
| 1 | 1 | hands | 14.7 | same | 14.7 |
| 2 | 1 | **nominal** | 19.2 | hands (**right** alone, outline 4 px) | 28.4 (right-arm frame) |
| 3 | 1 | hands | 11.8 | same | 11.8 |
| 4 | 4 | hands | 10.1 | same | 10.1 |
| 5 | 4 | hands | 12.7 | hands (left, left, both) | **1.7** |
| 6 | 5 | hands | 11.0 | same (medoid stays a both-arm fit) | 11.0 |
| 7 | 2 | **nominal** | 19.2 | hands (left, left) | **2.5** |
| 8 | 1 | **nominal** | 19.2 | hands (left) | **3.7** |
| 9 | 3 | **nominal** | 19.2 | hands (left, left) | **4.5** |
| 10 | 0 | nominal | 19.2 | nominal | 19.2 |
| 11 | 2 | hands | 21.5 | same | 21.5 |
| 12 | 1 | nominal | 19.2 | nominal | 19.2 |
| 13 | 1 | hands | 7.2 | same | 7.2 |
| 14 | 1 | hands | 6.9 | same | 6.9 |

- With P1, 13 of 15 episodes get a camera from the arms (main: 9).
- Ep 2 got the right arm's camera. It passed every guard (rms 2.3 mm, coverage 0.94, outline 4 px, turn 4.0 deg) and lies within
  the right arm's own spread. It is right for right-arm picks and 2.8 cm off for left-arm picks. That is the limit of any single
  camera while the arms disagree.

## The 14 deg counter (all episodes)

- **Drop 3:** counter plane tilt 13.8 to 14.9 deg (episode medians; per frame 11.5 to 15.7). It is pitch only: |normal y| < 0.045.
  The counter at the band is at pelvis z +0.035 to +0.039, plane rms 1.3 mm.
  - Through the left-arm camera the tilt is 12.6 deg; through the both-arm camera 14.4.
  - FK confirms it without the camera: the lowest hand points over many frames lie at z -0.004 near x 0.12 to 0.18, and at +0.022
    to +0.036 near x 0.22 to 0.38.
- **Robot night (14.11):** no depth was logged. I took the 5 detections of run 1's first look and put their top pixels in the
  picture on rays through the logged xy; the y came back within 2 mm of the logged values. The tops give a counter slope dz/dx of
  0.06 to 0.11, i.e. **3.4 to 6.4 deg (±3)**.
  - The pictures differ: in drop 3 the objects look about 15 % larger, the boxes fill the top of the image and no background
    shows. It is the same table with the same screw holes.
- **Lean, not table:** the camera-to-arm relation is the same (2.6 deg) in both sessions. So the extra about 9 deg lies between
  the robot body and the table: a waist, leg or stand lean.
- **What it does to the planner:**
  - Counter clearance is one z per object (correct at the object). 4 cm along x the table is ±1 cm, so far fingertips keep about
    2 mm of the 12 mm clearance.
  - `rim_z` took the highest corner of all boxes: 0.213 against a band counter of 0.037. The planner made that a 177 mm wall and
    planned releases around z 0.25. Fixed in `4b7a6c9`.
  - Ask the organisers whether the finals posture is the robot night's or drop 3's.

## Q2 Box row and rim (every 10th frame)

| camera | looks | row found | boxes found | near wall x (median, p10-p90) | centres (x, y) | rim z: old -> new | wall over band counter: old -> new mm | counter at band / at row |
|---|---|---|---|---|---|---|---|---|
| nominal | 488 | 488 | 487 | 0.480 (0.475-0.480) | (0.585, -0.242) (0.580, 0.000) (0.585, +0.244), sd 2-3 / 12-16 mm | 0.213 -> 0.173 | 177 -> 137 | 0.036 / 0.082 |
| left-arm fit | 269 | 268 | 258 | 0.460 (0.460-0.465) | (0.565, -0.235) (0.565, +0.007) (0.570, +0.249), sd 16-19 / 30-35 mm | 0.211 -> 0.175 | 171 -> 136 | 0.040 / 0.077 |
| both-arm fit | 173 | 173 | 162 | 0.460 (0.450-0.465) | (0.570, -0.225) (0.565, +0.018) (0.565, +0.263) | 0.217 (old) | | 0.037 / 0.080 |

- The near wall is 0.46 to 0.48, as on the robot night (0.48 to 0.53 nominal), right at the cube reach limit (about 0.46 to 0.48).
- The rim fix (`4b7a6c9`) takes the rim over each box's entry strip: the near wall and 6 cm past it, full width.
  - Sim: the rim changes by 1.0 mm median and 4 mm max (dr_train, lowhands, realsim_looks).
  - calibration_bench rim vs truth over the same strip: +8.0 mm (worst +5.8); under the 7 deg `real` shift +6.9 mm (worst +4.3).
  - perception_bench detections are unchanged.
- The remaining 137 mm is real: at 14 deg the entry rims stand that high over the band counter.

## Q3 Detector on real depth and RGB

Run on every 30th frame of every episode, with the episode's left-arm camera forced: 159 looks, 683 detections, 661 inside the
object area.

| named (colour, shape) | count | | named | count |
|---|---|---|---|---|
| yellow cube | 139 | | blue (purple) cylinder | 73 |
| red (orange) shape None | 116 | | red cylinder | 61 |
| blue triangle | 87 | | blue None | 58 |
| yellow triangle | 46 | | yellow None | 42 |
| red cube | 23 | | yellow cylinder 10, blue cube 2, black 1, colour None 3 | |

**Against the pictures** (episode starts 0, 3, 5, 11; 19 objects in view):
- Every colour was right: orange read as red, purple as blue, yellow as yellow.
- Every named shape was right (cubes, cylinders, upright prisms). 6 had shape None, mostly near cylinders and one purple prism.
- Missed: objects cut by the bottom image edge, and a yellow prism lying against the box row (ep 3).

**Top height above the fitted (tilted) counter** (visible >= 0.5, n 591):

| shape | height mm | sd mm |
|---|---|---|
| all | 48.7 (p10 45.0, p90 49.7) | |
| cube | 48.8 | 0.9 |
| cylinder | 48.9 | 0.7 |
| triangle | 47.9 | 1.9 |

- The tops are consistent to well under the 5 mm needed.
- The absolute height is unverified. The scene file says 45 mm, drop 2's RGB estimate was 46, and the organisers said "a 50 mm
  cube". `det.size` is measured (0.048 to 0.049), so the planner's counter under each object is right either way.
- **Phantoms:** 22 detections outside the object area. They are box tape in the gap between the centre and right boxes
  (x 0.53 to 0.60, y -0.12, z 0.13 to 0.17), some named "red triangle" at visible 0.6 to 0.8. ObjectArea drops them. No phantom
  came from the hands or the table edge.

## Q4 Grip signal from the operator's grasps

Score = our grip score on joints [1, 3, 5] (each hand's own order). It is the shortfall behind the command in the closing
direction, minus the free open-hand bias (left -0.013 / -0.046 / -0.033, right -0.029 / -0.016 / -0.006). Our policy subtracts a
closed empty-hand error instead (robot night: 0.014 to 0.026 summed), which would lower every row slightly.

| segment (hand closed > 0.6 rad, lifted part) | count | score (median per segment) |
|---|---|---|
| left-hand picks carried to x 0.41 | 9 (ep 6-14) | 0.32, 0.33, 0.47, 0.45, 0.32, 0.57, 0.41, 0.32, 0.43 |
| right-hand picks carried to x 0.41 | 6 (ep 0-5) | **0.104** (ep 0, yellow prism, held: gone from the table, frames 250 -> 300), 0.36, 0.56, 0.36, 0.44, 0.39 |
| right-hand miss (lifted to 0.14 empty, prism left on the table) | 1 (ep 5) | 0.031 |
| right hand resting half-closed (closure 0.6-0.9) while the left works | 7 (ep 7, 11-14) | 0.037-0.077 |
| open, steady | 1374 frames | 0.00 (p90 0.05) |

- **The load sits on one finger_0 joint:** median shortfall 0.22 to 0.25 rad there; thumb_1 -0.02 to -0.04; the other finger
  about 0.01. The operator closes 1.2 to 1.65 rad summed over the finger_0 joints.
- **Soft threshold 0.15:** it separates 13 of 14 holds from every empty reading. The weak prism pinch (0.104) would be read as
  empty, and the pick abandoned with the object in the hand.
- 0.10 would still clear every empty reading (max 0.077) but leaves little margin.
- `grip_check_patience` can drop to 1 or 2 now that real held readings exist. Keep 3 if the threshold stays at 0.15.
- Caveat: these are the operator's closures, not our cage's.

## Q5 Right thumb_0

**The sign is right.** Fitting the right hand alone (translation, left-arm camera) on the 262 frames with |thumb_0| > 0.3:

| thumb_0 | hand coverage | thumb-link coverage | covers more |
|---|---|---|---|
| as measured | 0.57 | 0.31 | in 97 % of frames |
| negated | 0.40 | 0.01 | |

So the URDF with the measured value is the physical thumb, and the operator commands both thumbs negative (left median -0.67,
right -0.57). That supports our `right_hand()` (thumb_0 keeps its sign), not INTERFACE.md's "thumb_0 mirrored too".

**New: the real right thumb_0 has a hard stop at -0.50 rad.**
- Measured -0.06 to -0.50 over all 4823 frames, while the command reaches -1.05. Commands below -0.7 still read -0.49.
- The left reaches -0.97.
- `ridge_opposed` commands -0.63, so on the right hand the thumb will stop 0.13 rad short.

## Q6 Tactile

216 taxels (9 pads × 12 per hand):
- 150 read a constant 30000 (dead or unwired).
- The other 66 are noise except on the left hand when closed: largest deviation from the taxel median 464 when closed vs 112 when
  open (p90 2982); any taxel more than 5000 off in 1 % of left frames. On the right hand nothing responds (88 closed vs 72 open).
- Not usable as a contact signal, and not in our obs anyway.

## Hover look (grasp_feedback servo) on real depth

The `HoverMeasurer` translation fit (links `wrist_` + `hand_`) through the left-arm camera, on hands 7.5 to 20 cm over the counter:

| hand | looks | all guards pass | failures | rms mm | coverage | inliers | shift mm |
|---|---|---|---|---|---|---|---|
| left | 223 | **94 %** | | 2.0 | 0.85 | 0.85 | (-0.7, 0.2, -0.2) |
| right | 137 | **13 %** | rms > 3 mm: 117, coverage 23 | 3.4 | 0.71 | 0.71 | (4.5, 15, 0.6) |

- The right-hand shift matches the 17 mm offset above. The servo would correct it only on the 13 % of right-arm looks that pass.
- Without the wrist links the right rms is still 3.2 mm: the right hand model itself is off (thumb stop, thumb_2).

## Fixes, ranked

1. **Done, `31ad9b4` (P1):** when no fit of both arms is believable, fit each arm alone. A one-arm fit is believed only if its
   own hand's RGB outline is within 8 px (`calibration.hands.one_arm`).
   - Drop 3: arm camera in 13 of 15 episodes, up from 9.
   - Sim: every fit believed today is unchanged (dr_train ×6 conditions, lowhands ×3, movinghands, closedhands, 431 realsim_looks);
     one realsim frame gained (4.4 mm vs 27.6 nominal); no wrong minimum admitted.
2. **Done, `4b7a6c9` (rim):** rim over the boxes' entry strips. -40 mm of modelled wall height at 14 deg, 10 to 18 mm at the robot
   night's 3 to 6 deg; at most 4 mm in sim.
3. **P2, on hold until tonight's robot export:** per-episode translation of the other arm's hand. It is the median over still looks
   of the hand-only translation through the chosen camera (sd 5 to 9 mm on drop 3), applied to that arm's targets and releases.
   - This is the fix for the about 2 cm right-arm error that no single camera removes.
   - The `arms (left|right|both)` and `outline:` lines of the robot export decide whether the finals robot shows the same offset.
4. **Grip:** consider `grip_threshold.soft` 0.10 to 0.12 (one real held prism read 0.104; empties max 0.077). Lower patience only
   once our own cage's held reading is seen on the robot.
5. **Right thumb_0 stop at -0.50 rad:** clamp the right-hand mirror of `ridge_opposed` (-0.63) or accept the 0.13 rad shortfall.
   Lying prisms only.
6. **Servo on real depth:** the right hand passes the hover guards 13 % of the time (rms 3.4 mm vs 3 mm). Do not count on it for
   the right arm until P2 or a better right-hand model.
7. **Ask the organisers:** is the finals posture the robot night's (3 to 6 deg) or drop 3's (14 deg lean)? Where will the box row
   stand (near wall 0.46 to 0.48 here, at the cube reach limit)?

Not justified on this evidence: loosening `min_coverage` or `min_inliers` (the compromise they reject is about 1 to 1.5 cm off for
each arm), and right-arm joint offsets (they failed the RGB check).

## P2: per-arm hand offset (`0bcb96d`)

What runs: at every still look with open fingers, each arm's palm and fingers (no wrist links) are fitted by translation only
through the camera the episode uses (`HandShift`, the hover look's fit with `links: [hand_]`). A fit counts as a sample if it has
rms <= 4 mm, coverage >= 0.6, inliers >= 0.5, spread <= 3 mm and a shift <= 5 cm. The detector keeps up to 6 samples per arm and
moves them rigidly onto the current camera when the episode's camera changes. The median goes into `SceneLayout.hand_offsets` only
with >= 2 samples, with a length between the 10 mm dead band and 4 cm, and in xy only (`correct_z: false`). The planner plans that
arm against the target, release point, bins, counter, rim and neighbours, all moved by minus the offset (`_scene_moved`, in
`plan_pick` and `reaches_bin`). So the dry run checks the joints that are sent, and the joint-space sag compensation stays
separate. A servo final pick onto its corrected target is already in the arm frame, so it is not shifted again
(`arm_frame=True`); its open-loop fallback is shifted.

**Drop 3, the policy's own still looks** (primer + look, as `episodes.py`), samples seen through each episode's left-arm-alone camera:

| hand | episodes with samples | median of episode medians, mm | sd of episode medians, mm | all samples |
|---|---|---|---|---|
| right | 10 | **(-8.2, -17.0, +1.7)** | 5.2, 4.5, 2.6 | 17, median (-9.3, -17.5, +1.7) |
| left | 11 | (+0.6, -0.3, +0.3) | 2.8, 1.5, 1.7 | 25 |

- This matches the analyst's (-7, -17, -2) in x and y. The z sd is larger than its value, so z is not applied.
- Applied (right arm) in 5 of 15 teleop episodes: eps 4, 5, 6, 7, 9, from 2 to 4 samples each. The others have 0 or 1 sample.
  Teleop gives only 0 to 5 still looks; at home in the finals there are up to 12.
- Through the episode's own camera the offset follows the camera. Ep 4 has a both-arm medoid camera and gets (-17, -18) mm;
  ep 7 has the left-arm camera and gets (-14, -10). The left hand under a both-arm camera reaches -21, +15 mm (ep 11, 1 sample).
  This is the case P2 exists for.
- Requiring open fingers dropped one ep 11 right sample at (+18, +1, +18) mm. Fingers were curled, so the fit had the palm only.

**Sim no-op** (fresh detector per seed, each still frame shown twice 0.2 s apart):

| set | seeds | camera | per-look xy left / right, median / max mm | median over >= 2 samples, max mm | applied |
|---|---|---|---|---|---|
| dr_train | 120 | scene | 2.1 / 4.7, 4.1 / 5.4 | (1 frame per seed) | 0 |
| realsim_looks (still, 12 looks) | 8 | hands | 0.6 / 4.9, 1.9 / 7.0 | 2.7 / 4.7 | 0 |
| movinghands (still) | 4 | scene | 2.5 / 4.2, 2.8 / 8.6 | 3.5 / 3.7 | 0 |
| lowhands, lookpose | 20, 45 | scene | <= 2.0, <= 5.2 | | 0 |
| closedhands | 20 | scene | 2 samples, 8.6 max (fingers curled; now not sampled) | | 0 |

- Mock seeds 3/5/7 hashes are unchanged (94e5e360 / 0a5ced74 / 264852e6) with 0 clips. The mock renders no arms, so it takes no samples.
- Mock with forced offsets (left +12 mm x, right (-7, -17) mm): 0 clips and 0 plan clips. Every right-arm grasp is at the seen
  target + (7, 17) mm, every left-arm one at -12 mm. In the mock the hand really is at FK, so the shifted picks missed there;
  that is expected.
- The `HoverMeasurer` refactor keeps its results: 62 of 62 drop-3 looks are identical before and after.

**Robot evidence** (servo hover looks in the 15.11 robot exports, links wrist + hand): the right-hand shift is (-15.8, +15.9),
(-0.4, +26.4), (+15.2, +14.3), (0, +7.9) and (-0.7, +1.9) mm. The left hand is mostly within 3 mm (worst (+6.7, -12.2)). So the
finals robot's right hand also stands 8 to 26 mm towards -y of its FK. The 05:20 video showed the right cage_deep coming down on
the cube's far/right edge.

Log lines: `hand offset (side): seen minus FK ...: sample n | rejected [...]` per look, and `hand offsets (real minus FK through
the episode's camera): left <verdict> ...; right <verdict> ...` whenever a verdict or median changes.
