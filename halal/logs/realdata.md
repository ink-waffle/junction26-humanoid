# Real-robot data log (workstream `realdata`)

Analysis of the organisers' first real-robot drop, Hugging Face dataset `zupermiez/g1_pick_cube` (LeRobot v3.0, 20
teleop episodes, 11,720 frames at 30 fps, head + two wrist RGB cameras, 28-dim state and action, no depth, no labels,
no calibration bundle). Everything below is measured on that drop with the scripts in `halal/eval/real/`
(section "Reproduce"). Outputs that are not tracked: `runs/real_frames/` (sample PNGs, overlays, `plots/`).
Joint vectors of one hand are in LEFT order `thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1`.
Error always means `measured − commanded`.

## Ten things that would break a sim-tuned policy

1. **Free fingers do not track to 0.00.** Steady error of an empty hand is −0.04…+0.08 rad per joint (noise only
   0.001). The sim "holding" signal of 0.03–0.08 rad is inside that band. The signal exists (0.1–0.4 rad per joint),
   but only with a closure target well past contact, after ≥ 0.4 s of steady command, and against a calibrated bias.
2. **Fingers are slow.** 80 ms lag (thumb_0 150 ms), 90 % of a step after 330–400 ms (thumb_0 445 ms). `close_hold_s`
   0.25 lifts before the grip has formed. The operator waits 1.05 s (median) between closing and lifting.
3. **Episodes do not start at `ARM_HOME`.** Start poses are up to 1.7 rad away (elbows +0.06…+1.2 instead of −0.5).
   A first command that is not the measured pose is clipped by the 1 rad/s filter on every tick.
4. **The nominal head calibration is off by tens of pixels.** The robot's own hands appear (+20, −22) px from their
   FK projection; the bins appear 60–85 px higher than `scene.yaml` + `calibration.yaml` predict.
5. **The camera-to-table geometry is not constant.** The camera looks 1.5–7.5° steeper at the table than nominal,
   differently per recording session; the view shifts 5–20 px between episodes and 20–40 px when the hand pushes
   on the table. Bins are 5–10 cm (ep17–19: 13–15 cm) farther than nominal, the table is at z ≈ +0.02 or 0.00.
6. **Colours and materials differ.** No grid mat (bare glossy table), cardboard bins wrapped in orange tape, foam
   objects in purple, ORANGE and yellow. Baseline `red` = the bins (95 % of the tape), orange objects are in no
   class, `yellow` finds 41 % of the yellow pixels, `blue` also fires on the robot's own hand and forearm.
7. **thumb_0 is not sign-mirrored between the hands** (URDF and the operator's commands agree); the organisers'
   `mirror_left_to_right` negates it. Harmless at thumb_0 = 0, wrong for every posture that uses the thumb swing.
8. **Arm lag and droop.** Arms follow 80–125 ms behind (about 0.06 rad per rad/s); the right shoulder pitch rests
   0.03–0.07 rad below its command. Use measured `arm_q` for FK, never the command, and wait after a move.
9. **Images are late and partly stale.** The head image shows the joint state of about 2.2 frames earlier (72 ms), wrist
   images 2.7 frames (90 ms); 4 % of head frames and 24 % of wrist frames are repeats while the arm moves. A moving
   hand is blurred over 15–20 px.
10. **The demo actions cannot be replayed through the safety filter.** 9.6 % of frames command an arm joint faster
    than 1 rad/s (single-frame jumps up to 28 rad/s), finger commands jump up to 25 rad/s, and 18 % of the frames
    command a finger 0.001 rad beyond its URDF limit (a clipped step each).

## What the drop contains

- Left hand only. 12 clean pick-and-place of a purple object into the LEFT bin: cube in ep0, 2–9, cylinder in ep11,
  14, 17. ep1 clamps the cube and tosses it into the bin. ep10, 12, 13, 15, 16, 18 never lift (fumbling, cylinder
  knocked over in ep18, hand pressed flat on the table in ep13), ep19 is 9.5 s of standing still.
- Three placements of robot and table: ep0–9, ep10–16, ep17–19 (bins at different image heights, see B5/B7).
- People are in view: a person's hand resets the cube inside ep0, bystanders fill the wrist images.
- Our labels (hold windows, empty-hand windows, bin corners) are in `halal/eval/real/labels.py`.

## A1 Finger tracking error and the contact signal

Left hand, all values in rad. "Free" = fingertips ≥ 10 cm above pelvis z = 0 and outside every labelled contact
window; "steady" = no hand command change for 10 frames. Plot: `plots/a1_finger_error_by_situation.png`.

| | thumb_0 | thumb_1 | thumb_2 | middle_0 | middle_1 | index_0 | index_1 |
|---|---|---|---|---|---|---|---|
| noise at rest (sd, ep19) | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| free, steady: median error | +0.035 | −0.033 | +0.010 | −0.021 | −0.021 | −0.038 | −0.019 |
| free, steady: p5 … p95 | −0.06…+0.09 | −0.05…−0.02 | −0.03…+0.02 | −0.06…+0.02 | −0.03…+0.01 | −0.06…+0.02 | −0.03…0.00 |
| closed on nothing (3 windows): mean | +0.066 | −0.014 | −0.001 | −0.021 | +0.011 | −0.039 | +0.016 |
| holding in the air (12 windows): median | −0.125 | −0.139 | −0.044 | +0.148 | +0.074 | +0.030 | +0.042 |
| holding: p5 … p95 | −0.56…+0.17 | −0.26…−0.05 | −0.16…0.00 | −0.03…+0.30 | −0.08…+0.30 | −0.06…+0.24 | −0.04…+0.38 |
| holding minus free bias, "blocked" direction | 0.16 | 0.11 | 0.05 | 0.17 | 0.09 | 0.07 | 0.06 |

- A free finger sits 0.02–0.04 rad more closed than commanded; a commanded 0 reads −0.02…−0.03 on the left hand and
  +0.01 on the right. thumb_0 has a scale error of about 10 % (commanded −1.0 reads −0.93; right hand: error =
  −0.10 × command − 0.03).
- A held object pushes every joint the other way (the joint stops short of its commanded closure). Which joints
  carry the load depends on the grasp: thumb_1 alone separates only 10 of 12 holds at a 0.08 threshold.
- Contact score = Σ over the 7 joints of max(0, blocked error − free bias), blocked error = −closing sign × error.
  Median per labelled window: holding in the air 0.50–1.41 (12 of 12), holding on the counter before the lift
  1.19–1.60, closed on nothing 0.07–0.14; free and steady: median 0.05, p95 0.17. A threshold of 0.30 separates all
  labelled windows (`plots/a1_contact_score.png`). Per frame, 96 % of the holding frames are above it; the 3.8 % of
  free steady frames above it belong to an unlabelled grasp attempt in ep18. While the fingers move the score
  reaches 0.5–0.75 from lag alone (15 % of moving frames above 0.30).
- These margins come from the operator commanding 0.15–0.4 rad past contact (torque = kp 1.5 × error = 0.2–0.6 Nm
  per joint, held for 3–6 s). A shallow squeeze gives sim-sized errors that drown in the bias. The error also creeps:
  index_1 kept closing by 0.15 rad over 5 s around a foam cube (`plots/a1_left_hand_ep04.png`).
- Lag of measured behind commanded speed (cross-correlation): 78–91 ms for six joints, 150 ms for thumb_0. Response
  to a fast command step (0.5–1.2 rad): 10 % after 26–57 ms, 50 % after 137–155 ms, 90 % after 331–401 ms
  (thumb_0: 77 / 201 / 444 ms; index_0 has a single step: 29 / 99 / 519 ms). Peak measured speed 5 rad/s on the
  distal finger joints, 2–3.5 rad/s on the proximal joints and the thumb. `plots/a1_finger_lag.png`.

Consequences
- Do not threshold a single joint at 0.03–0.08. Use the score above with threshold 0.30, evaluated only after the
  hand command has been still for ≥ 0.4 s (0.5 s if thumb_0 moved).
- Self-calibrate instead of trusting our bias numbers: once at the start (palm down, at home) close on nothing with
  the exact grasp target, wait 0.5 s, store the per-joint error e0; in a grasp test Σ max(0, s·(e − e0)) > 0.30 with
  s = closing direction. This removes bias, the thumb_0 scale error and gravity sag together.
- Command the closure ≥ 0.2 rad beyond contact on the pinching joints, otherwise there is nothing to measure.
- After the last hand command change wait ≥ 0.45 s before lifting; plan hand ramps at ≤ 2.5 rad/s (the fingers
  cannot follow faster than 2–5 rad/s anyway).

## A2 Operator grasps

- Preshape just before closing (median measured): `[-0.76, 0.90, 0.15, -0.38, -0.02, -0.21, -0.02]`: thumb swung
  out sideways with thumb_0, fingers nearly straight; thumb tip 109–127 mm from the fingertips.
- Three postures while holding (commanded → measured):
  - A, three-finger wrap on the cube (ep0, 2, 3, 4, 6, 8, 9 and the cylinder in ep11):
    `[0.0, 0.92, 0.2, -0.6, -1.27, -0.43, -1.34]` → `[-0.12, 0.74, 0.14, -0.47, -1.16, -0.39, -1.2]`
  - B, thumb + index (ep5, 7): `[-0.36, 0.92, 0.28, -0.55, -0.6, -0.45, -1.45]` →
    `[-0.39, 0.82, 0.15, -0.40, -0.60, -0.42, -1.27]`
  - C, thumb + middle, cylinder from the side (ep14, 17; also ep1): `[0.37, 0.92, 0.06, -0.7, -1.2, -0.3, -0.02]` →
    `[-0.12, 0.76, 0.03, -0.50, -0.96, -0.28, -0.02]`
- Against the baseline `closed` `[0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6]`: the operator keeps thumb_1 at +0.92 (thumb
  stretched forward towards the fingertips) and thumb_2 small, bends the proximal finger joints only −0.4…−0.6 and
  the distal ones −1.1…−1.4 (fingers hook around the object), and opens and closes the thumb with thumb_0
  (−0.8…−1.0 open, 0 closed).
- FK: with thumb_1 = 0.92, thumb_2 = 0.2 the thumb tip moves −78 mm per rad of thumb_0 across the palm: thumb_0 =
  −0.37 puts it exactly opposite the index tip (palm z +29 mm), +0.37 opposite the middle tip. That is a
  two-finger pinch without the 29 mm lever arm. In the baseline thumb posture the same thumb_0 moves the tip only
  ±14 mm.
- **Right hand:** mirror as `[thumb_0, −thumb_1, −thumb_2, −index_0, −index_1, −middle_0, −middle_1]`, thumb_0
  NOT negated. Both thumb_0 joints rotate about +y of their palm frame and the index sits at palm z +28.5 mm on both
  hands; FK gives thumb tip z +29 mm on the right hand for thumb_0 = −0.37 and −29 mm (opposite the middle finger)
  for the +0.37 that `mirror_left_to_right` produces. The operator's resting hands confirm it: median commanded
  thumb_0 is −0.63 left and −0.50 right while thumb_1 is +0.92 / −0.92. The right hand never grasps in this drop.
- Fingertip gaps from FK while holding: thumb–index 38–60 mm, thumb–middle 36–58 mm for the wrap (thumb–middle
  18 mm in ep3; thumb–index 10–14 mm in ep5 and ep11: the object sits deeper than the tips). Objects are about 5 cm.
- Palm at the grasp, cube (ep0–9): approach axis (palm normal) 15–26° from straight down, fingers pointing across
  the body (yaw −66…−84° for the left hand, ep0 −39°) and 13–25° downwards (ep0 level), palm origin at x 0.28–0.32,
  y 0.07–0.14, z 0.13–0.21; fingertips at z 0.04–0.09. A top-down claw turned inward, not forward.
- Palm at the grasp, cylinder (ep11, 14, 17): from the side, palm normal 45–70° from down, fingers forward-inward
  (yaw −42…−51°), palm origin only 7–10 cm above pelvis z = 0.
- Carry and release: lift 7–18 cm (median 12), carry 2.7–5.8 s at 0.13–0.27 m/s (p95 of palm speed per carry), palm
  at most 0.23–0.32 m above pelvis z = 0. Release with the palm at (0.40, 0.21, 0.23) and the fingertips at
  (0.52, 0.18, 0.17) median; the palm never goes beyond x = 0.41 in the whole drop (looks like a teleop limit).

Consequences
- Worth testing in sim: operator thumb (thumb_1 ≈ 0.9, thumb_2 ≈ 0.2), closure with the distal joints, thumb_0 as
  the open/close axis and as the selector of the opposing finger.
- The inward finger direction is what a human chooses for top-down picks in front of the body; it costs wrist range
  (left wrist pitch and yaw reach their limits, A3).

## A3 Arms

- Start poses (`plots/a3_arm_start_poses.png`): median `[-0.36, 0.23, 0.02, 0.45, 0.07, -0.59, -0.36]` left and
  `[-0.33, -0.26, -0.13, 0.81, -0.15, -0.81, -0.12]` right against `ARM_HOME` `[0, ±0.35, 0, -0.5, 0, 0, 0]`. The
  largest joint deviation per episode is 0.72–1.73 rad. Palms start at x 0.21–0.39, |y| 0.13–0.25, z 0.06–0.24
  (home: 0.22, ±0.18, 0.21), i.e. lower and with bent wrists. The data does not support "the real robot starts at
  the sim home pose"; these are teleop sessions, so the finals may differ (open question).
- Commanded speed (`plots/a3_arm_command_speed.png`): p95 0.35–1.01 rad/s, p99 0.8–2.3 rad/s on the left arm, single
  frames up to 27.8 rad/s; 9.6 % of all frames have an arm joint above 1 rad/s. Measured speed peaks at 2.9–4.1 rad/s.
- Tracking: lag 79–124 ms per joint; while moving the error is −54…−78 ms × commanded speed (0.06–0.08 rad behind
  at 1 rad/s). At rest the left arm is within 0.006 rad, the right shoulder pitch is −0.031 ± 0.032 (−0.068 in
  ep19) and the right elbow −0.009 ± 0.024 (−0.036 in ep19).
- Limits: left wrist pitch is driven to −1.61 commanded / −1.62 measured (URDF −1.614, filter −1.564) and left wrist
  yaw to −1.61 / −1.57; 22 frames command inside the filter margin, 2 measured samples are beyond the URDF limit.
- Reach used (pelvis frame): left palm x 0.18–0.41, y 0.03–0.29, z 0.03–0.38; fingertips x up to 0.59, radial
  distance up to 0.63 m. The palm never enters the counter keep-out (lowest palm z 0.032, wrist 0.036).
- Counter height: in ep0–9 no fingertip goes below z = +0.028 and tips hold the 5 cm cube at z 0.04–0.07, in
  ep10–16 fingertips touch down at z 0.00–0.02 (and read −0.045 while the flat hand pushes on the table in ep13, the
  body gives way). With the camera-based estimate (B7) the table is at about +0.02 in ep0–9 and 0.00 in ep10–16.

Consequences
- Start every episode from the measured `arm_q` and ramp to our home at ≤ 1 rad/s.
- After a move wait ≥ 0.15 s for the arm plus ≥ 0.1 s for the camera before trusting an image; `settle_s` 0.3 is the
  minimum, 0.5 is safer for the right arm.
- Compute grasp heights from the observed table (depth), not from `scene.yaml` z = +0.01.

## A4 Other traps in the joint data

- Measured hand joints outside the URDF range: left thumb_2 reads down to −0.022 (2.6 % of frames below its limit
  of 0), left thumb_0 to −1.055 (limit −1.047), right index_1 to −0.004. Clamp measured hand joints before FK or
  limit checks.
- Teleop hand commands overshoot the open limit by 0.001 rad in up to 18 % of the frames (left index_1; 8–13 % on
  three other finger joints).
- Left/right: thumb_0 is not mirrored (A2); free finger bias is smaller on the right hand (+0.01) than on the left
  (−0.02…−0.04); the right arm droops, the left does not.
- Units are radians, state and action share the documented order, no NaN, 4 identical consecutive state rows,
  timestamps exactly 1/30 s apart.

## B5 What the cameras see

- Head camera, 640×480, upright, image left = robot's left, as in sim. Three orange bins in a row at the top, objects
  below them, the left hand and silver forearm bottom-left, the right hand at the right edge. Both hands are in view
  at rest in every episode; the left arm hides up to a quarter of the image and a black cable loops around its wrist.
- Not the printed scene: no 5 cm grid mat, the table is bare, glossy and light brown with specular highlights and
  screw heads; the bins are cardboard boxes wrapped in orange tape (brown inside, loose flaps on the rim, walls about
  7–8 cm) with orange tape strips on the table in front of them.
- Objects: six per scene, cube + cylinder in purple, orange and yellow; no triangle, no red, no blue. Matte foam,
  about 5 cm. Two rows at about pelvis x 0.25–0.29 and 0.34–0.38, y +0.04…−0.17, often touching each other.
- Everything sits higher in the image than in sim: bottom of the bins' near wall at v ≈ 226 (sim ≈ 300), objects at
  v 260–450 (sim 380–480); objects are not cut off by the bottom edge here.
- The view moves: bottom of the centre bin's near face including its tape strip at v 251 (ep0), 227–241 (ep1–9),
  242–263 (ep10–16), 169–181 (ep17–19). Inside an episode the bins shift 1–7 px with the arm, 20–40 px in ep11 and
  ep13.
- Samples: `runs/real_frames/epNN_{first,grasp,last}_fFFFF_<camera>.png` for all episodes and cameras,
  `runs/real_frames/series/` with ten head frames each of ep04, ep11, ep17.

## B6 Colours

Head camera, OpenCV HSV, p5 / p50 / p95 over the first frame of all 20 episodes (`plots/b6_hue_saturation.png`):

| class | H | S | V |
|---|---|---|---|
| purple object | 109 / 112 / 124 | 88 / 144 / 164 | 99 / 220 / 252 |
| orange object | 9 / 13 / 14 | 119 / 161 / 229 | 148 / 250 / 254 |
| yellow object | 22 / 38 / 45 | 66 / 89 / 172 | 128 / 223 / 255 |
| bin tape | 2 / 5 / 8 | 99 / 179 / 218 | 76 / 119 / 220 |
| table | any | 6 / 21 / 44 | 116 / 153 / 210 |
| robot hand | ~100 (blue cast) | 31 / 124 / 228 | 11 / 43 / 59 |

- Lit top faces of the yellow objects are washed out and greenish (H 38–45), their shaded sides are H 26–28. The
  silver forearm reads as saturated light blue (H ≈ 100, S ≈ 150, V ≈ 230).
- Baseline thresholds on these frames (`plots/b6_baseline_masks.png`), mean pixels per frame:

| baseline mask | takes from |
|---|---|
| `red` | 49,700 px of bin tape incl. the strips on the table (95 % of it) + 4,600 px of other reddish pixels; 0 px of the orange objects |
| `yellow` | 3,900 of 9,450 yellow-object px (41 %, the shaded sides) |
| `blue` | 9,330 of 9,570 purple-object px (98 %) + 3,400 px of the robot hand + 13,300 px of forearm and background |

- Orange objects and orange tape are only 4–8 hue units apart; the objects are brighter (V p50 250 vs 119) and
  slightly less red.
- Exposure and white balance are stable: mean brightness 103–110 across episodes (99–120 inside an episode as the
  arm moves in), R/G 1.08–1.12, B/G 0.96–0.98.
- Motion blur: the static scene is sharp; at 20–24 px/frame (0.3 m/s) the hand and a carried cube are smeared over
  15–20 px, so the exposure is most of the frame period. Rolling-shutter skew is not separable from that blur.

Consequences
- Bins cannot be removed by geometry from `scene.yaml` (B7): segment them by colour + size + position (three large
  blobs in a row) and exclude them explicitly. Mask the robot's own arm with FK or by its darkness.
- If the finals use these foam objects, colours are purple / orange / yellow and need their own classes; hue alone
  does not separate orange objects from the bins. Ask the organisers which objects and bins the finals use.
- Classify only from frames taken while the arms are at rest.

## B7 Head camera pose and scene placement

- Camera against the robot's own hands (`head_fit.py`): FK points of both hands were fitted to the dark hand
  silhouette on 844 frames of ep0–16. With `calibration.yaml` the mean distance of the finger-axis points to the
  silhouette is 9.2 px, after the fit 5.1 px; the hands move by about (+20, −22) px (median) between the two, about
  1.5 cm at hand distance. The fit cannot split pitch from height: any pitch correction from −10° to +2° fits almost
  equally well (cost 5.0–5.4 px) if the camera moves 0.8 cm per degree along its y axis. Rotation-only solution:
  −2.2° about the optical x axis (steeper), −1.8° about y, −1.4° about z. Six-parameter solution: −7.4°, −1.9°,
  −0.4° and −4.0 cm in y. The fits of ep0–9 and ep10–13 agree with each other.
- Camera against the table (`scene_fit.py`, centre bin rim as a 21 cm square, PnP on hand-read corners, ±1.3° for
  3 px reading noise): the optical axis is 55.0° below the table plane in ep0 and 49.1° in ep11; nominal is 47.6°.
  The angle between camera and table therefore differs by about 6° between the two sessions, while the camera-to-arm
  relation stays the same: the upper body leans differently against the table from session to session.
- Scene placement, assuming 21 cm bins: near edge of the centre bin at pelvis x 0.45–0.48 in ep0–16 (0.47–0.50
  from the rim PnP) and 0.53–0.55 in ep17–19 (nominal 0.40) for both the nominal and the six-parameter camera; table
  height +0.02 (ep0), +0.005
  (ep11) with the six-parameter camera, which agrees with the fingertip contacts of A3 within 1 cm (the nominal
  camera gives +0.05 / +0.03, which the fingertips contradict).
- In pixels: `scene.yaml` through `calibration.yaml` puts the centre bin's near bottom edge at v = 311, the image has
  it at v ≈ 226 in ep0, ≈ 236 in ep11 and ≈ 157 in ep17. The camera correction accounts for 20–35 px of that
  (depending on the pitch/height split), the rest is the table standing farther away.
- Overlays: `runs/real_frames/overlay/` (nominal) and `runs/real_frames/overlay_fitted/` (six-parameter camera).

Consequences
- Do not use the nominal bin positions or the nominal calibration to mask bins, reject detections or choose release
  points. Detect the bins and fit the table plane (depth) in the first frame of every episode.
- Correct hand-eye error online: the fingertips are in view at home, so the offset between FK-projected and observed
  fingertips can be measured once per episode and applied to grasp targets (about 1.5 cm here).
- Release points must come from the observed bins: with bins 5–10 cm farther, "5 cm inside the nominal near wall"
  (x = 0.45) is on top of or in front of the real near wall.

## B8 Wrist cameras

- Mounted at the base of the palm looking along the fingers: the thumb fills the left third of the left-wrist image,
  the fingers the lower right, about half of the image is the robot's own hand. The right-wrist image is the mirror
  arrangement and mostly shows the venue.
- A held object sits between thumb and fingers in the image centre and bottom. Share of purple pixels in the fixed
  ROI u 220–480, v 200–480 (`plots/b8_wrist_roi_fraction.png`): holding the cube 0.29–0.53 (median per hold, 9 holds),
  holding the cylinder 0.02–0.31 (3 holds; the colour washes out to lavender when the camera over-exposes), empty
  hand in the air p50 0.006, p95 0.08, p99 0.24, empty hand near the counter p95 0.30 (other objects in view).
- A looser colour range (H 105–150, S ≥ 30) lifts all 12 holds to 0.30–0.61 but also the empty hand to p95 0.21.

Consequences
- A colour blob in the wrist ROI is a usable second vote for "still holding" once the hand is ≥ 12 cm above the
  counter, with the colour taken from the object just picked and a threshold near 0.2 on a 0.3 s median. It is not
  reliable enough alone (exposure, purple venue lights), and it cannot tell "between the fingers" from "held" near
  the counter.

## C9 Timing and data quality

- State and action are resampled to exactly 30 Hz. Camera frames are "newest at the tick": while an arm moves 4.4 %
  of the head frames and 23–24 % of the wrist frames repeat the previous one (typically every second frame in a
  burst), so the wrist cameras deliver about 23 fps.
- Image delay against the joint state, from correlating image change with joint-derived motion: head 2.2 frames
  (72 ms, 1.6–3.0 per episode), left wrist 2.7 frames (91 ms), right wrist 2.6 frames (88 ms).
- The first frame of an episode can show a hand already holding or touching an object (ep12, ep15), the last frame
  is often mid-motion; ep0 contains a manual reset of the cube.

Consequences
- When stopping to look: arm lag (≤ 125 ms) + image delay (≤ 100 ms) + one possible stale frame: use the third
  frame after the measured arm speed has dropped to zero, not the first.

## Decisions taken while working

- Video decoding ran on a laptop copy of the drop (485 MB), not on the shared sim instance, which was saturated
  (load 40 on 30 cores) by the sim evaluations; the whole decode takes under two minutes there.
- Labels were made by hand from contact sheets of head + left-wrist frames; hold windows are conservative (start
  after the lift, end before the release).
- The head calibration result is reported as a pixel offset plus two candidate decompositions, because the hands do
  not constrain pitch and height separately; the table-contact check favours the six-parameter one.

## Open questions for the organisers

1. Do the finals use the printed grid mat, the 3D-printed PLA objects in red / yellow / blue including triangles, and
   the measured 21 cm bins, or the foam blocks, cardboard bins and bare table of this drop?
2. How is the robot held (gantry, seated)? The camera-to-table angle differs by 6° between sessions of this drop and
   the view moves when the hand pushes on the table. Will the table be placed against the robot by measurement
   (bins' near edge at 0.40 m), and is the counter at z = +0.01 in the pelvis frame?
3. Does the runner move the arms to `ARM_HOME` before the first `act()`, and are the hands open at that point? The
   drop starts up to 1.7 rad away.
4. Which sign does right `thumb_0` have on the robot? URDF and teleop data say "same as left"; INTERFACE.md says
   mirrored.
5. When does the measured head calibration (AprilTag) arrive? This drop has none and the nominal one is 20–35 px
   off at the hands.
6. Are the hand gains kp 1.5 / kd 0.2 in the finals, and will `hand_max_lead` stay off? Is there a limit on how long
   or how deep a finger may squeeze?
7. Wrist camera rate (about 23 fps here) and head exposure settings in the finals; will later drops include depth,
   success labels and `aalto_episodes.jsonl`?
8. Were the teleop speed caps (0.9 rad/s arms, 3 rad/s fingers) active when this drop was recorded? The commands
   exceed them.

## Reproduce

Needs the drop under `runs/drops/g1_pick_cube` (or `HALAL_REAL_DROP`), the URDF under `third_party/`, and an
environment with numpy, scipy, pinocchio, OpenCV, PyAV, pyarrow, matplotlib, pyyaml.

```
python -m halal.eval.real.video_features   # per-frame image change and mean colour -> runs/real_frames/cache/
python -m halal.eval.real.finger_signal    # A1 tables, plots a1_*
python -m halal.eval.real.grasps           # A2 postures, palm poses, right-hand mirror check
python -m halal.eval.real.arms             # A3, A4 tables, plots a3_*
python -m halal.eval.real.extract_frames   # B5 sample PNGs
python -m halal.eval.real.colours          # B6 tables, plots b6_*
python -m halal.eval.real.head_fit         # B7 camera against the hands, overlay_fitted/
python -m halal.eval.real.overlay          # B7 nominal overlays
python -m halal.eval.real.scene_fit        # B7 bins and table
python -m halal.eval.real.wrist_blob       # B8
python -m halal.eval.real.timing           # C9
```
