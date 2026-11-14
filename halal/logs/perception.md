# Perception log (workstream `perception`)

Benchmark: `python -m halal.eval.perception_bench --detector <name> --frames runs/frames/<set> [--split heldout]`
and `--armless 10`. Frame sets (sim, `AALTO_DR=train`, privileged object poses and DR parameters per frame; on the
instance under `/data/halal/runs/frames/` and `/data/wt/perception/runs/frames/`):

| set | frames | what |
|---|---|---|
| `dr_train` | 120 (seeds 2000–2119) | first look at `ARM_HOME`, hands open. Held out: seeds with `seed % 4 == 3` (30), never tuned on |
| `lookpose` | 45 | no hand in view: baseline robot home (15), left arm away / right at home (15), `motion.look_arm_q` (15) |
| `lowhands` | 20 | both hands ~9 cm above the counter over the object band |
| `motion` | 1017 | every 20th tick of 8 organiser-baseline episodes (`sim_eval --record-frames`): arms anywhere, held objects, objects in bins |
| `armless` | 40, rendered | organiser mock world (no arm drawn) × 4 joint poses that put the kinematic hands in view |

All errors are of the top-face centre against the simulator's object pose. "Camera part" = error caused by the
camera model in use, "detector part" = the rest.

## Result

`dr_train`, first look, 120 frames / 472 objects (held-out 30 frames / 114 objects in brackets):

| | organiser detector (`baseline`) | `topface` |
|---|---|---|
| recall / precision | 0.989 / 0.692 (0.991 / 0.677) | 1.000 / 1.000 (1.000 / 1.000) |
| colour right | 1.000 | 1.000 |
| top xy, mm: median / p95 / max | 11.6 / 24.0 / 29.9 (13.6 / 25.8 / 29.9) | 1.7 / 2.4 / 2.7 (1.6 / 2.5 / 2.7) |
| top z, mm: median / p95 / max | 5.4 / 15.2 / 19.9 (4.3 / 11.9 / 14.1) | 0.0 / 0.3 / 0.3 (0.0 / 0.3 / 0.3) |
| shape: decided / right of decided | none | 0.892 / 1.000 (0.851 / 1.000) |
| yaw (cube, triangle), deg: median / p95 | none | 0.4 / 1.5 (0.5 / 1.5) |
| runtime per look, Mac CPU | 21 ms | 72 ms first look (30 ms of it the hand fit), 31 ms later looks |

`motion` (1017 frames, one detector per episode): baseline recall 0.912, precision 0.662 (1305 phantoms: held
objects, objects in bins, bin walls), xy 12.8 / 21.4 (median / p95). `topface` recall 0.907, precision 1.000 (0
phantoms), colour 2537 of 2537, xy 1.9 / 9.9 / 27.3; fully visible objects: recall 1.000, xy 1.9 median.
In the sim, same policy, fixed task, 8 seeds, DR (measured by the policy workstream): baseline 12/31 objects
sorted (39 %), `topface` 27/31 (87 %).

## What the detector does

1. Camera pose for the episode (`calibration.py`): counter plane → roll, pitch, height; front-wall line of the bin
   row → yaw, x; where that wall starts and stops → y ("scene anchor", depth only, 3 ms). In parallel the robot's own
   forearms and hands, posed by FK of the measured joints, are registered against the depth image (`hands.py`). The
   scene anchor is used unless the hands contradict it by more than `agree_within`; then the hand pose; else nominal.
2. Counter plane refitted at every look in that camera frame; `top[2]` = plane under the object + 45 mm.
3. Blobs at object-top height, split by colour class; around each, a 1 mm top-plane grid in which every cell is
   "top face seen" (+1), "seen empty" (−1) or "hidden / outside the image" (0); the three outlines at every yaw are
   correlated with it (`shapes.py`). Hidden cells cost evidence, not accuracy.
4. Colour from the most saturated 30 % of the object's pixels, hue basins, white balance on the bare counter.
5. Whatever coloured stands on the counter and no fit explains is reported with `shape=None`, `visible=0`.

Output semantics: `confidence` = how sure the colour is; `visible` = share of the top face seen (how well the pose
is known); `shape` / `yaw` = None when undecidable; `layout()` = counter height, bin centres and what fixed them.

## Facts found on the way

- **The sim does not render with calibration.yaml's principal point.** Fitted against ground truth: (319.5, 239.5),
  f = 604.83, not (325.04, 250.93): top-face height residual 0.03 mm (std 0.10) versus +5.9 mm with the yaml values.
  Equivalent to a nominal pitch error of 1.0° and 0.5° yaw on top of DR. The detector uses the yaml values (they are
  presumably right on the real robot); the per-episode calibration absorbs the mismatch up to a constant 1.4 mm.
- **DR camera jitter is the baseline's main error:** 11.3 mm median, 31.4 max at the objects; its own error is 5.1.
- **At `ARM_HOME` the thumbs hide part of the top face of 33 % of the objects** (157 of 472), up to 90 %.
- **Lit top faces lie about hue:** red drifts up to +20° towards orange, blue −34° towards cyan on the washed-out top;
  the most saturated pixels of the same object stay within ±14° / −25°.
- **A toppled triangle is 45 mm high along a ridge only:** nothing of it looks like a top face.

## Ideas in the order tried

Numbers are `dr_train` dev (90 frames) unless stated; held-out was looked at after each step, never tuned on.

**1. Self-calibration against plane and bins.** Hypothesis: the camera part of the error can be removed from depth
alone because counter and bins are exactly where scene.yaml has them. Camera part at the objects, 120 frames:
11.3 / 22.5 / 31.4 mm (median / p95 / max) → 1.4 / 1.8 / 2.0; z 5.3 / 16.8 / 21.0 → 1.0 / 1.1 / 1.2; all 120 frames
fitted; with the sim's true intrinsics 0.5 / 0.8 / 1.1. Verdict: kept. The counter height measured by the camera
alone is off by minus the DR camera z (correlation 0.999, ±10 mm), so height must come from an anchor.

**2. Outline fit on a top-plane evidence grid.** Hypothesis: fitting the known outlines to "seen top / seen empty /
hidden" cells gives centre, shape and yaw, and survives occlusion. First version (blob centroid replaced by fit):
xy 3.2 / 4.1 / 4.5, shape right 100 % of 98 % decided, yaw 0.4° median; recall 0.955 (objects under a thumb dropped
by a score threshold). Bug on the way: shapes were ranked by score per own area, so every cube tied with the circle
and triangle inside it; ranked by absolute area since.

**3. Silhouette bias.** The seen top face sits 1.8 mm towards the camera (front-face rays within the depth tolerance
count as top, the far fillet is seen past). Analytic correction from depth tolerance, fillet radius and viewing
angle: detector part 2.0 → 0.5 mm median, total xy 3.2 → 1.7. Kept.

**4. Low-visibility objects.** Objects with 8–16 % of the top visible still land within 5 mm because the seen-empty
cells pin the outline. Dropping the score threshold: recall 0.955 → 1.000, no phantom. No shape is named below 30 %
visible: 25 of the 26 wrong shapes in `motion` lay below (one cylinder at the image border called a triangle);
costs 5 % of decided shapes on `dr_train`.

**5. Colour.** Most-saturated pixels instead of the top-face median: hue spread of red ±20° → ±14°. Hue basins
(red 335–15°, yellow 40–90°, blue 195–250°, confidence falling to 0 over 6° outside) instead of nearest reference:
hues between basins (orange) come back with confidence 0 and are not returned. Stress on 60 frames / 236 objects
(image-space changes on top of DR): hue rotation ±6°, ±12°: 0 wrong, 229–236 found; ±20°: 0 wrong, 160–204 found;
±30°: 14 and 21 wrong, some above confidence 0.5. Saturation ×0.4: 0 wrong, 48 below 0.34. Gain ×0.6–1.5 and
colour casts of ±20 % per channel: no effect. Sim sets: no wrong colour in 3274 detections, 99.8 % at ≥ 0.8.
Recommended policy threshold 0.5. The confidence is a monotone score, not a calibrated probability: the sim offers
no colour errors to calibrate on.
Mistake corrected: confidence used to be colour × outline score, so half-hidden objects with a certain colour got
0.3–0.7; it is colour only now.

**6. Camera pose from the robot's own arms.** Hypothesis: the only reference rigid with the arm frame is the arm.
Surface samples of forearm, wrist, palm and finger links (16 357 points, `assets/hand_surface.npz`, from the URDF
meshes) posed by FK; 6-DOF point-to-plane registration of the depth pixels near them, prior 2 cm / 3° to nominal.
Hands only, full detector, 120 frames: xy 2.2 / 2.9 / 3.8, z 3.7 / 4.2 / 4.4 (nominal: 11.1 / 22.4 / 29.1 and
6.2 / 17.9 / 22.2); with the sim's true intrinsics the camera part is 0.6 / 1.4 / 2.4. Verdict: good enough to grasp
by itself in sim. Unknown on the real robot: needs real depth.

Failures of the hand fit, each found by a counter-example:
- *Believed with no arm in the image* (mock world, `arm_q = 0`: it matched the objects under the kinematic hands;
  counter 17 mm, bins 35 mm, objects 28 mm off). Fix: only depth more than 65 mm above the counter plane is arm
  evidence, and a fit is believed only with depth on ≥ 50 % of the visible model, ≥ 50 % of the depth pixels around
  the arms on the model, rms ≤ 4 mm, predicted spread ≤ 5 mm, camera moved ≤ 6° / 5 cm. `--armless 10`: believed 0 of
  40. True fits in sim: coverage 0.83–0.88, inliers 0.70–0.83, rms 1.7–2.5 mm.
- *Wrong minimum* in 2 of 20 `lowhands` frames (22.5 and 16.5 mm; coverage 0.66 / 0.70 gave it away). Fix: restart
  from the camera shifted ±15 mm per axis until 80 % of the model is covered, keep the result with most pixels on the
  model: 20 of 20 within 4.8 mm. Wider first gates made it worse (the sim wrist carries a camera box the URDF lacks).
- *Closed hands* (policy's calibration close, live sim): +11–13 mm in x, the hand pose overrode the scene anchor.
  Not reproduced offline (no closed-hand frames at rest). Mitigation: finger links are left out of the model when an
  index or middle joint is bent beyond 0.5 rad; at `ARM_HOME` the remaining surface (≈ 210 visible points) is too
  small and the fit rejects itself. Forearm and palm alone are not a substitute there: 11 of 20 trusted at 6.1 median
  / 19.0 max on `lowhands`. Root cause open.
- A Cauchy weight on residuals made the fit worse (max 3.5 → 6.4 mm); removed, the 8 mm final gate stays.

**7. Which pose to believe.** First rule: scene anchor if the hands agree within 10 mm, else hands. In sim the two
differ by 5.8 mm median / 6.3 max from the principal point alone, so a hand fit 5 mm worse than usual flipped the
choice to the worse estimate (the closed-hand case). Shipped: `agree_within` = 25 mm, hands are a check, not a rival.
Simulated on 60 frames with the measured errors of both fits and the table displaced by s (xy error at the band,
mean / max mm):

| rule | s = 0 | 5 | 10 | 15 | 20 | 30 mm |
|---|---|---|---|---|---|---|
| switch at 10 mm | 1.3 / 1.5 | 5.1 / 6.2 | 6.0 / 9.8 | 2.8 / 3.3 | 2.8 / 3.3 | 2.8 / 3.3 |
| switch at 25 mm (shipped) | 1.3 / 1.5 | 5.1 / 6.2 | 10.0 / 11.2 | 15.0 / 16.2 | 20.0 / 21.1 | 2.8 / 3.3 |
| blend, 90 % hands | 2.4 / 2.9 | 2.4 / 3.3 | 2.5 / 3.8 | 2.6 / 4.2 | 2.9 / 4.7 | 3.5 / 5.7 |
| hand fit off by +12 mm, means only: switch 10 / switch 25 / blend | 14.3 / 1.3 / 12.8 | 14.3 / 5.1 / 12.8 | 13.0 / 10.9 / 12.8 | 14.3 / 14.6 / 12.8 | 15.5 / 16.8 / 12.9 | 14.3 / 18.0 / 13.0 |

Verdict: a blend is only better than a switch if the hand fit is as good as measured in the benign case; it then
costs 1 mm in sim and saves centimetres on a displaced table. With a hand fit that can be 12 mm off it is the worst
choice in sim. The 25 mm switch is right for sim today; on the real robot it leaves a table displaced by 10–20 mm
uncorrected. Once the hand fit is verified on real depth, `agree_within` 10 mm (or the blend) is a config change.
The camera fit is retried at later looks while no arm has been seen and never replaced by one with less evidence
(`CameraFit.evidence`: 0 nothing, 1 scene anchor unconfirmed, 2 arms, 3 scene anchor confirmed by arms).

**8. Scene layout as output.** `SceneLayout(counter_z, bins, source)` from `Detector.layout()`. When the hands fixed
the camera, bins are found by geometry (`bins.py`): per bin the nearest vertical surface that fills the visible part
of its lane, then the wall's seen ends; a front wall more than 10 cm off the row median is discarded; unseen bins
are placed by the scene spacing. Sim with the scene anchor switched off: bin centres 4.9 mm median / 7.5 max,
counter height 0.6 mm. First version took the far wall for the front wall whenever a forearm covered the lane;
counting only cells the camera can see fixed it. Limits: lanes are ±8 cm around the scene y.

**9. Touching objects** (the real recordings show them; the sim layout does not). One blob gave one seed and fits
bridged two objects. Now blobs are split by colour class and pixels of another class count as "not this object";
after each fit its cells are claimed and the rest is fitted again. Mock world: rows of 3 and 4 touching objects, two
touching cubes of one colour, front-to-back at 20°: all found, ≤ 2.2 mm. Two touching objects of one colour are
separated only by the fit order, not by any seam.

**10. Under motion** (`motion`): no phantom in 1017 frames; a candidate with pixels 8–35 mm above its own top over its
footprint (a rim, a wall, a hand around it) is not an object, which also took the nominal-pose fallback from 0.71 to
0.996 precision. Recall by view: fully visible 1.000, partly hidden by an arm 0.757, cut by the image border 0.787.
xy of outline fits (median / p95 / max): fully visible 1.9 / 2.5 / 7.7, partly hidden 1.9 / 8.0 / 25.8, cut by the
border 2.0 / 6.2 / 18.5. By `visible`: ≥ 0.8: 1.8 / 2.5 / 3.0; 0.5–0.8: 1.6 / 2.6 / 7.7; 0.3–0.5: 1.9 / 2.5 / 8.0;
< 0.3: 2.7 / 11.4 / 25.8. So grasp at `visible` ≥ 0.3, better ≥ 0.5; below, look again with the arm away.
Toppled-object fallback (blob, `visible = 0`): recall 0.828 → 0.907 (one triangle lay on its side for 107 frames),
no phantom added; 220 such detections, centre 10–14 mm median, 27.3 max off.

**11. Look pose.** `lookpose`: recall and precision 1.000, xy 1.6 / 2.4 / 2.7, shape decided 0.98 (0.89 at
`ARM_HOME`: no thumb over the band). In sim the first look needs no hand in view. `lowhands`: recall 1.000, xy
1.7 / 2.7 / 22.7 (the maximum is an object under a hand, reported by the blob fallback), shape decided 0.61.

## Known failure cases

- Object mostly under a hand: centre up to 2.5 cm off, flagged by `visible` < 0.3 or 0 (blob). `lowhands` seed 2407
  (yellow triangle, blob, 22.7 mm); `dr_train` seeds 2017, 2005, 2009 (8–16 % visible, within 5 mm there).
- Toppled object: colour and a rough centre only (`motion` seed 2206, yellow triangle; ~1 cm).
- Cylinder cut by the lower image border: shape None; one case in 1891 decided shapes still wrong (`motion` seed 2202).
- Hue rotated by ±30° or more: wrong colours at high confidence (stress test only).
- Closed hands at the first look: hand fit biased by ~12 mm (live sim); harmless with the 25 mm rule, cause open.
- Two touching objects of one colour whose row ends are both hidden: no cue for the seam.
- Not handled: an object of a foreign colour inside a basin (purple foam reads blue).

## Real robot: what the recordings changed, and what must hold

From the organisers' drop (RGB only, no depth, so only the colour stage could be run): yellow foam is in the yellow
basin, purple foam in the blue one, orange foam in none; the bin tape reads red over the whole bin; the dark hand,
the silver forearm and the dark background read blue; the wood in no basin. So on that scene colour cannot tell bins
from red objects or the arm from blue ones: only geometry does. Dark pixels (value < 60) get no colour class now.
The scene differs from scene.yaml by centimetres and degrees (bins at x 0.45–0.56, counter height varying by
sessions, optical axis 49–55° below the table): the scene anchor will fail its 3° / 3 cm guards there and everything
rests on the hand fit, which has never seen real depth. For it to pass its checks the real depth must give: valid
depth within 8 mm of the model on at least half of the visible forearm, palm and (open) finger surface, at least 300
such pixels more than 65 mm above the table, at least half of the depth pixels within 4 cm of the arms on the
model. Doubts: stereo depth on glossy black fingers; the wrist camera mount and cables the URDF lacks; FK accuracy
of the real arm; depth edges fatter than in sim (the silhouette correction of idea 3 assumes exact edges).

## Next

1. Frames at `ARM_HOME` with closed and half-closed hands on sim `c`: find the cause of the closed-hand bias.
2. First real depth drop: hand fit on real data, then `agree_within` and the blend decision.
3. Not done: the red < yellow < blue cluster-order check. In sim every object has its own ±7° hue shift, so two reds
   differ by up to 14° and cluster order adds nothing to the basins; "all three colours present" holds at the first
   look only. It would be a policy-level rule for the first look.
4. Sub-millimetre gains available in sim only: the true principal point (−1.4 mm); not taken, it is a sim fact.
