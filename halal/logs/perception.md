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
| `closedhands` | 20 | `ARM_HOME`, hands closed (10) and half closed (10), at rest |
| `movinghands` | 168 | every tick while both hands close and reopen at `ARM_HOME` (4 seeds) |

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
objects, objects in bins, bin walls), xy 12.8 / 21.4 (median / p95). `topface` recall 0.906, precision 1.000 (0
phantoms), colour 2536 of 2536, xy 1.9 / 10.0 / 27.4; fully visible objects: recall 1.000, xy 1.9 median.
In the sim, same policy, fixed task, 8 seeds, DR (measured by the policy workstream): baseline 12/31 objects
sorted (39 %), `topface` 27/31 (87 %).

## What the detector does

1. Camera pose (`calibration.py`), at every look: counter plane → roll, pitch, height; front-wall line of the bin
   row → yaw, x; where that wall starts and stops → y ("scene anchor", depth only, 3 ms). Once per episode, when the
   robot stands still, its own forearms and hands, posed by FK of the measured joints, are registered against the
   depth image (`hands.py`). The scene anchor is used unless the hands contradict it by more than `agree_within`;
   then the hand pose; else nominal.
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
colour casts of ±20 % per channel: no effect. Sim sets: no wrong colour in 3273 detections, 99.8 % at ≥ 0.8.
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
- *Fingers in motion* (policy's calibration close, live sim: +11–13 mm in x, the hand pose overrode the scene
  anchor). Reproduced on `movinghands`: while the fingers REOPEN the fit is off by +6…+15 mm in x and +4…+7 mm in
  z (up to 22.8 mm from the scene anchor) and still passes every check (rms 2.1–2.7 mm, coverage 0.75–0.89); while
  they close it stays within 3 mm; closed or half closed at rest it is as good as open (`closedhands`: 20 of 20 within
  2.5 mm per axis). So the cause is the image not matching the measured finger joints during motion, not the closed
  posture. Fix: the hand fit is only attempted when an earlier look of the episode, at least 0.1 s before, saw every
  arm and hand joint within 0.02 rad of where it is now (and `arm_dq` is small). `movinghands`, one detector per
  seed: source scene 168 of 168, xy 1.6 / 2.3 / 2.6. Consequence: the first look of an episode never uses the hands.
  Kept from the first mitigation: finger links are left out when an index or middle joint is bent beyond 0.5 rad
  (a hand that holds something); forearm and palm alone are no substitute at `ARM_HOME` (≈ 210 visible model points:
  the fit rejects itself; on `lowhands` 11 of 20 trusted at 6.1 median / 19.0 max mm).
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
The arm pose, once a look kept one, stays for the episode; until then it is tried at up to 4 still looks.

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

**12. A stale first frame must not stick.** Live: the policy's first observation of an episode was rendered before
the evaluator's reset (previous episode's camera pose, empty counter); the one-shot calibration locked onto it and
every detection of the episode was shifted by the difference of two DR camera poses (6–25 mm). Fix: the scene anchor
is refitted at EVERY look (3 ms) and the newest is used; the stored anchor (for looks in which walls or counter are
covered) and the arm pose are only kept from looks that found at least one object. Benchmark `--stale` (each
detector first sees the frame of another episode, no `t` reset): `dr_train` 1.7 / 2.4 / 2.7 mm, recall and
precision 1.000, identical to a fresh detector; held-out likewise. Price of refitting per look: on `motion` the
camera part rises from 2.5 to 3.7 mm at worst (arms over the bin row), 1.6 median unchanged.

**13. Objects at the near image border.** The bottom image row meets the object-top plane at x = 0.213 in sim
(0.204–0.221 with ±1° camera pitch; 0.221 with the yaml intrinsics). A top face is complete in the image for a
centre x ≥ 0.236–0.245. Sweep in the mock world: centre within 1.5 mm down to 17 % of the top face inside the image
(centre 15 mm beyond the edge), shape named down to ~50 % inside, 8 mm off at 6 %, nothing beyond. So in sim an
object pushed to x < ~0.20 is gone for the head camera; the head cannot tilt.

## Known failure cases

- Object mostly under a hand: centre up to 2.5 cm off, flagged by `visible` < 0.3 or 0 (blob). `lowhands` seed 2407
  (yellow triangle, blob, 22.7 mm); `dr_train` seeds 2017, 2005, 2009 (8–16 % visible, within 5 mm there).
- Toppled object: colour and a rough centre only (`motion` seed 2206, yellow triangle; ~1 cm).
- Cylinder cut by the lower image border: shape None (the one wrong case of `motion` seed 2202 went with H10).
- Hue rotated by ±30° or more: wrong colours at high confidence (stress test only).
- Object pushed towards the robot beyond x ≈ 0.20: outside the head image, not detected.
- A hand fit taken while fingers move would be biased by up to 15 mm; prevented by the stillness rule, which needs two looks.
- Two touching objects of one colour whose row ends are both hidden: no cue for the seam.
- By design since drop 2: orange reads `red`, purple reads `blue` (colour families, below); box tape and sim bins are `red` too.

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

## Real colours and drop 2

Organisers, Sat 14 Nov: the real objects are purple / yellow / orange foam, the sim keeps red / yellow / blue. Decision
(orchestrator): three colour FAMILIES under the canonical names, `red` = red and orange, `yellow`, `blue` = blue and
purple. Drop 2 (`zupermiez/g1_junction_drop_2`, 48 episodes, RGB only) has the finals' object set: purple cube, orange
cube, orange triangular prism, yellow cube, yellow cylinder, on a bare glossy table in front of three orange-taped boxes.

Scripts (`halal/eval/real/`, environment as in realdata.md): `drop2_frames` (PNG samples under `runs/real_frames/drop2/`),
`drop2_colours` (tables below; `--sheets DIR` writes the segmentation to check), `drop2_size` (cube edge).

### Colour statistics

First head frame of all 48 episodes. Objects segmented by hue seeds and size, only blobs on the table in front of the
boxes, shapes labelled by hand: 81 orange, 62 yellow, 44 purple instances. Hue = the detector's angle (deg), cvH = OpenCV
H; pixels pooled, hue p1 / p5 / p50 / p95 / p99, the rest p5 / p50 / p95.

| class | hue deg | cvH | S | V |
|---|---|---|---|---|
| orange cube | 19 / 21 / 25 / 30 / 31 | 11 / 13 / 15 | 146 / 180 / 232 | 124 / 208 / 252 |
| orange prism | 18 / 20 / 24 / 30 / 31 | 10 / 12 / 15 | 146 / 220 / 246 | 115 / 177 / 248 |
| orange, lit faces | 20 / 22 / 25 / 30 / 32 | 11 / 13 / 15 | 145 / 176 / 232 | 173 / 218 / 253 |
| orange, shaded faces | 18 / 19 / 24 / 28 / 30 | 10 / 12 / 14 | 156 / 214 / 244 | 114 / 149 / 196 |
| yellow cube | 41 / 47 / 77 / 88 / 91 | 23 / 39 / 44 | 71 / 85 / 152 | 117 / 210 / 248 |
| yellow cylinder | 41 / 45 / 80 / 97 / 103 | 22 / 40 / 48 | 57 / 77 / 150 | 119 / 211 / 255 |
| yellow, lit faces | 59 / 70 / 81 / 93 / 100 | 36 / 41 / 47 | 61 / 77 / 94 | 187 / 226 / 254 |
| yellow, shaded faces | 39 / 42 / 56 / 81 / 85 | 21 / 28 / 41 | 84 / 116 / 164 | 109 / 137 / 181 |
| purple cube | 216 / 217 / 223 / 246 / 261 | 108 / 111 / 124 | 104 / 151 / 183 | 99 / 210 / 246 |
| purple, lit faces | 216 / 217 / 222 / 227 / 230 | 108 / 111 / 113 | 133 / 153 / 183 | 184 / 219 / 247 |
| purple, shaded faces | p5 222, p50 236, p95 259, p99 266 | 111 / 118 / 130 | 80 / 128 / 188 | 94 / 110 / 160 |
| box tape | -4 / 2 / 10 / 18 / 28 | 1 / 5 / 10 | 91 / 169 / 215 | 62 / 115 / 211 |
| bare table | any (chroma < 0.15) | any | 6 / 19 / 35 | 120 / 152 / 202 |
| hand (V < 60) | mostly ~200 | 0 / 101 / 115 | 40 / 132 / 228 | 14 / 40 / 58 |
| forearm cover (cyan plastic, left wrist) | 181 / 189 / 200 / 208 / 211 | 95 / 100 / 104 | 84 / 149 / 203 | 64 / 136 / 251 |

The colour the detector names per object (median of the most saturated 30 %, white-balanced on the table; the table is
nearly neutral, so the balance moves hues by a degree or two), min / p5 / p50 / p95 / max:

| | n | hue deg | chroma min / p50 |
|---|---|---|---|
| orange cube | 41 | 20.2 / 21.7 / 24.5 / 29.3 / 30.2 | 0.58 / 0.77 |
| orange prism | 39 | 19.4 / 20.1 / 26.6 / 29.0 / 30.9 | 0.59 / 0.89 |
| yellow cube | 34 | 40.7 / 43.6 / 55.6 / 79.8 / 81.6 | 0.37 / 0.47 |
| yellow cylinder | 23 | 45.2 / 46.0 / 52.8 / 95.9 / 98.4 | 0.24 / 0.48 |
| purple cube | 44 | 217.0 / 217.3 / 220.1 / 223.4 / 225.0 | 0.57 / 0.66 |
| tape, 40 x 40 px patches | 831 | 0.9 / 4.2 / 8.7 / 14.7 / 27.6 | 0.39 / 0.76 |

- Orange against yellow: separable by hue, gap 30.9 to 40.7 deg, about 5 deg on either side of a boundary at 36. Thin,
  but without exception in 48 episodes of one room's light. Second cue on the real foam only: orange chroma >= 0.58,
  yellow <= 0.5 (the sim's yellow is as saturated as its red, so it is not used).
- Orange object against tape: medians 25 against 10 deg (cvH 13 against 5; drop 1 said 4 to 8 units: confirmed), but lit
  tape rims reach 18 (p95) to 28 (p99) and single patches read up to 27.6, inside the objects' 19 to 31. Hue cannot tell
  a lit piece of tape from an orange object.
- Purple: this camera renders it blue. Lit faces 216 to 230, named colour 217 to 225; shaded faces drift violet to 259
  (p95) and 266 (p99), never near 290. The old basin ended at 250.
- Yellow is greenish and pale here: lit tops 70 to 100 deg (the old basin ended at 90), saturation a third of the orange's.
- In a family at pixel level without being an object (mean pixels per frame): all tape, `red`, 60 k; the cyan forearm
  cover, `blue`, 13.6 k (hue 195 to 211: sim blue's lit tops reach 200, so hue cannot cut it); arm shadows and
  reflections on the glossy table, `blue`, chroma 0.18 to 0.29, 1.6 k on the bright table and more in the shadow;
  cardboard inside the boxes, `yellow`, hue 38 to 68; the black hand 135 (the value floor holds). All of these are
  left to geometry: table height, bin footprints, pixels above an object's top.

### What changed in the detector

1. **Basins** (`topface.color.basins`): red 335..33 (was 335..15), yellow 39..110 (was 40..90), blue 195..280 (was
   195..250), taper 6 deg. Drop 2 objects under the old basins: orange 58 `red` + 23 `yellow`, all at confidence 0;
   new: 81 of 81 `red` at 1.00, 44 of 44 purple `blue` at 1.00, 62 of 62 yellow `yellow` (59 at 1.00, minimum 0.59: a
   washed-out cylinder, chroma 0.24).
2. **Own-family sample** (`color.own_share`): when at least half of an object's pixels lie in one family, only those
   are sampled. Reason: touching real objects. The colour sample is everything within 30 mm of the centre, so a
   touching neighbour contributes an edge up to 7.5 mm deep, and the most saturated 30 % of "pale yellow + a slice of
   orange" is the orange. Emulated on drop 2 (own pixels plus the neighbours' within n px; 13 px is about 7.5 mm):
   named another family or a hue between families 1 of 72 at 13 px and 15 of 93 at 20 px before, 0 and 0 after;
   2 of 122 at 30 px (a purple cube mostly hidden behind its neighbours).
3. **Wall guard** (`topface.wall_guard`), active only while the scene has not anchored the camera. With orange in the
   `red` family the sim's bin walls (hue 24) no longer fail the colour test, which had been a second line of defence:
   with the calibration switched off (nominal camera, 11 mm median error) the phantoms on `dr_train` / `motion` went
   from 14 / 15 (old basins) to 36 / 214, nearly all wall slivers from the toppled-object fallback. Now a fit's top
   cells must form a level patch (|cos| to the counter normal >= 0.7; wall faces 0.03 to 0.61, 99.5 % of true tops >=
   0.8) and a fallback blob needs depth in plan view (std >= 4 mm; wall slivers 1.0 median, toppled objects 7.7): 4 / 0
   phantoms, recall 0.996 / 0.895 (0.998 / 0.906 without the guard). Anchored, the guard does nothing.
4. **Object size** (`topface.size_range`, `size_snap`, `min_size_visible`; `ObjectDetection.size`). Top faces are looked
   for from 45 to 55 mm above the counter (band 39 to 61 mm, was 39 to 51); every blob is fitted on the plane at its
   own median height, which IS the scene file's 45 mm when within 2.5 mm of it. `size` = that height (edge length =
   height, m) when at least 30 % of the top face was seen, else None; an unexpected size with less than 30 % seen is
   not reported as an object (bin rims under a wrong camera pose looked like that). `top[2]` = counter + that height.
   Outlines are not scaled.

### Sim benchmarks, before (a158a92) and after

Recall / precision, colour right, xy median / p95 / max mm, shape decided / right of decided:

| set | before | after |
|---|---|---|
| dr_train (120 frames, 472 objects) | 1.000 / 1.000, 472/472, 1.65 / 2.45 / 2.74, 0.892 / 1.000 | 1.000 / 1.000, 472/472, 1.66 / 2.45 / 2.74, 0.892 / 1.000 |
| held-out (30, 114) | 1.000 / 1.000, 114/114, 1.62 / 2.45 / 2.69, 0.851 / 1.000 | same, median 1.63 |
| motion (1017, 2798) | 0.906 / 1.000, 2536/2536, 1.93 / 9.95 / 27.37, 0.746 / 0.999 | 0.906 / 1.000, 2536/2536, 1.94 / 9.95 / 27.37, 0.746 / 0.999 |
| lookpose (45, 188) | 1.000 / 1.000, 188/188, 1.62 / 2.40 / 2.65, 0.984 / 1.000 | same, p95 2.42 |
| lowhands (20, 77) | 1.000 / 1.000, 77/77, 1.68 / 2.66 / 22.67, 0.610 / 1.000 | same, median 1.70 |
| closedhands (20, 79) | 1.000 / 1.000, 79/79, 1.63 / 2.44 / 21.22, 0.937 / 1.000 | same, median 1.62 |
| movinghands (168, 588) | 1.000 / 1.000, 588/588, 1.60 / 2.34 / 2.66, 0.888 / 1.000 | same, median 1.61 |
| armless (40, 172) | 1.000 / 1.000, 172/172, 0.50 / 0.63 / 0.66, 1.000 / 1.000 | same |
| dr_train `--stale` | as dr_train | as dr_train |

Phantoms 0 and wrong colours 0 on every set, before and after. The hundredths move because lit pixels of red tops
(hue 15 to 33) now have a class, which shifts a few seeds. `size` in sim: reported for 0.913 of the detections on
`dr_train` (0.815 on `motion`), every one exactly 45.00 mm. The raw blob height reads 43.8 mm median (p5 43.2, p95 44.0):
1.1 mm low, the sim's principal point again (see "Facts"); that is what the 2.5 mm snap absorbs.

Other sizes, organiser mock world (no arms, 1 mm depth noise, 43 objects, detector expecting 45 mm):

| true size mm | found | xy median / max mm | `size` read mm | top z error mm | shape right of decided |
|---|---|---|---|---|---|
| 45 | 43/43 | 0.5 / 0.7 | 45.0 | 0.0 | 43/43 |
| 47 | 43/43 | 0.9 / 2.9 | 45.0 (snapped) | -2.0 | 20/20 (cylinders undecided) |
| 48 | 43/43 | 0.6 / 1.6 | 47.8 to 48.0 | -0.1 | 20/20 |
| 50 | 43/43 | 0.6 / 2.2 | 49.7 to 49.9 | -0.1 | 20/21 |
| 52 | 43/43 | 0.9 / 2.4 | 51.7 to 51.9 | -0.2 | 20/43 |
| 55 | 43/43 | 0.8 / 5.3 | 54.7 to 54.9 | -0.2 | 20/43 |

Position, colour and size hold over the range; the SHAPE does not above 50 mm (every cylinder is called a cube: the
45 mm outlines are not scaled with the size). Scaling them is the fix if the finals' objects turn out larger.

### The real palette under the sim's light (`--repaint`)

`perception_bench --repaint` turns the hue of every lying object's pixels (ground-truth outline) by `bench.repaint_deg`:
red +25 (orange foam), yellow +4, blue -7 (purple foam), everything else of the sim's randomisation staying on top.

| set | recall / precision | colour right | wrong colours | phantoms |
|---|---|---|---|---|
| dr_train | 1.000 / 1.000 | 446/472 (0.945) | 26, all orange named `yellow`, 18 at confidence >= 0.8 | 0 |
| motion | 0.904 / 1.000 | 2481/2530 (0.981) | 48, all orange named `yellow` | 1 |

Yellow and blue-as-purple: 0 wrong in 2190. Orange: 26 of 158 on `dr_train` cross into yellow, exactly the objects the
sim gave a hue shift of +3.2 to +7.1 deg (those named right: 0.1 to 7.0, median 3.3). So the families hold for purple
and yellow under the sim's randomisation, and for orange only up to about +5 deg: the sim's per-object hue shift of
+-7 deg is wider than the real gap allows. The real foam's own spread (19.4 to 30.9 over 81 objects and 48 episodes)
is inside the basin; what would break it is a hue shift of the whole image that the table white balance does not
remove. The boundary was not moved on this: 36 deg is the middle of the measured gap. A policy-level cue remains
for the finals: the scene holds two orange and two yellow objects, and the orange ones are by far the more saturated.

### Red against yellow by the scene (`color.warm_split`)

The fixed boundary at 36 deg has 5 deg of margin each way. Now a look's own object colours can vouch for it or move it.
Among the warm object hues of one look, a gap of at least `min_gap` (8 deg) counts if some turn of all hues by at most
`max_shift` (9 deg) puts its lower side in the red basin and its upper side in the yellow one. If such a gap needs no
turn, the basins stand (vouched). If exactly one gap fits and needs a turn, red and yellow meet in its middle. Several
fitting gaps, or none: nothing is concluded. What a look concludes holds for the episode (objects leave one by one).
Without a vouched boundary the confidence of a warm object is also limited by its distance to 36 deg (0 at the
boundary, full from 6 deg): a decision resting on the fixed boundary alone is not sure near it.

Why these numbers: sim, 1390 looks: red 16.3 at most, yellow 45.2 at least, so red to yellow is at least 34 deg and
always spans the basins' edge (vouched, same names as before). Two sim yellows differ by up to 12.9 deg and would pass
for "orange and yellow" only under a turn of 12.2 deg or more; two sim reds never. A look is not required to hold both
groups: 491 of the 1390 sim looks hold one.

| | fixed boundary | boundary from the look |
|---|---|---|
| sim sets as shipped (all nine) | unchanged | unchanged |
| drop 2 as recorded | orange 81/81, yellow 62/62, purple 44/44 | same, same confidences |
| drop 2, whole image turned +5 deg | orange 80/81 | 80/81 |
| drop 2, +10 deg | orange 32/81 | 73/81 |
| drop 2, +15 deg | orange 0/81 | 13/81 |
| drop 2, -5, -10, -15 deg | all right | all right |
| repainted `dr_train` (orange right / matched) | 131/154 | 147/157 |
| repainted `dr_train`, +5 / +10 deg | 99/156, 78/157 | 122/156, 94/158 |
| repainted `motion` | 604/652 | 632/652 |
| `dr_train` as rendered, -15 .. +15 deg | all right but yellow 150/155 at -15 | all right |

Yellow and purple stay right in every row; repainted blue is lost (confidence 0, never wrong) from -10 deg on: 5 of 158,
and 32 at -15. Reading: negative turns are harmless on drop 2 because of the own-family sample (the pale yellow tops
stay in the yellow basin and name the object); positive turns push orange over the boundary, the fixed scheme is gone at
+10, the look's boundary holds to +10 and is gone at +15, by design of `max_shift`. On the repainted sim it halves the
errors and no more: there the sim's +-7 deg per-object shift makes two oranges differ by more than orange and yellow
do, and then no single gap exists. That case is not in the real foam (19.4 to 30.9 over 81 objects).
`drop2_colours --sweep` and `perception_bench --repaint --rotate DEG` reproduce the rows.

### Prisms lying on a side (`topface.roof`, `ObjectDetection.lying`)

A prism on a rectangular side shows no top face, only a ridge; it used to come back as a fallback blob (`shape=None`,
`visible=0`, about 1 cm off). Now every such blob is tried as a roof (`halal/perception/roof.py`): the plane with most
points among those tilted 45 to 75 deg against the counter (120 three-point trials, 2 mm), a second one leaning against
it in the remaining points, and a flatness test (of the points within 6 mm of the plane, 70 % must be within 2 mm: the
side of a lying cylinder bends away). The ridge is where the two faces meet, or the first face's line at the 98th
percentile of its heights. It is a lying prism when the ridge is 30 to 52 mm up, the faces 20 to 65 mm long, the first
face at least 20 mm from eaves to ridge and 60 % of the blob on the faces. Reported as `shape="triangle"`,
`lying=True`, `top` = middle of the ridge, `yaw` = ridge direction (mod 180 deg), `size` = ridge height as measured (no
snap, no clip), `visible` = share of the blob on the faces. A lying cylinder is not named: it stays a blob.

`motion` (the only set with toppled prisms: 237 lying still on the counter, two episodes, ground truth from the
quaternion; the bench line "prisms lying on a side"):

| | before | after |
|---|---|---|
| lying prisms found at all | 200/237 | 200/237 |
| reported as lying prisms | 0 | 185 (0.781 of 237) |
| ridge direction error, deg: median / p95 / max | none | 0.3 / 2.8 / 3.7 |
| ridge centre xy, mm: median / max | blob centre, 10 to 14 median | 2.4 / 13.0 |
| `size` (truth: sharp ridge at 45, rounded by the 2.5 mm fillet) | none | 43.3 median, 38.9 to 44.2 (p5 to p95) |
| called lying without being one | none | 6 of 191 |
| whole set: recall / precision / colour | 0.906 / 1.000 / 2536 of 2536 | same |
| whole set: xy mean / p95 / max mm | 2.9 / 10.0 / 27.4 | 2.4 / 5.6 / 29.2 |
| whole set: shape decided / right | 0.746 / 0.999 | 0.822 / 0.997 |

The 6: five are matched to a cube 24 to 29 mm away from it (so they are something next to the cube, not the cube; the
frames were not looked at, and the set's xy maximum of 29.2 is one of them), one to a triangle the ground truth counts
as upright. The other eight sets are unchanged and contain no lying call.

Mock world (the organisers' renderer only draws boxes and cylinders; prisms ray-cast from scratch code into its image,
1 mm depth noise, no arms; 72 poses: 2 distances, 3 lateral positions, ridge every 15 deg):

| ridge height | seen as lying | ridge direction, deg: median / p95 / max | ridge centre xy, mm: median / max | `size` minus true, mm |
|---|---|---|---|---|
| 45 mm | 69/72 | 0.3 / 1.2 / 1.5 | 0.9 / 1.7 | -0.6 (-1.3 to -0.2) |
| 39 mm (real foam) | 71/72 | 0.3 / 1.0 / 1.4 | 0.8 / 1.4 | -0.5 (-1.2 to -0.2) |

The misses are ridge directions 15 and 90 deg at the lateral positions: they stay fallback blobs.
Not verified: real depth on foam (the 2 mm plane tolerance is sim noise; real stereo at 0.55 m is about 1.5 mm per
pixel and smears edges), a prism touching a box or another object. Not looked into: the 15 lying prisms in `motion`
that are found but stay blobs, and the 37 not found at all.

### Real object size

`drop2_size`: the centre box (21 cm outer, scene.yaml) is the ruler. Its six outline corners, read by eye in one frame
per camera view, give the camera's pitch, roll and height over the table; a cube's lit top face is then placed at the
height where its edge equals its height. Two views (episodes 36 to 47 and 16 to 18): optical axis 56.6 / 56.4 deg below
the table plane, camera 467 / 483 mm above the table, box wall 70 / 70 mm (sim 60), worst corner 7.5 / 6.1 px.
16 cube top faces (purple, orange): edge median 45.9 mm, p16..p84 44.7..46.6, min 42.9, max 47.0; box corners read 3 px
off move a value by about 1 mm. So the real cubes are the scene file's 45 mm (within 1 to 2 mm), not 50. Not measured:
cylinder diameter, prism, yellow cube (no clean square top). In most first frames the orange prism lies on a
rectangular side: a ridge, no top face, so the detector can only report it through the toppled-object fallback
(`shape=None`, `visible=0`).

### Unverified without real depth

- Everything geometric on the real scene: the counter plane on a glossy table, top faces of foam at 45 mm, the hand fit,
  the bin finder, the wall guard (thresholds are from sim depth; real walls are 70 mm and cardboard).
- `size` on real depth: expected within about 2 mm (D435 at 0.55 m: 1.5 mm per pixel, averaged over a face; what remains
  is bias between the table fit and the foam top), so real cubes should snap to 45 mm.
- The forearm cover and table shadows are `blue` by hue; that they never form a top face at 45 to 55 mm is an assumption.
- A prism lying on its side next to a taped box, and whether the fallback separates the two.
- The orange / yellow boundary under the finals' light: 5 deg of margin each way, measured in one room; the repainted
  sim shows what a +5 deg shift does (orange named `yellow` at full confidence).

## Real depth and a real table, emulated (14.11 afternoon)

No real depth has been published, so two bench conditions stand in for it. They are models, not measurements.

**`--depth mild|typical|bad`** (`halal/eval/depth_sensor.py`, `bench.d435`, seeded by `--depth-seed`, frame seed and tick): the
sim depth as a RealSense D435i would deliver it, worked out in the aligned 640x480 image.

| effect | source | mild / typical / bad |
|---|---|---|
| range noise, `Z^2 * subpixel / (f * B)`, B = 50 mm, f = 425 px (848x480, HFOV about 90 deg) | Intel, "BKMs for tuning D4xx cameras": subpixel rms 0.08 well tuned, below 0.1 expected, above 0.2 recalibrate | 0.05 / 0.08 / 0.2 px: 0.8 / 1.4 / 3.4 mm at 0.6 m, 70 % of it smooth over the matching window |
| disparity steps of 1/32 px | D400 depth format | 0.5 mm at 0.6 m |
| disparity bias (calibration) | datasheet: within 2 % at 2 m, which is 0.2 px | 0 / 0.05 / 0.2 px, sign drawn per frame |
| no depth left of every nearer surface | stereo occlusion; left imager is the reference, colour sits 15 mm beyond it, so 65 mm to the right imager | 40 px beside an arm 0.35 m away over the counter, 3 px beside an object |
| edges moved, foreground fattened at edges, thin parts smoothed | block matching window | jitter 0.5 / 1 / 2 px, fattening 0 / 1 / 2 px, median 3 / 5 / 5 px |
| flying and dropped pixels at depth steps | Intel support threads on object outlines | 5 + 10 / 15 + 30 / 40 + 50 % of edge pixels |
| glossy black Dex3 links (by FK through the true camera) | assumption: IR absorbed or mirrored | fingers return 80 % / palm and fingers 40 % / palm and fingers nothing, and every other dark pixel 30 % |
| holes on reflecting surfaces | assumption (bare glossy table) | 0.5 / 2 / 6 % of the image in 10 px blobs |

Not modelled: the wrist camera mount and cable the URDF lacks, FK error of the real arm, motion blur, exposure.
The sim frames already carry the organisers' DR depth noise (1 mm + 0.2 %, 1 % holes); the model comes on top.

**`--shift near|real|far`** (`halal/eval/scene_shift.py`, `bench.shifts`): the frame re-rendered with the table where the
recordings have it: bins 5 / 7 / 14 cm further, counter 20 / 15 / 0 mm higher, table pitched 1.5 / 7 / 9 deg (far edge up, about
the middle of the object band). Arms and background stay, table points are moved and splatted, uncovered counter is drawn as
a bare plane, ground truth moves along. A raised table brings the objects up under the thumbs of `ARM_HOME`, so the `hidden`
objects are harder than in the recorded frames (recall with clean depth: 0.92 to 0.98).

**`python -m halal.eval.calibration_bench`** (same options) runs every calibration stage on each frame as a first look would.

### Stages, `dr_train` dev (90 frames), table as recorded | `--shift real`; error at the object band, mm, median / p95 / max

| stage | clean | mild | typical | bad |
|---|---|---|---|---|
| counter plane, height error | 0.6 max 0.7, 1.0 max 1.1 | 0.5, 1.0 | 1.2 max 1.4, 1.7 max 1.9 | 3.2 max 3.8, 3.6 max 4.1 |
| scene anchor accepted | 90, 0 | 90, 0 | 90, 0 | 90, 0 |
| scene anchor, band | 1.5 / 1.6 / 1.7 | 1.5 / 2.0 / 2.2 | 1.7 / 2.3 / 2.7 | 3.9 / 5.0 / 5.8 |
| hand fit believed (before) | 90, 90 | 90, 90 | 90, 90 | 74, 76 |
| hand fit, band, all fits | 3.9 / 4.3 / 4.4, 3.7 / 4.0 / 4.2 | 4.2 / 4.6 / 4.8, 3.9 / 4.2 / 4.4 | 4.0 / 5.1 / 5.5, 3.7 / 4.7 / 5.1 | 4.4 / 8.5 / 17.5, 4.2 / 8.1 / 17.3 |
| bins through the hand camera (before) | 3.0 / 5.0 / 7.0, 2.9 / 5.8 / 7.6 | 1.0 / 5.0 / 8.1, 2.9 / 4.4 / 6.5 | 2.2 / 5.1 / 10.6, 2.2 / 5.3 / 13.4 | 5.0 / 12.6 / 43.1, 2.9 / 67.9 / 212.1 |
| counter height under the band, hand camera | 3.6, 3.6 | 3.8, 3.8 | 3.8 max 4.5, 3.8 max 4.5 | 3.5 max 10.5, 3.3 max 8.9 |

3.6 mm of every hand-camera height is the sim's principal point (see "Facts"), not the method. Every check of the hand fit but
coverage passed on every frame at every level. Guards of the scene anchor do their job: 0 of 90 accepted on a displaced table.

Detector end to end (two looks, before any change), recall / precision, xy median / p95 / max, z median / p95, shape decided / right:

| | clean | mild | typical | bad |
|---|---|---|---|---|
| table as recorded | 1.000 / 1.000, 1.7 / 2.4 / 2.7, 0.0 / 0.3, 0.91 / 1.000 | 1.000 / 1.000, 1.5 / 2.4 / 21.5, 0.0 / 0.3, 0.84 / 1.000 | 1.000 / 1.000, 1.3 / 3.0 / 22.5, 0.0 / 0.3, 0.80 / 0.996 | 0.994 / 0.997, 2.2 / 8.2 / 26.1, 0.1 / 0.3, 0.61 / 0.991 |
| `--shift real` | 0.978 / 1.000, 1.7 / 2.7 / 26.0, 3.8 / 4.2, 0.89 / 0.997 | 0.950 / 1.000, 2.1 / 3.2 / 18.3, 4.1 / 4.7, 0.87 / 0.997 | 0.958 / 1.000, 2.3 / 4.5 / 17.1, 4.0 / 4.9, 0.82 / 1.000 | 0.950 / 0.977, 4.3 / 18.3 / 23.5, 3.5 / 11.4, 0.57 / 1.000 |

Camera part under `--shift real`: 2.1 / 2.6 / 3.2, 2.1 / 2.7 / 3.1, 2.3 / 3.3 / 3.7, 4.0 / 17.3 / 24.4 (bad: 14 frames on the nominal
pose). Everything beyond 8 mm up to typical is an object partly under a thumb. Upright prisms there are called `lying` under
noise: 4 / 5 / 7 of 358 objects at mild / typical / bad on the table as recorded, none with clean depth. Colour: 1.000 throughout.

Reading: with the black links returning 40 % in patches, the hand fit holds (5 mm p95 at the band) and the displaced table costs
nothing beyond the hand fit's own error. The stages that broke were at `bad`: the coverage check of the hand fit, and the bin finder.

### Hardening

**H1. Coverage of the hand fit counts only surface that returned depth.** A model point with no depth at its pixel is no
evidence that the arm is elsewhere; it used to count against the fit, so hands that return nothing meant "nominal".
`coverage` = covered / (covered or depth at its pixel), `known` reports the share with any evidence, restarts still go by the
whole-model share. `min_coverage` 0.5 -> 0.7 on the new basis (dev: true fits 0.77 at least at typical, 30 mm fits of `lowhands`
at bad 0.55 to 0.60). bad, `--shift real`, dev: believed 76 -> 90 of 90, camera part p95 17.3 -> 7.1, max 24.4 -> 18.8, object xy
p95 18.3 -> 9.5. Held-out, bad: believed 27 -> 30 of 30, camera part p95 10.9 -> 6.8, max 21.6 -> 14.9. Clean, mild and typical:
the same numbers on every set; `--armless 10` believed 0 of 40, also with typical and bad depth. Left over: at bad, 3 of 90 dev
fits are 12 to 17 mm off and pass every check (coverage 0.69 to 0.77, one rejected by the new threshold).

**H2. Bin row found jointly.** Each lane used to need 6 seen cells for its own front wall; with arms and holes over the side
lanes the far wall (21 cm on) was the first that qualified, and the row's median followed two wrong lanes. Now the row's front
wall is the nearest x that fills the seen cells of all three lanes together, and each bin's own wall is looked for within
`row_window` (3 cm) of it with 3 seen cells. Bins through the hand camera, bad, `--shift real`: 2.9 / 67.9 / 212.1 -> 2.9 / 9.6 /
65.1, frames with a bin more than 2 cm off 9 -> 3 of 90 (wrong y from a wall end); table as recorded: 5.0 / 12.6 / 43.1 -> 5.0 / 10.9 /
43.1. Clean, mild, typical: unchanged within 1 mm (typical, real: 2.9 / 4.0 / 13.4).

**H3. `SceneLayout.rim_z`.** Heights (above the counter plane) of the points over the found bin footprints are counted in 2 mm
bins; the rim is the highest bin with half the count the walls show below 45 mm, lifted to the highest footprint corner. Signed
error against the true highest rim, dev, median / p95 / max (lowest): clean +9.3 / +10.1 / +10.2 (+7.5), typical +9.9 / +12.5 /
+15.4 (+7.5), bad +11.8 / +18.5 / +25.8 (+6.8); `--shift real`: +9.4 / +10.4 / +10.7 (+8.3), +10.3 / +13.9 / +16.4 (+7.4), +11.6 /
+18.8 / +27.7 (+5.5). Never low. Through the true camera the same frame reads +2.5 mm (the upper edge of the height bin): the
other 7 mm are the sim principal point again (hand camera 3.6 mm high and pitched 0.6 deg, which lifts the far corners).

**H4. Bins laid along the row, not looked for in lanes.** The recordings have the row 3 to 7 cm sideways with gaps of about
2.2 cm (scene: 5 cm). `--shift narrow | left3 | right3 | left7 | right7` add that to `real` (`bins_y`, `gap`); since then the
displaced boxes are drawn anew from their geometry (moved pixels left torn walls where an arm had hidden them), so the bin
figures of the table above (boxes as moved pixels) are not comparable to the digit. With lanes fixed at the scene y (H2), bin
centre error, dev, clean depth, median / p95 / max mm: narrow 4.2 / 28.2 / 28.2, +-3 cm 3.5 / 5.8 / 5.8 and 3.1 / 5.8 / 58.0,
+-7 cm 42.1 / 98.0 / 98.0 (every frame more than 3 cm off); with mild to bad depth 58 mm at +-3 cm in every frame.
First try, not kept: cut the row's front wall into boxes at the seen gaps. Clean: 3.5 / 5.8 / 7.6 at every offset, but with
typical depth 1 to 2 frames of 90 were 3 to 6 cm off: a 2 cm gap lies in the IR shadow of the box edge and is hidden, not seen
past, and the outer ends of the side boxes are behind the arms.
Shipped: (1) the row's front wall is the nearest x where wall fills half of what is seen over the row's whole width
(`row_fill`); (2) along it, three boxes of the scene file's width with one common gap are laid where they best explain the
cells (+1 wall, -1 seen past, hidden cells say nothing): slide +-15 cm and gap 0 to 7 cm in 2 mm steps, then 0.5 mm steps
around the best, a weak prior to the scene file for ties; (3) each box's x from its own front wall within 3 cm of the row.

| bins through the hand camera, dev, median / p95 / max mm (frames of 90 with a bin > 3 cm off) | clean | typical | bad |
|---|---|---|---|
| table as recorded, lanes (H2) | 3.0 / 5.0 / 7.0 | 1.0 / 5.0 / 10.6 | 5.0 / 10.9 / 43.1 (3) |
| table as recorded, now | 2.9 / 5.4 / 6.1 | 2.3 / 5.7 / 7.3 | 5.3 / 11.6 / 220.0 (1) |
| `real`, lanes | 2.9 / 5.8 / 5.8 | 2.9 / 4.2 / 13.4 | 2.9 / 9.9 / 61.1 (1) |
| `real`, now | 4.1 / 6.2 / 6.3 | 3.7 / 6.7 / 7.7 | 4.1 / 11.0 / 22.3 (0) |
| `left7`, lanes | 42.1 / 98.0 / 98.0 (90) | 42.1 / 98.0 / 98.0 (90) | 42.1 / 98.0 / 99.6 (90) |
| `left7`, now | 3.2 / 6.3 / 6.3 | 3.5 / 6.2 / 7.8 | 4.5 / 11.6 / 238.4 (2) |

narrow, left3, right3, right7 with the 2 mm search alone: clean 6.7 max, typical 8.5 to 12.3 max, no frame over 3 cm; bad 2 to 3
frames of 90 (a row with no gap seen fits one box pitch aside, or the smeared front wall never fills half of one x cell and
the far wall is taken). Held-out, 30 frames: `real` / `left7` / `right7`, typical: 3.8 / 6.8 / 7.7, 3.5 / 6.6 / 7.8, 4.1 / 7.3 / 9.2;
bad: 16.2, 11.2, 13.7 at most. Cost against the lanes: about 1 mm in the median on a table with the scene's own spacing.
Assumptions left: the three boxes have the scene file's width and one common gap (boxes placed by hand with unequal gaps
are off by half the difference), and the row is not turned by more than the 3 cm window allows.
Detector end to end, dev, `real` / `narrow` / `left7` / `right7`, two looks: clean recall 0.978, precision 1.000, xy 1.7 / 2.7 /
26.0 in all four; typical 0.958, 1.000, 2.3 / 4.5 / 17.1 in all four. `lowhands --shift real --depth typical`: precision 0.926
-> 1.000 (the phantoms were torn pieces of moved bins).

Added after H4 (`row_smear`): the row's x is first judged over three x cells taken together (wall where any of them shows wall),
then sharpened inside them: bad depth spreads a wall over 1 to 2 cm and no single cell reached half. Table as recorded, bad:
5.3 / 11.6 / 220.0 -> 5.3 / 11.6 / 39.2; every other figure of the table above, and all of clean to typical, unchanged to the digit.

**H5. A roof face must still be sloped after its refit** (`roof.py`, one line). Cause of the upright prisms called `lying`:
`sloped_face` picks the best trial plane among those tilted 45 to 75 deg, then refits it to its points and never looked at the
tilt again. The front face of an upright prism under a thumb (its top hidden or in IR shadow) is caught by a 74 deg trial plane
and refits to 84 to 86 deg: a wall, reported as a lying prism 17 mm from the object with `visible` 0.95.

| | before | after |
|---|---|---|
| `dr_train` dev, called lying without being one: clean / mild / typical / bad (of 358) | 0 / 4 / 5 / 7 | 0 / 0 / 0 / 1 |
| `motion` clean: called lying without being one, lying prisms seen as such | 6, 0.781 | 0, 0.814 |
| `motion` clean: shape right of decided, xy p95 / max | 0.997, 5.6 / 29.2 | 1.000, 5.4 / 27.4 |
| `motion` mild: false lying, seen as such, shape right | 8, 0.717, 0.997 | 0, 0.776, 1.000 |
| `motion` typical | 7, 0.688, 0.997 | 1, 0.743, 0.999 |
| `motion` bad | 8, 0.447, 0.991 | 0, 0.443, 0.996 |

So the six "called lying without being one" of the lying-prism section above were this bug, not something next to a cube.
More true lying prisms are seen because a wall-like second face no longer displaces the ridge. Ridge under typical depth
(`motion`): direction 1.1 / 4.5 / 8.5 deg, centre 2.7 median / 18.9 max mm (clean 0.3 / 2.7 / 3.7, 2.4 / 13.0). Recall, precision and
colour of `motion` and all other sets unchanged. What is left of the objects under a thumb with noisy depth: they come back as
fallback blobs (`visible` 0, centre about 2 cm off, on the front face) instead of top-face fits with `visible` 0.1; neither is
graspable from that view, both need the look with the arm away.

`motion` under depth noise, whole set (recall / precision / xy median / p95): clean 0.906 / 1.000 / 1.9 / 5.4, mild 0.855 / 1.000 /
1.6 / 5.4, typical 0.853 / 0.998 / 1.4 / 6.2, bad 0.849 / 0.991 / 2.6 / 11.7.

### Looks of one episode in sequence (`--still`) and recorded placements (`scene.json`)

`perception_bench --still` keeps, per episode, the frames whose joints stand where the previous recorded frame had them (the
moments a policy looks), one detector per episode in recording order, and every run now prints the layout per look against
the truth: source, bin centres, counter height, `rim_z`, and every change of source inside an episode. A set recorded from a
displaced sim (the real-look sim) carries `scene.json` with its offsets and wall height; ground truth and layout truth follow it.

`motion --still` (8 organiser-baseline episodes, 135 looks, arms anywhere, objects leaving one by one):

| | sources | objects: recall / precision, xy median / p95 | bins by hands, median / p95 / max mm | counter mm | rim mm, signed |
|---|---|---|---|---|---|
| as recorded, clean | scene 135 | 1.000 / 1.000, 1.8 / 2.6 | (scene file) | 0.0 | none |
| `--shift real`, clean | nominal 5, hands 130 | 1.000 / 1.000, 1.5 / 7.3 | 4.2 / 6.3 / 6.8 | 3.3 | +6.7 (+6.1 to +8.0) |
| `--shift real`, typical | nominal 5, hands 130 | 0.998 / 0.983, 2.1 / 4.3 | 3.9 / 6.7 / 7.8 | 3.2 | +9.6 (+7.3 to +13.0) |
| `--shift left7`, typical | nominal 5, hands 130 | 0.998 / 0.981, 2.1 / 4.3 | 3.7 / 6.1 / 10.6 | 3.2 | +9.5 (+7.3 to +12.1) |
| `--shift right7`, typical | nominal 5, hands 130 | 0.998 / 0.981, 2.1 / 4.3 | 4.1 / 7.0 / 8.4 | 3.2 | +9.7 (+7.1 to +12.6) |
| `--shift real`, bad | nominal 5, hands 130 | 0.990 / 0.975, 5.4 / 11.1 | 5.0 / 11.9 / 15.3 | 1.8 max 6.4 | +10.0 (+4.7 to +15.0) |

The only source change in any episode is nominal -> hands after the first still look (the hand fit wants a second one: 5 of
the 8 episodes have a still first frame); once the arm pose is kept it holds to the end whatever the arms do, no look has a bin
more than 3 cm off, and the camera part stays 2.0 / 2.7 (typical). The "phantoms" under the shift (8 looks of seed 2201) are a
yellow cube that leaned on a bin wall in the recording and stands on the counter in the re-rendered scene: a bench artefact.

**H6. The arm pose accumulates over looks** (`topface.arm_fits`, `SelfCalibration.consensus`). The first believed hand fit used
to fix the camera for the episode. Now believed fits are gathered at up to 5 still looks and the one nearest to all others at
the object band is used (with one or two fits: the first). Why: at bad depth 3 % of single fits are over 10 mm and pass every
check. The same frame under six depth seeds (45 dev frames, `--shift real`), band error median / p95 / max mm: single look
typical 3.7 / 4.6 / 6.4, bad 4.3 / 8.1 / 16.8 (8 of 270 over 10 mm); nearest-to-all of three looks 3.7 / 4.4 / 4.4, 4.2 / 8.2 / 12.5;
of five 3.7 / 4.1 / 4.4, 4.2 / 8.5 / 11.0. Two of 45 frames stay over 10 mm at bad: there most seeds are off (seeds 2010, 2074),
so it is the frame, not the noise. `motion --still --shift real`: clean and typical unchanged within 0.1 mm, bad camera part
5.5 / 6.6 -> 5.0 / 6.4. All clean sets including `motion`: identical. Cost: up to four more hand fits per episode (30 ms each,
200 ms when every restart is tried).
Tried and dropped: left arm and right arm fitted separately as a cross-check. At `ARM_HOME` the right arm shows 211 model
points (300 are needed for a fit), so there is nothing to compare.

**H7. Objects of no colour family are reported, with `color` None.** A photo of the venue table shows black cubes, cylinders
and prisms next to the foam. `perception_bench --blacken blue` paints one colour's objects black (grey value x 0.15). Before:
0 of 158 (`dr_train`) and 0 of 65 (`lookpose`) black objects detected, clean or typical depth: the top-face fit found them
and the colour confidence of 0 dropped them, so the policy could not even avoid them. Now every geometry-accepted object
comes back; its `color` is None and its `confidence` below `min_confidence` when no family fits (black, grey, a hue between
families). Black objects: `dr_train` recall 1.000 clean, 0.962 typical (the misses are under a thumb), xy 1.7 / 2.5 and 1.3 / 2.9,
shape right 1.000 of 0.87 and 0.79 decided; `lookpose` 1.000 / 1.000, clean and typical. The red and yellow objects of the same
frames: unchanged. All unpainted sets including `motion`: identical line by line, so no hand, wall or shadow has become a
colourless object in 1401 sim frames. Not found: a black prism lying on its side (the fallback blob starts from coloured pixels).
Not verified: black foam in real depth (if it returns as little as the black hand, it has no top face and stays unseen).

**H8. A fourth colour, black** (orchestrator's order after the venue photo; detector and parser together, neither alone).
Detector (`color.black`): a geometry-accepted object is black when most of its body pixels are dark against the bare counter
(value below 0.5 x the counter's median brightness); confidence 0 at half of the pixels, 1 at 80 %. A dark object is black
whatever its hue: the black hand has a blue cast that would otherwise name it blue. Stragglers (no top face) are never black.
- `--blacken blue`, black objects found / named black of those found / at confidence >= 0.5: `dr_train` clean 158 of 158 / all / all;
  typical 152 of 158 / all / all; bad 89 of 158 / all. `lookpose` 65, 65, 46 of 65, all named black. At bad 1 of 403 (`dr_train`)
  and 3 of 169 (`lookpose`) rightly named detections of any colour fall below 0.5. At bad the depth model takes
  70 % of every dark pixel's depth, so black objects lose their top face: that is the real risk for black foam, not the colour.
- Nothing else turns black: every unpainted sim set including `motion` is identical line by line; with `--repaint` (real
  palette) on `dr_train`, `lowhands`, `closedhands`, `lookpose`: 0 black among 815 detections; real drop 2, 187 foam objects of
  48 first frames (purple 44, orange 81, yellow 62): dark share 0.000 at most (the table reads 152, the limit is 76; the
  objects' darkest pixels are above it), while the hand's dark pixels have share 1.00. So the rule separates by a wide margin,
  and what keeps the hand from being a black object is geometry alone, as before.
Parser (`halal/language/grammar`): `EXTRA_COLOURS = ("black",)`. The grammar's object types are the nine of the scene unless
the instruction names black; then black joins them for that instruction ("everything else", negation and key rebuilding
include it). Words: black, blacks ("dark" is not taken). All 10 846 rendered cases of the three existing sets parse to the
same mapping and confidence as before, compared one by one; scores unchanged (dev 2781 cases 1.000 exact, 1.934 points; test
2853, 1.000, 1.909; fresh 4087, 1.000, 1.935; real 1125, 1.000, 1.655). New `halal/eval/paraphrases_black.yaml`, 12 entries, 249
cases: 1.000 exact, all confident (`language_bench --split test --file halal/eval/paraphrases_black.yaml`; the bench scores the
black object types whenever a text names black).
What follows for the policy: a black object is a target only if the mapping reaches it: through a `black` key, or through a
shape key or `default` ("all cubes left", "everything else right"), exactly like any colour. Under a colours-only
instruction that does not name black it has no bin and stays an obstacle.

**H9. `agree_within` 25 -> 12 mm** (the change "Next 1" was waiting for). A table displaced by 1 to 1.5 cm passes every guard of
the scene anchor (3 deg, 3 cm), and with hands and anchor "agreeing" within 25 mm the scene file was believed: the camera was
moved to put the row where the file has it, and with it every object. `--shift nudge` (bins +1 cm, counter +5 mm, 0.5 deg) and
`push` (bins +1.5 cm only), dev: anchor accepted 90 of 90, pose by scene, band error 13.5 / 13.6 / 13.6 and 16.0 / 16.0 / 16.0 mm,
while the hand fit of the same frames is 3.9 / 4.2 / 4.4. How far anchor and hand fit are apart at the band, median / p95 / max:
table as recorded: clean 5.8 / 6.1 / 6.3 (`lowhands` 7.2), mild 6.1 / 6.7 / 7.2, typical 6.0 / 7.3 / 9.7, bad 6.8 / 11.9 / 26.1;
`nudge` 17.8 / 18.2 / 18.5 (typical 19.8 max), `push` 19.0 / 19.4 / 19.7, `slight` (2 cm) 30. So 12 mm lies between everything
measured on the true table up to typical depth and the smallest displacement tried. After: `nudge` / `push` use the hands in
90 of 90, camera part 2.4 / 2.9 / 3.5 and 2.4 / 3.0 / 3.5 clean, 2.5 / 3.5 / 4.1 and 2.5 / 3.6 / 4.2 typical. All clean sets including
`motion` and the table as recorded under typical depth: identical. Cost: at bad depth on the true table 4 of 90 looks now go by
the hands (camera part p95 4.1 -> 5.4, max 6.2 -> 16.9). In sim 6 mm of the disagreement is the principal point; on the real
robot a displacement under about 1 cm in the direction the hands cannot tell still goes by the scene file.

**H10. No shape for an outline that is partly seen as not top face** (`topface.max_seen_empty`). First finding of the real-look
sim frames (`realsim_looks`, seed 1001): a yellow cylinder knocked onto its side lay against a cube for 38 recorded frames and
came back as an upright `triangle`, `visible` 0.7, centre within 5 mm. The strip of its back that lies within the depth
tolerance of the top plane is 26 x 45 mm, and the triangle outline is what fits a strip best. What tells it from a flat top:
the outline cells not seen as top face are SEEN (the back curving away), not hidden. Share of the outline seen as not top,
detections with a right shape: at most 0.076 over `dr_train`, `motion`, `lowhands`, `lookpose`, `realsim_looks`, clean to
typical depth (1900 shapes; bad depth: p95 0.19, max 0.26); the 38 lying-cylinder frames: 0.22 to 0.34; the one wrong shape
left in `motion` (a cylinder cut by the image border, called a triangle): 0.38. Above 0.15 the shape (and yaw) is now None.
`realsim_looks` (3 seeds so far): shape right of decided 0.925 -> 1.000. `motion`: the one wrong shape is gone (464 cylinders
right, 1 -> 0 wrong), everything else of every clean set identical. Cost: at bad depth about one decided shape in eight
(held-out, real, bad: decided 0.552 -> 0.524), none up to typical. A lying cylinder is thus an object of unknown shape with a
good centre; it is not named as lying.

**H11. On the nominal pose the layout is what that camera sees, not the scene file's.** When neither the scene anchor nor the
hands fix the camera (every first look on a displaced table, and every look if the hand fit fails), bins used to be taken from
the scene file. On a displaced table that left the far part of the real boxes outside the excluded footprints, and their
cardboard insides (the yellow family) came back as `yellow` blobs at full colour confidence: `realsim_looks`, first looks,
typical depth: 2 per look at x 0.68. Bins, counter height and rim are now found by geometry whenever the source is not
"scene"; objects and bins then share the nominal camera's error instead of differing by the table's displacement.
First looks, `dr_train` dev, `--shift real`: precision clean 0.989 -> 1.000, typical 0.887 -> 1.000 (44 phantoms -> 0; `left7`
typical 0.958 -> 1.000); bins 67.9 mm (98 to 119 with the lateral offset) -> 11.7 / 21.4 / 30.1 (the nominal camera's own 11 mm
included), counter 15.0 -> 6.6 (max 18); `rim_z` reported there too: +12.7 median, -4.0 to +33 (so on the nominal pose it can
be a few mm low). Recall, xy and every clean set: unchanged. A look anchored to the scene file still gets the file's layout.

### The real-look sim, whole episodes (`realsim_looks`)

Recorded by the realsim workstream: policy v18, `sort_fixed`, seeds 1000 to 1007, `REAL_SIM=all` (bins +6 cm, counter +1 cm,
walls 7 cm, real colours, bare table, half of the prisms lying), soft hands, every 10th tick: 2582 frames, 1326 of them still
looks. On the instance under `/data/wt/infra/runs/frames/realsim_looks/`; the bench reads the placement from `scene.json`
({"offsets": {"counter": [0, 0, 0.01], "bins": [0.06, 0]}, "boxes": {"wall_height": 0.07}}). Detector as of H11, one per episode.

| | all frames, sim depth | typical | bad | still looks, sim depth | typical | bad |
|---|---|---|---|---|---|---|
| recall / precision | 0.917 / 1.000 | 0.838 / 1.000 | 0.811 / 0.997 | 1.000 / 1.000 | 0.941 / 1.000 | 0.929 / 0.997 |
| xy mm, median / p95 | 2.2 / 3.6 | 3.2 / 12.4 | 5.7 / 14.3 | 2.5 / 3.5 | 3.3 / 15.0 | 6.3 / 15.8 |
| camera part, median / p95 | 2.2 / 3.6 | 2.5 / 4.0 | 4.7 / 8.0 | 1.9 / 3.6 | 2.4 / 4.5 | 5.8 / 8.0 |
| shape decided / right | 0.863 / 1.000 | 0.714 / 0.999 | 0.420 / 0.986 | 0.894 / 1.000 | 0.694 / 1.000 | 0.411 / 0.996 |
| lying prisms seen as such | 0.897 of 1063 | 0.739 | 0.454 | 1.000 of 600 | 0.820 | 0.493 |
| ridge direction deg, centre mm (medians) | 0.7, 1.5 | 0.9, 2.6 | 1.7, 4.1 | 0.8, 1.8 | 0.9, 2.5 | 1.7, 4.6 |
| called lying without being one | 0 | 0 | 0 | 0 | 0 | 0 |
| bins by hands mm, median / p95 / max | 3.3 / 6.1 / 9.4 | 3.2 / 6.7 / 13.5 | 5.7 / 11.4 / 109.4 | 3.3 / 5.8 / 6.9 | 2.9 / 6.4 / 10.9 | 6.4 / 11.4 / 16.5 |
| counter mm, median | 2.9 | 3.3 | 3.5 | 2.8 | 3.0 | 1.3 |
| rim mm, signed: median (lowest, highest) | +9.2 (+7.5, +10.6) | +10.5 (+6.8, +14.6) | +14.5 (+5.1, +26.6) | +9.2 (+7.8, +11.6) | +10.6 (+6.8, +14.4) | +13.4 (+5.3, +25.4) |

- Layout source: nominal at the first frame of each episode (8 frames), hands at all 2574 others, at every depth level; no
  other change in any episode. The policy's own log of that run agrees: 181 looks "layout from hands", none from scene, the
  hand fit "trusted" with rms 1.7 to 1.8 mm, covered 0.85 to 0.92.
- The xy tail under noisy depth (p95 12 to 15 mm) is lying prisms that fall back to blobs (seen as lying 1.000 -> 0.82 at
  typical on still looks) and objects under a hand; upright, clear objects stay as in `dr_train`.
- Wrong shapes at typical: 4 of 2702 decided (two cylinders as cubes, a cube and a cylinder as triangles).
- At bad depth 4 of 2574 moving frames have a bin more than 3 cm off (one 109 mm), no still look.
- Runtime per look in this hands mode: 60 ms median, 87 p95, 166 max on the Mac.
This set gave H10 (lying cylinder called a triangle) and H11 (cardboard insides as yellow blobs on the nominal pose).
Not yet run: the control set without offsets (`realsim_looks_nooff`, announced for later).

### Real data, when it comes: `python -m halal.eval.real_replay runs/halal/<run>`

Reads a runner run folder (`episode_NNN.npz` per tick: t, arm_q, hand_q, arm_dq and, with `--log-images`, head_rgb (T, 480, 640, 3)
and head_depth (T, 480, 640) uint16; `episode_NNN.jsonl` for the instruction; `episode_NNN.mp4` from `--video`; `container.log`).
Looks are the still moments (`perception.replay`), one detector per episode as the policy runs it. Output: a digest of our
server's log (layout source of its looks, its "arms:" lines, warnings), then per episode a summary and one line per look, a few
overlay PNGs and `report.json` under `<run>/replay/`.
- With depth, per look: camera source, scene anchor fitted or not, the hand fit with rms / coverage / known share / pixels on
  the model / which check failed, its distance to the anchor, counter height, rim, bins, detections. Summary: counter plane rms
  and tilt against the pelvis, medians of the hand fit against their limits, per arm part (fingers, palm, wrist, forearm) the
  share of visible model points that returned depth and that have depth on the model, and the share of DARK pixels around the
  arms that returned depth, which needs no camera fit and names the depth level of `bench.d435` the hands resemble.
- Overlay: left RGB with the bins at counter (cyan) and rim (yellow) height, detections, arm model; right depth (invalid purple)
  with the arm model points green (depth on the model), yellow (depth elsewhere), red (none).
- Without depth (frames from the mp4): the colour stage (white balance on pale bright pixels, blobs per family with the colour
  they would be named) and the outline of the black links: share of their model points on dark pixels and distance to them
  through the nominal camera, and after the best image shift (the head calibration's error at the hands, in pixels).
Tested: (1) a real runner run with the mock backend, `--log-images --video`: the format reads, but that backend's images are
black (the env's mock renders nothing), so such frames are skipped; (2) runner-format folders built from `realsim_looks` seed
1002 with our server's log of that episode as `container.log`:

| depth | camera by | hand fit rms / covered / known | fingers returned | palm | wrist | dark pixels returned | level named |
|---|---|---|---|---|---|---|---|
| sim | hands | 1.7 mm / 0.87 / 1.00 | 0.98 | 1.00 | 0.99 | 0.99 | mild |
| typical | hands | 1.5 / 0.85 / 0.95 | 0.41 | 0.63 | 0.92 | 0.58 | typical |
| bad | hands | 2.4 / 0.93 / 0.62 | 0.06 | 0.12 | 0.47 | 0.10 | bad |
| RGB only | n/a | outline: 0.47 of the black links on dark pixels, 3.3 px; after a shift of (+12, -8) px 0.52, 2.3 px | | | | | |

The sim arm is dark all over, so its dark-pixel share mixes hand and forearm; on the real robot only the hand is dark. In the
sim's light the hand reads brighter than 60 in half of its pixels, hence the low outline share there; the real hand reads 43.

**H12. A barely seen top face gives way to a lying prism** (`straggler.yield_visible`). Why the lying prisms of `realsim_looks`
dropped from 1.000 to 0.82 under typical depth: noise and edge fattening widen the ridge at 45 mm into a strip that survives
the top-face seed's opening; the outline fit accepts it as a top face seen 3 to 16 % (shape None, 16 mm off the ridge centre)
and its claim radius removes the blob before the roof fit can see it. In all 39 misses of a 200-look sample no roof was even
tried. Now an ACCEPTED top face seen less than 20 % claims nothing: if the blob there fits a roof, the lying prism replaces
it; if not, the face stays and the blob is dropped as before. Faces the geometry rejects (a hand around an object, a wrong
size) claim as before; without that the first version put four held objects and one false lying into clean `motion`.

| | lying seen as such, before -> after | xy p95 mm |
|---|---|---|
| `realsim_looks` still looks (600 lying), sim depth | 1.000 -> 1.000 | 3.5 -> 3.5 |
| typical | 0.820 -> 1.000 | 15.0 -> 5.5 |
| bad | 0.493 -> 0.578 | 15.8 -> 13.2 |
| `motion` (237 lying), mild / typical / bad | 0.776 -> 0.776, 0.743 -> 0.759, 0.443 -> 0.468 | 5.4, 6.2 -> 6.0, 11.7 -> 11.2 |

Called lying without being one: unchanged (0 on `realsim_looks` at every level; `motion` 0 / 1 / 0 -> 0 / 1 / 1; `dr_train` 0 / 0 / 1).
All clean sets including `motion`: identical line by line. What limits `motion` under noise is a different thing: prisms the
roof fit does not accept (one face seen, support) and prisms under an arm.

**H13. The hand fit may move the camera 10 deg / 7 cm (was 6 deg / 5 cm).** The realsim workstream built the head calibration
error of the recordings into the sim (B7's six-parameter solution: -7.4, -1.9, -0.4 deg about the optical axes, -4 cm along
optical y). Tried here first by handing the calibration stages a head pose that is wrong by exactly that (40 `dr_train` frames):
the hand fit finds the camera to 3.5 mm median / 4.2 max at the band, and was thrown away in 40 of 40 by its own `turn` guard
(6 deg), leaving the nominal pose, 31 mm off. With the limits at 10 deg / 7 cm: believed 40 of 40, 3.5 / 4.2 mm; -9 deg and -5 cm:
40 of 40, 3.7 / 5.7; the opposite sign (+7.4 deg, +4 cm): 40 of 40, 5.0 median / 17.0 max; -12 deg and -6 cm: rejected (39 of 40).
The prior of the fit (3 deg / 2 cm) does not stop it from getting there. Rotation-only error (-2.2, -1.8, -1.4 deg), 3 deg, 2 cm:
believed before and after, 3.6 to 4.4 mm. All clean sets including `motion`: identical; `--armless` believed 0 at clean, typical
and bad depth. The limit exists to refuse a fit that walked off; the other checks (coverage, pixels on the model, rms) did not
need it in any frame here.

### Real robot, first runs (`halal/logs/robot.md`): white balance on wood, coverage under the floor gate

**H14. No white balance on a counter that is not grey** (`topface.color.grey_chroma`, robot.md fix 4). The robot's table is light
wood. The colour stage divided by the bare counter's median colour, and on wood that turned the orange cylinder of run 1 into
"yellow" at confidence 1.00 (look 2). The camera's own white balance already gets the foam right there. Now the gains are ones when
the counter's median colour has a chroma above 0.15. The detector logs the counter colour and its decision once per episode and on
every change ("bare counter RGB [..], chroma 0.32: not grey, colours as the camera gives them").

| counter (median colour of the detector's bare-counter pixels) | chroma |
|---|---|
| sim mat: `dr_train` / `lookpose` / `lowhands` / `closedhands` / `movinghands` / `motion`, every frame, max | 0.046 / 0.034 / 0.046 / 0.050 / 0.025 / 0.034 |
| `realsim_looks` (bare table, real colours), every 3rd frame | p95 0.070, max 0.090 |
| drop 2 table, 48 first frames, no chroma pre-filter (table minus foam, tape, hand, forearm) | min 0.007, median 0.043, max 0.100 |
| robot wood, frozen run-1 picture: x 0.18 to 0.38 through the nominal camera, objects and hands masked / the analyst's patch | 0.25 / 0.32 |

Frozen run-1 picture, five objects by full silhouette, the detector's colour stage (hue deg, family, confidence):

| | orange cube | orange cylinder | yellow cube | purple cubes |
|---|---|---|---|---|
| no white balance | 25.8 red 1.00 | 25.2 red 1.00 | 51.7 yellow 1.00 | 219.2, 217.3 blue 1.00 |
| balanced on the wood, 154/132/104 (before) | 30.6 red 1.00 | **43.1 yellow 1.00** | 63.0 yellow 1.00 | 223.1, 222.1 blue 1.00 |
| balanced on the counter region, 158/151/119 (before) | 26.7 red | 27.0 red | 53.7 yellow | 225.7, 224.3 blue |
| now (either counter) | as "no white balance" | | | |

Sim: every bench of the matrix below identical to the digit (the gains cannot change below 0.15). Drop 2 (`drop2_colours`, cached
first-frame PNGs): identical, purple 44/44, orange 81/81, yellow 62/62, `--sweep` included. Left as it was: `real_replay`'s RGB-only
colour stage balances on pale pixels (chroma < 0.15 by construction), so on wood its blob colours still differ from the detector's.

**H15. Hand-fit coverage judged on the model points clear of the counter** (`HandFit.clear`, robot.md fix 5). With the hands lying on
the table (real episode start), the floor gate (6.5 cm) drops their depth pixels while the model points still counted, so coverage
could not exceed 0.42 to 0.54. Coverage and `known` are now taken over the visible model points that, imaged by the fitted camera,
would pass the same gate. The "arms:" line also gives how many points that was, and the turn (deg) and shift (mm) of the fit; the
calibration bench and `real_replay` report the same. The restart rule (`settled_coverage`) still goes by the whole visible model
(`HandFit.settled`). A first version that restarted on the clear share stopped low hands after the first start, and the kept fit
got worse (emulation below, gate 0.11: 2.1 / 2.3 / 2.5 -> 4.5 / 9.8 / 11.3 mm).

Sim, hands above the gate: the matrix (`dr_train` all `--looks 2`, `lookpose`, `lowhands` clean / bad / `--shift real --depth typical`,
`closedhands`, `movinghands`, `motion`, `dr_train` dev `--shift real` typical and bad, `--armless 10` clean / typical / bad, and the
calibration bench on `dr_train` dev clean / real typical / real bad, held-out real bad, `lowhands` clean and real bad, `closedhands`,
`movinghands`) is identical to the digit. One exception: `lowhands --shift real --depth bad`, where the raised table reaches under the
fingers in 2 of 20 frames: coverage and known move in the fourth decimal, same decisions. `--armless`: believed 0 of 40 at every
level. `motion`, calibration bench on all 1017 frames: 5 of 929 fits have a clear share under 0.99, same 813 believed.

Low hands, emulated: `lowhands` (lowest hand point 8.0 cm, hand median 11.6 cm above the counter) with the floor gate raised. A
gate of 0.11 to 0.14 m is the hands 4.5 to 7.5 cm lower against the 6.5 cm gate. That is still milder than the robot's start:
there the forearms are out of view and the clear share was 0.16 to 0.39. Band error mm, median / p95 / max:

| gate, depth | clear share | believed before -> after | believed fits after | nominal |
|---|---|---|---|---|
| 0.11 clean | 0.80 | 20 -> 20 | 2.1 / 2.3 / 2.5 (identical fits) | 17.2 / 22.1 / 24.9 |
| 0.12 clean | 0.86 | 18 -> 20 | 4.9 / 13.4 / 15.1 | |
| 0.13 clean | 0.69 | 0 -> 20 | 11.8 / 19.9 / 23.8 | |
| 0.14 clean | 0.52 | 0 -> 20 | 17.0 / 34.8 / 35.6 | |
| 0.12 typical | 0.85 | 1 -> 14 | 4.2 / 13.7 / 14.3 | |
| 0.13 typical | 0.69 | 0 -> 16 | 14.6 / 18.4 / 21.3 | |
| 0.14 typical | 0.53 | 0 -> 20 | 18.6 / 36.3 / 37.2 | |
| 0.13 bad | 0.72 | 5 -> 20 | 15.0 / 25.3 / 25.6 | |
| 0.14 bad | 0.54 | 3 -> 19 | 17.2 / 25.2 / 27.8 | |

So a fit on evidence cut by the gate is weak (the fits themselves are identical before and after; only acceptance changed), and on
the sim's own small camera error it is no better than nominal: of the 92 fits newly believed at gates 0.11 to 0.14 (clean and
typical), 35 are worse than nominal (by up to 16 mm), with a median gain of 2 to 3 mm. With a real-sized head error it pays.
The calibration stages were handed a head pose turned 3 deg about the optical x axis (`Camera(K, T @ E)`, nominal then 38 / 47 / 51
mm off at the band). +3 deg is the sign the robot's hand outline shows: the model sits 30 px below the real fist. Results there:
99 newly believed, 14 worse than nominal (by up to 9 mm), median gain 17 to 19 mm. At -3 deg: 66 newly believed, none worse,
median gain 17 to 18 mm. At gate 0.14 and +3 deg the believed fits are still 39 to 43 mm off (nominal 38 to 51). Tried and dropped:
fitting only the model points clear of the gate (consistent truncation). It made the fits worse (gate 0.12 clean: 10.2 / 20.7 /
36.9 mm).

What this means for the robot's three fits (robot.md; their clear shares are from FK through the nominal camera, so estimates):
A (palms only, 171 to 0 clear points) still fails spread and inliers. B (964 visible, about 376 clear) now passes every guard, at an
accuracy the emulation puts anywhere from a few mm to the nominal error. C (right arm at home): coverage becomes about 0.74 to 0.96
and passes, so C is now stopped by inliers alone (0.33 against 0.5), plus turn if it moved more than 10 deg. The "arms:" line now
says how far. Home fits under a real-sized error are fine: `dr_train` dev, head pose turned +3 / -3 / +5 / -5 deg, clean and typical,
table as recorded and `--shift real`: believed 90 of 90 everywhere, band 2.8 to 4.8 mm median, max 7.9 mm at 3 deg and 15 mm at
+5 deg. In `lowhands` at +3 deg even the unchanged fit (gate 0.065) is believed only 16 of 20 clean and 2 of 20 typical (coverage
rejects fits that did not converge, 17 to 39 mm). So leaving the table before fitting (robot.md fix 1) stays the better order.

**H16. Arm-fit consensus prefers the fits clearest of the counter** (`calibration.hands.clear_margin` 0.05, orchestrator's order). The
policy now takes its first look with the hands still on the table, to find the boxes, and only then goes home and looks again, so
a weak start-pose fit (H15) comes first. With two fits the H6 consensus did not take "the first" as written above. It took the
one with the smaller distance to the other, which is asymmetric and close to a coin toss: the low fit won 16 of 90 and 39 of 85
times below. Now only fits whose clear share (`HandFit.clear_share`, clear / visible model points) is within 0.05 of the clearest
are candidates, and of them the one nearest to all others. Only fits within 0.05 of fully clear fill the quota of five
(`topface.arm_fits`), so low-hand fits cannot use up the quota before the robot gets home. `arm_pose` returns the `HandFit`.

Sequence emulation (scratch): one detector per frame, `--shift real` (no scene anchor). First a still look with the floor gate
raised (low hands: `lowhands` gate 0.13 / 0.14, `dr_train` dev gate 0.21, clear share about 0.5 to 0.7), then a still look with
the gate at 0.065 (hands clear, the home fit). Camera actually used, band error mm (median / p95 / max):

| set, gate, head error, depth | nominal | H15 only (ae7b524): home fit used in | used | H16: home fit used in | used | before H15 |
|---|---|---|---|---|---|---|
| `dr_train` dev 0.21, 0, clean | 14.4 / 25.7 / 28.6 | 74 of 90 | 3.7 / 6.8 / 10.6 | 90 of 90 | 3.7 / 4.0 / 4.2 | 3.7 / 4.0 / 4.2 |
| `dr_train` dev 0.21, 0, typical | | 46 of 85 | 3.7 / 5.0 / 9.4 | 85 of 85 | 3.7 / 4.8 / 5.1 | 3.7 / 4.8 / 5.1 |
| `dr_train` dev 0.21, +3 deg, clean | 36.1 / 46.7 / 53.8 | 79 of 90 | 4.3 / 13.2 / 23.4 | 90 of 90 | 4.2 / 5.1 / 6.2 | 4.2 / 5.1 / 6.2 |
| `dr_train` dev 0.21, +3 deg, typical | | 65 of 83 | 4.3 / 9.5 / 16.7 | 83 of 83 | 4.3 / 5.7 / 7.5 | 4.3 / 5.7 / 7.5 |
| `lowhands` 0.13 / 0.14, 0, clean and typical | 16.9 / 21.9 / 24.6 | 19 to 20 of 20 | up to 2.1 / 3.8 / 13.3 | 20 of 20 | 2.0 to 2.1 / 3.2 / 3.7 to 4.0 | same |
| `lowhands` 0.13, +3 deg, typical | 37.0 / 45.7 / 49.7 | 2 of 2 (15 home fits rejected) | 39.0 / 43.1 / 45.3 | same | same | 37.0 / 45.7 / 49.7 (nominal) |

Wherever a home fit is believed, the camera is now exactly what it was before H15. Where only the low fit exists, H15's trade-off
stays: `lowhands` at +3 deg is the case where the home fit itself does not converge (H15). There the lone low fit (29 to 36 mm) is
used instead of nominal (37 to 50 mm), about equal. Regression matrix (the H15 list): identical to the digit again, `--armless` 0.

**H17. "outline:" log line** (`halal/perception/outline.py`, `perception.outline`; robot.md fix 6, step 0, log only). At every still look
with a hand-fit attempt the detector logs, per hand, the image shift that best lays the black links of the FK-posed model onto the
dark pixels (value < 60) through the nominal camera. It gives the shift in px and deg, the on-dark share and mean distance before
and after, the points, and how far the two hands disagree. That is the head error at the hands even when the depth fit is
rejected. The search is ±60 px in 4 px steps, then every pixel around the best: about 10 ms. `real_replay`'s RGB-only looks use the
same code per hand; its outline settings moved to `perception.outline`. Frozen run-1 picture: left (−1, −34) px, on dark
0.91 -> 0.96; right (+63, −2) px, 0.60 -> 0.71. The hands disagree by about 70 px, so that picture cannot vouch for a correction.
Matrix: identical (the H15 list). Step 1 (the outline as a camera source) waits for this line from the robot.

**H18. Turned starts for the hand fit** (`calibration.hands.turned_starts`: pitch ±3 and ±6 deg). The shifted starts did not reach
the right minimum when the hands are low and the head pose is off by +3 deg (H15: `lowhands`, typical, 2 of 20 believed).
`SelfCalibration.hand_fit` now runs a second descent from camera turns, but only when the fit from the shifted starts fails a
check, and takes it only if it passes all of them. So no fit believed before changes. The calibration bench and `real_replay`
use the same two passes.

| `lowhands`, head pose turned (deg), depth | believed, shifted starts only | believed now | band mm of believed fits now |
|---|---|---|---|
| +3, clean | 16 of 20 | 20 | 3.0 / 17.8 / 18.4 (the 18 mm fits were believed before too) |
| +3, typical | 2 of 20 | 20 | 2.4 / 3.5 / 4.1 (turns of ±3 deg alone: 12 of 20) |
| +5, clean / typical | 8 / 2 of 20 with ±3 deg turns | 20 / 20 | 2.3 / 16.7 / 17.0, 2.9 / 6.3 / 9.9 |
| −3, −5 | 20 / 20 | unchanged | |

- Home fits with a turned head pose (`dr_train` dev, `--shift real`, typical, ±3 / ±5 deg): all 90 believed as before, identical.
- Sequence (H16) at +3 deg, `lowhands` gates 0.13 / 0.14, typical: the camera used went from 39.0 / 43.1 / 45.3 (home fits not
  converging, the low fit used) to 2.3 / 3.4 / 3.9 mm, with the home fit used in 17 of 17 and 14 of 14.
- Matrix: every perception bench identical; `--armless` 0.
- Calibration bench `lowhands --shift real --depth bad`: 17 -> 20 believed. The three new fits are 1.3, 5.1 and 8.3 mm off (were 33,
  29, 33, rejected; nominal 17 to 25).
- `motion`, calibration bench on all 1017 frames: 813 -> 830 believed. Of the 17 new fits, 10 are better than nominal and 7 worse,
  3 of them by 6 to 7 mm. Those frames have moving arms (no stillness test in that bench); the detector never fits them, and its
  `motion` result is identical.

**H19. The episode keeps the last bin row it found** (red team, assumptions.md row 4). `BinFinder.find` used to return the scene
file's bins on any look that found no row, under a layout source of hands or nominal, so nothing downstream could tell. It now
returns None there. The detector keeps the last row found in the episode, with its rim, and places it where this look's camera
puts the same pixels (`camera.deproject(then.project(..))`). So a row seen through the nominal camera at the start is carried
over into the hand camera of a later look. Only with no row found yet in the episode are the file's bins used.
`SceneLayout.bins_from` says which (`seen` | `kept` | `scene`); the detector logs every change, and file bins without the camera
anchored to the file are a WARNING. The calibration bench counts frames with no row instead of scoring the file's bins.

| | bins from | bin centre error mm, median / p95 / max |
|---|---|---|
| `dr_train` dev `--shift real` typical: a full look, then one with no depth beyond x 0.40 (nominal camera), before | scene file | 67.9 / 67.9 / 67.9 |
| same, now | kept 90 of 90 | 3.5 / 6.7 / 7.9 |
| the blind look alone (no row ever seen), before and now | scene file (now with the warning) | 67.9 / 67.9 / 67.9 |

The matrix is identical apart from the new label; every non-anchored look in it finds its row. `realsim_looks` (all 2582 frames,
typical and bad; `--still` bad) is identical too, and its row was found at every look. So the 109 mm bin of that set (seed 1006,
tick 570, bad depth, arms moving) is a row found in the wrong place, not this fallback. One such look out of 2574, none among the
1318 still looks.

**H20. The warm split only from seen objects clear of the bins, moved only under a real yellow** (red team, assumptions.md row 8).
The episode's red / yellow boundary was taken once from every draft, blobs and box tape included, and then kept. Two oranges at
22 and 31 deg with no yellow, or tape at 20 and an orange at 31, split at about 26. The 31 deg orange was then yellow at confidence
1.0 for the episode. Now only drafts with a seen face (`visible` > 0: top faces and lying prisms, no blobs) at least
`warm_split.bin_clearance` (3 cm) from every bin footprint of the look vote. A moved boundary needs the upper side of its gap in
the yellow basin (>= 39 deg); otherwise the split is None and the basins with their taper decide. The cases with the real
classifier:
- 22 + 31 or 20 + 31: None, and 31 is red at 0.82 (was a 26 deg split, 31 yellow at 1.0);
- 22 + 31 + 55: 36, all right at 1.0;
- the drop-2 sweep's moved splits (+5 to +15 deg, yellow above): unchanged.

| | before | after |
|---|---|---|
| drop 2 (`drop2_colours --sweep`, cached first frames) | purple 44/44, orange 81/81, yellow 62/62, sweep as in the table above | identical |
| robot frame (frozen run 1, five objects) | split 36, all right at 1.00 | identical |
| `--repaint` `dr_train` (orange right of matched), `--rotate` 0 / 5 / 10 | 147, 122, 94 of 158 | identical |
| `--repaint` `motion` | orange 632 of 653 (21 wrong, 20 at confidence >= 0.5) | 648 of 653 (5 wrong, 4) |
| `--repaint` `lookpose` | 57 of 66 | identical |
| matrix (plain sim) | | identical |

### Parked: camera pose from the arm outline (for arms that return no depth at all)

Without any depth on the arms there is no hand fit and the pose is nominal; nothing in the chain covers that. Design sent to
the orchestrator and parked (17:40) after an offline experiment, nothing of it is in the tree:
- Engage only when depth is absent on the arms (`known` low or no fit), never when depth contradicts the model; same stillness
  and keeping rules; a new source "outline" between "hands" and "nominal".
- The counter plane cannot supply pitch, roll or height here: it ties the camera to the table, which stands 1.5 to 9 deg
  pitched against the robot in the recordings; a grasp needs the camera against the arms.
- Crude version tried (dark pixels, value < 60, against the FK surface model; two-sided distance transforms; 5 starts; prior
  3 deg / 2 cm), `dr_train` dev, 24 frames, band error mm median / p95 / max: nominal 15.5 / 27.9 / 29.0; rotation only 8.6 / 15.1 /
  16.3; 6 DOF 8.5 / 14.2 / 18.0. Not good enough to ship (own bar: p95 under 10).
- Oracles (the true image position of every model point given): rotation only 6.5 / 14.5 / 16.3, so the crude fit is at that
  floor and the rest is the camera position (sim DR: 6 mm std, 9.6 max per axis), which a rotation cannot absorb (0.6 x the
  position error stays at the objects). 6 DOF from one view: 2.3 / 2.4 / 2.4; with every link's image position off by a random
  1 px 2.4 median / 2.9 max, by 3 px 3.6 / 5.2 (8 `motion` episodes); two arm postures of one episode 2.9 / 4.1, all 2.3 / 3.1.
  So one view does separate position from rotation (forearm at 0.25 m to fingertips at 0.45 m) if the fit has point-level
  correspondences; an area overlap does not give them.
- Version worth building if it is ever needed: contour registration. Model contour points matched to the nearest dark / light
  edge along their image normal, one equation each, Gauss-Newton on 6 DOF with the hand fit's prior, gate for edges with no
  model contour (wrist camera box, cable). Guards: model on dark, dark explained by model, residual, left and right hand
  agreeing, `--armless` 0. Can be checked on the RGB-only drops. Estimate 2 to 3 hours; expected 4 to 6 mm median if edges are
  good to 3 px (B7's cruder fit left 5.1 px on the real frames).

### Clean depth after H1 to H5

Every row of "Sim benchmarks" above reproduced to the digit (`dr_train`, held-out, `lookpose`, `lowhands`, `closedhands`,
`movinghands`, `armless`, `--stale`, `--looks 2`); `motion`: recall 0.906, precision 1.000, colour 1.000 as before, the rest as in H5.

## Next

1. Step 1 of the outline (a camera source "outline" between hands and nominal): design in the orchestrator's thread. It waits
   for real "outline:" lines: whether left and right agree on the robot (they disagree by about 70 px on the frozen picture).
2. First real depth drop: hand fit on real data (`agree_within` is 12 mm since H9; the blend is still open). Check
   first which of mild / typical / bad the black links resemble (`known` in the "arms:" log line is the share that returned).
3. Why the image disagrees with the measured finger joints while they reopen (a tick of skew, or the sim's finger
   actuator lag showing in the render but not in `hand_q`): not investigated beyond the measurement above.
4. Not done: the red < yellow < blue cluster-order check. In sim every object has its own ±7° hue shift, so two reds
   differ by up to 14° and cluster order adds nothing to the basins; "all three colours present" holds at the first
   look only. It would be a policy-level rule for the first look.
5. Sub-millimetre gains available in sim only: the true principal point (−1.4 mm); not taken, it is a sim fact.
