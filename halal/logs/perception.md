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
- Cylinder cut by the lower image border: shape None; one case in 1891 decided shapes still wrong (`motion` seed 2202).
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

## Next

1. First real depth drop: hand fit on real data, then `agree_within` (10–12 mm instead of 25) and the blend decision.
2. Why the image disagrees with the measured finger joints while they reopen (a tick of skew, or the sim's finger
   actuator lag showing in the render but not in `hand_q`): not investigated beyond the measurement above.
3. Not done: the red < yellow < blue cluster-order check. In sim every object has its own ±7° hue shift, so two reds
   differ by up to 14° and cluster order adds nothing to the basins; "all three colours present" holds at the first
   look only. It would be a policy-level rule for the first look.
4. Sub-millimetre gains available in sim only: the true principal point (−1.4 mm); not taken, it is a sim fact.
