# Real-look sim log (workstream `realsim`, Sat 14.11)

Goal: a dev-only sim variant that looks and behaves like the real setup, and full physics episodes of the published
image through it, to see what breaks when the pieces that were tested alone meet. Nothing under `aaltonomous/` or
`third_party/` is edited: everything is patched in memory when the sim starts.

## The variant

`REAL_SIM=<lying,colours,boxes,table,offsets | all> bash halal/ops/sims.sh start <tag>` (composes with `HAND_GAINS`;
a sim started without `REAL_SIM` never loads the code). Values: `halal/ops/real_sim.yaml` (`REAL_SIM_CONFIG=<yaml>`
swaps the file). Code: `halal/ops/real_sim.py`.

| Feature | What it does | How |
|---|---|---|
| `lying` | a share (0.5) of the triangular prisms lies on a rectangular side, ridge yaw uniform | the scene service's `place` marks layout entries (seeded by the entry's own x, y, yaw, so it can be recomputed from a trial's JSON), `assign` gives them the lying orientation and centre height |
| `colours` | orange / pale yellow / purple objects under the names red / yellow / blue | `Scene.load()` of the sim process returns an edited scene dict: geometry and the scene service's scoring read the same one |
| `boxes` | 70 mm walls, tape colour outside, 0.5 mm cardboard panels on the inner faces and floor | scene dict + the scene config classes rebuilt with the extra panels |
| `table` | no printed grid mat, grey glossy table | same rebuild |
| `offsets` | counter +1 cm in z, box row +6 cm in x (real: 0 to +2 cm, +5 to +10 cm) | scene dict |
| real words | "orange" / "purple" in every instruction; sort_fixed = yellow left, purple center, orange right, for the sentence AND the scoring | client side: `python -m halal.eval.proxy --evaluator halal.eval.real_eval` |

Not modelled: the real prism's proportions (45 x 45 mm, ridge about 39 mm; the sim prism is 52 mm across with the
ridge at 45), tape strips on the table, real depth noise, pad friction, the camera-to-table angle of the recordings.

### Colours are fitted on the rendered image

The sim's tone mapping turns warm albedos towards yellow and washes saturation out: an albedo at hue 25 deg renders
at 44 deg, which is yellow for the detector. The yaml therefore holds albedos chosen so that the head frame matches
drop 2 (fixed layout, nominal light, median over the object's pixels):

| | albedo | rendered hue / S / V | drop 2 |
|---|---|---|---|
| orange | 220, 45, 10 | 24-25 deg (p5-p95 16-32) / 0.66-0.75 / 0.81-0.84 | 25 / 0.70 / 0.85 |
| yellow | 242, 229, 85 | 55-56 deg / 0.27-0.36 / 0.78-0.87 | 55 / 0.33 / 0.85 |
| purple | 10, 35, 175 | 219 deg (213-229) / 0.61 / 0.79 | 220 / 0.59 / 0.82 |
| tape | 60, 13, 7 | 10 deg (6-21) / 0.70 / 0.48 | 10 / 0.66 / 0.45 |
| cardboard | 150, 125, 80 | 44 deg / 0.18 / 0.75 | 38 to 68, low chroma |

Tried on the way: orange albedo 242, 132, 53 (hue 25) rendered at 44 deg; 215, 75, 10 (hue 19) at 37 deg.

## Legs

All: image through the proxy (`serve_image.sh`), `sort_fixed`, seeds 1000-1007 (31 objects, 5 of them triangles), DR
`train`, 120 s, sim `c`. One feature per leg unless stated. Stock = the organisers' sim: +3.88, 31/31 (v16).

| Leg | Image | Variant | Objects | Score | Wrong | Off | Gate | Lost objects and cause |
|---|---|---|---|---|---|---|---|---|
| rs-lying | v16 | `lying`, fraction 1.0 (5 lying prisms) | 27/31 | +3.38 ± 0.99 | 0 | 0 | PASS | 4 lying prisms: seen as `lying triangle` with the right colour every time, 3 picks each with `cage_lying`, grip score 0.000 after every lift, then left; 1 of 5 picked and sorted (seed 1007) |
| rs-colours | v16 | `colours` | 31/31 | +3.88 ± 0.78 | 0 | 0 | PASS | none |
| rs-boxes | v16 | `boxes` | 31/31 | +3.88 ± 0.78 | 0 | 0 | PASS | none |
| rs18-all-soft | v18 | all (with offsets), soft hands, real words | 23/31 | +2.88 ± 0.60 | 0 | 0 | PASS, 4 time limits | 3 lying prisms (3 empty `cage_lying` picks each); 5 purple objects bound for the center box never planned ("no plan for any of N targets" until the time limit: seeds 1001 x2, 1002, 1003, 1005) |
| rs18-nooff-soft | v18 | lying, colours, boxes, table, soft hands, real words | 28/31 | +3.50 ± 0.87 | 0 | 0 | PASS | the 3 lying prisms, same way; nothing else |

v18 with offsets: every look says `layout from hands`, releases at x 0.493 to 0.513, all 23 released objects landed
inside (the scene-file fallback of v16, below, is gone). The 5 purple objects are an effect of the offsets alone: the
same objects plan at the first look when the row stands where the scene file says. Three other purple objects did go
to the shifted center box (releases at +0.493 / +0.503, -0.030). v18's server log does not print the planner's reason
per attempt, so the cause (reach or rim clearance at a center box 6 cm farther and 1 cm higher) is not known.

Lying prisms: a failed pick pushes the prism 1 to 8 cm; none left the counter, none entered a wrong bin, no
neighbour was lost, and each costs 25 to 30 s.

## Filmed episodes (v16, soft hands `1.5,0.2,0.0075`, real words; `real_eval --record-video NAME`)

| Variant | Task, seed | Outcome |
|---|---|---|
| all, with offsets | fixed, 1004 | 2/5, time limit: see below |
| lying, colours, boxes, table | fixed, 1004 | 4/5, done 89 s; the lying prism left after 3 empty picks |
| same | fixed, 1001 | 5/5, done 69 s |
| same | language (unseen paraphrase), 1003 | 4/4, +8/8, done 61 s |
| same | language (unseen paraphrase), 1008 | 3/5, +6/10, done 110 s; two lying prisms left after 3 empty picks each |

### Defect found: release points fall back to the scene file (v16, offsets)

Seed 1004, box row 6 cm farther than `scene.yaml`. Looks at t = 3.8, 24.9, 34.9 s say `layout from hands` and the
releases are at x 0.513, 0.493, 0.513 (right for the shifted row). From t = 48.5 s every look says `layout from
scene` and the release slot is x 0.463, the scene file's: the yellow cylinder released there at t = 70 s landed on
the counter in front of the box, at (0.431, +0.305). Without offsets the same seed says `layout from scene` from the
first look and 0.463 is right. In the same episode an orange cube slipped out of the soft hand in the carry (grip
score 0.022) and came to rest 10 cm short of the right box; the planner then refused it until the time limit
("hand -0.4 cm above a bin rim during descend").
