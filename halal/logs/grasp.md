# Grasp and planner log (workstream `grasp`)

Newest entries at the bottom. Sim runs: sim `b`, `AALTO_DR=train`, harness `python -m halal.eval.grasp_bench`
(specs and raw records on the instance under `/data/wt/grasp/runs/grasp/<run>/`), tables by
`python -m halal.eval.grasp_report <run>`. "Hard DR" = seeds chosen so every placed object has friction scale ≤ 0.82
and mass scale ≥ 1.2 (the low-grip corner of the training DR). One pick = plan → pregrasp → descend → close → lift →
carry → release over the bin; `ok` = the scene checker says `correct`.

## Facts found on the way

- **The sim hand is 67× stiffer than the real one.** Isaac hand actuators: implicit PD, stiffness 100 Nm/rad, damping 10,
  effort limit 300 (`third_party/unitree_sim_isaaclab/robots/unitree.py`); the DDS kp 1.5 / kd 0.2 are dropped
  (`action_provider_dds.py` takes positions only). The baseline's 0.05 rad error is ~5 Nm ≈ 50 N on a 50 g object.
  In sim, holding is a matter of geometry (spin, squirt, wedge), not of friction × force.
- **Right hand finger slots are swapped in sim.** DDS slots 3,4 move the sim's right middle finger, 5,6 the right
  index (upstream mapping); `INTERFACE.md` and the real robot have index first. We command index and middle
  identically in every posture, so grasps are unaffected; contact checks must look at both fingers.
- **Finger geometry (URDF meshes).** Distal links are 52 mm long, 18 mm thick, 26 mm wide. Index and middle cover
  palm z 15.5–41.5 mm and −41.5…−15.5 mm: the gap between them is 31 mm, narrower than every object (45 mm). The
  thumb (z ±13 mm) pokes the object with its end cap. In the baseline `closed` posture the finger pads reach
  x = 0.092 m, the thumb tip x = 0.065 m: a 27 mm commanded gap for a 45 mm object.
- **Closing drags the object.** With the baseline grasp point (x = 0.099) the fingers come down in front of the
  object and pull it ~22 mm back onto the thumb; the object ends at palm x ≈ 0.078–0.082.
- **The organiser planner ignored `det.top[2]`** (grasp height = scene counter + half an object). Ours takes the
  counter height from the detection (`motion.counter_from_detection`).
- **Arm sag:** the palm sits ~6 mm lower than commanded at the grasp (measured object-in-palm y −0.078 vs −0.084).

## Sweep 1: organiser pinch, hard DR, exact pose (before the `OMP_NUM_THREADS=1` sim restart)

Hypothesis: the organiser pinch (thumb against the index finger, object 15 mm off the palm centre, default finger
hints) holds cubes and cylinders and spins triangles. Setup: 3 yaws (−40, 0, 40°) × 2 spots × both arms, own-side bin,
`close_hold_s` 0.25.

| grasp | shape | ok / n | failure modes |
|---|---|---|---|
| pinch | cube | 12 / 12 | — (median spin 11°, tilt 13°) |
| pinch | cylinder | 9 / 12 | 2 lost in lift, 1 not lifted (one squirted sideways past the middle finger) |
| pinch | triangle | 1 / 12 | 11 not lifted |

Why triangles fail: the triangle always lies apex-forward (layout yaw ±45° means apex within ±50° of +x). The closing
fingers push it towards z = 0, the apex slips between the two fingers, the base ends at x ≈ 0.070 where the thumb tip
(0.065–0.07) barely touches: finger error after close is 0.01 rad instead of 0.04–0.06. It is not spinning out of a
firm pinch, it is never pinched. Verdict: cubes fine, cylinders marginal, triangles need a different closure.

Finger signal (LEFT order thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1; |measured − target| at the
lift and carry marks): held n = 44: median [0.003, 0.041, 0.023, 0.025, 0.008, 0.002, 0.000], largest joint error
0.016–0.071; empty n = 28: every joint ≤ 0.001.

## More facts (sweeps 2–3)

- **The gap between the two fingertips at contact height is 40 mm** (the tips taper; the 26 mm link width is higher
  up). A cube face spans it; a cylinder nests 10 mm into it (centre at palm x = 0.083 with the fingers closed), a
  triangle pushed apex-first nests 34 mm (centroid at 0.098–0.103, base at 0.083–0.088).
- **The sim hand lags like a first-order system, τ = damping / stiffness = 0.1 s**: 0.16–0.22 rad behind during a
  2.5 rad/s close ramp, 0.014 rad left 0.3 s after it ends. A contact reading needs ≥ 0.35 s after the last hand command.
- **Thumb contact is 10 mm below the object's centre, finger contact above it**: lifted objects tilt 10–35° (triangles
  up to 50°) and stay held. Not fixed yet (a thumb tip 12 mm higher would level the couple).
- **`mirror_left_to_right` of the organisers negates thumb_0**; FK shows the right thumb then lands on the wrong side
  for any thumb_0 ≠ 0. `halal.motion.grasps.right_hand()` keeps thumb_0 and reproduces the left fingertip
  constellation exactly. No shipped posture uses thumb_0 yet.
- Released objects land within ±2 cm of the release point (44 drops); the nominal 5 cm near inset leaves the worst
  one 0.65 cm from the near wall, so the inset only shrinks when the arm cannot reach (`motion.drop_inset.near`).
- Bin reach (offline): every bin plans up to a near edge at pelvis x = 0.47 (0.48 for the side bins); the release
  point itself cannot go beyond x ≈ 0.50 (centre bin) with a top-down hand.

## Sweep 2: object centred between the fingers ("cage"), hard DR, exact pose and 10 mm position error

Hypothesis: with the object at palm z = 0 both fingers and the thumb touch it (three contacts) instead of thumb
against one finger; the organisers avoided this because objects wedged between the fingers with their old release
posture. Setup: yaws −30 / 0 / 30°, 2 spots, `close_hold_s` 0.25, organiser postures, grasp point (0.092, −0.084, 0).
In this sweep the left arm got every exact case and the right arm every 10 mm case (bench ordering, fixed afterwards).

| grasp | shape | exact | 10 mm error | failure modes |
|---|---|---|---|---|
| cage (any heading) | cube | 6/6 | 6/6 | — |
| cage | cylinder | 0/6 | 4/6 | not lifted: thumb overlaps the nested cylinder by 4 mm only |
| cage | triangle | 0/6 | 1/6 | not lifted: thumb does not reach the base |
| cage, cube face / triangle face to the fingers | cube | 6/6 | 6/6 | — |
| same | triangle | 2/6 | 4/6 | thumb on a vertex: the triangle slides sideways (z ±15–30 mm) and is not pinched |
| wedge: triangle apex between the fingers, thumb tip to 0.087 | triangle | 0/6 | 0/6 | base ends at 0.083: 4 mm overlap, not lifted |
| organiser pinch | cube | (12/12 sweep 1) | 6/6 | — |
| organiser pinch | cylinder | (9/12) | 3/6 | not lifted |
| organiser pinch | triangle | (1/12) | 2/6 | not lifted |
| organiser pinch, cube face-aligned (oracle) | cube | 6/6 | | — |
| oracle triangle pinch, fingers 105° from the apex | triangle | 1/6 | | not lifted |

Verdict: the centred hold is at least as good as the pinch for cubes and survives 10 mm of position error; cylinders
and triangles need the thumb deeper, by a shape-dependent amount. No object stayed wedged at release (0 of 31 lifted).

## Sweep 3: thumb depth ("squeeze") with the centred hold; tuned sim settings; close hold 0.45 s

`squeeze` s = rad on thumb_1 past the organiser `closed` posture along `motion.squeeze_axis` (thumb_1 +1, thumb_2 −1.15):
the thumb tip advances 41 mm per rad at constant height (s = 0.75: tip at palm x 0.094 instead of 0.064).
Setup: hard DR, exact pose, yaws −30 / 0 / 30°, 2 spots, arms alternating, default finger hints, 6 trials per cell.

| squeeze | cube | cylinder | triangle | finger error held (thumb_1 median) |
|---|---|---|---|---|
| 0.00 | (12/12 sweep 2) | 0/6 | 0/6 | — |
| 0.15 | 6/6 | 5/6 (lost in carry) | 0/6 | 0.074 |
| 0.30 | 6/6 | 6/6 | 0/6 | 0.079 |
| 0.45 | 6/6 | 6/6 | 0/6 | 0.108 |
| 0.60 | 6/6 | 6/6 | 0/6 (3 lost in lift) | 0.124 |
| 0.75 | 6/6 | 5/6 (stayed wedged in the open hand) | 6/6 | 0.114 (triangle 0.033) |
| 0.75, fingers along the apex | | | 6/6 | 0.032 |
| 0.90, fingers along the apex | | | 5/6 (stayed wedged in the open hand) | 0.068 |

Reference cells after the sim restart (organiser pinch, old 0.25 s hold): cube 12/12 (sweep 1: 12/12), cylinder 8/12
(sweep 1: 9/12): the tuned sim settings did not change grasp outcomes.

Findings:
- The stiff sim hand tolerates over-squeeze on the cube across the whole range (30/30, thumb_1 error up to 0.20 rad
  ≈ 20 Nm): nothing squirts or spins out, spin stays ≤ 28°.
- Each shape has a threshold set by how far it nests: cube 0, cylinder ≈ 0.15–0.3, triangle ≈ 0.7. Just above the
  threshold the grip is weak (triangle at 0.75: error 0.032–0.035) but 12/12 held.
- The only failures above the thresholds are objects that stay wedged between the straightening fingers at release
  (cylinder at 0.75: 1/6, triangle at 0.90: 1/6; none in 54 picks at ≤ 0.60).
- Triangles work with any finger heading once the thumb is deep enough (default hints 6/6, apex-aligned 6/6): the
  apex self-centres between the fingertips.
- Contact signal in sim: every failed-to-lift pick had a largest grip-joint error ≤ 0.02 rad at lift start (mostly
  0.001–0.003), every held one ≥ 0.032.

Verdict and what `candidate_grasps` returns now: `cage` for every shape with its own depth, cube 0.30, cylinder 0.45,
triangle 0.80, unknown shape 0.75 (works for all three: 17/18 + 6/6); the organiser pinch stays as second candidate
for cubes only. n is 6 per cell; confirmation with ≥ 24 per shape and with position error is the next sweep.

## Planner: coverage, duration, planning time (offline, `python -m halal.eval.plan_bench`)

Every object of N sampled layouts × every bin × both arms, first candidate grasp that plans, exact detections.

| planner | pairs | plannable (either arm) | pick duration mean / worst | planning time ok plans | planning time failed plans |
|---|---|---|---|---|---|
| organiser (`--organiser`), home → home, hold 0.25 s | 393 | 100 % | 9.46 s / 14.50 s | 0.15 s mean, 1.40 s max | 3.61 s mean, 5.19 s max |
| ours at ffbd071 (joint-space carry, shortcuts, reach check), home → home, hold 0.25 s | 393 | 100 % | 9.12 s / 13.83 s | 0.19 s, 2.10 s max | 0.00 s, 0.23 s max |
| ours, look pose → retreat (`--start look --end retreat`), hold 0.45 s | 165 | 100 % | 7.53 s / 9.43 s | 0.10 s, 1.07 s max | 0.01 s |
| ours with the low sequence (`motion.sequences: [low, organiser]`), look → retreat | 165 | 100 % | 6.11 s / 9.40 s | 0.10 s, 1.96 s max | 0.01 s |

Phase means, organiser → ours low sequence: to_pregrasp 1.79 → 1.56 s, descend 1.18 → 0.67, close 0.53 → 0.80 (longer
hold for lagging fingers), lift 1.13 → 0.66, carry 2.11 → 1.10, open 0.80 → 0.90, retreat 0.46 → 0.43, home 1.46 → none.

- Left bin: left arm only; right bin: right arm only; centre bin: both arms, from anywhere in the object band. The
  wrong-side arm is rejected by a cached reach check in 0 ms (the organiser planner searched 3.6 s per attempt).
- Vertical moves cost ~6 rad/m at the wrist (keeping the palm level turns wrist yaw as the forearm tilts), so descend
  and lift time is proportional to their height: 13.75 cm in the organiser sequence, 7 cm in ours.
- to_pregrasp is dominated by wrist roll: 0 at the look / home pose, ~1.4 rad with the palm down. A look pose with the
  wrists pre-rolled would save ~0.5 s on each side of a pick (offline: mean joint distance to pregrasp 1.51 → 0.97 rad);
  not adopted yet, needs a sim image check.
- Forcing the grasp's finger heading on pregrasp, descend and lift was tried and reverted: no time gained.
- The low sequence is 88 % of the plans; the rest fall back to the organiser sequence when the rising joint-space carry
  would bring the held object within 1.5 cm of a bin rim. It is OFF by default until the sim A/B.

## Sweep 4: one depth for both hand regimes? (squeeze 1.0, release together vs thumb first)

Hypothesis: if the stiff sim hand tolerates the depth the soft real hand needs (1.0 = the thumb's joint limit), the
policy needs no stiffness switch; a release that opens the thumb 0.2 s before the fingers frees objects wedged
between the fingertips. Setup: cage, hard DR, exact pose, 12 trials per cell.

| variant | cube | cylinder | triangle |
|---|---|---|---|
| squeeze 1.0, release together | 2/12 | 6/12 | 12/12 |
| squeeze 1.0, thumb first | 3/12 | 8/12 | 11/12 |
| squeeze 0.75 (the unknown-shape depth), thumb first | | 11/12 | 12/12 |

All 30 failures are the same thing: the object ends between the two straightened fingers, tilted ~90–100°, and
rides home with the hand (final height 0.26–0.29 m). At 1.0 the thumb drives a cube or cylinder to palm x 0.105–0.116,
past the tapered fingertips into the 31 mm gap between the finger links; at ≤ 0.75 it stays at 0.083–0.104 in the
40 mm tip gap and drops out when the hand opens. Opening the thumb first changes nothing (the fingers hold the object,
not the thumb).

Verdict: no single depth. The stiff hand keeps the per-shape depths; the soft hand needs deeper targets because its
force is bounded by kp × error. The thumb-first release was removed from the code again. An empty hand at squeeze
1.0 reads 0.037 rad on thumb_2 (target at the joint limit): contact checks must use thumb_1 and the finger_0 joints.
The unknown-shape depth 0.75 is confirmed as a fallback: cylinder 16/18 and triangle 24/24 over sweeps 3–4, cube 6/6.

Where failed picks leave the object (sweeps 1–3, objects that never left the counter): organiser pinch median shift
17–24 mm, on average 11–26 mm towards the robot; cage with too shallow a thumb 3–9 mm with no preferred direction.
