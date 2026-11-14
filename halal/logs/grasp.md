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

## Objects that stay in the hand at release (policy episodes: 3 in 16), and the staggered release

Observation from the policy's full episodes with the cage grasp: a triangle at 0.80 (twice) and a cube at the
unknown-shape depth 0.75 were "released" and stayed in the hand; one later fell into the wrong bin.

Mechanism (traces of sweeps 3–4): the object ends between the two distal finger links (31 mm apart), tilted ~95–107°,
and rides up with them when both fingers straighten. A triangle's apex is a 30° half-angle wedge (self-locking for
µ > 0.58, so on rubber too); a cube corner 45°; a cylinder's side enters 6 mm. Held objects tilt 15–50° because the
thumb presses 10 mm below their centre and the fingers above it, which walks the apex or corner from the 40 mm tip
gap up into the 31 mm link gap during lift and carry. The fingers read nothing after opening.

Sweep 10a, hypothesis: if one finger opens before the other the two links no longer face each other and the object
drops. Worst case of sweep 4 (cage at squeeze 1.0, hard DR, exact pose, n = 24), thumb and one finger open first, the
other finger after it (here fully sequential with a 0.25 s pause):

| release | cube | cylinder | wedged in the hand |
|---|---|---|---|
| together (sweep 4) | 2/12 | 6/12 | 16/24 |
| staggered | 12/12 | 11/12 (1 lost in the carry) | 0/24 |

All 23 landed inside the bin: x 0.438–0.571 (release point 0.453, inner walls 0.403–0.607), |y| 0.194–0.264 (release
0.228, walls 0.158–0.362); 15 of 24 came to rest on their side. Verdict: it ejects. The fully sequential version costs
0.95 s per release; `motion.release_stagger_s` is now the START delay of the second finger (overlapping ramps, one
hold), measured in the next sweeps.

## Soft hand (kp 1.5): what can and cannot be measured, and the estimate

- The sim's hand gains (stiffness 100, damping 10, armature 0.1, effort 300; printed in the sim's start-up table)
  cannot be changed from a client, the command line, Kit arguments, a DR yaml or the scene service. The dev-only
  launcher `halal/ops/soft_hand_sim.py` (`HAND_GAINS="kp,kd[,armature]" sims.sh start <tag>`) patches the upstream
  robot config in memory, as the organisers' own `aalto_sim_main.py` does for its fixes.
- Gains to use: `HAND_GAINS="1.5,0.27,0.012"`. kp is the runner's. The real free step response (50 % at ~140 ms, 90 %
  at 300–380 ms) is a critically damped second-order system at ~11 rad/s, i.e. inertia 0.012 kg m² and damping 0.27
  at kp 1.5: 0.2 is the controller's kd, 0.07 stands in for joint friction, the armature is fitted, not measured.
  Still no dead zone, backlash or stiction: "much closer than stiffness 100", not the real finger.
- Static estimate (thumb tip 259 N/m, two fingers 490 N/m, in series 170 N per metre of commanded overlap; the thumb
  advances 41 mm per rad of squeeze): thumb-only closure gives cube 3.1 + 7.0 s N, cylinder 0.7 + 7.0 s, triangle
  7.0 (s − 0.6): at the stiff-hand depths 5.1 / 3.8 / 1.4 N, and the triangle cannot exceed 2.8 N (thumb limit).
  Needed for our fastest motion (8 rad/s² ≈ 3.2 m/s², 50 g, µ 0.5): 0.67 N (cube), 0.44 N (cylinder, triangle).
- Hence `candidate_grasps(..., hand="soft")`: same depths, but the fingers close too (`squeeze_axis` finger_0 −0.5 per
  rad, fingertips 47 mm/rad): ~7.5 N on every shape by the same model. Soft motion profile for the policy: arm
  acceleration 4 rad/s² for lift and carry (gravity is 3/4 of the load, so slowing down buys at most 26 %).

## Sweeps 10 and 11: release at the policy's depths, plain versus staggered; how short the stagger can be

Metric added after the policy's release check showed objects still in the cage after a plain opening: "out at open" =
the carried object has dropped out of the hand by the END of the open phase, before any retreat motion.

Sweep 10 (hard DR, 5 mm position error in a random direction, both arms, n = 24 per cell; stagger = second finger
after the first has fully opened plus 0.25 s):

| cell | binned | out at open | stuck in the hand |
|---|---|---|---|
| cube at 0.75 (unknown-shape depth), plain release | 21/24 | 21/24 | 3 |
| cube at 0.75, staggered | 24/24 | 24/24 | 0 |
| triangle at 0.80, plain | 24/24 | 24/24 | 0 |
| triangle at 0.80, staggered | 24/24 | 24/24 | 0 |

Sweep 11 (worst case again: squeeze 1.0, exact pose, n = 24 per cell; `release_stagger_s` = start delay of the second
finger, ramps overlap): 0.25 s: cube 12/12, cylinder 12/12, all out at open; 0.5 s: cube 11/12 (one not lifted),
cylinder 12/12, all carried ones out at open. Plain release in the same case (sweep 4): 8/24.

Traces: with the stagger the object is on the bin floor 0.4–0.5 s after the opening starts, i.e. during the first
finger's swing. That swing carries it forward first: landing minus release point over the staggered cells
(n = 119): x −27…+118 mm, outward y −48…+63 mm (plain release, cube 0.75 and triangle 0.80: x −21…+83, y −37…+29).
The bin's inner walls are at x −50 / +154 and y −70 / +134 from the release point, the object is 45 mm wide: every
object landed inside, the closest 0.5 mm from a wall.

Verdict: `motion.release_stagger_s: 0.25` is the default from here (+0.25 s per pick). Not reproduced in the bench: the
policy's triangles that stayed in the cage at 0.80 after a plain opening (0 of 48 here at ≤ 5 mm error); the
tolerance sweep adds plain-release cells with 10–15 mm forward error.

## Sweep 5: confirmation and position tolerance of the candidate grasp (cage, per-shape depth, staggered release)

Setup: hard DR, both arms, own-side bin, organiser pick sequence, release stagger 0.25 s. Position error = perceived
minus true object position along pelvis x (forward) and y (outward). 178 picks.

| condition | cube | cylinder | triangle |
|---|---|---|---|
| exact pose, layout yaw −30 / 0 / 30° | 12/12 | 12/12 | 12/12 |
| cube corner-on (yaw ±20°) / triangle at yaw ±45° | 8/8 | | 5/8 (2 not lifted, 1 lost in lift) |
| x error −10 mm | 4/4 | 4/4 | 4/4 |
| x error +10 mm | 4/4 | 4/4 | 4/4 |
| x error +15 mm | 4/4 | 4/4 | 4/4 |
| y error −10 mm (inward) | 4/4 | 4/4 | 4/4 |
| y error +10 mm (outward) | 4/4 | 4/4 | 3/4 (not lifted) |
| same grasp, plain release, exact | 6/6 | 6/6 | 6/6 (one still in the hand at the end of the open phase) |
| plain release, x error +10 / +15 mm | 8/8 | 8/8 | 8/8 |
| cube at 0.75, 5 mm error, stagger | 12/12 | | |
| cube at 0.75, stagger, opening at 1.0 rad/s instead of 2.0 | 10/12 (2 stuck in the hand) | | |

- Out at open with the stagger: 120 of 120 carried objects. A slower opening is worse (the object rides the finger);
  the release-speed setting was removed again.
- Tolerance window: every shape holds through ±10 mm in x and y and +15 mm in x (59/60). The weak spot is the
  triangle's orientation, not its position: with the apex ~70° off the finger heading (layout yaw ±45°) 3 of 8 were
  not lifted. Next: finger heading along the nearest triangle vertex when the yaw is known.
- Failed picks left the object 29–32 mm from where it was (max 70 mm), upright, on the counter.
- Landing minus release point with the stagger at these depths (n = 108): x −28…+77 mm, outward y −40…+48 mm.
  The release point moved to 6 cm from the near wall and 8.5 cm from the side walls (`motion.drop_inset`) so that this
  box clears every wall by ≥ 9 mm; the near inset still shrinks (5, 4, 3 cm) when the arm cannot reach, and now does
  so on a failed full plan too, not only on the reach check.
- Reach with the cage grasp (its grasp point is 7 mm nearer the palm than the pinch's): every bin plans up to a near
  edge at x 0.46, the side bins up to 0.47.

## Sweep 12: first measurements on the soft-hand sim

Sim started with `HAND_GAINS="1.5,0.2,0.0075"` (kp and kd as the real runner, armature fitted): free step response of
thumb_1 through AaltoEnv + safety filter 50 % at 0.17–0.20 s, 90 % at 0.33–0.37 s, no overshoot (real hand: ~0.14 s
and 0.30–0.38 s); the first guess "1.5,0.27,0.012" was slower (0.23 / 0.47 s). Fingers sag 0.014–0.04 rad under their
own weight with nothing held, the size of the real hand's bias. Hard DR, exact pose, staggered release.

| closure on the soft hand | cube | cylinder | triangle |
|---|---|---|---|
| `hand="soft"`: stiff-hand depths, fingers closing too; arm 8 rad/s² | 8/12 (2 lost in lift, 2 in carry) | 12/12 | 12/12 |
| the same at 4 rad/s² | 6/6 | 6/6 | 6/6 |
| stiff-hand closure (thumb only) | 1/6 | 1/6 | 0/6 |
| thumb only at squeeze 1.0 | 3/6 | 3/6 | 1/6 |

- On a soft hand the fingers must close together with the thumb; the thumb alone does not hold anything reliably.
- The cube is the weak shape there (two flat contacts; it slips 16 mm in the hand, the others 5–6 mm) and it is the
  one that needs the gentler arm: all 4 losses at 8 rad/s², none at 4. `carry_acc: 4.0` is now part of the cage's
  `soft` overrides and the planner re-times lift and carry with it.
- Contact signal on the soft hand (largest of thumb_1, finger_0 error at lift and mid-carry): held 0.14–0.27 rad
  (thumb_1 median 0.22), empty 0.006–0.056: a threshold of 0.10 separates them; the stiff-sim threshold 0.025 does not.
- Every carried object was out of the hand within the open phase (50/50).

## Sweeps 13a and 15: triangles with the fingers along the nearest vertex

Analysis first (114 default-hint triangle picks of sweeps 3, 5, 10, 13a): the default hints put the fingers straight
forward; success by the apex's angle off the finger heading: within ±40° 100/101, 40–60° off 9/13. Near 60° a flat side
faces the fingertips and the thumb meets a vertex, the unstable configuration of sweep 2.

Sweep 13a (hard DR, exact pose, layout yaw −45 / −30 / 0 / 30 / 45°, both arms): fingers along the nearest vertex
(`headings_deg: {triangle: [0, 120, 240]}`, needs the yaw) 30/30, spin ≤ 11°; the same with a 10° error on the
perceived yaw 10/10; default hints 19/20 (one not lifted at −45°), spin 27–68°.

Sweep 15, the policy's repeatable miss (layout seed 1001, triangle at pelvis (0.304, −0.102), layout yaw −36°, right
arm, 4 of 4 runs empty) and its mirror image for the left arm, hard DR: default hints 12/12, vertex-aligned 12/12.
NOT reproduced. That episode's object has friction ×0.95 and, after the DR pose jitter, yaw −39.6°: every sweep so far
used the slippery-and-heavy DR corner (friction ≤ ×0.82), where a misaligned triangle slides into the gap between the
fingertips. The opposite corner (grippy and light) is the hard one for self-centring and had not been tested; the
bench now has `dr: sticky` for it.

Verdict: `candidates.triangle: [cage_apex, cage]` (vertex-aligned when the yaw is known, default hints otherwise).

## Sweep 14: which way may the hand class err; the low pick sequence in sim

Stock (stiff) sim, hard DR, exact pose unless noted, staggered release.

Decision table, true hand × assumed class (cage grasp; cube / cylinder / triangle; no object stayed in the hand in any
cell):

| true hand | assumed stiff | assumed soft |
|---|---|---|
| stiff sim | 12/12, 12/12, 12/12 (sweep 5) | 12/12, 12/12, 6/12 (triangle: 3 not lifted, 2 lost in lift, 1 in carry) |
| soft sim (sweep 12) | 1/6, 1/6, 0/6 | 8/12, 12/12, 12/12 at 8 rad/s²; 6/6, 6/6, 6/6 at 4 rad/s² |

Verdict: a soft hand taken for stiff loses nearly everything, a stiff hand taken for soft loses half the triangles
only (the soft triangle closure, thumb 0.80 plus fingers 0.4 rad deeper, crushes it on the stiff hand). When in
doubt the class should be soft. Next: a shallower soft triangle depth that suits both hands.

Low pick sequence (`motion.sequences: [low, organiser]`) against the organiser-sequence cells of sweep 5:

| condition | low sequence | organiser sequence |
|---|---|---|
| exact pose | 35/36 (triangle 11/12) | 36/36 |
| x error ±10 mm | 23/24 (triangle 7/8) | 24/24 |
| duration home → retreat, as executed | 5.48 s | 8.66 s |

Both low-sequence losses are triangles (default hints) lost in the carry. Verdict: 3.2 s faster per pick, no cost
for cubes and cylinders (40/40); stays off until triangles are confirmed with the vertex-aligned grasp under it.

## Obstacle-aware plans (offline)

`plan_pick(..., obstacles=...)`: plans whose fingers come within 47 mm of another object's centre while below its top
during descend and close are rejected. The policy's episode where the vertex-aligned grasp knocked over a neighbour
7.5 cm away (target at (+0.301, −0.098), right arm, heading +40°, neighbour at (+0.299, −0.023)): with the neighbour
passed, the vertex-aligned plan is rejected and the default-hint cage plans with a clear heading. A neighbour at the
minimum 8 cm blocks every default hint only in the forward-inward diagonal. Sim validation: sweep 19.

## Sweep 16: the grippy-and-light DR corner (friction ×1.0–1.1, mass ×0.7–1.0)

Hypothesis: grasps that rely on the object sliding into place between the fingertips are harder when it grips the
counter and the fingers; every earlier sweep used the slippery-heavy corner. Stock sim, both arms, staggered release.

| cell | ok / n |
|---|---|
| candidate grasps as the policy gets them: cube / cylinder / triangle (vertex-aligned) | 12/12, 12/12, 12/12 |
| triangles at layout yaw −45 / −30 / 0 / 30 / 45°, vertex-aligned | 20/20 |
| the same, default hints | 19/20 (one not lifted, toppled) |
| the policy's repeatable miss at its true pose (yaw −39.6°), vertex-aligned | 12/12 |
| the same pose, default hints | 12/12 |

Verdict: no weakness in this corner; all carried objects out of the hand within the open phase (99/99). The policy's
miss with default hints is still not reproduced by the bench (the policy's own run with the vertex-aligned grasp
holds it); vertex-aligned triangles stand at 96/96 over sweeps 13a, 15 and 16.

## Real bin walls are 70 mm, the scene file says 60 mm (offline, 188 plans over 10 layouts)

Perception measured the real boxes: walls ~70 mm. Clearance over a 70 mm rim with the plans as they are (nominal:
before the 6 mm arm sag and the 10–50° tilt of a held object, which lowers its lowest corner by up to ~10 mm):

| sequence | held object's bottom over the wall, min / p5 | lowest hand point over the wall, min | plans that stay low |
|---|---|---|---|
| organiser, walls as in the file | 29 / 40 mm | 46 mm | |
| low, walls as in the file | 3 / 7 mm | 24 mm | 81 % |
| organiser, `min_wall_height: 0.07` | 39 / 44 mm | 55 mm | |
| low, `min_wall_height: 0.07` | 19 / 21 mm | 35 mm | 55 % |
| low, `min_wall_height: 0.07`, `lift_height: 0.08` | 14 / 17 mm | 33 mm | 74 % |

Verdict: the organiser sequence clears a 70 mm rim as it is; the low sequence would hit it. `motion.min_wall_height`
(walls count as at least this tall in everything the planner derives from them: release height, joint-carry floor,
rim checks of both sequences, the organiser's bin check) is in the planner at 0.06 = no change; 0.07 raises the
release by 10 mm, to be measured in sim (sweep 20) before it becomes the default.

Bench: planning with obstacles can take more than a second, which tripped the env's watchdog stop (sweep 19 died in
its first episode); the bench now plans in a thread while the arms hold.

## Sweep 17: the soft-hand sim again, both DR corners (`HAND_GAINS="1.5,0.2,0.0075"`)

Hypotheses: the candidates as the policy gets them with `hand="soft"` (fingers closing too, lift and carry at 4 rad/s²)
hold every shape in both DR corners; a cube at 0.45 is a usable second candidate now that the pinch is not offered on a
soft hand; a shallower soft triangle might suit the stiff hand as well (stiff half: sweep 20). Exact pose, own-side
bin, staggered release, 182 picks.

| cell | slippery-heavy corner | grippy-light corner | largest grip joint error, held |
|---|---|---|---|
| candidates, cube (0.30) | 11/13 (1 not lifted, 1 lost in lift) | 11/12 (1 not lifted) | 0.16–0.27 |
| candidates, cylinder (0.45) | 19/19 | 12/12 | |
| candidates, triangle (vertex-aligned, 0.80) | 17/18 (1 lost in lift) | 11/12 (1 lost in lift) | |
| cube at 0.45 | 19/20 (1 not lifted) | | 0.22–0.31 |
| triangle at 0.40 | 4/9 (lost in lift and carry, 4 of 5 toppled) | | 0.05–0.06 |
| triangle at 0.50 | 19/19 | | 0.10–0.11 |
| triangle at 0.60 | 12/12 | | 0.15–0.17 |

- Every carried object was out of the hand within the open phase (140/140).
- Sum of |error| on thumb_1 and both finger_0 joints with a triangle held, lift end / mid-carry: at 0.80 0.53–0.55 /
  0.52–0.54, at 0.60 0.31–0.32 / 0.30–0.31, at 0.50 0.19–0.21; an empty hand reads 0.01–0.12 at the lift end and 0.02
  in the carry. The policy's threshold (0.15 on the sum) leaves a factor 2 at 0.60 and none at 0.50.
- All six losses look the same in the traces, in both corners: the grip reads firm (0.22–0.33), and when the lift
  starts the cube slides DOWN out of the hand: the object's distance below the palm goes 83 → 93 → 115 mm within 0.4 s
  while the finger errors fall to 0. Good picks show the start of the same slide (83 → 86–95 mm, tilt 15–20°) and
  then catch. The thumb presses 10 mm below the centre and the fingers above it; a kp 1.5 hand cannot resist that
  couple once the counter no longer supports the object. Soft-hand totals with sweep 12: cube 28/31, cylinder
  37/37, triangle 34/36.

Verdict: the soft candidates stand (about one cube or triangle in twelve needs the policy's retry); cube at 0.45 is
offered as the second soft candidate; the soft triangle stays at 0.80 (0.60 is the lowest depth the policy's contact
threshold can see). Open: a closed posture with the thumb tip level with the fingertips.

## Sweep 18: cross-body carry on the soft hand (one arm at a time)

Hypothesis: a cube carried across the body is lost because of the arm speed (a policy episode lost one). Soft-hand
sim, slippery-heavy corner, cubes at y −0.10 and 0.00 of the picking arm's side, own-side bin, soft candidates
(lift and carry at 4 rad/s²), arm speed 0.97 rad/s against 0.6 rad/s (here on every phase).

| arm speed | binned | slip in the hand, close → mid-carry (median) | pick duration |
|---|---|---|---|
| 0.97 rad/s | 11/12 (1 lost in the carry, at y 0.00) | 23 mm (17 at y −0.10, 24 at y 0.00) | 9.0–9.6 s |
| 0.6 rad/s | 11/12 (1 lost at the start of the lift) | 10 mm | 12.3–13.5 s |

Verdict: not decided by n = 12. The slower arm halves the slide of the cube in the hand, the losses are one each
(the slow one is the lift-start slide of sweep 17, which speed does not touch). Slowing only lift and carry would
cost ~1.6 s per pick; not adopted without a larger A/B.

## Sweep 19: a second object beside the target, with and without `obstacles` (stock sim)

Hypothesis: the obstacle check (digit centre lines stay 47 mm from another object's centre while below its top)
prevents in physics what a blind plan does to a neighbour. Triangle neighbours 7.5–8 cm from the target, DR as it
comes, candidates as the policy gets them; the bench records the planned digit gap to the neighbour's true centre
(`PickPlanner.digit_gap`). Fine = target binned and neighbour neither moved more than 10 mm nor toppled.

| case | variant | planned | target binned | neighbour moved > 10 mm / toppled | fine |
|---|---|---|---|---|---|
| the policy's knock-over: triangle, vertex-aligned heading towards a neighbour 7.5 cm inward | blind | 12/12 | 12 | 0 / 0 (gap 24–35 mm, largest shift 4 mm) | 12/12 |
| | obstacles | 6/12 | 5 | 0 / 0 | 5/12 |
| cube, neighbour 8 cm behind (towards the robot) | blind | 6/6 | 0 | 3 / 1 (gap 2–4 mm: under the thumb) | 0/6 |
| | obstacles | 6/6 | 6 | 0 / 0 | 6/6 |
| cube, neighbour diagonally ahead (6, 5.5 cm) | blind | 6/6 | 5 | 6 / 6 (gap 24–31 mm) | 0/6 |
| | obstacles | 6/6 | 6 | 0 / 0 | 6/6 |
| cube, neighbour 8 cm ahead (valid picks only) | blind | 3/3 | 3 | 2 / 2 (gap 24–25 mm; untouched at 31 mm) | 1/3 |
| | obstacles | 2/2 | 2 | 0 / 0 | 2/2 |

- Every pick planned with `obstacles` left the neighbour exactly where it was (26 of 26: 0.0 mm).
- The preshaped thumb comes down 8 cm behind the grasp point: a neighbour there is hit from above and the target is
  not even gripped (0/6). This is the commonest blind conflict in sampled layouts.
- The policy's case is not reproduced: its vertex-aligned plan passes a flat side of the neighbour at 24–35 mm and
  shifts it by at most 4 mm, while the obstacle check rejects that heading and finds no other in 6 of 12 (the default
  hints then lose one of six). At the same gap a neighbour that turns a vertex to the finger is toppled 6/6.
  Contact needs gap < the neighbour's extent in that direction + ~12 mm (finger half-width): triangle 15–30 mm, cube
  22.5–32 mm, cylinder 22.5 mm; the 47 mm circle is the never-touch value for a vertex or a cube corner plus 3 mm.
- Cost of the circle, offline (40 layouts, 169 targets, every neighbour present): the blind first candidate passes
  within 47 mm of a neighbour in 31 % of the picks (within 26 mm: 12 %); with `obstacles` 17 % of the targets have no
  plan at 47 mm, 11 % at 40 mm, 5 % at 34 mm.
- Half of the "ahead" cells were void: a neighbour at pelvis x 0.38 spawns against the bin wall and is lying before
  the pick starts (gaps of 64–258 mm). Neighbours must stay at x ≤ 0.36.

Verdict: the check works and stays at 47 mm for objects of unknown shape and heading. What it rejects needlessly are
flat sides and cylinders; the footprint (shape and yaw of the neighbour) would say which. Gap scan with cube, cylinder
and triangle neighbours: sweep 26.

## Sweep 24: the policy's first side-bin slot, 2 cm inward of the release point, against the walls

Question from full episodes: slots that keep two objects of one bin apart put the first one 2 cm inward (65 mm from
the inner side wall); the landing box (inward y down to −59 mm in the grippy-light corner) says an object could
reach that wall. Stock sim, grippy-light corner, side bins, candidates; new metric: distance of the object's centre
to the nearest inner wall when its bottom passes the rim (under 22.5 mm it touches the rim).

| release point | shape | binned | centre to wall passing the rim, min / p5 | touches the rim | at rest, min |
|---|---|---|---|---|---|
| slot −2 cm | cube | 12/12 | 37 / 40 mm | 0 | 37 mm |
| | cylinder | 12/12 | 28 / 38 mm | 0 | 28 mm |
| | triangle | 24/24 | 28 / 29 mm | 0 | 28 mm |
| default (85 mm from the side wall) | cube | 11/12 | 45 / 46 mm | 0 | 45 mm |
| | cylinder | 12/12 | 49 / 50 mm | 0 | 49 mm |
| | triangle | 12/12 | 36 / 41 mm | 0 | 36 mm |

Verdict: no wall incident in 48 releases at −2 cm, smallest margin over touching 5.5 mm. The −59 mm extreme is one
release in about 160 on record (p5 −45 mm); at this slot it leaves the centre 6 mm inside the wall's inner face: the
object scrapes the wall and falls inside. The release point stays; the bench takes `drop_shift` per variant.

## Obstacles as outlines (offline; sim evidence: sweep 26, queued)

`plan_pick(..., obstacles=...)` now takes detections as well as bare top points. A detection is kept off by its
outline (square for a cube with a yaw, triangle with the apex along its yaw, circle of half its size for a cylinder;
circle of `motion.obstacles.radius` when the shape or yaw is unknown or the prism is lying); digit centre lines, now
sampled at five points per distal link, stay `finger_radius + margin` = 17 mm from it. Bare tops behave exactly as
before (47 mm from the centre). Offline over 40 layouts (169 targets, every neighbour present): the blind first plan
passes within 13 mm of a neighbour's true outline (touching) in 20 % of the picks; targets left without a plan: 17 %
with bare tops, 12 % with detections; planning time unchanged (0.4 s mean, 4.6 s worst). The bench passes
`avoid: footprint` and records the planned clearance to the neighbour's true outline (`neighbour_clear_mm`).
Not yet used by the policy (it passes tops) and not yet measured in sim.

## Objects close to a bin's near wall (offline; sim cell: sweep 27, queued)

Policy episode: a cube that came to rest 3.2 cm (centre) before the centre bin was refused 47 times ("hand −0.4 cm
above a bin rim during descend"). The preshaped fingers come down ~4 cm ahead of the object along the finger heading:
with the fingers forward that is on the wall. The default hints do include an inward direction, but only as a weak
preference (weight 0.3) that the IK does not follow there. Offline, cube at 2.5–5 cm before the walls, cage:

| centre before the wall | default hints | fingers 90° inward, grasp point at 0.6 of the height | 90° outward, same |
|---|---|---|---|
| 5 cm | 4/6 (the rest plan at 60° inward) | | |
| 4 cm | 0/6 | centre bin 5/5, side bins 0/5 (the IK flips the hand) | side bins 7/7 |
| 3 cm | 0/6 | centre bin 5/5 | side bins 5/5, between the bins 0/2 |
| 2.5 cm | 0/6 | centre bin 2/5 | 2/7 |

At 90° the hand is tilted 22–36° (reach), so with the grasp point at mid-height the closed fingertips are 8–9 mm above
the counter when the lift starts and the organiser's check (12 mm) refuses the plan; 4.5 mm higher it passes.
Strategies can now fix their headings (`inward_headings_deg: [90, -90]`, tried in that order, not filtered by the
natural-heading range; `_check_finger_yaw` rejects the one the arm cannot turn to). Not in the candidates yet: a
tilted hand at that heading has never held an object in sim (sweep 27: `side` at the wall, `side_mid` in the middle
of the table, `reference` = today's candidates).

## Sweep 23: prisms lying on a rectangular side (23a, 23b; 23d / 23e queued)

Why: on the real table the foam prism lies on a side in most first frames, and in sim a triangle toppled by a failed
grasp lies the same way. Perception reports it as `shape="triangle", lying=True`, yaw = ridge direction, top = ridge
centre. Sim prism: 52 mm across the ridge, 45 mm along it, ridge 45 mm high (real: 45 x 45, ridge ~39 mm).

Bench: variant key `topple_deg` tips the upright prism with a scripted closed-hand push through its upper third, from
the vertex nearest that direction towards the opposite face, and reads the true pose back; the ridge ends up along
the push. From x 0.30 it worked 8/8; from x 0.265 (sweep 23b) 60 % of the pushes did not plan and the prism stayed
upright, so only records with `lying` true count below.

Geometry: along the ridge the two end faces are vertical and 45 mm apart like a cube's, but triangular: 26 mm wide at
mid-height, narrower than the 40 mm between the fingertips, so a centred cage straddles them. Across the ridge the
45 mm long sloped faces span the fingertips like a cube face, but they lean 30° inward: the squeeze pushes the roof
down and the fingers up the slope, friction only (needs µ > 0.58).

Lying and planned picks, 23a = soft-hand sim (smoke), 23b = stock sim (80 picks; sweeps 23d / 23e,
from x 0.29 on both sims, are queued and land in `runs/grasp/` on the instance):

| grasp | stock | soft | what happens |
|---|---|---|---|
| unknown-shape cage, default hints, 0.75 (the policy's fallback, `cage_lying`) | 1/5 | | |
| cage along the ridge, 0.30 | 0/4 | 0/3 | grip signal 0.00–0.19: the fingers straddle the end face |
| organiser pinch along the ridge | 0/5 | | grip signal 0 |
| cage across the ridge, 0.45 / 0.60 / 0.60 with the grasp point at 0.4 of the height / 0.80 | 0/2, 1/1, 0/3, 2/2 | 0/2 at 0.60 | soft: grip reads firm (0.25–0.33), the roof stays on the counter |
| pinch along the ridge, thumb swung opposite one finger (`thumb_0` −0.63), 0.30 | 2/5 | | |
| the same with the grasp point at 0.4 of the height | 4/5 | | |
| the same with the thumb at half its swing (−0.315) | 2/3 | | |

The opposed pinch: object under one finger (`point_in_palm` z 0.0285), preshape `[-0.63, -0.5, 1.5, -0.4 x4]`, closed
`[-0.63, 0.1, 1.01, -1.0, -0.6, -1.0, -0.6]` (thumb tip at palm x 65 mm, z 28.5 mm), squeeze axis
`[0.46, 1.0, -1.23, 0, 0, 0, 0]` (46 mm per rad at constant height and z).

Verdict so far: nothing is confirmed (n ≤ 5 per cell). The opposed pinch holds 8 of 13 over its three variants, best
with the lower grasp point (4/5); across the ridge only the deepest closure held (0.80: 2/2, shallower 1/6). The
opposed pinch depends on `thumb_0`, which on the real hand did not follow below −0.26 in one of the organisers' recordings.
`candidate_grasps(..., lying=True)` therefore still returns the unknown-shape cage.

## Sweep 27: objects 3.5–4 cm before a bin's near wall, fingers sideways (stock sim, DR as it comes)

Hypothesis: the cage with the fingers turned 90° (inward first, then outward) and the grasp point at 0.6 of the height
plans where no default-hint grasp does, and the tilted hand (22–36°) still holds. Cubes and cylinders, centre 3.5–4 cm
before the centre and side bins; the same grasp in the middle of the table; today's candidates as reference.

| cell | planned | binned | spin / tilt (median) | centre to wall passing the rim, min |
|---|---|---|---|---|
| candidates (reference), at the wall | 0/8 | | | |
| sideways cage, at the wall | 16/16 | cube 8/8, cylinder 8/8 | 14° / 20°, 4° / 19° | 39 mm |
| sideways cage, middle of the table | 8/8 | cube 4/4, cylinder 4/4 | 24° / 21°, 13° / 29° | 38 mm |

Grip signal as firm as the forward cage (largest joint error 0.064–0.138). Verdict: `cage_side` is the LAST candidate
for cube, cylinder and unknown shape on both hand classes (it only runs when nothing else plans; the soft half is
queued as s27b). Triangles are not covered (no cell).

## Sweep 20: bin walls counted as 70 mm, and the low pick sequence with the candidates (stock sim, hard DR)

| cell | cube | cylinder | triangle | centre to wall passing the rim, min | pick duration |
|---|---|---|---|---|---|
| `min_wall_height: 0.07`, organiser sequence | 12/12 | 12/12 | 12/12 | 37 mm | 8.07–8.25 s |
| low sequence, walls as in the file | 12/12 | 12/12 | 12/12 + 16/16 at yaw ±45° | 28 mm | 6.10–6.23 s |
| low sequence, `min_wall_height: 0.07`, `lift_height: 0.08` | 12/12 | 12/12 | 12/12 | 38 mm | 5.97–6.32 s |

All 124 carried objects were out of the hand within the open phase; landing minus release point x −32…+65 mm, outward
y −48…+44 mm (inside the box of sweeps 5 and 11); no release touched a rim.

Verdict: `motion.min_wall_height: 0.07` is the default from here: 36/36, no cost in success, landing or time (the
organiser sequence took 8.66 s in sweep 14). The policy's own transit checks (`bin_clearance`) see the taller wall too.
The low sequence with vertex-aligned triangles stands at 40/40 triangles and 88/88 picks (sweep 14 lost 2 of 20
default-hint triangles), 2.1 s faster per pick. It is NOT switched on here: nominal clearance of the held object over a
70 mm rim is 14–17 mm before arm sag and tilt, fine in sim (0 touches in 36) but thinner than the head calibration
error on the real robot, and it has never run on the soft-hand sim (queued: s20c). To use it:
`sequences: [low, organiser]`, `lift_height: 0.08`.

## Sweeps 23d, 23f, 23g: lying prisms, the decision (stock 23d + 23f, soft-hand sim 23g; DR as it comes)

Counted: picks where the prism was lying after the push AND the grasp planned (the push leaves a third of them within
6 cm of a bin wall or under the shoulder, where nothing plans). Both arms mixed; 23f / 23g: 6 grasps x 2 push
directions x 8.

| grasp (fingers along the ridge unless noted, grasp point at 0.4 of the ridge height) | stock | soft |
|---|---|---|
| thumb swung opposite one finger (`thumb_0` −0.63), squeeze 0.30 | 18/23 (23b 4/5, 23d 4/5, 23f 10/12) | 9/14 |
| the same, squeeze 0.45 | 6/9 | 11/15 |
| the same at mid height, 0.30 | 6/10 | |
| thumb at half swing (−0.315), object midway between thumb and finger (palm z 22 mm), 0.20 / 0.40 | 9/12, 4/10 | 4/14, 6/15 |
| thumb at half swing, object under the finger (a thumb that stalls halfway), 0.30 | 4/7 | |
| organiser pinch (`thumb_0` 0), no squeeze | 15/21 (23d 4/7, 23f 11/14) | 0/14 |
| organiser pinch, squeeze 0.30 | 5/14 | 4/16 |
| cage ACROSS the ridge, 0.60–0.80 | 12/24 | 0/2 (23a) |
| unknown-shape cage, default hints (the old fallback) | 5/19 | |

- Along the ridge the grip is two contacts on the vertical end faces; it holds by geometry like a cube pinch but the
  faces are triangles, so the contacts must be low and opposite each other. Deeper squeezes push the prism out
  sideways on the stiff hand; the soft hand needs the depth for force (grip signal 0.2–0.45 held).
- Across the ridge every success is the prism re-seating in the hand (spin 40–170° between close and lift): the
  sloped faces only hold by friction. The fallback cage succeeds only when its default heading happens to be across
  the ridge (4/6 there, 0/9 along it: the fingertips straddle the end face, grip signal 0).
- A thumb that only gets halfway loses about half on the stiff hand and two thirds on the soft one; without
  `thumb_0` nothing holds on the soft hand. The real hand's `thumb_0` tracking at −0.63 is unverified (one organiser
  recording stalled at −0.26 when commanded below −0.9): the policy should watch `thumb_0` when it preshapes.
- Coverage (offline): an arm turns its fingers along the ridge for ridges within ±45° of its natural heading
  (25° inward), about half at ±60°, almost never beyond: a third of the ridge directions per arm, a sixth with the
  better arm of the two (ridge roughly along pelvis y), fall through to the cage. `height_frac` 0.4 also plans for
  the real 39 mm ridge.

Verdict: `candidates.stiff.lying: [ridge_opposed, ridge_pinch, cage_lying]` (18/23 and 15/21 against 5/19 for the old
fallback); `cage_lying` is marked `last_resort`. The soft list: sweep 23h below.

## Sweep 27b: the sideways cage on the soft-hand sim

`cage_side` with the soft closure, DR as it comes: 3.5–4 cm before a bin wall 14/16 (cube 7/8, one not lifted;
cylinder 7/8, one lost in the lift), middle of the table 8/8; all 22 carried objects out of the hand within the open
phase; grip signal 0.16–0.24 held, ≤ 0.014 empty. Verdict: `cage_side` stays the last soft candidate (stock: 24/24).

## Sweep 23h: lying prisms on the soft-hand sim with a thumb that stalls at `thumb_0` −0.26

Question: the soft list runs on the real hand, whose `thumb_0` stalled at about −0.26 in one organiser recording.
What is left of the opposed pinch then, and does anything without the swing hold? Lying and planned picks, 2 push
directions x 8 per grasp, DR as it comes.

| grasp (soft closure: the finger closes too, lift and carry at 4 rad/s²) | held |
|---|---|
| opposed pinch, full swing (−0.63), squeeze 0.60 | 11/12 |
| the same grasp with the thumb held at −0.26 in preshape and closure (the stall), 0.45 | 9/16 |
| thumb at −0.26 and the object midway between thumb and finger (palm z 20.5 mm), 0.45 / 0.65 | 8/12, 7/14 |
| cage across the ridge, 0.90 (`thumb_0` 0) | 0/10 |

With 23g the full swing reads 9/14, 11/15, 11/12 at 0.30, 0.45, 0.60: the soft hand gains with depth. A stalled thumb
costs a third of the picks (9/16) and moving the grasp point to suit it does not help (15/26 over both depths);
nothing without the swing holds on the soft hand (0/10 here, 0/14 and 4/16 for the pinches in 23g).

Verdict: `candidates.soft.lying: [ridge_opposed, cage_lying]` with the soft squeeze at 0.60. One grasp serves both
cases: 11/12 when the thumb follows, 9/16 when it stalls at −0.26.
