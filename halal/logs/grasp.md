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
