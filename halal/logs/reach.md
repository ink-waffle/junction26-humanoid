# Reach log (workstream `reach`): releasing into boxes that stand further away than the scene file says

Newest entries at the bottom. Frame: pelvis, metres. "+N cm" = the bin row N cm further from the robot than `scene.yaml`
(near edge 0.40, inner near wall 0.403). Offline numbers: `python -m halal.eval.reach_bench`; sim numbers: sim `d`,
`REAL_SIM=boxes,offsets` (counter +1 cm, 7 cm walls, bins as stated), harness `halal.eval.grasp_bench` with the spec key `scene`.

## The arm's limit (offline, 14.11)

Farthest x of the cage grasp point (= object centre) the planner's IK holds (position tolerance 6 mm included), left arm,
by release height z; today's release is z 0.1425 (4 cm above a 7 cm rim, counter at +0.01):

| release point | z 0.1425 | 0.17 | 0.20 | 0.23 | 0.26 | 0.29 |
|---|---|---|---|---|---|---|
| centre bin, y 0 | 0.501 | 0.510 | 0.516 | 0.521 | 0.525 | 0.526 |
| centre bin, y +0.03 (the arm's own side) | 0.509 | 0.516 | 0.524 | 0.529 | 0.531 | 0.533 |
| side bin, y 0.243 | 0.509 | 0.518 | 0.524 | 0.529 | 0.532 | 0.534 |

- It is the straight arm: shoulder (0, 0.14, 0.29) to palm 0.41 m, palm to object 0.125 m. A search over every hand
  orientation (joint limits only) ends at the same ceiling: 0.52 at y 0, 0.54 at y 0.10–0.15.
- Palm tilt is not the limit: allowing 35° → 120° from top-down moves the table by 0–2 mm. Height is: the shoulder is at
  z 0.29, so every 4–5 cm of release height gives about 1 cm of reach, up to z ≈ 0.27.
- Letting go in a forward motion gains nothing: at 0.97 rad/s per joint the grasp point moves forward at 0.15–0.3 m/s
  near the bins (0.06–0.12 m/s at x 0.49, the arm is nearly straight); the best ballistic landing over release points
  x 0.38–0.49, z 0.17–0.27 and elevations 0–60° is x 0.51–0.53, the static ceiling, and the real fingers add 0.3–0.45 s
  of release-time spread. Rejected.
- Consequence: an object centre cannot be brought further than x 0.526 (centre bin) / 0.534 (side bins, centre bin on
  the arm's own side). Inner near wall at +8 cm: 0.483 (4–5 cm inside), +10: 0.503 (2.3–3.1 cm; an upright object needs
  2.25 cm to clear the wall), +12: 0.523 (0.3–1.1 cm: on the rim). A release inside a box at +12 cm does not exist.
- The teleop operator shows the same picture (FK of the recorded joints at 59 releases of both drops): palm never
  beyond x 0.41–0.42; the point between the fingertips at x 0.53–0.55 for the side boxes and 0.48–0.52 for the centre
  box, z 0.19–0.34 (12–25 cm above the table), slower than 0.1 m/s. Drop 1 ep17 (boxes estimated at 0.53–0.55): that
  point at x 0.533. The operator holds at the fingertips and drops from high up just inside the near wall.

## `motion.release_rise`: release higher when the release point is out of reach (planner, offline)

`release_rise` lists extra release heights. For every near inset (largest first) the planner takes the lowest rise from
which the arm reaches the release point (binary search; reach grows with height), carries straight to that height in
joint space (floor of the carry unchanged: the nominal release height) and shortens the retreat by the rise (never
below `min_retreat_up`). Default `[0.0]`: 456 plans (4 layouts, every object × bin × arm × hand class, bins +0 and
+3 cm) are bit for bit those of main.

The reach check itself changed for every configuration: it now runs with the grasp's own point in the palm, the finger
heading free and the carry's IK seeds, at the release point including the slot shift, cached per point (not dropped at
every `set_layout`). Before, it used the organiser pinch point (7 mm further out along the fingers), so near the limit
it let insets through whose plan then failed in the IK chain, once per finger hint.

`reach_bench`, cube, look pose → retreat, every bin × suitable arm × both policy slots × 2 spots (16 plans per row):

| bins | main: plans, inset, planning time mean / max | reach check fixed, rise [0] | rise [0, 0.03, 0.06, 0.09, 0.12] |
|---|---|---|---|
| +0 cm | 16/16, 6 cm, 0.08 / 0.25 s | 16/16, 6 cm, 0.09 / 0.25 s | 16/16, 6 cm, rise 0, 0.08 / 0.27 s |
| +3 cm | 16/16, 5–6 cm, 0.49 / 1.75 s | 16/16, 5–6 cm, 0.17 / 0.43 s | 16/16, 6 cm, rise 0–3 cm, 0.26 / 0.98 s |
| +6 cm | 12/16, 3–4 cm, 4.66 / 9.07 s | 12/16, 3–4 cm, 0.49 / 1.17 s | 16/16, 4–6 cm, rise 6–9 cm, 1.31 / 2.82 s |
| +8 cm | 0/16 | 0/16, 0.30 / 0.73 s | 12/16, 3–4 cm, rise 6–9 cm, 0.83 / 1.67 s |
| +10, +12 cm | 0/16 | 0/16 | 0/16 with insets ≥ 3 cm, 0.6 / 1.4 s |

- The 4 misses at +6 (main) and +8 (with rise) are the centre bin's slot on the far side of the arm (y ∓0.03); the
  other slot plans, so every bin keeps an arm.
- Planning time is the first call per release point (cold cache); the pick itself is as long as before (8.8–8.9 s).
- At the limit the IK ends 4–5 mm short of the release point (inside its 6 mm tolerance): the real inset is that much
  smaller than the table says.

## Sweep r1 (side bins): boxes +6 cm, risen release against today's (sim d, stock hand, DR as it comes)

Sim: `REAL_SIM=boxes,offsets`, bins +6 cm, counter +1 cm, 7 cm walls; bench spec key `scene` with the same values.
Both arms, each into its own side bin, policy candidates, cube / cylinder / triangle 12 each per cell, code a7b7c93.

| config | planned | binned | inset, rise | landing minus release point, x | centre to the nearest wall passing the rim, min | rim contacts |
|---|---|---|---|---|---|---|
| `release_rise: [0, 0.03, 0.06, 0.09, 0.12]` | 36/36 | 36/36 | 6 cm, +6 cm | −34…+56 mm (median +4) | 26 mm | 0 |
| today (`[0.0]`) | 36/36 | 36/36 | 4 cm, 0 | −15…+51 mm (median +7) | 25 mm | 0 |

Reading: at +6 cm the side bins work either way in sim. The rise buys 2 cm of inset; the landing spread from 6 cm
higher is 90 mm wide against 66 mm, well inside the 204 mm box, and no object touched a rim (contact starts under
22.5 mm). No verdict on the config value yet: the centre bin (where main loses plans at +6), +8 cm, the rim releases at
+10 / +12 and the soft hand are still running.

## Sweeps r1 (centre bin, +6 cm) and r2 (side bins, +8 cm), stock hand

Same sim settings as above with the bins as stated; centre bin: one arm at a time, release at y 0 (no slot shift),
4 picks per shape and cell; side bins at +8: 12 per shape.

| bins, bin | config | planned | binned | inset, rise | landing minus release point, x | centre to wall passing the rim, min | rim contacts |
|---|---|---|---|---|---|---|---|
| +6, centre | rise list | 12/12 | 12/12 | 5 cm, +6 cm | −25…+66 mm (median +13) | 25 mm | 0 |
| +6, centre | today | 12/12 | 12/12 | 4 cm, 0 | −12…+33 mm (median +5) | 28 mm | 0 |
| +8, side | rise list | 36/36 | 36/36 | 4 cm, +6 cm | −17…+62 mm (median +5) | 23 mm | 0 |

- +8 cm side bins: main plans nothing there (offline 0/16); with the rise every pick planned and was binned. The
  smallest margin at the rim is 0.5 mm over touching: an inset of 4 cm is the edge of contact-free, as expected from
  an object 45 mm wide.
- From 6 cm higher the landing spread is 80–90 mm wide (60–65 mm at the nominal height), median still within 13 mm of
  the release point: objects do not fly on.

## v18's centre-box refusals in the real-look sim (offline, 14.11)

Run `rs18-all-soft` (bins +6 cm, counter +0.02, soft hands): five objects bound for the centre box got "no plan for
any of N targets" for the rest of their episodes, in the same millisecond as the look and without a per-arm reason.
Cause: v18's `reaches_bin` answered for `self.near_inset` as the previous plan left it. After a side-bin pick (inset
5 cm, release x 0.513 as in the log) it is False for both arms on the centre box, which only reaches insets 3–4 cm
there, so the policy's arm filter left no arm and `plan_pick` was never called. With `plan_pick` called directly v18
plans all 24 cells (6 objects × 2 arms × 2 centre slots; 6–11 s per far-slot call). Main since a7b7c93: the filter
stays True; rise `[0.0]` plans 12/24 (each arm's near slot, inset 4 cm; the far slot is refused by the landing rule at
inset 3 cm), the rise list 24/24 (insets 4–6 cm), 18 s for the 24 calls against 106 s.

## Sweeps r2 (centre bin, +8 cm) and r3 (side bins, +10 cm), stock hand

`rim` = the rise list with `drop_inset.near: [0.02, 0.01]`. Expected value per object (EV) = (binned − off the counter −
wrong bin) / carried, sort_fixed points.

| bins, bin | config | planned | binned | counter / off | EV | inset, rise | landing minus release point, x | centre to wall passing the rim, min | rim contacts |
|---|---|---|---|---|---|---|---|---|---|
| +8, centre | rise list | 12/12 | 12/12 | 0 / 0 | +1.00 | 3 cm, +6 cm | −8…+110 mm (median +24) | 22 mm | 1 |
| +10, side | rise list (insets ≥ 3 cm) | 8/24 | 8/8 | 0 / 0 | +1.00 | 3 cm, +9 cm | −2…+102 mm (median +31) | 28 mm | 0 |
| +10, side | `rim` | 24/24 | 24/24 | 0 / 0 | +1.00 | 2 cm, +6 cm | +3…+129 mm (median +25) | 23 mm | 0 |

- At +10 cm with insets ≥ 3 cm only the cubes planned, and only with the pinch candidate (its point in the palm is
  7 mm further out than the cage's); cylinders and triangles: "arm cannot reach the bin" (16 of 24).
- With an inset of 2 cm every pick planned and all 24 landed inside. The objects do not fall straight down at the
  reach limit: they leave 3–129 mm forward of the release point (median +25 mm; at the nominal boxes the median is
  +5…+7 mm), so the centre is 23 mm or more from the near wall when it passes the rim although it was released 20 mm
  from it. The arm is nearly straight and tilted there; the forward drift is consistent in sim (0 of 24 short) but its
  cause is not established and nothing says the real hand does the same.
