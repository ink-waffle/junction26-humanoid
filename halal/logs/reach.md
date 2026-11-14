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

## Sweeps r3 (centre bin, +10 cm) and r4 (side bins, +12 cm): releases on the rim, stock hand

Sim EV is an UPPER BOUND for the real robot: these releases only land inside because the objects leave the hand
forward at the reach limit (median +46…+58 mm here), which is unexplained and unverified for the real soft hand.

| bins, bin | config | planned | binned | on the counter / off | EV | inset, rise | landing minus release point, x (binned) | by shape |
|---|---|---|---|---|---|---|---|---|
| +10, centre | rise list (insets ≥ 3 cm) | 0/12 | | | | | | "arm cannot reach the center bin" |
| +10, centre | `rim` [0.02, 0.01] | 12/12 | 10/12 | 2 / 0 | +0.83 | 1 cm, +6 cm | +15…+106 mm (median +58) | cube 2/4, cylinder 4/4, triangle 4/4 |
| +12, side | `rim` [0.02, 0.01, 0.0] | 24/24 | 14/24 | 10 / 0 | +0.58 | 0 cm, +6 cm | +27…+87 mm (median +46) | cube 0/8, cylinder 6/8, triangle 8/8 |

- The centre bin at +10 cm (release at y 0) reaches only 1 cm inside the wall; the side bins at +12 cm only the wall's
  inner face (inset 0). Every miss came to rest on the counter in front of the box (0 points); nothing left the counter
  and nothing reached another box in 36 rim releases.
- Shape decides: triangles 12/12, cylinders 10/12, cubes 2/12 (why the cube falls short was not traced).

## Sweeps r4 (centre bin, +12 cm) and r5 (soft hand, side bins, +6 cm)

| bins, bin, hand | config | planned | carried | binned | on the counter / off | inset, rise | landing minus release point, x | centre to wall passing the rim, min |
|---|---|---|---|---|---|---|---|---|
| +12, centre, stock | `rim` [0.02, 0.01, 0.0] | 4/12 | 4 | 0/4 | 4 / 0 | 0 cm, +6 cm | | all four on the rim |
| +6, side, soft (`HAND_GAINS` 1.5, 0.2, 0.0075) | rise list | 36/36 | 35 | 35/35 | 0 / 0 | 6 cm, +6 cm | −31…+60 mm (median +2) | 29 mm, 0 contacts |

- +12 cm centre: only cubes planned (pinch candidate, inset 0), none went in; cylinders and triangles "arm cannot
  reach the center bin". The centre box at +12 cm is out of reach for every release.
- Soft hand at +6 cm: the risen release behaves as on the stock hand (stock: 36/36, −34…+56 mm, 26 mm). One cube was
  lost during the carry (a grasp matter; it is not in the 35).

## Sweep r5 (soft hand, side bins, +8 cm) and the landing box of risen releases

| bins, bin, hand | config | planned | carried | binned | on the counter / off | inset, rise | landing minus release point, x | centre to wall passing the rim, min |
|---|---|---|---|---|---|---|---|---|
| +8, side, soft | rise list | 36/36 | 33 | 33/33 | 0 / 0 | 4 cm, +6 cm | −17…+79 mm (median −2) | 23 mm, 0 contacts |

Three objects were lost before the release (not lifted, lost in the lift, lost in the carry: grasp matters).

Landing minus release point over every binned release of r1–r5 (outward = away from the robot's midline):

| hand, release | n | x min…max | x p5 / median / p95 | outward y min…max |
|---|---|---|---|---|
| stock, today's height | 48 | −15…+51 mm | −13 / +6 / +32 | −49…+46 mm |
| stock, risen 6–9 cm | 152 | −34…+129 mm | −17 / +17 / +88 | −57…+89 mm |
| soft, risen 6 cm | 68 | −31…+79 mm | −17 / 0 / +42 | −31…+83 mm |

- Backward the risen release is no worse than the box the motion log has for the nominal height (−32 mm): −34 mm.
  Forward it reaches +129 mm on the stock hand, with the long tail from the releases at the reach limit (insets
  ≤ 2 cm, medians +25…+58 mm); sideways the spread grows to −57…+89 mm.
- The soft hand shows no forward drift at +6 / +8 cm (median 0); it was not run at the reach limit, so whether the
  drift that carries the rim releases inside exists there is unknown. Sim EV of the rim releases stays an upper bound.

Verdict so far: `release_rise: [0, 0.03, 0.06, 0.09, 0.12]` holds on both hands up to +8 cm (164/164 carried objects
binned at +6 / +8, no rim contact that kept an object out). Insets under 3 cm: pending the joint proposal with motion.
Sim d is back on stock settings.

## Sweep r6: soft hand, side bins, +10 cm (the real hand's class at the reach limit)

| config | planned | carried | binned | on the counter / off | inset, rise | landing minus release point, x | centre to wall passing the rim, min |
|---|---|---|---|---|---|---|---|
| rise list, insets ≥ 3 cm | 0/24 | | | | | | "arm cannot reach the bin" (the soft list has no pinch) |
| `rim` [0.02, 0.01] | 24/24 | 22 | 22/22 (cube 6/6, cylinder 8/8, triangle 8/8) | 0 / 0 | 2 cm, +6 cm | +11…+148 mm (median +40) | 29 mm, 0 contacts |

- The forward drift at the reach limit exists on the soft-hand sim too (median +40 mm; stock +25 mm at the same
  point), although it is absent at +6 / +8 cm (median 0): it belongs to the nearly straight arm, not to the hand class.
  Its cause is still not traced, and the soft-hand sim has no backlash or stiction: the EV stays an upper bound.
- Two objects were lost before the release (lift, carry). Sim d is back on stock settings.
- Percentiles of landing minus release point, binned releases (min / p2.5 / median / p97.5 / max, mm): stock risen,
  inset ≥ 3 cm (n 104): x −34 / −26 / +8 / +75 / +110, outward y −52 / −45 / +8 / +71 / +89; soft risen, inset ≥ 3 cm
  (n 68): x −31 / −20 / 0 / +52 / +79, y −31 / −26 / +6 / +37 / +83; stock risen at the limit, inset ≤ 2 cm (n 48):
  x +3 / +6 / +46 / +117 / +129, y −57 / −46 / −4 / +60 / +82.

## Sweep r7: soft hand, centre bin, +10 cm (a 1 cm inset: the rim release on the real hand's class)

Same sim settings (soft: `HAND_GAINS` 1.5, 0.2, 0.0075), centre bin, one arm at a time, release at y 0, 4 picks per
shape, code as for r1–r6 (no landing rule). Columns as r3/r4; "landing" is over every carried object.

| bins, bin, hand | config | planned | carried | binned | on the counter / off / other bin | EV | inset, rise | landing minus release point, x | centre to wall passing the rim, min | by shape (binned / carried) |
|---|---|---|---|---|---|---|---|---|---|---|
| +10, centre, soft | `rim` [0.02, 0.01] | 12/12 | 11 | 11/11 | 0 / 0 / 0 | +1.00 | 1 cm, +6 cm | +16…+107 mm (median +49) | 26 mm | cube 3/3, cylinder 4/4, triangle 4/4 |

- Per shape, landing minus release point x (median): cube +16…+19 (+18), cylinder +21…+107 (+82), triangle +48…+62 (+51) mm.
  Stock hand at the same point (r3): cube 2/4 (both misses rest 27–28 mm in front of the near wall), cylinder 4/4,
  triangle 4/4.
- One cube was not lifted (grasp matter; not in the 11). The forward drift at the reach limit is there on the soft hand
  too (median +49 mm). EV stays an upper bound for the real robot (no backlash / stiction in the soft sim).
