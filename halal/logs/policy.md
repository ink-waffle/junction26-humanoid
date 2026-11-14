# Policy workstream log

State machine, policy server and submission image (`halal/policy/`, `halal/server.py`, `halal/docker/`).
All sim numbers: sim `a`, `AALTO_DR=train`, seeds 1000–1007, 120 s limit, unless stated. Newest entries at the bottom
of each section.

## Floor

| Policy | Leg | Score | Objects | Outcomes | Gate |
|---|---|---|---|---|---|
| organiser baseline (server) | fixed | +1.62 ± 0.86 | 13/31 | 13 correct, 18 counter | pass (30.0 Hz, rt 60–83 Hz) |

Where the floor loses objects (from its runner logs, `runs/eval/baseline_sort_fixed_train_20261114_002421`):
- Finger signal while the hand command is at `closed`: 23 of 41 grasps over six episodes read 0.000 rad one second
  after closing (nothing in the hand) and were still carried to the bin and back: ~6 s lost each. Seed 1004: 9
  grasps, all empty, 0/5; seed 1005: 7 of 10 empty.
- A held object reads 0.041–0.18 rad (sum over the hand's joints of the shortfall behind the closed posture),
  constant through the carry; 0.25 s after closing an empty hand still reads 0.05–0.11 (lag), at 0.5 s 0.002–0.016
  (0.06–0.07 when the object is slipping out), at 1.0 s exactly 0.000.
- Seed 1002: the second blue object came to rest on the first one in the right bin at z = 0.080 (rim 0.070) and
  scored 0: both were released at the same point.
- Failed grasps shove objects up to 15 cm; the organiser detector then sees them elsewhere and the attempt counter
  (by position) starts again.

## What the policy is (v0)

`SortPolicy` (own state machine, not a subclass of the organiser policy): look pose → look (detector) → targets
(instruction mapping, object area, confidence, attempts left) → first plannable target nearest its bin, arm on the
bin's side first, each candidate grasp → execute in 10-tick chunks → look again. Perception and planning run in one
low-priority background thread; `act()` never waits (worst measured 4.6 ms) and answers 6-tick holds meanwhile.
`done` only after two looks in a row without a sortable object. A pick starts only if its release fits before the
time limit; if the episode outlasts the assumed limit the policy carries on.

Local proof (Mac, kinematic mock world through the organisers' runner at wall-clock 30 Hz): 5/5 and 3/3 sorted,
0 clips, command rate 29.9 Hz, `policy_hz_rt` ~290 Hz; `replay --synthetic 100`: pass. Image `halal/policy:v0`
(1.48 GB, Python 3.12 and the package versions of the `isaac` env): pre-check through the container passes
(startup 1.6 s, 1014 Hz, p50 5.6 ms).

Failure injection (scratch scripts outside the repo, mock world): blank head frames for the first 40 calls, no
depth at all, detector raising on two looks, planner raising once / always, reset during a 1 s plan followed
directly by the next episode, assumed limit 20 s with a real limit of 60 s, start pose far from home with closed
hands, an object lost during the carry, a mock without any finger signal: no exception reaches the runner, 0
clipped steps, the affected object is left on the counter at worst.

## Ideas, one at a time

### Geometric target filter (in v0)
Why: the organiser detector reports washed-out bin rims and things behind the bins as yellow objects (precision
0.69 on the perception benchmark). Rule: a target must lie on the counter top (2 cm inside its edge) and not in a
bin or within 3 cm of a bin footprint; reach is left to the planner. First sim episode (seed 1000, organiser
detector): 21 of 27 detections over five looks rejected, all of them at x ≈ 0.63 (behind the bins); none picked.

### Release slots (in v0)
Why: seed 1002 of the floor run (above). Each bin has two release points 7 cm apart, used in turn
(`policy.release_shifts`, passed as `plan_pick(drop_shift=...)`). Reach checked offline: sideways shifts of
6–9 cm plan with every arm that reaches the bin (+0.5 s); +x shifts from 6 cm on do not (IK `lower`).
Verdict: pending the sim legs below.

### Hand calibration close + grasp verification (v1)
Why: the floor wastes ~6 s on every empty carry and never learns that a grasp failed.
What: once per episode, at the look pose, both hands close on nothing (1 s); the trace gives per hand the
empty-hand tracking error e0, the lag behind the command and the class (stiff = sim, soft = real). After the lift
of each pick `score = Σ_j max(0, s_j (e_j − e0_j))` over the hand's seven joints (e = commanded − measured at the
closed posture, s = closing direction) is compared with the class threshold (`policy.grip_threshold`: stiff 0.02,
soft 0.3); an empty hand goes straight back to the look pose (checked joint move), and the same test runs through
the carry to notice a drop. If the first three picks of an episode all read empty and none ever read held, the
signal is distrusted and the check is switched off for the episode (a mock without contact physics, or a hand
whose signal we misjudge, then behaves like v0 instead of sorting nothing).
Verdict: pending.
