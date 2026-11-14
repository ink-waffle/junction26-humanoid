# Night log, Aaltonomous Humanoid (Halal Team)

Newest status first; experiments below, grouped by idea. All sim numbers are with training domain randomisation
(`AALTO_DR=train`) on seeds 1000–1007 unless stated. Proxy = `fixed + known + 1.5 × unseen` (mean episode score per leg).
Per-workstream detail: `halal/logs/*.md`. How things are organised: `halal/README.md`.

## Status

### Sat 14.11, 02:45
- **Best version:** none of ours yet. Floor = organiser baseline (rung 1 only), numbers below.
- **Now:** five workstreams in parallel, each in its own git worktree (`ws/*` branches) and on its own sim:
  policy core (state machine, verification, retries, hand-in image), motion (grasp research per shape, faster
  planner), perception (benchmark on 120 DR frames with ground truth, camera self-calibration, shape + yaw),
  real-data analysis (organisers' first drop), language (rule parser + held-out paraphrase benchmark).
- **Instance:** two sims with DR at real-time factor ~0.27 each (one alone: 0.45). Credits: ~€199 at launch
  (Fri 22:10 UTC), ~€2.6/h, so ~€187 left; the Verda console is down (their domain), the instance is unaffected.
- **Organisers (Discord, read 02:30):** no hand-in form, no deadlines, no sim image yet. First real data drop is
  out (20 teleop episodes "pick up cube", RGB only). Finals instructions: colour first, shape-based ones possible,
  no negation. Real hand pads grip less than the sim's.

## Floor: organiser baseline and oracle

| Policy | Sim | Seeds | Objects | Score | Gate |
|---|---|---|---|---|---|
| classical baseline (image) | nominal | 1000–1001 | 6/8 | +3.00 | pass |
| oracle (privileged poses) | nominal | 1000–1002 | 12/12 | +4.00 | pass |
| classical baseline (server) | DR train | 1000–1002 (run in progress) | 6/12 | +2.00 | pass |

Under DR the baseline loses a third of what it sorted in the nominal scene on the same seeds (2/3, 2/5, 2/4
versus 3/3, 3/5). It never reads the instruction, so on `sort_language` it scores about −1 per object whenever the
mapping differs from the fixed one.

## Ideas

(filled in as results arrive)
