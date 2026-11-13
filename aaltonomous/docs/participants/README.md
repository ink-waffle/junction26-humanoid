# Aaltonomous Humanoid — Junction 2026 challenge brief

*Version 2026-11-13. Times and links are posted in the challenge channel on the Junction Discord.*

## The pitch

**Sim is free, reality is scarce.** Väiski is a Unitree G1 humanoid with two Dex3 three-finger hands. It stands
behind a counter with a stiff lower body. Your code has to make it sort coloured objects into bins, **fully
autonomously**, using mostly simulation and a handful of real demonstrations. Then it has to survive reality: the
best teams run their policy on the real robot in the finals.

The hardware has moved fast; the intelligence hasn't. Real robot data is scarce and compute isn't. Any method is
allowed: classical perception + IK, a VLM planner over your own skills, imitation learning / VLA fine-tuning, RL,
or improving the sim itself.

## The task in one paragraph

3–5 printed objects (cube, cylinder or triangular prism; red, yellow or blue; about 45 mm) lie at random positions on the
counter. Three identical orange bins sit in a row behind the objects at fixed positions: left, center and right. Every 1/30 s your policy
gets head RGB-D, two wrist RGB images, the joint states and a text instruction. It returns absolute joint targets for
both 7-DoF arms and both 7-DoF hands. Every target passes through our safety filter. An episode lasts at most 120 s.

| Rung | Task ID | What |
|---|---|---|
| 1 | `sort_fixed` | Fixed mapping: red → left, yellow → center, blue → right |
| 2 | `sort_language` | A text instruction defines the mapping; bins are named by position |
| 3 (stretch) | `insert_shapes` | Each object goes through the matching hole of a lid on one bin (sim; real only as a bonus) |

Details: [RULES_AND_SCORING.md](RULES_AND_SCORING.md).

## Timeline (Fri 2026-11-13 → Sun 2026-11-15)

| When | What |
|---|---|
| Fri, kickoff | Live robot demo, brief, sim + baseline |
| Fri evening | **Drop 0**: first real-robot data (raw teleop demos) |
| Fri–Sat | Work in sim on cloud GPUs |
| Sat | More drops, e.g. **Drop 1**: venue lighting, empty-scene RGB-D, real depth noise on our objects ([how it is captured](../../data/drop1/CHECKLIST.md)). Organisers pick the finalists |
| Sun | Final submission (Docker image), **real-robot finals**, award |

## Sim and data: rough on purpose

The sim is **deliberately rough**. We will improve the basic calibration during the event (wrist camera mounts,
camera intrinsics, box and object sizes), but making it match reality — looks, lighting, depth noise, contact
physics — is part of the challenge. The real data drops are your reference.

Real data drops are **raw**: they include failed, messy and partly broken episodes. Each episode's success label,
operator notes and warnings are in `meta/aalto_episodes.jsonl`; deciding what to train on is up to you.

Updates to the sim image, calibration and data are **versioned** (`v1`, `v2`, …) and announced on Discord with a
one-line changelog. Values may change between versions; the obs/action keys and the data format will not.

## Read next

| Doc | For |
|---|---|
| [Quickstart](../../cloud/QUICKSTART.md) | From a fresh cloud instance to the baseline running in sim |
| [RULES_AND_SCORING.md](RULES_AND_SCORING.md) | Rungs, success rule, points, finalists, what's allowed |
| [INTERFACE.md](INTERFACE.md) | The frozen interface: obs/action keys, joint order, limits, cameras, scene geometry |
| [SUBMISSION.md](SUBMISSION.md) | The Docker policy-server contract and how to test it |
| [STARTER_PATHS.md](STARTER_PATHS.md) | Pointers per route (classical, VLM, IL/VLA, RL, real2sim) |
| [Data card](../../data/DATA_CARD.md), [recording setup](../../data/RECORDING.md) | The real demonstrations |
| [FAQ.md](FAQ.md) | Compute, data, baseline, common questions |

## Contact

Organisers: DigiTwin (Aalto). Questions, times, data links and announcements: the challenge channel on the Junction Discord.
