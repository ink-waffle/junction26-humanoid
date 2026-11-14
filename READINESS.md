# Halal Team: readiness for the real robot

Aaltonomous Humanoid, Junction 2026. Team: Damir Kairov, Maksim Kruglov. Code: this repository (`halal/`).
Image (public, CPU only, 1.5 GB, starts in 0.5 s):
`ghcr.io/ink-waffle/halal-policy@sha256:ede4203ebfb1647edd2a136c53417888e9b2ad3b68113d593d1fbb7dce79989d`

## In simulation, through the submitted container

Organisers' evaluator, training domain randomisation, 8 seeds per task, sim gate on every episode.

| Task | Organiser baseline | Ours (v19) |
|---|---|---|
| `sort_fixed` | +1.62 (13/31 objects) | +3.88 (31/31) |
| `sort_language`, training template | not applicable | +7.00 (28/28) |
| `sort_language`, unseen paraphrase | not applicable | +7.00 (28/28) |
| Gate (0 clipped steps, ≥ 27 Hz, real-time rate) | pass | pass on 24 of 24 episodes, real-time rate ≥ 149 Hz |

No object in a wrong bin and none knocked off the counter in any measured episode of the last five versions.

## What we did about the gap between the sim and the robot

- **Hands.** The sim's finger actuators are about 67× stiffer than the robot's (kp 1.5 / kd 0.2). We built a
  second sim with the real gains, fitted to the finger step response in the teleop drops (50 % at 0.14 s, 90 % at
  0.3–0.4 s). The policy closes each empty hand once per episode, recognises which kind of hand it has, and
  chooses grasp closure, lift acceleration and contact threshold from that. On the soft-hand sim: 31/31 objects,
  gate pass, hands recognised in 16 of 16 episodes.
- **Camera.** The head camera pose is re-estimated in every episode from depth alone: counter plane, the box row,
  and the robot's own forearms against forward kinematics (the only reference tied to the arm). Position error
  under ±1 cm / ±1° camera jitter: 1.7 mm median, 2.7 mm max, against 11.6 / 29.9 mm for the baseline detector.
  Boxes and counter are used where they are seen, not where the scene file puts them.
- **Colours.** Red / yellow / blue in sim, purple / yellow / orange on the robot: three hue families measured on
  drop 2 (orange 81/81, purple 44/44, yellow 62/62 objects named correctly), with the orange–yellow boundary
  taken from the scene's own hue gap. Orange objects are told from the orange box tape by geometry.
- **Grasping.** One three-finger grasp per shape with a measured closure depth, fingers along a vertex for
  triangular prisms, staggered finger release; grasp checked from finger tracking error after the lift, with
  the threshold calibrated on the empty hand. Tolerates ±10 mm of position error (59/60 picks).
- **Instructions.** Deterministic parser, no network: 100 % on the 8 training templates; 97.9 % on 4087 unseen
  paraphrases written blind by a second author (2.1 % wrong with confidence); understands the drop-2 style
  ("Put the orange cube into the right box.") and leaves an object alone rather than guess.
- **Safety.** Every trajectory is dry-run through the organisers' safety filter before it is sent: 0 clipped
  steps in every evaluated episode. The policy starts from the measured pose, whatever it is, and holds still
  when it has no depth or no confident plan.

## What a short robot slot would tell us

1. Whether the hand recognition and the contact signal read on the real fingers as they do in the drops.
2. Whether the camera estimate from the forearms holds on real depth (the drops have none).
3. One pick of each shape into a box.

How the numbers were obtained, including what failed: `NIGHT_LOG.md` and `halal/logs/`.
