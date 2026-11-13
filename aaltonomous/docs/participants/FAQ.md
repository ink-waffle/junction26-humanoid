# FAQ

*Version 2026-11-13.*

### Compute

**What machine should I use?** One **RTX-class** cloud GPU per team for Isaac Sim (RTX Pro 6000 / A6000 / L40S
class). Isaac renders the cameras with RT cores: **A100 / H100 have none** and are the wrong choice for the sim (fine
for training). Get whatever GPUs you can from Verda (Junction's cloud partner); we don't allocate compute.

**On-demand or spot?** **On-demand only.** Spot instances can disappear within seconds and take your run with them.

**Where do I keep checkpoints?** On a **persistent block volume** (or the shared file system), not the instance
disk. If an instance dies, attach the volume to a new one and resume. Also push your best weights to a private
Hugging Face repo as a backup.

**How do I get started on the instance?** [Quickstart](../../cloud/QUICKSTART.md). Work over SSH + `tmux`; Isaac
runs headless and still renders the cameras.

**Can I run the sim on my laptop?** Only with an RTX GPU. Without one, use the `mock` backend for plumbing
(`python -m aaltonomous.deploy.runner --backend mock ...`) and the cloud for anything visual.

**What runs the finals?** One PC with an **RTX 5090 (32 GB)**, wired to the robot. Your container must fit in it.

### Data

**Where is the data?** Real teleop demos in LeRobot format: [Data card](../../data/DATA_CARD.md), how they were
recorded: [RECORDING.md](../../data/RECORDING.md). Drop 0 at kickoff, Drop 1 (venue lighting, empty-scene RGB-D,
real depth noise) on Saturday. Download links are posted on the Junction Discord.

**How much real data is there?** About 20 episodes per rung plus a few deliberately messy ones. That is the point
of the challenge: sim is free, reality is scarce.

**Can I use other datasets / pretrained models?** Yes, anything public. See
[STARTER_PATHS.md](STARTER_PATHS.md#public-data-for-pretraining).

### The task

**Do I have to use both arms?** No, but you may. The side bins are each easiest for the arm on that side.

**Are the bin positions fixed?** Yes: three identical orange boxes, left, center and right, at known positions
([INTERFACE.md §7](INTERFACE.md#7-scene-geometry)). Only the objects move between episodes.

**Do I get object poses?** Not in a submission. The sim's scene service has them (useful for training, RL rewards,
data generation), but at eval time your policy only sees `obs`.

**Is there force / tactile feedback?** Not in `obs`. Only joint positions (`hand_q`) tell you what the fingers did.

**Does the robot walk?** No. Legs and waist are held stiff at a fixed pose; only arms and hands move.

**Why does my policy get clipped?** Read `runs/<team>/<run>/episode_NNN.jsonl`: every clip has a reason
(`arm_velocity`, `arm_position`, `workspace`, `invalid:...`). The usual cause is jumping to a target faster than
1 rad/s. The sim gate allows **0** clips.

**My model takes seconds per step. Is that allowed?** Yes. While a call is in flight the robot holds its last
target, for up to 30 s per call (then the episode ends, `policy_timeout`). But those held ticks don't count toward
the sim gate's 27 Hz command rate, so to pass the gate: think in a background thread and keep returning targets
(e.g. hold), or return long action chunks. An error reply is different: 1 s of errors ends the episode (watchdog).
See [INTERFACE.md §3](INTERFACE.md#timing-30-hz-chunks-holds).

**Can my container call an external API (e.g. a hosted VLM)?** Yes, at your own risk. We provide no network: the
finals machine has whatever connection the venue gives it, or none. Bring your own keys. A failed or slow call
counts like any other failure; no reruns. Have an offline fallback.

### The baseline

**What does the baseline do?** `aaltonomous/baselines/classical/`: colour thresholding on the head RGB image + head
depth → object top centre in the pelvis frame → top-down "claw" grasp planned with IK → carry to the bin of the same
colour → release → back home. Plans are checked against the safety filter before execution (0 clips). Rung 1 only;
it ignores shape and the instruction text. Objects that one arm can't take to their bin are first put down where the
other arm can finish.

**How well does it score?** Isaac sim, rung 1, nominal scene (no randomisation), 2026-11-11, seeds 0–7: **76–82 %
of objects sorted** (29/38 as the submission image through the evaluator, 31/38 in-process), 0 safety clips,
command rate 30 Hz. Cylinders and cubes almost always work; most misses are triangles (the pinch slips) and the
120 s limit. Sim-tuned settings are in the `sim` profile of `config.yaml`. On the real robot it has done a pick into a bin.
It is meant as a floor and a code reference: rung 2, randomised scenes and speed are where to beat it.

**Is the baseline available as a submission image?** Yes: `aaltonomous/docker/baseline.Dockerfile` builds it
(CPU only, ~1.5 GB, starts in < 2 s), and `baselines/classical/policy_server.py` is the wrapper. It is the reference
submission; see [SUBMISSION.md §7](SUBMISSION.md#7-reference-submission-the-classical-baseline).

### Rules and finals

**How are finalists chosen?** Organisers evaluate submissions in sim (held-out variant); only
submissions that pass the sim gate qualify. See [RULES_AND_SCORING.md](RULES_AND_SCORING.md#4-finalist-selection-sim).

**What is the held-out variant?** The same task with different textures, lighting and object poses that you never
train on. Its configuration is not published.

**What if the robot breaks during the finals?** The organisers decide (rerun or sim score).
