# Rules and scoring

*Version 2026-11-13. Organisers may clarify details during the event; announcements on the Junction Discord.*

## 1. Rungs

| Rung | Task ID | Instruction | Mapping |
|---|---|---|---|
| 1 | `sort_fixed` | Always `"Put the red objects in the left bin, the yellow objects in the center bin and the blue objects in the right bin."` | Fixed: red → left, yellow → center, blue → right; shape ignored |
| 2 | `sort_language` | A sentence from a template set with paraphrases | Defined by the instruction. Bins are always named by position (`left`, `center`, `right`) |
| 3 (stretch) | `insert_shapes` | `"Put each object through the hole of the same shape in the lid."` | Circle / square / triangle hole in a lid on one bin. Sim; real robot only as a bonus if time allows |

- The three bins are identical orange boxes with no markings, named only by position: **left, center, right**.
- Rung 2 training templates are in `aaltonomous/interface/tasks.py` (`TEMPLATES`, 8 of them). The instruction
  defines the mapping; evaluation and finals instructions may be phrased differently from the training templates.
  Expect paraphrases you haven't seen.

## 2. Episode

- **Reset:** 3–5 objects from the 9 types (at most 2 of one colour), random position and yaw (±45°) in the object
  band, at least 8 cm between centres. Arms start in a fixed home pose, hands open. On the real robot staff place
  the objects from a printed layout card made by the same sampling code (`tasks.sample_layout`).
- **Time limit:** 120 s. The episode ends at the limit, when your policy sends `done: true`, or on a safety stop.
- **Success rule** (one checker, `aaltonomous/scene/scene.py`, same in sim, eval and for the judges):
  > An object counts as IN a bin if its centre is inside the bin's inner walls, below the rim, and it is at rest
  > (< 2 cm/s) at episode end. Off the counter (or more than 3 cm below its surface) = knocked off. Anything else =
  > left on counter.

  On the real robot, staff judge each object by eye with the same rule.

## 3. Points per object

| Outcome | `sort_fixed` | `sort_language` |
|---|---:|---:|
| In the correct bin | +1 | +2 |
| In a wrong bin | −1 | −1 |
| Left on the counter (or still moving at the end) | 0 | 0 |
| Knocked off the counter | −1 | −1 |

`insert_shapes`: +3 through the matching hole, −1 forced/jammed on the lid or dropped beside it, 0 otherwise.

**Episode score** = sum over objects. **Safety abort** (more than 30 clipped steps, see §6) ends the episode; points
scored so far are kept.

## 4. Finalist selection (sim)

- **Sim score** = mean episode score over seeded episodes on the **held-out sim variant**: same task, different
  textures, lighting, object poses and camera jitter than anything you can train on. Seeds are secret.
- Organisers pick the finalists from the sim leaderboard.
- **Sim gate** — only submissions that run cleanly get real-robot time. Defaults of
  `aaltonomous/sim/evaluate.py`:
  - no safety abort, no watchdog / safe stop, no policy timeout (> 30 s per call), no crash, no lost connection;
  - **0 clipped steps** (`--gate-max-clips 0`): every target must already respect the safety limits, including the
    velocity cap;
  - **command rate ≥ 27 Hz** (`--gate-min-hz 27`): ticks that execute a target from your policy ÷ episode time. Ticks
    where the runner has to hold because your server hasn't answered yet do **not** count. Chunks count per tick.
    In sim, the 30 Hz loop and the 120 s limit run on the **sim clock** (the sim is slower than real time), so a second check applies:
  - **your own latency must also allow ≥ 27 Hz** (`policy_hz_rt`: command ticks ÷ wall time your server spent
    answering). This is what the real robot will see: a policy that only keeps up because the sim is slow fails.

## 5. Real-robot finals

- Finals: **3 episodes**, scored on the real robot only:
  1. `sort_fixed`
  2. `sort_language` with a known (training) template
  3. `sort_language` with an unseen instruction and layout — **weight ×1.5**
- **Final ranking** = real-robot score. Sim score breaks ties.
- Finals settings (objects, layout, time limit) are not announced in advance.

## 6. Safety (enforced, not negotiable)

- **Nobody touches the robot.** Teams never get direct access to Väiski, DDS, SSH or the robot network. You hand in a
  Docker image; our runner executes it through our safety filter while staff stand at the e-stop.
- Every target goes through the safety filter: joint limits minus a margin, a **velocity cap of 1 rad/s (arms) and
  3 rad/s (hands)**, a workspace box and a counter keep-out box for the wrists and palms, a watchdog, and an abort
  after more than 30 clipped steps. Details: [INTERFACE.md](INTERFACE.md#5-safety-filter).
- Staff may stop any run at any time with the e-stop. A stopped run keeps the points scored so far.
- If the robot fails or a run is stopped through no fault of the team, the organisers decide what happens (rerun
  or sim score).
- Robot cooldown breaks between runs are at staff discretion.

## 7. What's allowed

| Allowed | Not allowed |
|---|---|
| Any method: classical, VLM/LLM planning, IL, VLA fine-tuning, RL, hybrids | Commanding the robot except through a submission run by the organisers |
| Any public pretrained model or dataset | Training or tuning on the held-out sim variant or its seeds (we only tell you it exists) |
| Training on our demos, the sim, and all drops | Trying to extract or reverse-engineer held-out configs, seeds or templates |
| Generating your own sim data (Mimic, DR, RL rollouts) | Using privileged sim state (scene service object poses) inside a submission: submissions only see `obs` |
| Using both arms | Hard-coding layouts; the layouts are random |
| Calling external APIs from your container, **at your own risk**: we provide no network (see [FAQ](FAQ.md)) | |

## 8. Award

- **Winner** — best real-robot finals score. Prizes are announced by Junction.
