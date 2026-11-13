# M9 demo recording: operator checklist (Thursday 2026-11-12)

Three people: the **operator** (Quest), the **spotter** (e-stop and R3 remote, watches the robot), and the
**recorder** (laptop keyboard, resets the layout). The robot is in the final setup: seated or on the gantry,
counter, mat, bins and objects, and the same cameras and lighting as for the challenge.

Shortcut on the recording laptop: `bash scripts/record.sh teleop|check|rec r1|rec r2|convert` wraps steps 1 and 3
(robot interface auto-detected; convert uses env `lerobot07` = py3.12 + `lerobot[dataset]==0.7.0`, CPU torch).

## 0. One-time setup (recording laptop, env `uni3`)
```bash
conda activate uni3 && cd ~/UNITREE-G1-RESEARCH && git pull
python -m unittest aaltonomous/data/test_record.py                   # 11 OK, offline
python -m aaltonomous.scene.cards --task sort_fixed    --n 25 --seed 1000 --figs cards_r1 --out cards_r1.md
python -m aaltonomous.scene.cards --task sort_language --n 25 --seed 2000 --figs cards_r2 --out cards_r2.md
```
Print both card sets. Card *k* of a set is recorder card *k* (`--card-seed` is the set's first seed).

## 1. Start (every session)
1. **PC2 cameras in Junction mode** (head RGB-D on :55560, wrists on teleimager): `bash scripts/deploy_rgbd_pc2.sh`
2. **Teleop** (terminal A). The speed caps keep the demos inside the safety filter (1 rad/s arms, 3 rad/s fingers):
   ```bash
   python scripts/teleop_safe.py --arm=G1_29 --ee=dex3 --input-mode=hand --img-server-ip=192.168.123.164 \
       --network-interface=enx00e04cd30dba --track-speed 0.9 --finger-speed 3 --display-mode=pass-through
   ```
   With the head camera in RGB-D mode, teleimager does not stream it, so the operator sees the scene through
   Quest passthrough. **Check 1 below.**
3. **Recorder** (terminal B):
   ```bash
   python -m aaltonomous.data.record_demos --backend real --network-interface enx00e04cd30dba --check
   ```
   It must print `CHECK PASS`: lowcmd about 250 Hz, dex3 cmd and state about 100 Hz, lowstate 500 Hz, all 4 images
   live, and the `head RGB-D (depth)` line at >= 15 Hz with >= 30 % valid depth pixels. The recorder refuses to start
   an episode (`!! not starting: no head depth`) whenever no RGB-D frame has arrived for 0.2 s, so depth cannot
   silently go missing; if it does mid-episode, the `== saved` line warns (stale or all-zero depth ticks).
   Then:
   ```bash
   python -m aaltonomous.data.record_demos --backend real --network-interface enx00e04cd30dba \
       --task sort_fixed --card-seed 1000 --session r1_$(date +%m%d) --operator <name> \
       --notes "track 0.9, finger 3"
   ```

## 2. Per episode
1. The recorder shows card *k*. Place the objects from the printed card, using the mat grid.
2. The operator homes the arms (`r` to start or resume, `h` for hands), holding them still above the counter.
3. Recorder `n` → the operator sorts → recorder `s` once the last object is released and the arms are back near the start.
4. Label it: `y` success (every object in its correct bin), `f` failure, `d` discard (a teleop glitch, SPACE/limp,
   or someone in the view). The recorder then advances to the next card. `]`/`[` skip a card, `c` shows it again.
5. Read the line `== saved ... filter clips=N`. If N is far above 0, the operator was faster than the filter: slow down.

Aim for 20 to 25 successful episodes per rung, plus 3 to 5 deliberately messy ones (a regrasp, or a recovery after a
drop; still label them `y` if they end correct). Rung 2: restart the recorder with `--task sort_language
--card-seed 2000 --session r2_...`. The card gives the instruction, and the operator must follow it (bin colours
may contradict it). Names on disk are `data/demos/<session>/ep<NNN>_<task>_s<cardseed>/`. A crash loses only the
episode in progress (left as `.partial`); restart with the same `--session` to append.

## 3. Convert (twinflow or any machine with env `lerobot`)
```bash
rsync -a <recording-laptop>:UNITREE-G1-RESEARCH/data/demos/ data/demos/
conda activate lerobot
python aaltonomous/data/to_lerobot.py data/demos/r1_* data/demos/r2_* --repo-id aalto/junction_demos_drop0 \
    --root data/lerobot/junction_demos_drop0 --format both
```
`--format both` writes the LeRobot v3.0 dataset (with head depth) at `--root` and a GR00T copy (LeRobot v2.1 +
`meta/modality.json` + `new_embodiment_config.py`, RGB only) at `<root>_groot`. Ship both in the drop. `--format groot`
needs only numpy, pyarrow, opencv and ffmpeg (`aaltonomous/data/to_groot.py`). Episodes recorded with
`--no-head-rgbd` are skipped unless `--allow-no-depth`.
Fill in `DATA_CARD.md` (counts, issues) and ship it with the calibration bundle as Drop 0.

## Thursday checks (cannot be verified without the robot; do them before the first real episode)
1. **Teleop runs with the head camera in RGB-D mode.** Run `teleop_safe.py` as in step 1.2 after
   `deploy_rgbd_pc2.sh`. The page must load on the Quest and hands must track. If upstream fails with the head
   camera disabled, fallback: `deploy_rgbd_pc2.sh --stop`, record with `--no-head-rgbd` (head RGB from teleimager,
   **no depth**), and note it in the data card. Those episodes are marked `head_depth: false` and the converter
   skips them unless `--allow-no-depth` (their depth video would be all zeros).
2. **Same-host DDS.** The recorder `--check` must see `rt/lowcmd` and `rt/dex3/*/cmd` published by teleop on the
   same laptop (CycloneDDS loopback on `enx00e04cd30dba`). In sim on twinflow this works (domain 9, localhost).
   If cmd shows `MISSING` while the state topics are ok, stop and investigate. Do **not** record actions from state.
3. **Joint mapping on hardware.** After `r` + `h`, raise the operator's left hand. In `--check` output, `action arm`
   index 0-6 (left) must change, and `action hand` 0-6 must follow the left fingers.
4. **Laptop keeps 30 Hz.** On the first episode the `== saved` line must show no `late ticks` warning. If it does,
   close the camera_view tab and other load. Teleop IK plus recording together is the untested load on the i5.
