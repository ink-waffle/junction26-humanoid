# Drop 1 venue capture: one-page procedure (Friday 2026-11-13, ~1 h)

One organiser can do it alone except step 6 (arm poses need a teleop operator + spotter; optional).
The robot only **streams**: `capture.py` never commands it (DDS subscribe-only). Robot on the gantry / seated,
final counter, mat, bins, lights. Steps and positions: `checklist.yaml`. Time per step in brackets.

## 0. Before (10 min, recording laptop, env `uni3`)
```bash
cd ~/UNITREE-G1-RESEARCH && git pull && conda activate uni3
python -m unittest aaltonomous/data/drop1/test_drop1.py          # 5 OK, offline
bash scripts/deploy_rgbd_pc2.sh                                    # PC2: head RGB-D :55560 + wrist cams
python -m aaltonomous.data.record_demos --backend real --network-interface enx00e04cd30dba --check  # images live
```
Have: the 18 printed objects, the printed layout cards (seeds 1000, 1001, 2000), a white A4 sheet, phone charged.

## 1. Capture (~45 min)
```bash
python -m aaltonomous.data.drop1.capture --backend real --network-interface enx00e04cd30dba \
    --out data/drop1 --session venue_1113 --operator <name>
```
Each step prints what to set up. **Enter** = capture, **s** = skip, **q** = quit, or type a note then Enter
(e.g. "sun through window"). After a warning: Enter = keep, r = redo. Crash or break: rerun the same command,
finished steps are skipped. In a hurry: `--frames-scale 0.5 --skip-optional`.

| # | step | what you do | time |
|---|---|---|---|
| 1 | `empty_on` | mat + 3 bins only, event lights, nobody in view, arms at rest | 4 min |
| 2 | `phone_hdri` | **phone**: panorama / HDRI from the robot's head position; counter from front, left, right, top; mat close-up (glare); every light source; white A4 on the counter | 8 min |
| 3-8 | `obj_<colour>_a/b` | 3 objects of one colour at x = 30 / 45 / 60 cm, y = 15.5 cm (mat grid), order as printed | 6 × 2 min |
| 9-11 | `layout_1000/1001/2000` | place the printed layout card | 3 × 2 min |
| 12 | `layout_2000_off` | same layout, **venue lights off**, our lamps on | 2 min |
| 13 | `empty_off` | remove the objects, lights as in 12 | 3 min |
| 14 | `empty_ownlamps_off` *(opt.)* | venue lights on, **our lamps off**; then restore event lighting | 2 min |
| 15-17 | `pose_*` *(opt.)* | **teleop operator** holds the arms still (start pose, above mat, above bins); spotter at e-stop | 7 min |

Phone tips: lock exposure/white balance where possible, panorama at the head camera height (~1.25 m), one HDR
bracket set if the phone has it. Name nothing; just copy all files into one folder afterwards.

## 2. Stats + package (~5 min, any machine with the repo)
```bash
python -m aaltonomous.data.drop1.stats data/drop1/venue_1113          # stats/stats.json + PNGs
python -m aaltonomous.data.drop1.package data/drop1/venue_1113 --phone-dir ~/phone_drop1 --out data/ --date 2026-11-14
```
Look at `stats/masks_obj_*.png` (green contour = colour mask; magenta = colour failed, projection used) and the
`calib_offset_px` median in the README: above ~15 px the head extrinsics are off, fix calibration (M7) and say so.
Over 2 GB: `--max-frames 20` and/or `--no-wrist`. Upload `drop1_2026-11-14.tar` for Saturday morning.
