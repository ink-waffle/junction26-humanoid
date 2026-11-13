"""Recorded demo episodes (record_demos.py, format aalto-demo-v1) -> LeRobot dataset (v3.0 format, lerobot 0.7.0),
and/or a GR00T copy (LeRobot v2.1 + meta/modality.json, what Isaac-GR00T fine-tunes on; see to_groot.py).

    conda activate lerobot     # separate env (py3.12; lerobot 0.7 needs >= 3.12), see DATA_CARD.md
    python aaltonomous/data/to_lerobot.py data/demos/<session> [more dirs] --repo-id aalto/junction_demos_drop0 \
        --root data/lerobot/junction_demos_drop0 [--include-failures] [--overwrite] [--format v3|groot|both]
    python aaltonomous/data/to_lerobot.py --verify data/lerobot/junction_demos_drop0     # load + print one frame

--format both writes the v3 dataset at --root and the GR00T copy at <root>_groot (--groot-root to change it).
--format groot needs no lerobot install (numpy, pyarrow, opencv, ffmpeg).

Features (spec §4 keys; vectors are [left 7, right 7] in joint_spec.yaml order, names list every joint):
  observation.images.head_rgb / left_wrist_rgb / right_wrist_rgb   video 480x640x3 (AV1, lerobot default)
  observation.images.head_depth   depth video 480x640x1, lerobot native depth: HEVC 12-bit lossless, LINEAR
                                  quantisation 0..4.095 m = exactly 1 mm per code, so uint16 mm round-trips
                                  bit-exact up to 4095 mm (farther is clipped; 0 = invalid stays 0)
  observation.state   (28)  = arm_q (14) ++ hand_q (14)
  observation.arm_dq  (14)
  action              (28)  = arm_q_target (14) ++ hand_q_target (14)   absolute rad, the COMMANDED targets
  task                      = the episode's instruction string
Per-episode extras (success, layout, mapping, card seed, operator, warnings, safety-filter check) go to
<root>/meta/aalto_episodes.jsonl, keyed by episode_index.
Only successful episodes are converted unless --include-failures; .partial (unlabelled) episodes never are.
Episodes recorded without head depth (record_demos --no-head-rgbd, meta head_depth: false) are skipped unless
--allow-no-depth, because their head_depth video would be all zeros.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

H, W, FPS = 480, 640, 30
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
STALE_S = 0.2        # same as record_demos.STALE_S
DEPTH_MAX_M = 4.095  # linear 12-bit: 4095 codes over 0..4.095 m = 1 mm/code


def features(arm_names, hand_names):
    img = lambda c: {"dtype": "video", "shape": (H, W, c), "names": ["height", "width", "channels"]}
    f = {f"observation.images.{k}": img(3) for k in CAMS}
    f["observation.images.head_depth"] = {**img(1), "info": {"is_depth_map": True, "depth_unit": "mm"}}
    f["observation.state"] = {"dtype": "float32", "shape": (28,),
                              "names": [f"arm_q.{n}" for n in arm_names] + [f"hand_q.{n}" for n in hand_names]}
    f["observation.arm_dq"] = {"dtype": "float32", "shape": (14,), "names": [f"arm_dq.{n}" for n in arm_names]}
    f["action"] = {"dtype": "float32", "shape": (28,), "names": [f"arm_q_target.{n}" for n in arm_names]
                   + [f"hand_q_target.{n}" for n in hand_names]}
    return f


def find_episodes(srcs, include_failures, allow_no_depth=False):
    eps = []
    for s in srcs:
        for d in sorted(Path(s).glob("**/ep*")):
            if not d.is_dir() or d.name.endswith(".partial") or not (d / "lowdim.npz").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            if meta.get("success") is None or (not meta["success"] and not include_failures):
                print(f"skip {d.name}: success={meta.get('success')}")
                continue
            # head_depth missing = recorded before the recorder marked it (RGB-D was the default then)
            if meta.get("head_depth") is False and not allow_no_depth:
                print(f"skip {d.name}: recorded without head depth (--no-head-rgbd); --allow-no-depth keeps it")
                continue
            eps.append((d, meta))
    return eps


SIDECAR_KEYS = ("success", "task", "instruction", "mapping", "layout_id", "card_seed", "layout", "operator", "notes",
                "start", "duration_s", "warnings", "filter_check", "dds_rates_hz", "git", "head_depth", "depth")


def valid_range(low):
    """Frames [a, b) with finite action/state and live command streams (drops warm-up / teleop already stopped)."""
    keys = ("action_arm", "action_hand", "arm_q", "arm_dq", "hand_q")
    ok = np.all([np.isfinite(low[k]).all(1) for k in keys], axis=0)
    for k in ("age_lowcmd", "age_dex3_left_cmd", "age_dex3_right_cmd"):   # teleop stopped publishing
        if k in low:
            ok &= low[k] <= STALE_S
    if not ok.any():
        return 0, 0
    a = int(np.argmax(ok))
    b = a + int(np.argmin(ok[a:])) if not ok[a:].all() else len(ok)
    return a, b


def convert(srcs, repo_id, root, include_failures=False, overwrite=False, vcodec=None, allow_no_depth=False):
    import cv2
    from lerobot.configs import DepthEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = find_episodes(srcs, include_failures, allow_no_depth)
    if not eps:
        sys.exit("no labelled episodes found")
    root = Path(root)
    if root.exists():
        if not overwrite:
            sys.exit(f"{root} exists (use --overwrite)")
        shutil.rmtree(root)
    m0 = eps[0][1]
    kw = {}
    if vcodec:
        from lerobot.configs import RGBEncoderConfig
        kw["rgb_encoder"] = RGBEncoderConfig(vcodec=vcodec)
    ds = LeRobotDataset.create(repo_id, FPS, features(m0["arm_names"], m0["hand_names"]), root=root,
                               robot_type="unitree_g1_29dof_dex3", use_videos=True,
                               depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=DEPTH_MAX_M, use_log=False),
                               image_writer_threads=8, **kw)
    side = []
    for d, meta in eps:
        low = dict(np.load(d / "lowdim.npz"))
        a, b = valid_range(low)
        if b - a < FPS:
            print(f"skip {d.name}: only {b - a} valid frames")
            continue
        if b < len(low["t"]):
            print(f"note {d.name}: invalid/stale frames from {b}; kept frames {a}..{b} of {len(low['t'])}")
        state = np.concatenate([low["arm_q"], low["hand_q"]], 1).astype(np.float32)
        action = np.concatenate([low["action_arm"], low["action_hand"]], 1).astype(np.float32)
        for i in range(a, b):
            fr = {f"observation.images.{k}": cv2.imread(str(d / k / f"{i:06d}.jpg"))[..., ::-1] for k in CAMS}
            dep = cv2.imread(str(d / "head_depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            fr["observation.images.head_depth"] = dep[..., None]
            fr["observation.state"], fr["action"] = state[i], action[i]
            fr["observation.arm_dq"] = low["arm_dq"][i].astype(np.float32)
            fr["task"] = meta["instruction"]
            ds.add_frame(fr)
        ds.save_episode()
        idx = ds.meta.total_episodes - 1
        side.append({"episode_index": idx, "source": str(d), "frames": [a, b], **{k: meta.get(k) for k in SIDECAR_KEYS}})
        print(f"episode {idx}: {d.name} ({b - a} frames, success={meta['success']})")
    ds.finalize()
    with open(root / "meta" / "aalto_episodes.jsonl", "w") as f:
        for s in side:
            f.write(json.dumps(s) + "\n")
    print(f"-> {root}: {len(side)} episodes")
    return root


def verify(root, repo_id="local/verify"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    print(f"{root}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}")
    x = ds[0]
    for k, v in x.items():
        print(f"  {k:34s} {tuple(v.shape) if hasattr(v, 'shape') else repr(v)[:70]}"
              + (f" {v.dtype}" if hasattr(v, "dtype") else ""))
    return ds


def export_episode(ds, root, k, out):
    """Actions + states of episode k as read back by LeRobotDataset (not from the raw files)."""
    idx = [i for i in range(ds.num_frames) if int(ds.hf_dataset[i]["episode_index"]) == k]
    rows = ds.hf_dataset.select(idx)
    action = np.stack([np.asarray(r) for r in rows["action"]]).astype(np.float32)
    state = np.stack([np.asarray(r) for r in rows["observation.state"]]).astype(np.float32)
    side = [json.loads(line) for line in (Path(root) / "meta" / "aalto_episodes.jsonl").read_text().splitlines()]
    meta = next(s for s in side if s["episode_index"] == k)
    np.savez(out, action=action, state=state, meta=json.dumps(meta))
    print(f"exported episode {k}: {len(action)} frames -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="*", help="session dirs (or a parent) with ep* episode dirs")
    ap.add_argument("--repo-id", default="aalto/junction_demos")
    ap.add_argument("--root", default=None, help="output dir (default data/lerobot/<repo name>)")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--allow-no-depth", action="store_true",
                    help="also convert episodes recorded with --no-head-rgbd (their head_depth is all zeros)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--format", choices=("v3", "groot", "both"), default="v3",
                    help="v3 = LeRobot v3.0 (default); groot = LeRobot v2.1 + modality.json for Isaac-GR00T; both")
    ap.add_argument("--groot-root", default=None, help="GR00T copy dir (default: --root with --format groot, "
                    "<root>_groot with both)")
    ap.add_argument("--vcodec", default=None, help="RGB codec override, e.g. h264 (lerobot default: libsvtav1)")
    ap.add_argument("--verify", metavar="ROOT", help="only load ROOT with LeRobotDataset and print frame 0")
    ap.add_argument("--export-episode", type=int, default=None, help="with --verify: dump that episode's action/"
                    "state + aalto meta to --export-out (npz) for replay_sim.py in the sim env")
    ap.add_argument("--export-out", default="episode.npz")
    a = ap.parse_args(argv)
    if a.verify:
        ds = verify(a.verify)
        if a.export_episode is not None:
            export_episode(ds, a.verify, a.export_episode, a.export_out)
        return
    root = a.root or f"data/lerobot/{a.repo_id.split('/')[-1]}"
    if a.format in ("v3", "both"):
        convert(a.src, a.repo_id, root, a.include_failures, a.overwrite, a.vcodec, a.allow_no_depth)
        verify(root, a.repo_id)
    if a.format in ("groot", "both"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script: repo root for the import
        from aaltonomous.data import to_groot
        groot_root = a.groot_root or (root if a.format == "groot" else f"{root}_groot")
        to_groot.convert(a.src, groot_root, a.include_failures, a.overwrite, a.allow_no_depth)


if __name__ == "__main__":
    main()
