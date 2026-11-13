"""Recorded demo episodes (record_demos.py, aalto-demo-v1) -> GR00T dataset: LeRobot v2.1 + meta/modality.json.

Isaac-GR00T (N1.5/N1.6) fine-tunes on the "GR00T LeRobot" layout, which is LeRobot **v2.1** (one parquet and one
mp4 per episode) plus meta/modality.json. Its loader does not read LeRobot v3.0 (what to_lerobot.py writes by
default), so this writes v2.1 directly from the raw episodes. Same episodes, same frames, same vectors as the v3
dataset; no lerobot install needed (numpy, pyarrow, opencv, ffmpeg).

    python aaltonomous/data/to_lerobot.py data/demos/r1_* --root data/lerobot/drop0 --format both  # v3 + <root>_groot
    python aaltonomous/data/to_groot.py data/demos/r1_* --root data/groot/drop0                      # GR00T copy only

Layout (30 fps, vectors [left 7, right 7] in joint_spec.yaml order, absolute rad):
  observation.state (28) = arm_q ++ hand_q        action (28) = arm_q_target ++ hand_q_target (commanded)
  observation.arm_dq (14)                          task_index / annotation.human.task_description -> meta/tasks.jsonl
  videos: observation.images.{head_rgb,left_wrist_rgb,right_wrist_rgb}  H.264 yuv420p, keyframe every 2 frames
  meta/modality.json   state/action groups left_arm, right_arm, left_hand, right_hand; 3 video keys; the instruction
  meta/episodes_stats.jsonl  LeRobot's per-episode stats (also lets lerobot's v2.1 -> v3.0 converter read this copy)
  meta/aalto_episodes.jsonl  same per-episode extras as the v3 dataset
  new_embodiment_config.py   GR00T modality config: --modality-config-path <root>/new_embodiment_config.py
Head depth is NOT in this copy: GR00T's video path is 8-bit RGB. Depth stays in the v3 dataset and the raw PNGs.
Train:  python gr00t/experiment/launch_finetune.py --dataset-path <root> --embodiment-tag NEW_EMBODIMENT \\
            --modality-config-path <root>/new_embodiment_config.py ...   (Isaac-GR00T computes its stats itself)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aaltonomous.data.to_lerobot import CAMS, FPS, SIDECAR_KEYS, find_episodes, valid_range  # noqa: E402

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
ROBOT_TYPE = "unitree_g1_29dof_dex3"
ANNOTATION_KEY = "human.task_description"
ANNOTATION_COLUMN = f"annotation.{ANNOTATION_KEY}"
GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 21), "right_hand": (21, 28)}
ACTION_HORIZON = 16
QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def video_feature(cam):
    return f"observation.images.{cam}"


# ------------------------------------------------------------------ one episode
def encode_video(files, out_mp4, ffmpeg="ffmpeg"):
    """JPEGs -> H.264 yuv420p at FPS. A keyframe every 2 frames and no B-frames, as LeRobot encodes its own
    (VideoEncoderConfig g=2): training samples frames at random, and long GOPs made frames past the first episode
    undecodable on random access after LeRobot's v3 concatenation."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="groot-frames-") as td:
        for i, f in enumerate(files):
            os.symlink(os.path.abspath(f), os.path.join(td, f"{i:06d}.jpg"))
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                        "-i", os.path.join(td, "%06d.jpg"), "-frames:v", str(len(files)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-bf", "0", "-crf", "18",
                        "-movflags", "+faststart", str(out_mp4)], check=True)


def count_video_frames(path, ffprobe="ffprobe"):
    out = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def _stats(arr, axis, keepdims):
    arr = np.asarray(arr, dtype=np.float64)
    out = {"min": arr.min(axis=axis, keepdims=keepdims), "max": arr.max(axis=axis, keepdims=keepdims),
           "mean": arr.mean(axis=axis, keepdims=keepdims), "std": arr.std(axis=axis, keepdims=keepdims)}
    for name, q in QUANTILES:
        out[name] = np.quantile(arr, q, axis=axis, keepdims=keepdims)
    return out


def sample_indices(n):
    """LeRobot's rule for which frames image statistics come from (compute_stats.sample_indices)."""
    k = max(min(n, 100), min(int(n ** 0.75), 10_000))
    return np.round(np.linspace(0, n - 1, k)).astype(int).tolist()


def columns(state, action, arm_dq, task_index, episode_index, global_start):
    n = len(state)
    return {
        "observation.state": state, "action": action, "observation.arm_dq": arm_dq,
        # on the grid exactly: LeRobot checks timestamp * fps against frame_index within a tolerance
        "timestamp": (np.arange(n) / float(FPS)).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": np.arange(global_start, global_start + n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
        ANNOTATION_COLUMN: np.full(n, task_index, dtype=np.int64),
        "next.reward": np.zeros(n, dtype=np.float32),
        "next.done": np.arange(n) == n - 1,
    }


def write_parquet(path, cols):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({k: (pa.array(list(v.astype(np.float32)), type=pa.list_(pa.float32())) if v.ndim == 2
                          else pa.array(v)) for k, v in cols.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def episode_stats(cols, frames, episode_index):
    """meta/episodes_stats.jsonl line in LeRobot's shapes: a vector reduces over time to its own shape, a scalar
    keeps [1], a video is per channel [3, 1, 1] in [0, 1] from sampled frames; count = episode length."""
    import cv2
    n = len(cols["timestamp"])
    stats = {}
    for k, v in cols.items():
        s = _stats(v.astype(np.float64), 0, v.ndim == 1)
        s["count"] = np.array([n])
        stats[k] = s
    for cam, files in frames.items():
        imgs = np.stack([cv2.imread(str(files[i]))[..., ::-1] for i in sample_indices(len(files))])
        s = {m: v.reshape(3, 1, 1) for m, v in _stats(imgs.transpose(0, 3, 1, 2) / 255.0, (0, 2, 3), True).items()}
        s["count"] = np.array([n])
        stats[video_feature(cam)] = s
    return {"episode_index": episode_index,
            "stats": {k: {m: np.asarray(v).tolist() for m, v in s.items()} for k, s in stats.items()}}


# ------------------------------------------------------------------ meta files
def info_json(arm_names, hand_names, total_episodes, total_frames, n_tasks, hw):
    h, w = hw
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}  # noqa: E731
    names = lambda pre: [f"{pre[0]}.{n}" for n in arm_names] + [f"{pre[1]}.{n}" for n in hand_names]  # noqa: E731
    feats = {
        "observation.state": {"dtype": "float32", "shape": [28], "names": names(("arm_q", "hand_q"))},
        "action": {"dtype": "float32", "shape": [28], "names": names(("arm_q_target", "hand_q_target"))},
        "observation.arm_dq": {"dtype": "float32", "shape": [14], "names": [f"arm_dq.{n}" for n in arm_names]},
    }
    for cam in CAMS:
        feats[video_feature(cam)] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    feats.update({"timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
                  "index": scalar("int64"), "task_index": scalar("int64"), ANNOTATION_COLUMN: scalar("int64"),
                  "next.reward": scalar("float32"), "next.done": scalar("bool")})
    return {"codebase_version": CODEBASE_VERSION, "robot_type": ROBOT_TYPE, "total_episodes": total_episodes,
            "total_frames": total_frames, "total_tasks": n_tasks, "total_videos": total_episodes * len(CAMS),
            "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE, "chunks_size": CHUNK_SIZE,
            "fps": FPS, "splits": {"train": f"0:{total_episodes}"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
            "features": feats}


def modality_json():
    groups = {g: {"start": a, "end": b} for g, (a, b) in GROUPS.items()}
    return {"state": groups, "action": dict(groups),
            "video": {cam: {"original_key": video_feature(cam)} for cam in CAMS},
            "annotation": {ANNOTATION_KEY: {"original_key": "task_index"}}}


EMBODIMENT_CONFIG = '''"""GR00T modality config for the Aaltonomous G1 + Dex3 demos (written by aaltonomous/data/to_groot.py).

Register with:  --modality-config-path <dataset>/new_embodiment_config.py  --embodiment-tag NEW_EMBODIMENT
State/action (28): left_arm[7] right_arm[7] left_hand[7] right_hand[7], absolute rad in the dataset, 30 fps.
Arms use RELATIVE actions (derived from the absolute state by GR00T's processor); hands stay ABSOLUTE.
The policy server must still send ABSOLUTE targets (arm_q_target, hand_q_target) to the runner.
"""
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


def _action(rep):
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=__VIDEO_KEYS__),
    "state": ModalityConfig(delta_indices=[0], modality_keys=GROUPS, sin_cos_embedding_keys=["left_arm", "right_arm"]),
    "action": ModalityConfig(
        delta_indices=list(range(__HORIZON__)),
        modality_keys=GROUPS,
        action_configs=[
            _action(ActionRepresentation.RELATIVE),   # left_arm
            _action(ActionRepresentation.RELATIVE),   # right_arm
            _action(ActionRepresentation.ABSOLUTE),   # left_hand
            _action(ActionRepresentation.ABSOLUTE),   # right_hand
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["__ANNOTATION_COLUMN__"]),
}

MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''


def embodiment_config_text():
    return (EMBODIMENT_CONFIG.replace("__VIDEO_KEYS__", json.dumps(list(CAMS)))
            .replace("__HORIZON__", str(ACTION_HORIZON)).replace("__ANNOTATION_COLUMN__", ANNOTATION_COLUMN))


# ------------------------------------------------------------------ the dataset
def missing_tools():
    out = []
    for mod, pip in (("pyarrow", "pyarrow"), ("cv2", "opencv-python")):
        try:
            __import__(mod)
        except ImportError:
            out.append(f"{mod} (pip install {pip})")
    out += [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    return out


def convert(srcs, root, include_failures=False, overwrite=False, allow_no_depth=False):
    missing = missing_tools()
    if missing:
        sys.exit("to_groot needs: " + ", ".join(missing))
    eps = find_episodes(srcs, include_failures, allow_no_depth)
    if not eps:
        sys.exit("no labelled episodes found")
    root = Path(root)
    if root.exists():
        if not overwrite:
            sys.exit(f"{root} exists (use --overwrite)")
        shutil.rmtree(root)
    tmp = root.with_name(root.name + ".partial")     # renamed when complete: a crash leaves no half dataset
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        n_eps = _write(eps, tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(root)
    print(f"-> {root}: {n_eps} episodes (GR00T LeRobot v2.1)")
    return root


def _write(eps, root):
    import cv2
    tasks, side, ep_lines, stat_lines, total, hw = {}, [], [], [], 0, None
    m0 = eps[0][1]
    for d, meta in eps:
        low = dict(np.load(d / "lowdim.npz"))
        a, b = valid_range(low)
        if b - a < FPS:
            print(f"skip {d.name}: only {b - a} valid frames")
            continue
        if b < len(low["t"]):
            print(f"note {d.name}: invalid/stale frames from {b}; kept frames {a}..{b} of {len(low['t'])}")
        frames = {cam: [d / cam / f"{i:06d}.jpg" for i in range(a, b)] for cam in CAMS}
        gone = [str(f) for fs in frames.values() for f in fs if not f.exists()]
        if gone:
            raise SystemExit(f"{d.name}: {len(gone)} camera frames missing, e.g. {gone[0]}")
        shape = cv2.imread(str(frames[CAMS[0]][0])).shape[:2]
        if hw is None:
            hw = shape
        elif shape != hw:
            raise SystemExit(f"{d.name}: frames are {shape}, earlier episodes {hw}")
        idx = len(ep_lines)
        task_index = tasks.setdefault(meta["instruction"], len(tasks))
        cols = columns(np.concatenate([low["arm_q"], low["hand_q"]], 1)[a:b].astype(np.float32),
                       np.concatenate([low["action_arm"], low["action_hand"]], 1)[a:b].astype(np.float32),
                       low["arm_dq"][a:b].astype(np.float32), task_index, idx, total)
        chunk = idx // CHUNK_SIZE
        write_parquet(root / DATA_PATH.format(episode_chunk=chunk, episode_index=idx), cols)
        for cam, files in frames.items():
            mp4 = root / VIDEO_PATH.format(episode_chunk=chunk, video_key=video_feature(cam), episode_index=idx)
            encode_video(files, mp4)
            got = count_video_frames(mp4)
            if got != b - a:
                raise SystemExit(f"{d.name}: {cam} video has {got} frames, expected {b - a}")
        stat_lines.append(episode_stats(cols, frames, idx))
        ep_lines.append({"episode_index": idx, "tasks": [meta["instruction"]], "length": b - a})
        side.append({"episode_index": idx, "source": str(d), "frames": [a, b], **{k: meta.get(k) for k in SIDECAR_KEYS}})
        total += b - a
        print(f"episode {idx}: {d.name} ({b - a} frames, success={meta['success']})")
    if not ep_lines:
        raise SystemExit("no episode had enough valid frames")
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    jsonl = lambda name, rows: (meta_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
    jsonl("episodes.jsonl", ep_lines)
    jsonl("tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    jsonl("episodes_stats.jsonl", stat_lines)
    jsonl("aalto_episodes.jsonl", side)
    (meta_dir / "info.json").write_text(json.dumps(
        info_json(m0["arm_names"], m0["hand_names"], len(ep_lines), total, len(tasks), hw), indent=2))
    (meta_dir / "modality.json").write_text(json.dumps(modality_json(), indent=2))
    (root / "new_embodiment_config.py").write_text(embodiment_config_text())
    return len(ep_lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help="session dirs (or a parent) with ep* episode dirs")
    ap.add_argument("--root", required=True, help="output dir")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--allow-no-depth", action="store_true",
                    help="also convert episodes recorded with --no-head-rgbd (same rule as the v3 converter)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    convert(a.src, a.root, a.include_failures, a.overwrite, a.allow_no_depth)


if __name__ == "__main__":
    main()
