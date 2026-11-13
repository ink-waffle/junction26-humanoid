"""GR00T copy test (needs pyarrow, opencv, ffmpeg; skipped otherwise): fake recorded episodes -> to_groot ->
files, values and frame counts checked against the raw episodes. With lerobot installed (env `lerobot`) it also runs
LeRobot's own v2.1 -> v3.0 converter on the copy and loads every frame in random order.

    python -m unittest aaltonomous/data/test_to_groot.py
"""
import ast
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    import pyarrow.parquet as pq
    HAVE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
except ImportError:
    HAVE = False
try:
    import lerobot  # noqa: F401
    HAVE_LEROBOT = True
except ImportError:
    HAVE_LEROBOT = False


@unittest.skipUnless(HAVE, "needs pyarrow, opencv and ffmpeg")
class TestToGroot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aaltonomous.data import record_demos as R
        from aaltonomous.data.test_record import FakeCams, FakeDDS, run_ticks
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        rec = R.Recorder(FakeDDS(), FakeCams(), tmp, session="s", task="sort_language", log=lambda *_: None)
        for label, n in (("y", 35), ("f", 32), ("y", 40)):
            rec.handle_key("n")
            run_ticks(rec, n)
            rec.handle_key("s")
            rec.handle_key(label)
        nodepth = R.Recorder(FakeDDS(), FakeCams(depth_age=None), tmp, session="nd", log=lambda *_: None)
        nodepth.handle_key("n")
        run_ticks(nodepth, 31)
        nodepth.handle_key("s")
        nodepth.handle_key("y")
        cls.raw = [p for p in sorted((tmp / "s").iterdir()) if json.loads((p / "meta.json").read_text())["success"]]
        from aaltonomous.data import to_groot as G
        cls.G, cls.root = G, tmp / "groot"
        G.convert([tmp / "s", tmp / "nd"], cls.root)        # failures and the depthless episode excluded

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def meta(self, name):
        return json.loads((self.root / "meta" / name).read_text())

    def jsonl(self, name):
        return [json.loads(x) for x in (self.root / "meta" / name).read_text().splitlines()]

    def test_layout_and_info(self):
        info = self.meta("info.json")
        self.assertEqual((info["codebase_version"], info["fps"], info["total_episodes"]), ("v2.1", 30, 2))
        self.assertEqual(info["total_frames"], 75)
        self.assertEqual(info["total_videos"], 6)
        self.assertEqual(info["features"]["observation.state"]["shape"], [28])
        self.assertEqual(len(info["features"]["action"]["names"]), 28)
        self.assertNotIn("observation.images.head_depth", info["features"])
        self.assertFalse((self.root.parent / "groot.partial").exists())
        for e in range(2):
            self.assertTrue((self.root / info["data_path"].format(episode_chunk=0, episode_index=e)).exists())

    def test_values_match_raw_episodes(self):
        eps = self.jsonl("episodes.jsonl")
        tasks = {t["task_index"]: t["task"] for t in self.jsonl("tasks.jsonl")}
        start = 0
        for e, raw_dir in enumerate(self.raw):
            raw = np.load(raw_dir / "lowdim.npz")
            t = pq.read_table(self.root / f"data/chunk-000/episode_{e:06d}.parquet").to_pydict()
            n = len(raw["t"])
            self.assertEqual(eps[e]["length"], n)
            np.testing.assert_allclose(np.array(t["observation.state"]), np.c_[raw["arm_q"], raw["hand_q"]], atol=1e-6)
            np.testing.assert_allclose(np.array(t["action"]), np.c_[raw["action_arm"], raw["action_hand"]], atol=1e-6)
            self.assertEqual(t["index"], list(range(start, start + n)))
            self.assertEqual(t["frame_index"], list(range(n)))
            np.testing.assert_allclose(t["timestamp"], np.arange(n) / 30.0, atol=1e-6)
            instruction = json.loads((raw_dir / "meta.json").read_text())["instruction"]
            self.assertEqual({tasks[i] for i in t["task_index"]}, {instruction})
            self.assertEqual(t["annotation.human.task_description"], t["task_index"])
            self.assertEqual(eps[e]["tasks"], [instruction])
            start += n

    def test_videos_have_every_frame(self):
        for e, raw_dir in enumerate(self.raw):
            n = len(np.load(raw_dir / "lowdim.npz")["t"])
            for cam in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                mp4 = self.root / f"videos/chunk-000/observation.images.{cam}/episode_{e:06d}.mp4"
                self.assertEqual(self.G.count_video_frames(mp4), n)

    def test_modality_and_embodiment_config(self):
        mod, info = self.meta("modality.json"), self.meta("info.json")
        spans = sorted((g["start"], g["end"]) for g in mod["state"].values())
        self.assertEqual(spans, [(0, 7), (7, 14), (14, 21), (21, 28)])
        self.assertEqual(mod["action"], mod["state"])
        for v in mod["video"].values():
            self.assertIn(v["original_key"], info["features"])
        self.assertEqual(mod["annotation"], {"human.task_description": {"original_key": "task_index"}})
        src = (self.root / "new_embodiment_config.py").read_text()
        ast.parse(src)
        for cam in mod["video"]:
            self.assertIn(f'"{cam}"', src)
        self.assertIn('"annotation.human.task_description"', src)

    def test_episode_stats_shapes(self):
        st = self.jsonl("episodes_stats.jsonl")
        self.assertEqual([s["episode_index"] for s in st], [0, 1])
        s = st[0]["stats"]
        self.assertEqual(np.array(s["observation.state"]["mean"]).shape, (28,))
        self.assertEqual(np.array(s["timestamp"]["min"]).shape, (1,))
        img = s["observation.images.head_rgb"]
        self.assertEqual(np.array(img["mean"]).shape, (3, 1, 1))
        self.assertTrue(0.0 <= np.array(img["min"]).min() <= np.array(img["max"]).max() <= 1.0)
        self.assertEqual(img["count"], [35])

    def test_sidecar_and_refusals(self):
        side = self.jsonl("aalto_episodes.jsonl")
        self.assertTrue(all(x["success"] and x["head_depth"] for x in side))
        with self.assertRaises(SystemExit):
            self.G.convert([Path(self.tmp.name) / "s"], self.root)       # exists, no --overwrite

    @unittest.skipUnless(HAVE_LEROBOT, "needs env lerobot")
    def test_lerobot_reads_it_after_its_own_v3_conversion(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        v3 = Path(self.tmp.name) / "v3"
        shutil.copytree(self.root, v3)
        subprocess.run([sys.executable, "-m", "lerobot.scripts.convert_dataset_v21_to_v30", "--repo-id=local/g",
                        f"--root={v3}", "--push-to-hub=false"], check=True, capture_output=True)
        ds = LeRobotDataset("local/g", root=v3)
        self.assertEqual((ds.num_episodes, ds.num_frames), (2, 75))
        order = list(range(ds.num_frames))
        random.Random(0).shuffle(order)
        for i in order:                                   # random access, as training reads it
            x = ds[i]
            self.assertEqual(tuple(x["observation.images.head_rgb"].shape), (3, 480, 640))
            self.assertEqual(tuple(x["action"].shape), (28,))


if __name__ == "__main__":
    unittest.main()
