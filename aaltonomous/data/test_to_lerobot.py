"""Converter test (env `lerobot`; skipped elsewhere): fake recorded episodes -> LeRobot dataset -> load back.

    python -m unittest aaltonomous/data/test_to_lerobot.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import lerobot  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs env lerobot")
class TestToLeRobot(unittest.TestCase):
    def test_roundtrip(self):
        from aaltonomous.data import record_demos as R, to_lerobot as T
        from aaltonomous.data.test_record import FakeCams, FakeDDS, run_ticks
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as tmp:
            rec = R.Recorder(FakeDDS(), FakeCams(), tmp, session="s", task="sort_language", log=lambda *_: None)
            for label in "yfy":
                rec.handle_key("n")
                run_ticks(rec, 35)
                rec.handle_key("s")
                rec.handle_key(label)
            root = Path(tmp) / "ds"
            T.convert([Path(tmp) / "s"], "local/test", root)                  # failures excluded by default
            ds = LeRobotDataset("local/test", root=root)
            self.assertEqual((ds.num_episodes, ds.num_frames), (2, 70))
            raw = np.load(Path(tmp) / "s" / "ep000_sort_language_s1000" / "lowdim.npz")
            x = ds[3]
            np.testing.assert_allclose(x["action"].numpy(), np.r_[raw["action_arm"][3], raw["action_hand"][3]], atol=1e-6)
            np.testing.assert_allclose(x["observation.state"].numpy(), np.r_[raw["arm_q"][3], raw["hand_q"][3]], atol=1e-6)
            import cv2
            dep = cv2.imread(str(Path(tmp) / "s/ep000_sort_language_s1000/head_depth/000003.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(x["observation.images.head_depth"].numpy()[0], dep)   # mm, bit exact
            self.assertEqual(x["observation.images.head_rgb"].shape, (3, 480, 640))
            meta = json.loads((Path(tmp) / "s/ep000_sort_language_s1000/meta.json").read_text())
            self.assertEqual(x["task"], meta["instruction"])
            side = [json.loads(line) for line in (root / "meta" / "aalto_episodes.jsonl").read_text().splitlines()]
            self.assertEqual([s["card_seed"] for s in side], [1000, 1002])
            self.assertTrue(all(s["success"] for s in side))


if __name__ == "__main__":
    unittest.main()
