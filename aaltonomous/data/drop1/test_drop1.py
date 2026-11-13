"""Offline tests for the Drop 1 capture kit (mock backend: no DDS, no cameras, no sim).

    python -m unittest aaltonomous/data/drop1/test_drop1.py
"""
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.data.drop1 import capture as C
from aaltonomous.data.drop1 import package as P
from aaltonomous.data.drop1 import stats as S


class TestDrop1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src = C.MockSource(seed=1, rgb_sigma=2.0, depth_sigma_mm=3.0, hole_p=0.02, obj_hole_p=0.25)
        cls.sdir = C.run(src, cls.tmp.name, "t", yes=True, only=["empty_on", "phone_hdri", "obj_red_a", "layout_1000"],
                         frames_scale=0.34, fps=1000, log=lambda *a: None)
        cls.st = S.analyse(cls.sdir, plots=True, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checklist_objects_in_head_view(self):
        from aaltonomous.interface.calibration import Calibration
        from aaltonomous.scene.scene import Scene
        cal, sc = Calibration.load(), Scene.load()
        for st in C.load_checklist()["steps"]:
            for o in C.step_objects(st, sc):
                p = sc.mat_to_pelvis(np.array([o["x"], o["y"], sc.object_height(o["shape"]) / 2]))
                self.assertTrue(cal.visible("head", p), (st["id"], o))

    def test_files(self):
        d = self.sdir / "empty_on"
        meta = json.loads((d / "meta.json").read_text())
        self.assertEqual(meta["n_frames"], 20)
        self.assertEqual(len(list((d / "head_depth").glob("*.png"))), 20)
        z = np.load(d / "frames.npz")
        self.assertEqual(z["arm_q"].shape, (20, 14))
        self.assertTrue(np.all(np.diff(z["head_frame_id"]) > 0))      # every frame is a new frame
        dep = S.load_frames(d, "head_depth")
        self.assertEqual(dep.dtype, np.uint16)
        sess = json.loads((self.sdir / "session.json").read_text())
        self.assertIn("phone_hdri", [e["step"] for e in sess["log"]])
        self.assertFalse((self.sdir / "phone_hdri").exists())

    def test_noise_and_holes_recovered(self):
        e = self.st["steps"]["empty_on"]
        self.assertAlmostEqual(e["head_depth"]["std_mm_median"], 3.0, delta=0.5)
        self.assertAlmostEqual(e["head_depth"]["hole_frac_mean"], 0.02, delta=0.01)
        self.assertAlmostEqual(e["cams"]["head_rgb"]["noise_std_median"], 2.0, delta=0.4)
        objs = [o for o in self.st["objects"] if o["step"] == "obj_red_a"]
        self.assertEqual(len(objs), 3)
        for o in objs:
            self.assertEqual(o["mask_source"], "colour")
            self.assertAlmostEqual(o["hole_frac"], 0.25, delta=0.06)
        self.assertTrue((self.sdir / "stats/depth_maps_empty_on.png").exists())

    def test_package_and_size_check(self):
        out = Path(self.tmp.name) / "pkg"
        p = P.package(self.sdir, out, day="2026-11-14", log=lambda *a: None)
        with tarfile.open(p) as tf:
            names = tf.getnames()
            self.assertIn("drop1_2026-11-14/README.md", names)
            self.assertIn("drop1_2026-11-14/steps/empty_on/head_depth/000000.png", names)
            man = tf.extractfile("drop1_2026-11-14/MANIFEST.sha256").read().decode()
            readme = tf.extractfile("drop1_2026-11-14/README.md").read().decode()
        self.assertIn("TEST DATA", readme)                 # non-real sessions are labelled
        self.assertEqual(len(man.strip().splitlines()), len(names) - 2)
        with self.assertRaises(SystemExit):
            P.package(self.sdir, out, max_gb=1e-6, log=lambda *a: None)
        small = P.collect(self.sdir, max_frames=2, no_wrist=True)
        self.assertFalse(any("wrist" in a for a, _ in small))

    def test_read_only(self):
        src = Path(C.__file__).read_text()
        for bad in ("ChannelPublisher", "Write(", "ctrl_dual_arm", "make_backend"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
