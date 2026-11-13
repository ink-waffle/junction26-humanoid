"""Offline tests for record_demos.py (no DDS, no cameras): joint mapping, read-only DDS, episode files, crash safety.

    python -m unittest aaltonomous/data/test_record.py
"""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.data import record_demos as R


def motors(n, base, kp=80.0):
    return [types.SimpleNamespace(q=base + 0.01 * i, dq=-(base + 0.01 * i), kp=kp) for i in range(n)]


def fake_latest(now, age=0.0, arm_kp=80.0):
    t = now - age
    return {"rt/lowcmd": (types.SimpleNamespace(motor_cmd=motors(35, 1.0, arm_kp)), t),
            "rt/lowstate": (types.SimpleNamespace(motor_state=motors(35, 2.0)), t),
            "rt/dex3/left/cmd": (types.SimpleNamespace(motor_cmd=motors(7, 3.0, 1.5)), t),
            "rt/dex3/right/cmd": (types.SimpleNamespace(motor_cmd=motors(7, 4.0, 1.5)), t),
            "rt/dex3/left/state": (types.SimpleNamespace(motor_state=motors(7, 5.0)), t),
            "rt/dex3/right/state": (types.SimpleNamespace(motor_state=motors(7, 6.0)), t)}


class FakeDDS:
    def __init__(self, age=0.0):
        self.age, self.n = age, 0

    def sample(self):
        self.n += 1
        now = time.monotonic()
        return R.extract(fake_latest(now, self.age), now)

    def counts(self):
        return {t: self.n for t in R.TOPICS}


class FakeCams:
    """Like CameraTap with the head RGB-D server running. depth_age=None: recording without depth (--no-head-rgbd);
    stale_after=k: the RGB-D server stops after k frames (age grows, depth falls back to zeros)."""

    def __init__(self, depth_age=0.0, stale_after=None):
        self.k, self.age, self.stale_after = 0, depth_age, stale_after

    def depth_age(self):
        return self.age

    def images(self):
        self.k += 1
        if self.stale_after is not None and self.k > self.stale_after:
            self.age = 1.0
        img = np.full((480, 640, 3), self.k % 255, np.uint8)
        depth = np.full((480, 640), 500 + self.k, np.uint16)
        if self.age is not None and self.age > R.STALE_S:
            depth[:] = 0
        out = {"head_rgb": img, "left_wrist_rgb": img, "right_wrist_rgb": img, "head_depth": depth,
               "head_frame_id": self.k, "depth_valid": float(depth.any())}
        if self.age is not None:
            out["age_head_rgbd"] = self.age
        return out

    def close(self):
        pass


def run_ticks(rec, n):
    for _ in range(n):
        time.sleep(rec.tick())


class TestExtract(unittest.TestCase):
    def test_joint_order(self):
        row = R.extract(fake_latest(10.0, age=0.05), 10.0)
        # arm = rt/lowcmd motors 15..28 in joint_spec order; hands = own DDS order, left then right
        np.testing.assert_allclose(row["action_arm"], 1.0 + 0.01 * np.arange(15, 29))
        np.testing.assert_allclose(row["arm_q"], 2.0 + 0.01 * np.arange(15, 29))
        np.testing.assert_allclose(row["arm_dq"], -(2.0 + 0.01 * np.arange(15, 29)))
        np.testing.assert_allclose(row["action_hand"], np.r_[3.0 + 0.01 * np.arange(7), 4.0 + 0.01 * np.arange(7)])
        np.testing.assert_allclose(row["hand_q"], np.r_[5.0 + 0.01 * np.arange(7), 6.0 + 0.01 * np.arange(7)])
        self.assertEqual(row["body_q"].shape, (29,))
        self.assertAlmostEqual(row["age_lowcmd"], 0.05)
        self.assertEqual(R.ARM_IDX, list(range(15, 29)))

    def test_missing_stream_is_nan(self):
        row = R.extract({}, 0.0)
        self.assertTrue(np.isnan(row["action_arm"]).all() and np.isinf(row["age_lowcmd"]))


class TestResample(unittest.TestCase):
    def test_no_aliasing(self):
        # commander updates a 1 rad/s ramp at 30 Hz, republished at 250 Hz; ticks at 30 Hz with a phase offset
        tp = np.arange(0, 3, 1 / 250)
        log = [(t, [np.floor(t * 30) / 30] * 14) for t in tp]
        ticks = 0.5 + np.arange(60) / 30 + 0.0005
        latest = np.array([log[np.searchsorted(tp, t, side="right") - 1][1][0] for t in ticks])
        box = R.box_resample(log, ticks)[:, 0]
        # latest-message sampling gives 2x; the box average stays within the 4 ms publication jitter (~5 %)
        self.assertGreater(np.abs(np.diff(latest)).max() * 30, 1.9)
        self.assertLessEqual(np.abs(np.diff(box)).max() * 30, 1.06)
        self.assertLess(np.abs(box - ticks).max(), 0.04)
        self.assertTrue(np.isnan(R.box_resample(log, np.array([0.0]))).all())
        np.testing.assert_allclose(R.box_resample(log, np.array([10.0]))[0, 0], log[-1][1][0])  # hold


class TestReadOnly(unittest.TestCase):
    def test_dds_tap_never_publishes(self):
        subs = []

        class Sub:
            def __init__(self, topic, typ):
                self.topic = topic
                subs.append(self)

            def Init(self, handler, queue_len):
                self.handler = handler

        def no_pub(*a, **k):
            raise AssertionError("recorder created a DDS publisher")

        ch = types.ModuleType("unitree_sdk2py.core.channel")
        ch.ChannelFactoryInitialize = lambda *a, **k: None
        ch.ChannelSubscriber, ch.ChannelPublisher = Sub, no_pub
        idl = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        idl.HandCmd_ = idl.HandState_ = idl.LowCmd_ = idl.LowState_ = object
        saved = {k: sys.modules.get(k) for k in (ch.__name__, idl.__name__)}
        sys.modules.update({ch.__name__: ch, idl.__name__: idl})
        try:
            tap = R.DDSTap(1)
        finally:
            for k, v in saved.items():
                sys.modules.pop(k) if v is None else sys.modules.__setitem__(k, v)
        self.assertEqual(sorted(s.topic for s in subs), sorted(R.TOPICS))
        for s in subs:  # feed one message per topic through the handler
            s.handler(fake_latest(0.0)[s.topic][0])
        np.testing.assert_allclose(tap.sample()["action_arm"], 1.0 + 0.01 * np.arange(15, 29))
        # and no publisher symbol is even referenced in the module
        self.assertNotIn("ChannelPublisher(", Path(R.__file__).read_text())


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, dds=None, cams=None, **kw):
        return R.Recorder(dds or FakeDDS(), cams or FakeCams(), self.tmp.name, session="s", log=self.logs.append, **kw)

    def record(self, rec, n, label="y"):
        rec.handle_key("n")
        run_ticks(rec, n)
        rec.handle_key("s")
        rec.handle_key(label)
        return rec.saved[-1]

    def test_episode_saved_and_discard_and_partial(self):
        rec = self.rec(task="sort_language", card_seed=1000)
        rec.handle_key("n")
        self.assertEqual(rec.state, "recording")
        run_ticks(rec, 15)
        rec.handle_key("s")
        rec.handle_key("y")
        ep = Path(self.tmp.name) / "s" / "ep000_sort_language_s1000"
        self.assertTrue(ep.is_dir(), list((Path(self.tmp.name) / "s").iterdir()))
        meta = json.loads((ep / "meta.json").read_text())
        self.assertEqual((meta["success"], meta["n_frames"], meta["card_seed"], meta["task"]),
                         (True, 15, 1000, "sort_language"))
        self.assertTrue(meta["instruction"] and meta["layout"] and meta["layout_id"] == "sort_language/1000")
        d = np.load(ep / "lowdim.npz")
        self.assertEqual(d["action_arm"].shape, (15, 14))
        self.assertEqual(d["action_hand"].shape, (15, 14))
        self.assertTrue(np.all(np.diff(d["t"]) > 0.02))   # paced at ~30 Hz
        for k in R.CAMS:
            self.assertEqual(len(list((ep / k).glob("*.jpg"))), 15)
        import cv2
        dep = cv2.imread(str(ep / "head_depth/000000.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual((dep.dtype, dep.shape), (np.uint16, (480, 640)))
        self.assertIn("filter_check", meta)
        # next card, then discard while recording
        self.assertEqual(rec.card["seed"], 1001)
        rec.handle_key("n")
        run_ticks(rec, 3)
        rec.handle_key("d")
        self.assertEqual(len(list((Path(self.tmp.name) / "s").iterdir())), 1)
        # crash mid-episode: left as .partial, earlier episode untouched, a new run continues the numbering
        rec.handle_key("n")
        run_ticks(rec, 3)
        rec.close()
        names = sorted(p.name for p in (Path(self.tmp.name) / "s").iterdir())
        self.assertEqual(names, ["ep000_sort_language_s1000", "ep001_sort_language_s1001.partial"])
        self.assertEqual(self.rec().ep, 1)

    def test_refuses_without_teleop(self):
        rec = self.rec(dds=FakeDDS(age=1.0))
        rec.handle_key("n")
        self.assertEqual(rec.state, "idle")
        self.assertTrue(any("not starting" in m for m in self.logs))

    def test_depth_recorded_and_marked(self):
        ep = self.record(self.rec(), 10)
        meta = json.loads((ep / "meta.json").read_text())
        self.assertIs(meta["head_depth"], True)
        self.assertEqual(meta["depth"], {"stale_ticks": 0, "blank_frames": 0, "valid_frac_mean": 1.0})
        d = np.load(ep / "lowdim.npz")
        self.assertEqual(d["age_head_rgbd"].shape, (10,))
        self.assertTrue((d["depth_valid"] == 1.0).all())
        import cv2
        self.assertTrue(cv2.imread(str(ep / "head_depth/000009.png"), cv2.IMREAD_UNCHANGED).all())

    def test_refuses_without_depth(self):
        for age in (float("inf"), 1.0):          # server never seen / stopped a second ago
            self.logs.clear()
            rec = self.rec(cams=FakeCams(depth_age=age))
            rec.handle_key("n")
            self.assertEqual(rec.state, "idle")
            self.assertTrue(any("no head depth" in m for m in self.logs), self.logs)

    def test_depth_lost_mid_episode_is_flagged(self):
        ep = self.record(self.rec(cams=FakeCams(stale_after=4)), 10)
        meta = json.loads((ep / "meta.json").read_text())
        self.assertEqual((meta["depth"]["stale_ticks"], meta["depth"]["blank_frames"]), (6, 6))
        self.assertTrue(any("stale head depth" in w for w in meta["warnings"]))
        self.assertTrue(any("all-zero head depth" in w for w in meta["warnings"]))

    def test_no_head_rgbd_is_marked_and_skipped_by_converter(self):
        from aaltonomous.data.to_lerobot import find_episodes
        ep = self.record(self.rec(cams=FakeCams(depth_age=None)), 5)
        meta = json.loads((ep / "meta.json").read_text())
        self.assertIs(meta["head_depth"], False)
        self.assertEqual(meta["depth"], {})
        self.assertEqual(find_episodes([self.tmp.name], False), [])
        self.assertEqual(len(find_episodes([self.tmp.name], False, allow_no_depth=True)), 1)

    def test_failure_label_and_card_nav(self):
        rec = self.rec(card_seed=5)
        rec.handle_key("]")
        rec.handle_key("]")
        rec.handle_key("[")
        self.assertEqual(rec.card["seed"], 6)
        rec.handle_key("n")
        run_ticks(rec, 2)
        rec.handle_key("s")
        rec.handle_key("x")          # ignored while labelling
        rec.handle_key("f")
        meta = json.loads((Path(self.tmp.name) / "s" / "ep000_sort_fixed_s6" / "meta.json").read_text())
        self.assertIs(meta["success"], False)


if __name__ == "__main__":
    unittest.main()
