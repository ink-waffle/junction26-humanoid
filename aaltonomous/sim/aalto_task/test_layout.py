"""Offline tests for the Isaac task's pure side: python -m unittest aaltonomous/sim/aalto_task/test_layout.py
(no Isaac needed)."""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from aaltonomous.sim.aalto_task import layout as L


class TestLayout(unittest.TestCase):
    def setUp(self):
        self.sc = Scene.load()

    def test_names(self):
        n = L.object_names(self.sc)
        self.assertEqual(len(n), 18)
        self.assertEqual(len(set(n)), 18)
        self.assertEqual(L.parse_name("obj_red_triangle_1"), ("red", "triangle", 1))

    def test_assign_random_layouts(self):
        for seed in range(30):
            lay = L.random_layout(self.sc, seed)
            poses, active = L.assign(self.sc, lay)
            self.assertEqual(len(poses), 18)
            self.assertEqual(len(active), len(lay))
            for ob, n in zip(lay, active):
                self.assertEqual(L.parse_name(n)[:2], (ob["color"], ob["shape"]))
                p, q = poses[n]
                self.assertAlmostEqual(np.linalg.norm(q), 1.0)
                # resting on the mat, inside the counter
                self.assertAlmostEqual(p[2], self.sc.counter_z + self.sc.object_height() / 2 + 0.001)
                self.assertTrue(self.sc.on_counter(self.sc.pelvis_to_mat(p)))
            for n in set(poses) - set(active):  # parked: off the counter = never scored, behind the robot
                p, _ = poses[n]
                self.assertLess(p[0], -0.3)
                self.assertFalse(self.sc.on_counter(self.sc.pelvis_to_mat(p)))

    def test_parked_spacing(self):
        poses, _ = L.assign(self.sc, [])
        ys = sorted(p[1] for p, _ in poses.values())
        self.assertGreater(min(np.diff(ys)), self.sc.object_dims("triangle")["side"])

    def test_too_many_of_a_type(self):
        lay = [{"color": "red", "shape": "cube", "x": 0.3 + 0.1 * i, "y": 0.15, "yaw": 0.0} for i in range(3)]
        with self.assertRaises(ValueError):
            L.assign(self.sc, lay)

    def test_yaw_mapping_triangle_apex(self):
        # apex direction of the card (mat) must equal the apex direction of the sim object (pelvis)
        for yaw in (-0.7, 0.0, 0.3, 1.2):
            apex_mat = np.array([-np.sin(yaw), np.cos(yaw)])              # mesh +y rotated by yaw, mat axes
            apex_pel = apex_mat[::-1]                                     # mat (x, y) -> pelvis (y, x)
            t = L.mat_yaw_to_pelvis(yaw)
            np.testing.assert_allclose([-np.sin(t), np.cos(t)], apex_pel, atol=1e-12)

    def test_evaluate(self):
        sc = self.sc
        lo, hi = sc.bin_inner_box("left")
        in_left = sc.mat_to_pelvis((lo + hi) / 2 * np.array([1, 1, 0]) + [0, 0, 0.03])
        on_counter = sc.mat_to_pelvis([0.45, 0.15, sc.object_height() / 2])
        objs = [
            {"name": "a", "color": "red", "shape": "cube", "active": True, "pos": in_left, "speed": 0.0},
            {"name": "b", "color": "blue", "shape": "cube", "active": True, "pos": in_left, "speed": 0.0},
            {"name": "c", "color": "yellow", "shape": "cube", "active": True, "pos": on_counter, "speed": 0.0},
            {"name": "d", "color": "yellow", "shape": "cube", "active": False, "pos": [-0.6, 0, -0.7], "speed": 0.0},
        ]
        r = L.evaluate(sc, "sort_fixed", objs)
        self.assertEqual([o["outcome"] for o in r["outcomes"]], ["correct", "wrong", "counter"])
        self.assertEqual(r["score"], 0)
        self.assertFalse(r["all_correct"])
        # rung 2, (colour, shape) key sent as "colour/shape"
        r = L.evaluate(sc, "sort_language", objs[:2], {"blue/cube": "left", "default": "left"})
        self.assertTrue(r["all_correct"])
        self.assertEqual(r["score"], 4)


if __name__ == "__main__":
    unittest.main()
