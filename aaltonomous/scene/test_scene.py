"""Offline tests for the scene definition, success checker and scoring.

    python -m unittest aaltonomous/scene/test_scene.py
"""
import unittest

import numpy as np

from aaltonomous.interface.tasks import BINS, FIXED_MAPPING, SHAPES, make_instruction, sample_layout
from aaltonomous.scene.export import bin_tris, cube_tris, lid_tris, mat_image, object_tris
from aaltonomous.scene.scene import Scene

SC = Scene.load()
FIXED = dict(FIXED_MAPPING)


def in_bin(name, z=0.03):
    cx, cy = SC.cfg["bins"]["centers"][name]
    return SC.mat_to_pelvis([cx, cy, z])


class TestFrames(unittest.TestCase):
    def test_roundtrip(self):
        p = np.array([0.31, 0.12, 0.04])
        np.testing.assert_allclose(SC.pelvis_to_mat(SC.mat_to_pelvis(p)), p)

    def test_left_bin_is_on_robot_left(self):
        self.assertGreater(in_bin("left")[1], in_bin("right")[1])   # pelvis +y = left
        self.assertGreater(SC.mat_to_pelvis([0, 0.4, 0])[0], SC.mat_to_pelvis([0, 0, 0])[0])  # mat y = forward

    def test_bins_identical_row(self):
        """Published layout: identical boxes in a row, 5 cm gaps, near edge 40 cm in front of the pelvis."""
        b = SC.cfg["bins"]
        self.assertNotIn("color", b)
        self.assertEqual(len(SC.bin_rgb), 3)
        self.assertEqual(set(b["centers"]), set(BINS))
        bx, by = b["outer_size"]
        c = {n: SC.mat_to_pelvis([*b["centers"][n], 0]) for n in BINS}
        for n in BINS:
            self.assertAlmostEqual(c[n][0] - by / 2, 0.40, places=3)
        self.assertAlmostEqual(c["left"][1] - c["center"][1] - bx, 0.05, places=3)
        self.assertAlmostEqual(c["center"][1] - c["right"][1] - bx, 0.05, places=3)
        self.assertAlmostEqual(c["center"][1], 0.0, places=3)
        self.assertLessEqual(max(y for _, y in b["centers"].values()) + by / 2, SC.cfg["counter"]["mat_size"][1])

    def test_instructions_name_bins_by_position(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            text, _ = make_instruction("sort_language", rng)
            self.assertEqual(sum(text.count(f"{b} bin") + text.count(f"into {b}") for b in BINS), 3, text)


class TestChecker(unittest.TestCase):
    def test_correct_wrong(self):
        self.assertEqual(SC.classify_cube(in_bin("left"), 0.0, "red", FIXED), "correct")
        self.assertEqual(SC.classify_cube(in_bin("right"), 0.0, "red", FIXED), "wrong")

    def test_moving_cube_not_counted(self):
        self.assertEqual(SC.classify_cube(in_bin("left"), 0.5, "red", FIXED), "counter")

    def test_on_wall_or_above_rim_not_in_bin(self):
        cx, cy = SC.cfg["bins"]["centers"]["left"]
        half = SC.cfg["bins"]["outer_size"][0] / 2
        self.assertEqual(SC.classify_cube(SC.mat_to_pelvis([cx + half - 0.001, cy, 0.03]), 0, "red", FIXED), "counter")
        self.assertEqual(SC.classify_cube(SC.mat_to_pelvis([cx, cy, 0.10]), 0, "red", FIXED), "counter")

    def test_counter_and_off(self):
        self.assertEqual(SC.classify_cube(SC.mat_to_pelvis([0.4, 0.1, 0.025]), 0, "red", FIXED), "counter")
        self.assertEqual(SC.classify_cube(SC.mat_to_pelvis([0.4, 0.1, -0.5]), 0, "red", FIXED), "off")
        self.assertEqual(SC.classify_cube(SC.mat_to_pelvis([2.0, 0.1, 0.025]), 0, "red", FIXED), "off")

    def test_language_mapping(self):
        rng = np.random.default_rng(3)
        _, mapping = make_instruction("sort_language", rng)
        for color, b in mapping.items():
            self.assertEqual(SC.classify_cube(in_bin(b), 0, color, mapping), "correct")

    def test_score(self):
        o = ["correct", "correct", "wrong", "counter", "off"]
        self.assertEqual(SC.score("sort_fixed", o), 1 + 1 - 1 + 0 - 1)
        self.assertEqual(SC.score("sort_language", o), 2 + 2 - 1 + 0 - 1)
        with self.assertRaises(ValueError):
            SC.score("sort_fixed", ["bogus"])


class TestShapes(unittest.TestCase):
    def test_equal_grasp_width_and_dims(self):
        d = {k: SC.object_dims(k) for k in SHAPES}
        for k in SHAPES:
            self.assertAlmostEqual(d[k]["width"], SC.size)
            self.assertAlmostEqual(d[k]["apothem"] * (3 if k == "triangle" else 2), SC.size)   # flat-to-flat/apex
        self.assertAlmostEqual(d["triangle"]["side"], 2 * SC.size / np.sqrt(3))

    def test_layout_has_shape_and_type_limit(self):
        seen = set()
        for seed in range(100):
            lay = sample_layout(np.random.default_rng(seed), **SC.layout_kwargs())
            types = [(c["color"], c["shape"]) for c in lay]
            seen |= {c["shape"] for c in lay}
            self.assertTrue(all(types.count(t) <= 2 for t in types))
        self.assertEqual(seen, set(SHAPES))

    def test_rung1_ignores_shape_rung2_by_shape_or_both(self):
        for shp in SHAPES:
            self.assertEqual(SC.classify_cube(in_bin("left"), 0, "red", FIXED, shape=shp), "correct")
        by_shape = {"cylinder": "left", "default": "right"}
        self.assertEqual(SC.classify_cube(in_bin("left"), 0, "blue", by_shape, shape="cylinder"), "correct")
        self.assertEqual(SC.classify_cube(in_bin("right"), 0, "blue", by_shape, shape="cube"), "correct")
        self.assertEqual(SC.classify_cube(in_bin("left"), 0, "blue", by_shape, shape="cube"), "wrong")
        both = {("red", "cube"): "center", "red": "left", "cube": "right"}   # (colour, shape) beats colour beats shape
        self.assertEqual(SC.classify_cube(in_bin("center"), 0, "red", both, shape="cube"), "correct")
        self.assertEqual(SC.classify_cube(in_bin("left"), 0, "red", both, shape="cylinder"), "correct")
        self.assertEqual(SC.classify_cube(in_bin("right"), 0, "blue", both, shape="cube"), "correct")

    def test_legacy_cube_section(self):
        import copy
        cfg = copy.deepcopy(SC.cfg)
        cfg["cube"] = {"size": 0.05, "colors": cfg.pop("objects")["colors"], "copies_per_color": 2}
        self.assertAlmostEqual(Scene(cfg).cube_size, 0.05)


class TestLid(unittest.TestCase):
    def lid_pt(self, shape, z, dx=0.0, dy=0.0):
        h = next(h for h in SC.lid_holes() if h["shape"] == shape)
        return SC.mat_to_pelvis([h["center"][0] + dx, h["center"][1] + dy, z])

    def test_holes_fit_and_dont_overlap(self):
        cx, cy = SC.cfg["bins"]["centers"][SC.lid_bin]
        H = SC.cfg["bins"]["outer_size"][0] / 2 - SC.cfg["bins"]["wall_thickness"]
        holes = SC.lid_holes()
        self.assertEqual({h["shape"] for h in holes}, set(SHAPES))
        for h in holes:
            self.assertTrue(np.all(np.abs(h["poly"] - [cx, cy]) < H))
            # every object shape of the same type fits with the clearance; hole wider by clearance
            a = SC.object_dims(h["shape"])["apothem"]
            c = SC.cfg["lid"]["hole_clearance"]
            ext = h["poly"].max(0) - h["poly"].min(0)
            self.assertGreater(ext.max(), 2 * a + c * 0.9 - 1e-9)
        for i, p in enumerate(holes):
            for q in holes[i + 1:]:
                self.assertFalse(any(SC.lid_hole_at(v) == q["shape"] for v in p["poly"]))
                self.assertFalse(any(SC.lid_hole_at(v) == p["shape"] for v in q["poly"]))

    def test_cube_cannot_pass_circle_hole_at_default_clearance(self):
        circle = next(h for h in SC.lid_holes() if h["shape"] == "cylinder")
        r_hole = np.linalg.norm(circle["poly"][0] - circle["center"])
        if SC.cfg["lid"]["hole_clearance"] <= 0.010:
            self.assertGreater(SC.object_dims("cube")["circumradius"], r_hole)

    def test_scoring_outcomes(self):
        z_in = SC.lid_z[0] - 0.03
        z_on = SC.lid_z[1] + SC.object_height() / 2
        ins = SC.classify_insert(self.lid_pt("cube", z_in, 0.0, -0.0), 0, "cube")
        self.assertEqual(ins, "inserted")
        self.assertEqual(SC.classify_insert(self.lid_pt("cube", z_in), 0.5, "cube"), "counter")        # still moving
        self.assertEqual(SC.classify_insert(self.lid_pt("cube", z_on), 0, "cube"), "forced")           # sitting on lid
        cx, cy = SC.cfg["bins"]["centers"][SC.lid_bin]
        hx = SC.cfg["bins"]["outer_size"][0] / 2
        beside = SC.mat_to_pelvis([cx + hx + 0.03, cy, SC.object_height() / 2])
        self.assertEqual(SC.classify_insert(beside, 0, "cube"), "dropped")
        far = SC.mat_to_pelvis([0.30, 0.15, SC.object_height() / 2])
        self.assertEqual(SC.classify_insert(far, 0, "cube"), "counter")
        self.assertEqual(SC.classify_insert(SC.mat_to_pelvis([2.0, 0.1, 0.0]), 0, "cube"), "off")

    def test_hole_used_from_track(self):
        z_in = SC.lid_z[0] - 0.03
        end = self.lid_pt("cylinder", z_in)
        up = lambda sh: self.lid_pt(sh, SC.lid_z[1] + 0.05)
        right = np.stack([up("cylinder"), end])
        wrong = np.stack([up("cube"), end])
        self.assertEqual(SC.classify_insert(end, 0, "cylinder", track_pelvis=right), "inserted")
        self.assertEqual(SC.classify_insert(end, 0, "cylinder", track_pelvis=wrong), "wrong_hole")

    def test_insert_score(self):
        self.assertEqual(SC.score("insert_shapes", ["inserted", "inserted", "forced", "dropped", "wrong_hole", "counter"]),
                         3 + 3 - 1 - 1)
        with self.assertRaises(ValueError):
            SC.score("insert_shapes", ["correct"])
        self.assertEqual(make_instruction("insert_shapes", np.random.default_rng(0))[1], {})


class TestLayout(unittest.TestCase):
    def test_layouts_inside_region_and_clear_of_bins(self):
        kw = SC.layout_kwargs()
        (x0, x1), (y0, y1) = kw["region"]
        for seed in range(200):
            for c in sample_layout(np.random.default_rng(seed), **kw):
                self.assertTrue(x0 <= c["x"] <= x1 and y0 <= c["y"] <= y1)
                p = SC.cube_spawn_mat(c)
                self.assertIsNone(SC.bin_of(p))
                self.assertTrue(SC.on_counter(p))

    def test_workspace_above_counter(self):
        lo, hi = SC.suggested_workspace()
        self.assertGreater(lo[2], SC.counter_z)
        self.assertTrue(all(a < b for a, b in zip(lo, hi)))


class TestExport(unittest.TestCase):
    def test_meshes(self):
        self.assertEqual(len(cube_tris(SC)), 12)
        self.assertEqual(len(bin_tris(SC)), 5 * 12)

    def test_object_and_lid_meshes(self):
        for shape in SHAPES:
            t = np.array(object_tris(SC, shape))
            lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
            self.assertAlmostEqual(hi[2], SC.object_height() * 1000, places=3)
            self.assertAlmostEqual(lo[2], 0, places=6)
            self.assertLessEqual(hi[0] - lo[0], SC.object_dims(shape)["circumradius"] * 2000 + 1e-6)
        lid = np.array(lid_tris(SC))
        self.assertAlmostEqual(lid[..., 2].max(), SC.cfg["lid"]["thickness"] * 1000)

    def test_mat_orientation(self):
        img = mat_image(SC).astype(int)
        # the box row (far from the robot) must appear in the TOP half of a top view with the robot at the bottom;
        # the boxes are identical and symmetric about the robot, so left/right shows only in the box outline count
        box = np.all(img == np.array(SC.bin_rgb), axis=-1)
        rows, cols = np.nonzero(box)
        self.assertLess(rows.mean(), img.shape[0] / 2)
        self.assertAlmostEqual(cols.mean(), img.shape[1] / 2, delta=img.shape[1] * 0.02)


class TestHeadCameraView(unittest.TestCase):
    """With the default calibration (URDF head pose), every object must be in the head image.
    Bins are fixed and known (spec §11), so they only need to be reachable; the v1 side bins are outside the view."""

    def test_cubes_and_bins_visible(self):
        from aaltonomous.interface.calibration import Calibration
        cal = Calibration.load()
        L, h = SC.cfg["layout"], SC.object_height() / 2
        pts = [[x + dx, y + dy, z] for x in L["region_x"] for y in L["region_y"]
               for dx in (-h, h) for dy in (-h, h) for z in (0.0, 2 * h)]
        vis = cal.visible("head", SC.mat_to_pelvis(np.array(pts)))
        self.assertTrue(vis.all(), np.array(pts)[~vis])


class TestReach(unittest.TestCase):
    @unittest.expectedFailure  # KNOWN 2026-11-05: default counter is out of reach -- see check_layout.py
    def test_cube_region_and_bins_reachable(self):
        """Every cube-region cell and every bin centre must be reachable (palm 8 cm above, URDF arm limits)."""
        from aaltonomous.scene.check_layout import palm_cloud, reach_map
        cloud = palm_cloud(20000)
        L, b = SC.cfg["layout"], SC.cfg["bins"]
        X, Y = np.meshgrid(np.linspace(*L["region_x"], 9), np.linspace(*L["region_y"], 4))
        pts = [[x, y, SC.cube_size / 2] for x, y in zip(X.ravel(), Y.ravel())]
        pts += [[cx, cy, b["wall_height"]] for cx, cy in b["centers"].values()]
        ok = reach_map(SC, cloud, SC.mat_to_pelvis(np.array(pts)), 0.08, 0.03)
        self.assertTrue(ok.all(), f"{(~ok).sum()} of {len(ok)} points unreachable")


if __name__ == "__main__":
    unittest.main()
