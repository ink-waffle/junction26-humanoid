"""Bad-command tests for the safety filter (WP2 'done when', sim/offline part).
Run: python -m unittest aaltonomous/interface/test_safety.py"""
import unittest

import numpy as np

from aaltonomous.interface.safety import N, JointSpec, SafetyConfig, SafetyFilter

SPEC = JointSpec()
Z = np.zeros(N)


def act(arm=Z, hand=Z):
    return {"arm_q_target": arm, "hand_q_target": hand}


class T(unittest.TestCase):
    def mk(self, **kw):
        return SafetyFilter(SPEC, SafetyConfig(**kw))

    def test_spec(self):
        self.assertTrue((SPEC.arm_lower < SPEC.arm_upper).all() and (SPEC.hand_lower < SPEC.hand_upper).all())
        self.assertEqual(SPEC.hand_names[3], "left_hand_middle_0_joint")
        self.assertEqual(SPEC.hand_names[10], "right_hand_index_0_joint")

    def test_home_passes(self):
        f = self.mk()
        r = f.filter(act(hand=f.home_hand), 0.033)
        self.assertFalse(r.clipped)
        self.assertEqual(r.mode, "ok")

    def test_invalid_holds(self):
        f = self.mk()
        for a in (act(np.full(N, np.nan)), act(np.zeros(7)), {"hand_q_target": Z}, "junk"):
            r = f.filter(a, 0.1)
            self.assertTrue(np.allclose(r.arm_q_cmd, 0))
            self.assertTrue(any(s.startswith("invalid") for s in r.reasons))

    def test_velocity_and_limits(self):
        f = self.mk(workspace=False, max_clipped_steps=10**6)
        prev, t = np.zeros(N), 0.0
        for _ in range(300):
            t += 1 / 30
            r = f.filter(act(np.full(N, 10.0), np.full(N, 10.0)), t)
            self.assertLessEqual(np.abs(r.arm_q_cmd - prev).max(), 1.0 / 30 + 1e-9)
            prev = r.arm_q_cmd
        self.assertTrue((r.arm_q_cmd <= SPEC.arm_upper - 0.0499).all())
        self.assertTrue((r.hand_q_cmd <= SPEC.hand_upper).all())
        # mirrored hands: right middle_0 lower limit is 0, so it must not go negative
        r = f.filter(act(hand=np.full(N, -10.0)), t + 1)
        self.assertGreaterEqual(r.hand_q_cmd[12], 0.0)

    def test_workspace_box(self):
        f = self.mk(max_clipped_steps=10**6)
        rng = np.random.default_rng(0)
        zmin = f.cfg.workspace_min[2]
        for _ in range(2000):  # find an in-limit target that puts a palm below z_min
            q = rng.uniform(f.arm_lo, f.arm_hi)
            if f.ws.violation(q) > 0.05:
                break
        t = 0.0
        for _ in range(400):
            t += 1 / 30
            r = f.filter(act(q), t)
            self.assertLessEqual(f.ws.violation(r.arm_q_cmd), 1e-3)
        self.assertIn("workspace", r.reasons)
        self.assertGreaterEqual(f.ws.positions(r.arm_q_cmd)[:, 2].min(), zmin - 1e-3)

    def test_recover_from_outside(self):
        f = self.mk(max_clipped_steps=10**6)
        out = np.zeros(N); out[0] = out[7] = 1.5  # shoulder pitch forward-down: palms low
        f.reset(out, Z, 0.0)
        v0 = f.ws.violation(out)
        self.assertGreater(v0, 0)
        r = f.filter(act(Z), 1 / 30)
        self.assertNotIn("workspace", r.reasons)
        self.assertLess(f.ws.violation(r.arm_q_cmd), v0)

    def test_watchdog(self):
        f = self.mk()
        f.filter(act(), 0.0)
        self.assertEqual(f.tick(0.1).mode, "ok")
        self.assertEqual(f.tick(0.25).mode, "stale")
        self.assertEqual(f.tick(1.1).mode, "safe_stop")
        self.assertEqual(f.filter(act(), 1.2).mode, "safe_stop")  # latched until reset

    def test_safe_stop_home(self):
        f = self.mk(safe_stop="home", workspace=False)
        start = np.full(N, 0.5)
        f.reset(start, Z, 0.0)
        r = f.tick(2.0)
        self.assertAlmostEqual(np.abs(r.arm_q_cmd - start).max(), 0.3 / 30, places=9)

    def test_abort(self):
        f = self.mk(max_clipped_steps=3, workspace=False)
        for i in range(5):
            r = f.filter(act(np.full(N, 0.5)), i / 30)
        self.assertEqual(r.mode, "aborted")
        held = r.arm_q_cmd
        self.assertTrue(np.allclose(f.filter(act(Z), 1.0).arm_q_cmd, held))
        self.assertIn("abort", f.summary()["counts"])

    def test_hand_lead(self):
        f = self.mk(hand_max_lead=0.3, hand_max_vel=100)
        r = f.filter(act(hand=np.full(N, 1.0)), 1 / 30, hand_q=Z)
        self.assertLessEqual(r.hand_q_cmd.max(), 0.3 + 1e-9)
        self.assertIn("hand_lead", r.reasons)


if __name__ == "__main__":
    unittest.main()


class TestCounterKeepout(unittest.TestCase):
    """v1 scene: wrist/palm frames may not enter the table volume (+2 cm above it)."""

    def test_zero_pose_ok_palm_into_table_violates(self):
        from aaltonomous.interface.safety import WorkspaceChecker
        spec = SPEC
        ws = WorkspaceChecker(spec, SafetyConfig())
        q = np.zeros(N)
        self.assertEqual(ws.violation(q), 0.0)
        q[3] = q[10] = 0.5  # elbows +0.5 rad from zero: palms ~1 cm below the table top, 0.21 m forward
        self.assertGreater(ws.violation(q), 0.0)
        off = WorkspaceChecker(spec, SafetyConfig(counter_keepout_min=None))
        self.assertEqual(off.violation(q), 0.0)
