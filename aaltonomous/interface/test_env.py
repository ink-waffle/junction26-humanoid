"""AaltoEnv tests on the mock backend (WP1 offline part).
Run: python -m unittest aaltonomous/interface/test_env.py"""
import time
import unittest

import numpy as np

from aaltonomous.interface.env import AaltoEnv
from aaltonomous.interface.safety import N, SafetyConfig
from aaltonomous.interface.tasks import BINS, COLORS, make_instruction, sample_layout

KEYS = {"head_rgb": ((480, 640, 3), np.uint8), "head_depth": ((480, 640), np.uint16),
        "left_wrist_rgb": ((480, 640, 3), np.uint8), "right_wrist_rgb": ((480, 640, 3), np.uint8),
        "arm_q": ((N,), np.float32), "arm_dq": ((N,), np.float32), "hand_q": ((N,), np.float32)}
Z = np.zeros(N)


class T(unittest.TestCase):
    def test_obs_keys(self):
        obs = AaltoEnv().reset()
        for k, (shape, dt) in KEYS.items():
            self.assertEqual((obs[k].shape, obs[k].dtype), (shape, dt), k)
        self.assertIsInstance(obs["instruction"], str)
        self.assertEqual(obs["t"], 0.0)

    def test_open_hand_is_not_a_clip(self):
        env = AaltoEnv()
        env.reset()
        for _ in range(60):
            _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
        self.assertEqual(info["clipped_steps"], 0)

    def test_scripted_reach_and_chunk(self):
        # Elbow sweep 0 -> 1 rad passes the palms through the table top -> no counter keep-out in this test.
        env = AaltoEnv(safety=SafetyConfig(counter_keepout_min=None))
        env.reset()
        goal = np.zeros(N); goal[3] = goal[10] = 1.0  # elbows
        traj = np.linspace(0, 1, 60)[:, None] * goal  # 2 s chunk, within 1 rad/s
        obs, info = env.step({"arm_q_target": traj, "hand_q_target": np.zeros((60, N))})
        self.assertEqual(info["ticks"], 60)
        self.assertAlmostEqual(info["t"], 2.0, places=6)
        self.assertEqual(info["clipped_steps"], 0)
        self.assertLess(np.abs(obs["arm_q"] - goal).max(), 0.05)

    def test_done_and_time_limit(self):
        env = AaltoEnv(time_limit=0.5)
        env.reset()
        _, info = env.step({"arm_q_target": Z, "hand_q_target": Z, "done": True})
        self.assertEqual(info["end_reason"], "done")
        with self.assertRaises(RuntimeError):
            env.step({"arm_q_target": Z, "hand_q_target": Z})
        env.reset()
        info = {}
        for _ in range(20):
            _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
            if info["episode_over"]:
                break
        self.assertEqual(info["end_reason"], "time_limit")

    def test_bad_policy_aborts(self):
        env = AaltoEnv()
        env.reset()
        for _ in range(100):
            _, info = env.step({"arm_q_target": np.full(N, 5.0), "hand_q_target": Z})
            if info["episode_over"]:
                break
        self.assertEqual(info["end_reason"], "aborted")
        self.assertIn("arm_position", info["safety_summary"]["counts"])

    def test_watchdog_via_hold(self):
        env = AaltoEnv()
        env.reset()
        env.step({"arm_q_target": Z, "hand_q_target": Z})
        modes = [env.hold() for _ in range(40)]  # 1.33 s with no action
        self.assertIn("stale", modes)
        self.assertEqual(modes[-1], "safe_stop")
        _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
        self.assertEqual(info["end_reason"], "safe_stop")

    def test_realtime_rate(self):
        env = AaltoEnv(realtime=True)
        env.reset()
        t0 = time.monotonic()
        env.step({"arm_q_target": np.zeros((15, N)), "hand_q_target": np.zeros((15, N))})
        self.assertAlmostEqual(time.monotonic() - t0, 0.5, delta=0.05)

    def test_sim_clock_pacing(self):
        """With a backend clock (sim), ticks and t follow SIM time: a sim at 0.5x real time -> 2x wall per tick."""
        class HalfSpeedClock:          # sim time = 0.5 * wall
            def __init__(self):
                self.w0 = time.monotonic()

            def now(self):
                return 100.0 + 0.5 * (time.monotonic() - self.w0)

            def wait_until(self, t):
                while self.now() < t:
                    time.sleep(0.001)
        env = AaltoEnv(realtime=True)
        env.clock = env.backend.clock = HalfSpeedClock()
        env.reset()
        t0 = time.monotonic()
        _, info = env.step({"arm_q_target": np.zeros((6, N)), "hand_q_target": np.zeros((6, N))})
        self.assertAlmostEqual(time.monotonic() - t0, 0.4, delta=0.05)   # 6 ticks = 0.2 s sim = 0.4 s wall
        self.assertAlmostEqual(info["t"], 0.2, delta=0.02)
        self.assertEqual(info["safety_mode"], "ok")

    def test_tasks(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            cubes = sample_layout(rng)
            self.assertTrue(3 <= len(cubes) <= 6)
            self.assertEqual({c["color"] for c in cubes}, set(COLORS))
            text, mapping = make_instruction("sort_language", rng)
            self.assertEqual(set(mapping), set(COLORS))
            self.assertEqual(set(mapping.values()), set(BINS))
            self.assertNotIn("{", text)
        a, b = AaltoEnv(seed=3), AaltoEnv(seed=3)
        a.reset(); b.reset()
        self.assertEqual(a.layout, b.layout)

class TestDDSBackendSafety(unittest.TestCase):
    def test_arm_holds_measured_pose_until_armed(self):
        import numpy as np
        from aaltonomous.interface.backends import make_safe_arm_controller

        class Fake:
            measured = np.full(14, 0.3)

            def __init__(self):
                self.q_target = np.zeros(14)  # what upstream does first

            def get_current_dual_arm_q(self):
                return self.measured

        a = make_safe_arm_controller(Fake)()
        np.testing.assert_allclose(a.q_target, 0.3)   # not zero
        a.q_target = np.zeros(14)                       # ignored while unarmed
        np.testing.assert_allclose(a.q_target, 0.3)
        a.arm()
        np.testing.assert_allclose(a.q_target, 0.3)
        a.q_target = np.full(14, 0.5)
        np.testing.assert_allclose(a.q_target, 0.5)

    def test_debug_mode_failure_refuses(self):
        from aaltonomous.interface.backends import enter_debug_mode

        class Sw:
            def __init__(self, r): self.r = r
            def Enter_Debug_Mode(self): return self.r

        enter_debug_mode(Sw((0, {"name": ""})))
        for bad in ((None, None), (0, {"name": "ai"}), (3, {"name": ""})):
            with self.assertRaises(RuntimeError):
                enter_debug_mode(Sw(bad))


if __name__ == "__main__":
    unittest.main()
