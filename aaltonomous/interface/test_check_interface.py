"""Full check_interface run (incl. motion probes) on the mock backend.
Run: python -m unittest aaltonomous/interface/test_check_interface.py"""
import json
import tempfile
import unittest
from pathlib import Path

from aaltonomous.interface.check_interface import main


class T(unittest.TestCase):
    def test_mock_full(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            self.assertEqual(main(["--backend", "mock", "--report", str(p), "--passive-secs", "1"]), 0)
            rows = json.loads(p.read_text())["checks"]
        motion = [r for r in rows if "probe" in r["check"]]
        self.assertEqual(len(motion), 28)
        self.assertTrue(all(r["status"] == "PASS" for r in motion), [r for r in motion if r["status"] != "PASS"])
        self.assertFalse(any(r["status"] == "FAIL" for r in rows))

    def test_mock_seated_options(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            args = ["--backend", "mock", "--report", str(p), "--passive-secs", "1", "--no-counter-keepout",
                    "--raise-arms", "-0.3", "--hands-only", "--open-hands"]
            self.assertEqual(main(args), 0)
            rows = json.loads(p.read_text())["checks"]
        self.assertEqual(len([r for r in rows if r["check"].startswith("hand probe")]), 14)

    def test_safety_real_on_mock(self):
        from aaltonomous.interface.check_safety_real import main as safety_main
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            self.assertEqual(safety_main(["--backend", "mock", "--raise-arms", "-0.3", "--report", str(p)]), 0)
            self.assertEqual(json.loads(p.read_text())["counts"]["PASS"], 5)
