"""Offline tests: python3 -m unittest aaltonomous/leaderboard/test_leaderboard.py"""
import copy
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from aaltonomous.leaderboard import ingest, serve

FIX = Path(__file__).resolve().parent / "fixtures"
BASE = {t: json.loads((FIX / f"eval_{t}.json").read_text()) for t in ("sort_fixed", "sort_language")}


def cfg(**kw):
    c = ingest.load_config()
    c.update(kw)
    return c


class Runs:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.n = 0

    def add(self, team, task="sort_fixed", variant="eval_heldout", score=None, gate=True, trials=None,
            mtime=None, team_field=True, raw=None, episodes=0, summary=None):
        self.n += 1
        d = self.root / f"{team}_{task}_{variant}_20261113_10{self.n:02d}00"
        d.mkdir()
        if raw is not None:
            (d / "eval.json").write_text(raw)
        elif summary is not False:
            data = copy.deepcopy(BASE["sort_language" if task == "sort_language" else "sort_fixed"])
            s = data["summary"]
            s.update(team=team if team_field else None, task=task, variant=variant)
            if score is not None:
                s["score_mean"] = score
            if trials is not None:
                s["trials"] = trials
            if not gate:
                s["gate"].update(passed=False, failed_trials=[2])
            if summary:
                s.update(summary)
            (d / "eval.json").write_text(json.dumps(data))
        for i in range(episodes):
            t = BASE["sort_fixed"]["trials"][i]
            (d / f"episode_{i:03d}.jsonl").write_text(json.dumps({"event": {}}) + "\n" + json.dumps({"summary": t}) + "\n")
        if mtime and (d / "eval.json").exists():
            os.utime(d / "eval.json", (mtime, mtime))
        return d


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.r = Runs()
        self.addCleanup(self.r.tmp.cleanup)

    def board(self, **kw):
        return ingest.leaderboard(self.r.root, cfg(**kw))

    def test_real_fixture_reads(self):
        self.r.add("alpha")
        b = self.board()
        t = b["teams"][0]
        self.assertEqual((t["team"], t["rank"], t["gate"], t["total"]), ("alpha", 1, "pass", 2.4))
        self.assertEqual(t["tasks"]["sort_fixed"]["trials"], 5)

    def test_rank_sum_weights_and_finalists(self):
        self.r.add("alpha", score=3.0)
        self.r.add("alpha", "sort_language", score=6.0)
        self.r.add("beta", score=3.5)
        self.r.add("gamma", score=1.0)
        b = self.board(finalists=2)
        self.assertEqual([t["team"] for t in b["teams"]], ["alpha", "beta", "gamma"])
        self.assertEqual(b["teams"][0]["total"], 9.0)
        self.assertEqual([t["finalist"] for t in b["teams"]], [True, True, False])
        b = self.board(tasks={"sort_fixed": {"weight": 1.0, "label": "R1"}, "sort_language": {"weight": 0.0, "label": "R2"}})
        self.assertEqual(b["teams"][0]["team"], "beta")

    def test_gate_fail_not_ranked(self):
        self.r.add("alpha", score=5.0, gate=False)
        self.r.add("beta", score=1.0)
        b = self.board()
        a = next(t for t in b["teams"] if t["team"] == "alpha")
        self.assertEqual((a["gate"], a["rank"], a["eligible"], a["total"]), ("fail", None, False, None))
        self.assertEqual(b["teams"][0]["team"], "beta")
        self.assertFalse(a["tasks"]["sort_fixed"]["counts"])

    def test_best_skips_gate_fail_and_latest(self):
        self.r.add("alpha", score=2.0, mtime=1000)
        self.r.add("alpha", score=9.0, gate=False, mtime=2000)
        self.r.add("alpha", score=1.0, mtime=3000)
        self.assertEqual(self.board()["teams"][0]["total"], 2.0)
        self.assertEqual(self.board(select="latest")["teams"][0]["total"], 1.0)

    def test_partial_gate(self):
        self.r.add("alpha", score=3.0)
        self.r.add("alpha", "sort_language", score=6.0, gate=False)
        t = self.board()["teams"][0]
        self.assertEqual((t["gate"], t["total"], t["eligible"]), ("partial", 3.0, False))
        self.assertTrue(self.board(finalist_gate="any")["teams"][0]["finalist"])

    def test_lenient_gate_rejudged(self):
        self.r.add("alpha", summary={"gate": {"passed": True, "failed_trials": [], "max_clips": 50, "min_hz": 10},
                                     "safety_clipped_steps_max": 12})
        t = self.board()["teams"][0]
        self.assertEqual(t["gate"], "fail")
        self.assertIn("clips 12 > 0", t["tasks"]["sort_fixed"]["gate_why"])

    def test_min_trials_and_variant_filter(self):
        self.r.add("alpha", trials=2)
        self.r.add("beta", variant="nominal", score=9)
        b = self.board()
        self.assertEqual([t["team"] for t in b["teams"]], ["alpha"])
        self.assertIsNone(b["teams"][0]["rank"])
        self.assertEqual(self.board(variant=None)["teams"][0]["team"], "beta")

    def test_team_from_dirname_aliases_reference(self):
        self.r.add("team_with_under", team_field=False)
        self.r.add("Team_With_Under", "sort_language")
        self.r.add("oracle", score=99)
        b = self.board(aliases={"Team_With_Under": "team_with_under"})
        names = [t["team"] for t in b["teams"]]
        self.assertEqual(names, ["team_with_under", "oracle"])
        self.assertTrue(b["teams"][1]["reference"])
        self.assertEqual(b["teams"][0]["rank"], 1)
        self.assertEqual(len(b["teams"][0]["tasks"]), 2)

    def test_corrupt_running_and_odd_dirs(self):
        self.r.add("good")
        d = self.r.add("broken", raw='{"summary": {"score_me')
        os.utime(d / "eval.json", (time.time() - 600,) * 2)
        self.r.add("writer", raw="")                       # being written right now
        self.r.add("live", summary=False, episodes=2)        # eval still running
        self.r.add("weird", raw=json.dumps({"summary": {"team": "weird", "score_mean": "nan?", "gate": "yes"},
                                            "trials": "x"}))
        self.r.add("empty", raw="[]")
        (self.r.root / "not_an_eval").mkdir()
        (self.r.root / "notes.txt").write_text("hi")
        b = self.board()
        self.assertEqual([t["team"] for t in b["teams"]], ["good"])
        st = {r["team"]: r for r in b["running"]}
        self.assertEqual(st["live"]["status"], "running")
        self.assertEqual(st["live"]["trials"], 2)
        self.assertEqual(st["writer"]["status"], "writing")
        errs = {e["dir"].split("_")[0] for e in b["errors"]}
        self.assertEqual(errs, {"broken", "weird", "empty"})
        serve.render_html(b)                               # renders despite all of it

    def test_aborted_eval_hidden_and_trial_gate_reasons(self):
        self.r.add("good")
        d = self.r.add("crashed", summary=False, episodes=1)   # sim died mid-eval: intake worker marks the dir
        (d / "aborted.json").write_text("{}")
        bad = self.r.add("clipper", gate=False)
        data = json.loads((bad / "eval.json").read_text())
        data["trials"][0]["gate_why"] = ["clipped 31 > 0"]
        (bad / "eval.json").write_text(json.dumps(data))
        b = self.board()
        self.assertEqual(b["running"], [])
        self.assertNotIn("crashed", [t["team"] for t in b["teams"]])
        why = next(t for t in b["teams"] if t["team"] == "clipper")["tasks"]["sort_fixed"]["gate_why"]
        self.assertIn("clipped 31 > 0", why)

    def test_summary_missing_fields_falls_back_to_trials(self):
        data = copy.deepcopy(BASE["sort_fixed"])
        data["summary"] = {"team": "fb", "task": "sort_fixed", "variant": "eval_heldout"}
        self.r.add("fb", raw=json.dumps(data))
        t = self.board()["teams"][0]
        self.assertAlmostEqual(t["total"], 2.4)
        self.assertEqual(t["gate"], "pass")

    def test_missing_runs_dir(self):
        b = ingest.leaderboard(self.r.root / "nope", cfg())
        self.assertEqual(b["teams"], [])
        self.assertIn("No scored submissions", serve.render_html(b))


class TestServe(unittest.TestCase):
    def test_http_and_static(self):
        r = Runs()
        self.addCleanup(r.tmp.cleanup)
        r.add("alpha<script>", score=3.0)
        r.add("beta", score=2.0, gate=False)
        out = r.root / "out" / "leaderboard.html"
        board = serve.Board([r.root], html_out=out)
        srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(board))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        self.assertIn("http-equiv='refresh'", page)
        self.assertIn("alpha&lt;script&gt;", page)
        self.assertNotIn("alpha<script>", page)
        self.assertIn("FINALIST", page)
        self.assertIn("FAIL", page)
        j = json.loads(urllib.request.urlopen(base + "/leaderboard.json").read())
        self.assertEqual(j["teams"][0]["team"], "alpha<script>")
        static = out.read_text()
        self.assertNotIn("http-equiv='refresh'", static)
        self.assertIn("alpha&lt;script&gt;", static)
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope")


if __name__ == "__main__":
    unittest.main()
