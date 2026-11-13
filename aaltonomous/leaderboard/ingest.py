"""Scan sim eval result dirs (aaltonomous/sim/evaluate.py output) and build the leaderboard.

    python -m aaltonomous.leaderboard.ingest --runs runs/eval [--config leaderboard.yaml] [--json]

Each eval dir is `<team>_<task>_<variant>_<YYYYmmdd_HHMMSS>/` with `eval.json` (written once at the end) and
`episode_NNN.jsonl` (last line = that trial's summary, written per trial). A dir without eval.json is a running
(or crashed) eval; a corrupt eval.json is reported, never fatal. All eval.json fields are read defensively.
Ranking rules live in leaderboard.yaml (open decisions marked TBD there).
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "leaderboard.yaml"
TASKS = ("sort_fixed", "sort_language", "insert_shapes")
BAD_ENDS = ("aborted", "watchdog", "safe_stop", "policy_connection_lost", "policy_timeout", "crash", "interrupted")
DIR_RE = re.compile(r"^(?P<team>.+?)_(?P<task>" + "|".join(TASKS) + r")_(?P<variant>.+)_(?P<stamp>\d{8}_\d{6})$")

DEFAULTS = {
    "title": "Aaltonomous Humanoid · sim leaderboard",
    "variant": "eval_heldout",
    "tasks": {"sort_fixed": {"weight": 1.0, "label": "Rung 1"},
              "sort_language": {"weight": 1.0, "label": "Rung 2"},
              "insert_shapes": {"weight": 0.0, "label": "Rung 3"}},
    "metric": "score_mean", "select": "best", "min_trials": 5, "require_gate": True,
    "gate_max_clips": 0, "gate_min_hz": 27.0, "finalist_gate": "all", "finalists": 5,
    "reference_teams": ["oracle", "policy", "baseline"], "hide_reference": False, "aliases": {},
    "refresh_s": 20, "max_rows": 20, "stale_running_s": 900,
}


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    path = Path(path) if path else DEFAULT_CONFIG
    if path.exists():
        try:
            import yaml
            user = yaml.safe_load(path.read_text()) or {}
        except ImportError:                       # no PyYAML: allow a JSON config, else defaults
            try:
                user = json.loads(path.read_text())
            except ValueError:
                user = {}
        cfg.update({k: v for k, v in user.items() if v is not None or k == "variant"})
    cfg["tasks"] = {t: {"weight": float((v or {}).get("weight", 0)), "label": (v or {}).get("label", t)}
                    for t, v in (cfg.get("tasks") or {}).items()}
    return cfg


# ---------------------------------------------------------------- reading one eval dir

def _num(x, default=None):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _stamp_epoch(stamp):
    try:
        return time.mktime(time.strptime(stamp, "%Y%m%d_%H%M%S"))
    except (TypeError, ValueError):
        return None


def _last_json_line(p):
    try:
        lines = p.read_text(errors="replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def _summarize_trials(trials):
    """Fallback summary from trial dicts (running evals, or an eval.json without a usable summary)."""
    trials = [t for t in trials if isinstance(t, dict) and _num(t.get("score")) is not None]
    if not trials:
        return {"trials": 0}
    sc = [_num(t["score"]) for t in trials]
    mx = sum(_num(t.get("max_score"), 0) for t in trials)
    obj = sum(_num(t.get("n_objects"), 0) for t in trials)
    cor = sum(_num(t.get("n_correct"), 0) for t in trials)
    ends = {}
    for t in trials:
        ends[str(t.get("end_reason"))] = ends.get(str(t.get("end_reason")), 0) + 1
    clips = [_num((t.get("safety") or {}).get("clipped_steps"), 0) for t in trials]
    hz = [_num(t.get("command_hz", t.get("effective_hz")), 0) for t in trials]
    gates = [t.get("gate_ok") for t in trials]
    return {
        "trials": len(trials), "score_mean": sum(sc) / len(sc),
        "score_frac": (sum(sc) / mx) if mx else None, "object_success": (cor / obj) if obj else None,
        "end_reasons": ends, "safety_clipped_steps_max": max(clips), "command_hz_min": min(hz),
        "gate": {"passed": all(g is True for g in gates) if all(g is not None for g in gates) else None},
    }


def judge_gate(s, cfg):
    """(passed: bool|None, why: list). Trust evaluate's gate unless it ran with looser thresholds than cfg."""
    g = s.get("gate") if isinstance(s.get("gate"), dict) else {}
    passed = g.get("passed")
    why = []
    looser = (_num(g.get("max_clips"), 0) > cfg["gate_max_clips"]) or (_num(g.get("min_hz"), 1e9) < cfg["gate_min_hz"])
    if passed is False and g.get("failed_trials"):
        why.append(f"failed trials {g['failed_trials']}")
    if passed is None or looser:
        ends = s.get("end_reasons") if isinstance(s.get("end_reasons"), dict) else {}
        bad = [e for e in ends if e in BAD_ENDS and ends[e]]
        clips = _num(s.get("safety_clipped_steps_max"))
        hz = _num(s.get("command_hz_min", s.get("effective_hz_min")))
        if bad:
            why.append("end: " + ",".join(bad))
        if clips is not None and clips > cfg["gate_max_clips"]:
            why.append(f"clips {clips:g} > {cfg['gate_max_clips']}")
        if hz is not None and hz < cfg["gate_min_hz"]:
            why.append(f"{hz:g} Hz < {cfg['gate_min_hz']:g}")
        rt = _num(s.get("policy_hz_rt_min"))      # own-latency rate (wall), evaluate gate since sim-clock pacing
        if rt is not None and rt < cfg["gate_min_hz"]:
            why.append(f"policy latency allows {rt:g} Hz < {cfg['gate_min_hz']:g}")
        if clips is None and hz is None and not ends:
            return passed if not looser else None, why or ["gate unknown"]
        passed = (passed is not False) and not bad and not why
    return passed, why


def read_run(d, cfg, now=None):
    """One eval dir -> run dict, or None if it is not an eval dir. Never raises on bad content."""
    now = now or time.time()
    m = DIR_RE.match(d.name)
    ev = d / "eval.json"
    if d.name.startswith(".") or ((d / "aborted.json").exists() and not ev.exists()):
        return None                               # intake worker marked it: interrupted eval (sim down, timeout)
    eps = sorted(d.glob("episode_*.jsonl"))
    if not m and not ev.exists() and not eps:
        return None
    run = {"dir": d.name, "team": m["team"] if m else None, "task": m["task"] if m else None,
           "variant": m["variant"] if m else None, "started": _stamp_epoch(m["stamp"]) if m else None,
           "status": "done", "error": None}
    s, trials = None, None
    if ev.exists():
        try:
            data = json.loads(ev.read_text())
            s = data.get("summary") if isinstance(data, dict) else None
            trials = data.get("trials") if isinstance(data, dict) else None
            if not isinstance(s, dict):
                s = None
            run["finished"] = ev.stat().st_mtime
        except (OSError, ValueError, UnicodeDecodeError) as e:
            age = now - ev.stat().st_mtime if ev.exists() else 0
            run.update(status="writing" if age < 30 else "corrupt", error=f"eval.json: {type(e).__name__}")
    if s is None or _num(s.get("score_mean")) is None:
        if not isinstance(trials, list):          # no usable eval.json: per-trial summaries from the episode logs
            trials = [x.get("summary") for x in map(_last_json_line, eps) if isinstance(x, dict)]
        fb = _summarize_trials(trials or [])
        s = {**fb, **{k: v for k, v in (s or {}).items() if v is not None}} if s else fb
        if not ev.exists():
            last = max([p.stat().st_mtime for p in eps] + [d.stat().st_mtime])
            run["status"] = "stalled" if now - last > cfg["stale_running_s"] else "running"
            run["finished"] = last
        elif run["status"] == "done" and not s.get("trials"):
            run.update(status="corrupt", error=run["error"] or "eval.json has no scored trials")
    run["team"] = s.get("team") or run["team"] or "unknown"
    for k in ("task", "variant"):
        if s.get(k):
            run[k] = s[k]
    run.update(
        policy=s.get("policy"), trials=int(_num(s.get("trials"), 0)),
        score_mean=_num(s.get("score_mean")), score_std=_num(s.get("score_std")),
        score_frac=_num(s.get("score_frac")), object_success=_num(s.get("object_success")),
    )
    run["gate"], run["gate_why"] = judge_gate(s, cfg)
    if run["gate"] is False and isinstance(trials, list):    # evaluate's per-trial reasons ("clipped 31 > 0", ...)
        seen = [w for t in trials if isinstance(t, dict) for w in (t.get("gate_why") or []) if isinstance(w, str)]
        run["gate_why"] += [w for w in dict.fromkeys(seen) if w not in run["gate_why"]][:4]
    run.setdefault("finished", run["started"])
    return run


def scan(runs_dirs, cfg, now=None):
    out = []
    for root in ([runs_dirs] if isinstance(runs_dirs, (str, Path)) else runs_dirs):
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                r = read_run(d, cfg, now)
            except OSError as e:                   # dir vanished mid-scan etc.
                r = {"dir": d.name, "team": "unknown", "status": "corrupt", "error": repr(e)}
            if r:
                out.append(r)
    return out


# ---------------------------------------------------------------- ranking

def team_key(name, cfg):
    name = str(name).strip()
    al = {str(k).strip().lower(): v for k, v in (cfg.get("aliases") or {}).items()}
    return str(al.get(name.lower(), name))


def _metric(run, cfg):
    v = run.get(cfg["metric"])
    return v if v is not None else run.get("score_mean")


def build(runs, cfg, now=None):
    now = now or time.time()
    refs = {str(t).lower() for t in cfg.get("reference_teams") or []}
    teams, running, errors = {}, [], []
    for r in runs:
        if r.get("status") in ("running", "stalled", "writing"):
            running.append(r)
            continue
        if r.get("status") == "corrupt":
            errors.append(r)
            continue
        if cfg.get("variant") and r.get("variant") != cfg["variant"]:
            continue
        if r.get("task") not in cfg["tasks"]:
            continue
        teams.setdefault(team_key(r["team"], cfg), []).append(r)

    rows = []
    for name, rs in teams.items():
        per, total, last = {}, 0.0, max((r.get("finished") or 0) for r in rs)
        attempted_pass, attempted_fail = 0, 0
        for task, tc in cfg["tasks"].items():
            cand = [r for r in rs if r["task"] == task]
            if not cand:
                continue
            full = [r for r in cand if r["trials"] >= cfg["min_trials"] and _metric(r, cfg) is not None]
            ok = [r for r in full if r["gate"] is True] if cfg["require_gate"] else full
            pool = ok or full or cand
            if cfg["select"] == "latest":
                pick = max(pool, key=lambda r: r.get("finished") or 0)
            else:
                pick = max(pool, key=lambda r: (_metric(r, cfg) if _metric(r, cfg) is not None else -1e9,
                                                -(r.get("finished") or 0)))
            counts = bool(ok) and pick in ok
            val = _metric(pick, cfg) if counts else None
            if tc["weight"] > 0:
                if ok:
                    attempted_pass += 1
                else:
                    attempted_fail += 1
                if counts:
                    total += tc["weight"] * val
            per[task] = {"value": _metric(pick, cfg), "counts": counts, "trials": pick["trials"], "gate": pick["gate"],
                         "gate_why": pick.get("gate_why") or [], "object_success": pick.get("object_success"),
                         "runs": len(cand), "dir": pick["dir"], "finished": pick.get("finished"),
                         "short": None if full else f"{pick['trials']}/{cfg['min_trials']} trials"}
        if attempted_pass and not attempted_fail:
            gate = "pass"
        elif attempted_pass:
            gate = "partial"
        elif attempted_fail:
            gate = "fail"
        else:
            gate = "none"          # only weight-0 tasks
        eligible = gate == "pass" or (gate == "partial" and cfg["finalist_gate"] == "any")
        ref = name.lower() in refs
        if ref and cfg.get("hide_reference"):
            continue
        rows.append({"team": name, "total": round(total, 3) if attempted_pass else None, "gate": gate,
                     "eligible": eligible and not ref, "reference": ref, "tasks": per, "runs": len(rs),
                     "last_submission": last, "rank": None, "finalist": False})

    def order(t):
        best_first = max([v["finished"] or 0 for v in t["tasks"].values() if v["counts"]] or [0])
        return (not t["eligible"], t["reference"], -(t["total"] if t["total"] is not None else -1e9), best_first)
    rows.sort(key=order)
    rank = 0
    for t in rows:
        if t["eligible"]:
            rank += 1
            t["rank"] = rank
            t["finalist"] = rank <= int(cfg["finalists"])
    return {"title": cfg["title"], "generated": now, "variant": cfg.get("variant"), "metric": cfg["metric"],
            "select": cfg["select"], "finalists": cfg["finalists"],
            "tasks": {k: v for k, v in cfg["tasks"].items()}, "teams": rows,
            "running": [{k: r.get(k) for k in ("dir", "team", "task", "variant", "status", "trials", "score_mean",
                                               "finished")} for r in running],
            "errors": [{"dir": r.get("dir"), "error": r.get("error")} for r in errors],
            "config": {k: cfg[k] for k in ("min_trials", "require_gate", "gate_max_clips", "gate_min_hz",
                                           "finalist_gate")}}


def leaderboard(runs_dirs, cfg=None, now=None):
    cfg = cfg or load_config()
    return build(scan(runs_dirs, cfg, now), cfg, now)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/eval"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    b = leaderboard(a.runs, load_config(a.config))
    if a.json:
        print(json.dumps(b, indent=1))
        return
    for t in b["teams"]:
        cells = "  ".join(f"{k}={v['value']:.2f}{'' if v['counts'] else '*'}" if v["value"] is not None else f"{k}=-"
                          for k, v in t["tasks"].items())
        tot = f"{t['total']:.2f}" if t["total"] is not None else "-"
        print(f"{t['rank'] or '-':>3} {t['team']:<20} {tot:>7}  gate={t['gate']:<7} {cells}"
              f"{'  FINALIST' if t['finalist'] else ''}{'  (ref)' if t['reference'] else ''}")
    for r in b["running"]:
        print(f"  {r['status']}: {r['dir']} ({r['trials']} trials)")
    for e in b["errors"]:
        print(f"  error: {e['dir']}: {e['error']}")


if __name__ == "__main__":
    main()
