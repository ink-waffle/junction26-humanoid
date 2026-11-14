"""Offline planner statistics over sampled layouts: every object x bin x arm, coverage, phase durations, planning time.

    python -m halal.eval.plan_bench --layouts 60 [--organiser] [--end retreat] [--start look]     (no sim needed)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from aaltonomous.baselines.classical.planner import Plan, Planner
from aaltonomous.interface.tasks import BINS, sample_layout

from halal import config
from halal.motion import PickPlanner, candidate_grasps
from halal.perception import ObjectDetection

SIDES = ("left", "right")


def detections(planner: Planner, seed: int) -> list[ObjectDetection]:
    """One sampled layout as exact detections in the perception convention."""
    scene = planner.scene
    layout = sample_layout(np.random.default_rng(seed), shapes=tuple(scene.cfg["objects"]["shapes"]), **scene.layout_kwargs())
    found = []
    for item in layout:
        top = scene.mat_to_pelvis(scene.cube_spawn_mat(item)) + [0.0, 0.0, scene.object_height() / 2]
        yaw = {"cube": -np.pi / 2 - item["yaw"], "triangle": -item["yaw"]}.get(item["shape"])
        found.append(ObjectDetection(color=item["color"], top=top, shape=item["shape"], yaw=yaw))
    return found


def plan_one(planner: Planner, det: ObjectDetection, side: str, bin_name: str, organiser: bool, end: str, start: str) -> Plan:
    """First candidate grasp that plans from the start pose, or the organiser planner's own pinch."""
    arm = planner.home.copy() if start == "home" else np.asarray(config.load()["motion"]["look_arm_q"], float)
    hand = planner.post["open"].copy()
    if organiser:
        return planner._plan_side_hints(det, side, bin_name, arm, hand)
    plan = Plan(False, reason="no grasp")
    for grasp in candidate_grasps(det.shape, det.yaw, side):
        plan = planner.plan_pick(det, grasp, side, bin_name, arm, hand, end=end)
        if plan.ok:
            break
    return plan


def collect(planner: Planner, layouts: int, organiser: bool, end: str, start: str) -> list[dict[str, Any]]:
    rows = []
    for seed in range(layouts):
        for det in detections(planner, seed):
            for bin_name in BINS:
                for side in SIDES:
                    start = time.perf_counter()
                    plan = plan_one(planner, det, side, bin_name, organiser, end, start)
                    rows.append({"y": float(det.top[1]), "bin": bin_name, "side": side, "ok": plan.ok, "low": getattr(planner, "low", False),
                                 "wall": time.perf_counter() - start, "reason": plan.reason, "duration": plan.duration,
                                 "phases": {name: (stop - first) / 30.0 for name, first, stop in plan.phases}})
    return rows


def report(rows: list[dict[str, Any]]) -> None:
    pairs: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for k, row in enumerate(rows):
        pairs[k // len(SIDES)][row["side"]] = row
    print(f"(object, bin) pairs: {len(pairs)}")
    for bin_name in BINS:
        mine = [p for p in pairs.values() if p["left"]["bin"] == bin_name]
        share = {side: np.mean([p[side]["ok"] for p in mine]) for side in SIDES}
        either = np.mean([p["left"]["ok"] or p["right"]["ok"] for p in mine])
        print(f"  bin {bin_name:<6} left {share['left']:.0%}  right {share['right']:.0%}  either {either:.0%}")
    print(f"  any bin: either arm {np.mean([p['left']['ok'] or p['right']['ok'] for p in pairs.values()]):.1%}")
    best = [min((p[s] for s in SIDES if p[s]["ok"]), key=lambda r: r["duration"]) for p in pairs.values()
            if p["left"]["ok"] or p["right"]["ok"]]
    print(f"low sequence used in {np.mean([r['low'] for r in rows if r['ok']]):.0%} of the plans")
    print(f"plan duration (faster arm): mean {np.mean([r['duration'] for r in best]):.2f} s  worst {max(r['duration'] for r in best):.2f} s")
    for name in best[0]["phases"]:
        times = [r["phases"][name] for r in best]
        print(f"  {name:<12} mean {np.mean(times):.2f} s  worst {max(times):.2f} s")
    for label, group in (("ok", [r for r in rows if r["ok"]]), ("failed", [r for r in rows if not r["ok"]])):
        walls = [r["wall"] for r in group] or [0.0]
        print(f"planning wall time, {label} plans ({len(group)}): mean {np.mean(walls):.2f} s  p95 {np.percentile(walls, 95):.2f} s  max {max(walls):.2f} s")
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row["ok"]:
            reasons[row["reason"][:60]] += 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {count:4d} x {reason}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", type=int, default=60)
    ap.add_argument("--organiser", action="store_true", help="the organiser planner's own pinch instead of ours")
    ap.add_argument("--end", default="home", choices=("home", "retreat"))
    ap.add_argument("--start", default="home", choices=("home", "look"), help="arm pose the plans start from")
    args = ap.parse_args(argv)
    planner: Planner = PickPlanner.create()
    if args.organiser:
        planner = Planner(planner.cfg, planner.scene, planner.post, planner.home, planner.kin, planner.safety, planner.spec)
    report(collect(planner, args.layouts, args.organiser, args.end, args.start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
