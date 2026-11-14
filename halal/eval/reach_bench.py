"""Offline: which release the planner finds with the bin row moved away from the robot, per bin, arm and release slot.

    python -m halal.eval.reach_bench --offsets 0 0.03 0.06 0.08 0.10 0.12 [--rises 0 0.03 0.06 0.09 0.12] [--shape cylinder]     (no sim needed)
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from aaltonomous.interface.tasks import BINS

from halal import config
from halal.motion import PickPlanner, candidate_grasps
from halal.perception import ObjectDetection, SceneLayout, scene_layout

ARMS = {"left": ("left",), "center": ("left", "right"), "right": ("right",)}
OUTWARD = {"left": 1.0, "right": -1.0}
YAWS = {"cube": 0.0, "triangle": 0.0, "cylinder": None}


def moved(planner: PickPlanner, offset: float) -> SceneLayout:
    """The scene file's layout with the bin row `offset` further from the robot."""
    layout = scene_layout(planner.file_scene)
    return SceneLayout(layout.counter_z, {name: xy + [offset, 0.0] for name, xy in layout.bins.items()}, "scene")


def upright(planner: PickPlanner, side: str, spot: list[float], shape: str) -> ObjectDetection:
    """An upright object of this shape at a bench spot (forward, outward) on this arm's side."""
    top = np.r_[spot[0], OUTWARD[side] * spot[1], planner.scene.counter_z + planner.height]
    return ObjectDetection(color="red", top=top, shape=shape, yaw=YAWS[shape])


def rows(planner: PickPlanner, offset: float, hand: str, shape: str) -> list[dict[str, Any]]:
    """One plan per bin, arm, release slot and object spot with the bins at this offset."""
    settings, slots = config.load()["reach_bench"], config.load()["policy"]["release_shifts"]
    planner.set_layout(moved(planner, offset))
    start, out = np.asarray(planner.motion["look_arm_q"], float), []
    for bin_name in BINS:
        for side in ARMS[bin_name]:
            for shift in slots[bin_name]:
                for spot in settings["spots"]:
                    det = upright(planner, side, spot, shape)
                    began = time.perf_counter()
                    plan = planner.plan_pick(det, candidate_grasps(shape, det.yaw, side, hand)[0], side, bin_name, start,
                                             planner.post["open"].copy(), drop_shift=tuple(shift), end="retreat")
                    out.append({"bin": bin_name, "side": side, "ok": plan.ok, "reason": plan.reason, "wall": time.perf_counter() - began,
                                "inset": planner.near_inset, "duration": plan.duration,
                                "rise": plan.waypoints["drop"][2] - planner.release_height() if plan.ok else 0.0})
    return out


def report(offset: float, found: list[dict[str, Any]]) -> None:
    ok = [r for r in found if r["ok"]]
    line = f"bins {100 * offset:+5.1f} cm: {len(ok):2d}/{len(found)} plans"
    for bin_name in BINS:
        mine = [r for r in found if r["bin"] == bin_name]
        line += f"  {bin_name} {sum(r['ok'] for r in mine)}/{len(mine)}"
    if ok:
        line += (f"  inset {100 * min(r['inset'] for r in ok):.0f}-{100 * max(r['inset'] for r in ok):.0f} cm"
                 f"  rise {100 * min(r['rise'] for r in ok):.0f}-{100 * max(r['rise'] for r in ok):.0f} cm"
                 f"  pick {np.mean([r['duration'] for r in ok]):.2f} s")
    line += f"  planning {np.mean([r['wall'] for r in found]):.2f} s mean, {max(r['wall'] for r in found):.2f} s max"
    print(line, flush=True)
    for reason in sorted({r["reason"][:70] for r in found if not r["ok"]}):
        print(f"      {sum(r['reason'][:70] == reason for r in found):2d} x {reason}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.08, 0.10, 0.12])
    ap.add_argument("--rises", type=float, nargs="+", help="motion.release_rise for this run (default: the config's)")
    ap.add_argument("--hand", default="stiff", choices=("stiff", "soft"))
    ap.add_argument("--shape", default="cube", choices=tuple(YAWS))
    args = ap.parse_args(argv)
    planner = PickPlanner.create()
    if args.rises:
        planner.motion["release_rise"] = args.rises
    for offset in args.offsets:
        report(offset, rows(planner, offset, args.hand, args.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
