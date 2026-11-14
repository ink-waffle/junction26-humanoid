"""Tables from grasp bench runs: success and failure modes per variant, shape and condition; finger error held vs empty; where failures leave the object.

    python -m halal.eval.grasp_report runs/grasp/first [runs/grasp/second ...] [--by pos_noise yaw_noise_deg]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

JOINTS = ("thumb_0", "thumb_1", "thumb_2", "finger_0", "finger_1")
HELD_MARKS = ("lift", "carry")
TOPPLED_DEG = 45.0


def by_role(error: list[float]) -> list[float]:
    """LEFT-order finger error (7) as thumb joints plus the more loaded of index / middle per joint: the sim swaps them on the right hand."""
    e = np.abs(error)
    return [e[0], e[1], e[2], max(e[3], e[5]), max(e[4], e[6])]


def load(runs: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(line) for run in runs for line in (run / "picks.jsonl").read_text().splitlines()]


def success_table(picks: list[dict[str, Any]], by: list[str]) -> None:
    """Successes / trials and the failure modes for every (variant, shape, condition) group."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for pick in picks:
        if pick["present"]:
            groups[(pick["variant"], pick["shape"], *(pick[key] for key in by))].append(pick)
    print(f"{'variant':<16} {'shape':<9} {' '.join(by):<22} ok/n   spin  tilt  slip   failures")
    for key, group in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        good = sum(p["failure"] == "ok" for p in group)
        fails = Counter(p["failure"] for p in group if p["failure"] != "ok")
        lifted = [p for p in group if p.get("held", {}).get("lift")]
        spin, tilt, slip = (np.median([abs(p[k]) for p in lifted]) if lifted else np.nan for k in ("spin_deg", "tilt_deg", "slip_mm"))
        condition = " ".join(str(v) for v in key[2:])
        print(f"{key[0]:<16} {key[1]:<9} {condition:<22} {good:2d}/{len(group):<3d} {spin:5.0f} {tilt:5.0f} {slip:5.1f}   "
              + ", ".join(f"{name} {count}" for name, count in fails.most_common()))


def finger_table(picks: list[dict[str, Any]]) -> None:
    """Per variant: finger tracking error at the lift and carry marks with an object in the hand and without."""
    held: dict[str, list[list[float]]] = defaultdict(list)
    empty: dict[str, list[list[float]]] = defaultdict(list)
    for pick in picks:
        if "finger_error" not in pick:
            continue
        for mark in HELD_MARKS:
            in_hand = pick["present"] and pick.get("held", {}).get(mark)
            (held if in_hand else empty)[pick["variant"]].append(by_role(pick["finger_error"][mark]))
    print(f"\nfinger error |measured - target| (rad) at lift and mid-carry: {' '.join(JOINTS)}")
    for variant in sorted(set(held) | set(empty)):
        for label, rows in (("held", held[variant]), ("empty", empty[variant])):
            if rows:
                values = np.array(rows)
                print(f"{variant:<16} {label:<5} n={len(rows):<3d} median {np.round(np.median(values, 0), 3).tolist()}  "
                      f"largest joint: min {values.max(1).min():.3f}  max {values.max(1).max():.3f}")


def aftermath_table(picks: list[dict[str, Any]]) -> None:
    """Where failed picks leave the object: displacement from its start (pelvis x, outward y), toppled, still on the counter."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for pick in picks:
        if pick["present"] and pick["failure"] not in ("ok", "no_plan") and "final_pos" in pick:
            groups[(pick["variant"], pick["shape"])].append(pick)
    print("\nafter a failed pick: n, median shift (mm), largest shift, mean dx / outward dy (mm), toppled, left the counter")
    for key, group in sorted(groups.items()):
        outward = np.array([1.0 if p["side"] == "left" else -1.0 for p in group])
        shift = 1000 * (np.array([p["final_pos"][:2] for p in group]) - np.array([p["true_pos"][:2] for p in group]))
        shift[:, 1] *= outward
        distance = np.linalg.norm(shift, axis=1)
        tilts = [p["final_tilt_deg"] for p in group if "final_tilt_deg" in p]
        toppled = f"{sum(t > TOPPLED_DEG for t in tilts)}/{len(tilts)}"
        gone = sum(p["outcome"] == "off" for p in group)
        print(f"{key[0]:<16} {key[1]:<9} {len(group):2d}  {np.median(distance):5.0f} {distance.max():5.0f}   "
              f"{shift[:, 0].mean():+5.0f} / {shift[:, 1].mean():+5.0f}   toppled {toppled}  off {gone}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--by", nargs="*", default=[], help="extra pick fields to group by, e.g. pos_noise yaw_known side")
    args = ap.parse_args(argv)
    picks = load(args.runs)
    success_table(picks, args.by)
    finger_table(picks)
    aftermath_table(picks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
