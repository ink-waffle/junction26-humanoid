"""Tables from grasp bench runs: success and failure modes per variant, shape and condition; finger error held vs empty.

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

JOINTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
HELD_MARKS = ("lift", "carry")


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
            (held if in_hand else empty)[pick["variant"]].append(pick["finger_error"][mark])
    print(f"\nfinger error after close, measured - target (rad), LEFT order {' '.join(JOINTS)}")
    for variant in sorted(set(held) | set(empty)):
        for label, rows in (("held", held[variant]), ("empty", empty[variant])):
            if rows:
                values = np.abs(np.array(rows))
                print(f"{variant:<16} {label:<5} n={len(rows):<3d} median |e| {np.round(np.median(values, 0), 3).tolist()}  "
                      f"min of max-joint {values.max(1).min():.3f}  max of max-joint {values.max(1).max():.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--by", nargs="*", default=[], help="extra pick fields to group by, e.g. pos_noise yaw_known side")
    args = ap.parse_args(argv)
    picks = load(args.runs)
    success_table(picks, args.by)
    finger_table(picks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
