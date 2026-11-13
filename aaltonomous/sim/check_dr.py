"""Check domain randomization on a running Aalto-* sim started with AALTO_DR (no robot motion).

    python -m aaltonomous.sim.check_dr [--host 127.0.0.1] [--seeds 1 2 3] [--save DIR]

For each seed: scene service `randomize` (which groups applied), then a head RGB-D frame from :$AALTO_RGBD_PORT
(mean RGB, depth validity / noise on the counter). PASS if every group applied and the images differ between
seeds. --save writes the head RGB per seed as PNG for a visual check.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from aaltonomous.interface.head_rgbd import HeadRGBDClient
from aaltonomous.sim.scene_client import SceneClient


def frame(cli, after, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = cli.get()
        if f is not None and f[2] > after:
            return f
        time.sleep(0.05)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    sc = SceneClient(a.host)
    info = sc.info()
    print("sim:", {k: v for k, v in info.items() if k != "ok"})
    if not info.get("dr"):
        print("FAIL: sim runs without AALTO_DR")
        return 1
    cli = HeadRGBDClient(a.host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    fails, means = [], []
    sc.reset_random(0)
    for s in a.seeds:
        p = sc.randomize(s)["params"]
        bad = {k: v for k, v in p["applied"].items() if v is not True}
        t = time.time()
        time.sleep(0.5)
        f = frame(cli, t + 0.3)
        if f is None:
            fails.append(f"seed {s}: no head RGB-D frame")
            print(f"seed {s}: applied={p['applied']}  NO FRAME")
            continue
        rgb, depth = f[0], f[1]
        valid = depth > 0
        means.append(rgb.reshape(-1, 3).mean(0))
        print(f"seed {s}: applied={p['applied']}  key light x{p['lighting']['key_intensity_scale']:.2f}  "
              f"rgb mean {np.round(means[-1], 1)}  depth valid {100 * valid.mean():.1f}%  "
              f"depth median {np.median(depth[valid]) if valid.any() else 0:.0f} mm")
        if bad:
            fails.append(f"seed {s}: not applied {bad}")
        if a.save:
            import cv2
            os.makedirs(a.save, exist_ok=True)
            cv2.imwrite(os.path.join(a.save, f"head_seed{s}.png"), rgb[..., ::-1])
    cli.close()
    if len(means) >= 2 and max(np.abs(m - means[0]).max() for m in means[1:]) < 1.0:
        fails.append("head images do not change between seeds (USD changes not rendered?)")
    print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
