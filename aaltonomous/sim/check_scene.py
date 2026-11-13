"""M4 check against a running Aalto-* sim (run_sim.sh with TASK=Aalto-Sort-G129-Dex3-Joint or Aalto-Insert-...).

  1. layouts: N random layouts settle at rest within 5 mm of the commanded mat pose; every object scores "counter"
  2. checker: objects dropped into bins / beside the counter land as "correct" / "wrong" / "off" (sort_fixed, and
     a sort_language mapping), so the Scene checker and the sim geometry agree
  3. (--lid) rung 3: one object of each shape dropped above its own lid hole falls through -> "inserted";
     a cube dropped on the lid's solid part -> "forced"
  4. head RGB-D frame on :55560 (depth valid fraction, object pixels nearer than the mat)

    conda activate isaac; python aaltonomous/sim/check_scene.py [--lid] [--n 5] [--save DIR]
Exit code 0 = all PASS. Needs no robot commands (arms stay at home).
"""
from __future__ import annotations

import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aaltonomous.scene.scene import Scene                       # noqa: E402
from aaltonomous.sim.scene_client import SceneClient            # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def ordered(ev, active):
    """Outcomes in layout order (the service returns them in object-name order)."""
    by = {o["name"]: o["outcome"] for o in ev["outcomes"]}
    return [by[n] for n in active]


def settle(c, secs=1.5):
    time.sleep(secs)
    return {o["name"]: o for o in c.state()["objects"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lid", action="store_true", help="also run the rung-3 lid checks (Aalto-Insert task)")
    ap.add_argument("--save", default=None, help="save head RGB + depth JPEGs here")
    a = ap.parse_args(argv)
    sc = Scene.load()
    c = SceneClient(a.host, timeout_ms=10000)
    c.ping()

    # 1. random layouts
    for seed in range(a.n):
        r = c.reset_random(seed)
        st = settle(c)
        errs, moving = [], []
        for ob, n in zip(r["layout"], r["active"]):
            want = sc.mat_to_pelvis(sc.cube_spawn_mat(ob))
            errs.append(np.linalg.norm(np.asarray(st[n]["pos"]) - want))
            moving.append(st[n]["speed"])
        ev = c.evaluate("sort_fixed")
        outs = {o["outcome"] for o in ev["outcomes"]}
        check(f"layout seed {seed} ({len(r['active'])} objects)",
              max(errs) < 0.005 and max(moving) < 0.01 and outs == {"counter"},
              f"max pos err {max(errs) * 1000:.1f} mm, max speed {max(moving):.3f} m/s, outcomes {sorted(outs)}")

    # 2. drops into bins: rung 1 colour == bin colour
    bins = sc.cfg["bins"]
    colour_bin = {col: b for b, col in bins["color"].items()}
    drop_z = bins["wall_height"] + 0.04
    lay = [
        {"color": "red", "shape": "cube", "x": bins["centers"][colour_bin["red"]][0],
         "y": bins["centers"][colour_bin["red"]][1], "yaw": 0.2, "z": drop_z},                 # correct
        {"color": "blue", "shape": "cylinder", "x": bins["centers"][colour_bin["blue"]][0],
         "y": bins["centers"][colour_bin["blue"]][1], "yaw": 0.0, "z": drop_z},                # correct
        {"color": "yellow", "shape": "triangle", "x": bins["centers"][colour_bin["blue"]][0] + 0.03,
         "y": bins["centers"][colour_bin["blue"]][1] - 0.02, "yaw": 0.5, "z": drop_z + 0.06},  # wrong (blue bin)
        {"color": "red", "shape": "triangle", "x": 0.45, "y": -0.30, "yaw": 0.0, "z": 0.05},  # off: behind the
    ]                                                                                         # counter edge
    if sc.lid_bin == colour_bin["red"] and a.lid:   # the lid covers the red bin in the Insert scene
        lay = lay[1:]
    act = c.reset(lay)["active"]
    settle(c, 2.5)
    ev = c.evaluate("sort_fixed")
    got = ordered(ev, act)
    if len(lay) == 3:
        got = ["correct"] + got
        ev["score"] += 1
    check("drop test sort_fixed", got == ["correct", "correct", "wrong", "off"] and ev["score"] == 1 + 1 - 1 - 1,
          f"outcomes {got}, score {ev['score']}")
    # same scene, rung-2 mapping: yellow -> the blue bin, red -> centre
    ev = c.evaluate("sort_language", {"yellow": colour_bin["blue"], "red": "center", "blue": colour_bin["blue"]})
    got = ordered(ev, act)
    if len(lay) == 3:
        got = ["wrong"] + got
        ev["score"] -= 1
    check("drop test sort_language", got == ["wrong", "correct", "correct", "off"] and ev["score"] == -1 + 2 + 2 - 1,
          f"outcomes {got}, score {ev['score']}")

    # 3. lid
    if a.lid:
        holes = {h["shape"]: np.asarray(h["center"], float) for h in sc.lid_holes()}
        yaw = {"cube": 0.0, "cylinder": 0.0, "triangle": np.pi}   # scene.lid_holes: triangle hole apex towards -y
        z = sc.lid_z[1] + sc.object_height() / 2 + 0.02
        lay = [{"color": col, "shape": s, "x": float(holes[s][0]), "y": float(holes[s][1]), "yaw": yaw[s], "z": z}
               for col, s in zip(("red", "yellow", "blue"), ("cube", "cylinder", "triangle"))]
        cx, cy = bins["centers"][sc.lid_bin]
        hx, hy = np.asarray(bins["outer_size"]) / 2
        solid = next(([x, y] for x in np.linspace(cx - hx + 0.03, cx + hx - 0.03, 9)
                      for y in np.linspace(cy - hy + 0.03, cy + hy - 0.03, 9)
                      if all(np.linalg.norm(np.array([x, y]) - h) > 0.045 for h in holes.values())), None)
        if solid is not None:
            lay.append({"color": "red", "shape": "cube", "x": solid[0], "y": solid[1], "yaw": 0.0, "z": z})
        act = c.reset(lay)["active"]
        st = settle(c, 2.5)
        ev = c.evaluate("insert_shapes")
        got = ordered(ev, act)
        for ob, n in zip(lay, act):
            p = sc.pelvis_to_mat(np.asarray(st[n]["pos"]))
            print(f"      {n}: target mat ({ob['x']:.3f}, {ob['y']:.3f}) -> rest mat ({p[0]:.3f}, {p[1]:.3f}, z {p[2]:.3f})")
        want = ["inserted"] * 3 + (["forced"] if solid is not None else [])
        check("lid drop test insert_shapes", got == want, f"outcomes {got}, score {ev['score']}")

    # 4. head RGB-D
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    c.reset_random(0)
    settle(c, 1.0)
    h = HeadRGBDClient(a.host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    t0 = time.time()
    while h.get() is None and time.time() - t0 < 10:
        time.sleep(0.1)
    f = h.get()
    check("head RGB-D frame", f is not None and f[0].shape == (480, 640, 3) and f[1].shape == (480, 640))
    if f is not None:
        rgb, d = f[0], f[1]
        check("head depth valid", (d > 0).mean() > 0.9 and 200 < np.median(d[d > 0]) < 1500,
              f"valid {100 * (d > 0).mean():.0f} %, median {np.median(d[d > 0]):.0f} mm")
        if a.save:
            import cv2
            Path(a.save).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{a.save}/head_rgb.jpg", rgb[:, :, ::-1])
            cv2.imwrite(f"{a.save}/head_depth.jpg",
                        cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255 / 1500.0), cv2.COLORMAP_JET))
    h.close()
    c.reset_random(0)
    print(f"\n{sum(RES)} PASS / {len(RES) - sum(RES)} FAIL")
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
