"""Reach + head-camera view map of the counter: is every cube/bin position both SEEN and REACHABLE?

    python -m aaltonomous.scene.check_layout [--samples 40000] [--origin X Y Z]

Prints an ASCII top view of the mat (robot at the bottom, robot's left on the left, 2.5 cm cells):
  '#' seen + reachable   'r' reachable only   'v' seen only   '.' neither
  overlaid with  C = cube region, B = bin footprint.
Reach = some random arm configuration (within URDF limits, either arm) puts the palm frame within
--tol of a point --above m above the cell (approximate grasp from above; Dex3 fingers add a few cm).
Uses scene.yaml (counter pose) and interface/calibration.yaml (head camera) unless --origin overrides the
mat origin, so a candidate counter position can be tried before editing scene.yaml.
"""
from __future__ import annotations

import argparse

import numpy as np

from aaltonomous.interface.calibration import URDF, Calibration

from .scene import Scene

ARMS = {s: [f"{s}_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                       "wrist_roll", "wrist_pitch", "wrist_yaw")] for s in ("left", "right")}


def palm_cloud(n, seed=0, waist_q=(0.0, 0.0, 0.0)):
    import pinocchio as pin
    m = pin.buildModelFromUrdf(str(URDF))
    d = m.createData()
    rng = np.random.default_rng(seed)
    out = []
    for side, joints in ARMS.items():
        idx = [m.joints[m.getJointId(j)].idx_q for j in joints]
        fid = m.getFrameId(f"{side}_hand_palm_link")
        wid = [m.joints[m.getJointId(j)].idx_q for j in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
        for _ in range(n // 2):
            q = pin.neutral(m)
            q[wid] = waist_q
            q[idx] = rng.uniform(m.lowerPositionLimit[idx], m.upperPositionLimit[idx])
            pin.framesForwardKinematics(m, d, q)
            out.append(d.oMf[fid].translation.copy())
    return np.array(out)


def reach_map(sc, cloud, pts_pelvis, above, tol):
    from scipy.spatial import cKDTree
    tree = cKDTree(cloud)
    q = pts_pelvis + np.array([0, 0, above])
    dist, _ = tree.query(q)
    return dist < tol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--above", type=float, default=0.08, help="palm height above the cube centre (m)")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--scene", help="alternative scene yaml (default scene.yaml)")
    ap.add_argument("--origin", type=float, nargs=3, help="override mat_origin_in_pelvis (m)")
    a = ap.parse_args(argv)
    sc, cal = Scene.load(a.scene), Calibration.load()
    if a.origin:
        sc.cfg["counter"]["mat_origin_in_pelvis"] = list(a.origin)
    W, D = sc.cfg["counter"]["mat_size"]
    cell = 0.025
    xs = np.arange(cell / 2, W, cell)
    ys = np.arange(cell / 2, D, cell)
    X, Y = np.meshgrid(xs, ys)
    P = sc.mat_to_pelvis(np.stack([X, Y, np.full_like(X, sc.cube_size / 2)], -1).reshape(-1, 3))
    seen = cal.visible("head", P).reshape(X.shape)
    reach = reach_map(sc, palm_cloud(a.samples), P, a.above, a.tol).reshape(X.shape)

    L, b = sc.cfg["layout"], sc.cfg["bins"]
    in_c = (X >= L["region_x"][0]) & (X <= L["region_x"][1]) & (Y >= L["region_y"][0]) & (Y <= L["region_y"][1])
    in_b = np.zeros_like(in_c)
    for cx, cy in b["centers"].values():
        hx, hy = np.asarray(b["outer_size"]) / 2
        in_b |= (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)

    sym = np.where(seen & reach, "#", np.where(reach, "r", np.where(seen, "v", ".")))
    print(f"mat origin in pelvis {sc.cfg['counter']['mat_origin_in_pelvis']}  (far edge at top, robot at bottom)")
    for i in range(len(ys) - 1, -1, -1):
        row = ""
        for j in range(len(xs) - 1, -1, -1):          # robot's left (large mat x) on the left
            s = sym[i, j]
            row += ("C" if s == "#" else "c") if in_c[i, j] else ("B" if s == "#" else "b") if in_b[i, j] else s
        print(f"y={ys[i] * 100:5.1f} " + row)
    print("C/B = cube region / bin cell seen+reachable; c/b = NOT both. Legend: # both, r reach, v seen, . neither")
    for name, m in (("cube region", in_c), ("bins", in_b)):
        ok = (seen & reach)[m].mean() if m.any() else float("nan")
        print(f"{name}: {ok * 100:.0f}% of cells seen + reachable")


if __name__ == "__main__":
    main()
