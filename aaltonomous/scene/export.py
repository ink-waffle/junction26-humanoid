"""Export printable parts and the counter mat from scene.yaml (M3 physical scene, M4 sim assets).

    python -m aaltonomous.scene.export [--out aaltonomous/scene/assets]

Writes:
  object_cube.stl       the 3 object shapes (extruded square, cylinder, equilateral triangle prism) with 2-3 mm
  object_cylinder.stl   fillets, all derived from objects.size in scene.yaml; print copies_per_type per colour in
  object_triangle.stl   matte PLA (mm units, as slicers expect). Origin: centroid on the bottom face.
  lid_shapes.stl        rung-3 lid with circle / square / triangle holes (objects + lid.hole_clearance); origin at lid
                        centre, bottom face z=0. Sits on the rim of the bin named in scene.yaml lid.bin.
  bin.stl               open bin (floor + 4 walls), print 3 or buy boxes of the same inner size
  mat_texture.png       counter mat texture for sim (1 px = 1 mm), identical to the print
  mat_print.pdf         the same mat at true scale (print at 100 %, no "fit to page")
  scene_sheet.txt       measurement sheet: what to measure/place at the venue so the scene is rebuilt identically
Sim (Isaac Lab) can import the STLs directly (metres = mm / 1000) or convert them to USD.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .scene import HERE, SHAPES, Scene, in_convex, shape_polygon


# ------------------------------------------------------------------ meshes
def box_tris(lo, hi):
    """12 outward-facing triangles of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [v[list(t)] for t in f]


def write_stl(path, tris, name):
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / (np.linalg.norm(n) or 1.0)
            fh.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for p in t:
                fh.write(f"      vertex {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write(f"endsolid {name}\n")


def cube_tris(sc):
    """Sharp (unfilleted) cube, 12 triangles; kept for simple collision proxies. Printable STLs: object_tris()."""
    s = sc.cube_size * 1000
    return box_tris((-s / 2, -s / 2, 0), (s / 2, s / 2, s))


def _ring(sc_dims, shape, inset, n_arc=8):
    """Outline (M,2) of the rounded shape inset by `inset` (m->mm done by caller units): erode by the fillet r, then
    dilate by r - inset (rounded corners). Same vertex count for every inset so rings can be lofted."""
    r, a = sc_dims["fillet"], sc_dims["apothem"]
    rho = max(r - inset, 0.02)   # mm; >0 keeps ring vertices distinct after STL rounding
    if shape == "cylinder":
        ang = np.linspace(0, 2 * np.pi, 4 * n_arc * 2, endpoint=False)
        return np.stack([(a - inset) * np.cos(ang), (a - inset) * np.sin(ang)], -1)
    n = 4 if shape == "cube" else 3
    a0 = np.pi / 4 if shape == "cube" else np.pi / 2
    pts = []
    for i in range(n):
        th = a0 + 2 * np.pi * i / n
        v = (a - r) / np.cos(np.pi / n) * np.array([np.cos(th), np.sin(th)])
        phi = np.linspace(th - np.pi / n, th + np.pi / n, n_arc)
        pts += [v + rho * np.array([np.cos(f), np.sin(f)]) for f in phi]
    return np.array(pts)


def object_tris(sc, shape, n_prof=5):
    """Filleted object, mm. Lofted rings: quarter-circle fillet profile at bottom and top, straight sides between."""
    d = {k: (v * 1000 if k != "fillet" else v * 1000) for k, v in sc.object_dims(shape).items()}
    r, h = d["fillet"], d["height"]
    t = np.linspace(0, np.pi / 2, n_prof)
    prof = [(r * (1 - np.sin(x)), r * (1 - np.cos(x))) for x in t]               # (inset, z) bottom, z 0..r
    prof += [(i, h - z) for i, z in reversed(prof)]                               # top, z h-r..h
    rings = [np.c_[_ring(d, shape, i), np.full(len(_ring(d, shape, i)), z)] for i, z in prof]
    tris = []
    for A, B in zip(rings[:-1], rings[1:]):
        m = len(A)
        for k in range(m):
            k2 = (k + 1) % m
            tris += [np.array([A[k], A[k2], B[k2]]), np.array([A[k], B[k2], B[k]])]
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        c = np.array([0.0, 0.0, ring[0, 2]])
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            tris.append(np.array([c, b, a]) if flip else np.array([c, a, b]))
    return tris


def lid_tris(sc, step=2.0):
    """Lid plate with the three holes, mm, origin at the lid centre, bottom at z=0. Delaunay of a grid plus densified
    boundaries, triangles inside holes dropped (hole edges are accurate to well under 1 mm)."""
    from scipy.spatial import Delaunay
    b = sc.cfg["bins"]
    X, Y = np.asarray(b["outer_size"]) * 500
    th = sc.cfg["lid"]["thickness"] * 1000
    cx, cy = b["centers"][sc.lid_bin]
    holes = [(h["poly"] - np.array([cx, cy])) * 1000 for h in sc.lid_holes()]
    pts = [np.array([x, y]) for x in np.arange(-X, X + 1e-6, step) for y in np.arange(-Y, Y + 1e-6, step)]
    for poly in holes:
        for a, c in zip(poly, np.roll(poly, -1, axis=0)):
            for u in np.linspace(0, 1, max(int(np.linalg.norm(c - a) / 0.5), 2), endpoint=False):
                pts.append(a + u * (c - a))
    pts = np.array(pts)
    keep = ~np.array([any(in_convex(p, q) and min_edge_dist(p, q) > 0.4 for p in holes) for q in pts])
    pts = pts[keep]
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    tri = tri[~np.array([any(in_convex(p, q) for p in holes) for q in cen])]
    edges = {}
    out = []
    for t in tri:
        a, b_, c = pts[t]
        if (b_ - a)[0] * (c - a)[1] - (b_ - a)[1] * (c - a)[0] < 0:
            t = t[[0, 2, 1]]
        for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(i, j)] = edges.get((i, j), 0) + 1
        P = [np.r_[pts[i], 0.0] for i in t]
        out.append(np.array([P[0], P[2], P[1]]))                                   # bottom, faces -z
        out.append(np.array([P[0] + [0, 0, th], P[1] + [0, 0, th], P[2] + [0, 0, th]]))  # top
    for (i, j) in edges:
        if (j, i) not in edges:                                                    # boundary edge: add a side wall
            a0, b0 = np.r_[pts[i], 0.0], np.r_[pts[j], 0.0]
            up = np.array([0, 0, th])
            out += [np.array([a0, b0, b0 + up]), np.array([a0, b0 + up, a0 + up])]
    return out


def min_edge_dist(poly, q):
    e = np.roll(poly, -1, axis=0) - poly
    d = q - poly
    t = np.clip((d * e).sum(1) / (e * e).sum(1), 0, 1)
    return float(np.min(np.linalg.norm(d - t[:, None] * e, axis=1)))


def bin_tris(sc):
    """Floor plus four non-overlapping walls (separate shells; slicers merge them). mm, origin at floor centre."""
    b = sc.cfg["bins"]
    X, Y = (np.asarray(b["outer_size"]) * 1000) / 2
    t, ft, h = b["wall_thickness"] * 1000, b["floor_thickness"] * 1000, b["wall_height"] * 1000
    parts = [((-X, -Y, 0), (X, Y, ft)),                      # floor
             ((-X, -Y, ft), (X, -Y + t, h)), ((-X, Y - t, ft), (X, Y, h)),            # front/back walls
             ((-X, -Y + t, ft), (-X + t, Y - t, h)), ((X - t, -Y + t, ft), (X, Y - t, h))]  # side walls
    return [tri for lo, hi in parts for tri in box_tris(lo, hi)]


# ------------------------------------------------------------------ mat
def mat_image(sc, px_per_m=1000):
    """RGB uint8 mat texture as seen from ABOVE with the robot at the bottom edge facing up the image:
    row 0 = far edge (mat y max), last column = mat x = 0 (robot's right). Drawn in mat x order, then flipped."""
    c = sc.cfg["counter"]
    W, D = c["mat_size"]
    w, d = int(round(W * px_per_m)), int(round(D * px_per_m))
    img = np.full((d, w, 3), 205, np.uint8)            # light matte grey: cubes and bins stand out
    rng = np.random.default_rng(0)                      # fixed fine texture so depth/RGB aren't uniform
    img = np.clip(img.astype(int) + rng.integers(-6, 7, (d, w, 1)), 0, 255).astype(np.uint8)

    def to_px(x, y):  # mat metres -> (row, col)
        return int(round((D - y) * px_per_m)), int(round(x * px_per_m))

    pitch = c["grid_cm"] / 100
    for i in range(int(W / pitch) + 1):                 # grid lines every grid_cm, darker every 10 cm
        col = min(int(round(i * pitch * px_per_m)), w - 1)
        img[:, max(col - 1, 0):col + 1] = 120 if (i * c["grid_cm"]) % 10 == 0 else 165
    for j in range(int(D / pitch) + 1):
        row = min(int(round((D - j * pitch) * px_per_m)), d - 1)
        img[max(row - 1, 0):row + 1, :] = 120 if (j * c["grid_cm"]) % 10 == 0 else 165

    b = sc.cfg["bins"]                                  # bin footprints as 4 px outlines in the bin colour
    for name, (cx, cy) in b["centers"].items():
        rgb = sc.bin_rgb
        hx, hy = np.asarray(b["outer_size"]) / 2
        r0, c0 = to_px(cx - hx, cy + hy)
        r1, c1 = to_px(cx + hx, cy - hy)
        for sl in (np.s_[r0:r0 + 4, c0:c1], np.s_[r1 - 4:r1, c0:c1], np.s_[r0:r1, c0:c0 + 4], np.s_[r0:r1, c1 - 4:c1]):
            img[sl] = rgb
    L = sc.cfg["layout"]                                # cube region as a dashed dark outline
    r0, c0 = to_px(L["region_x"][0], L["region_y"][1])
    r1, c1 = to_px(L["region_x"][1], L["region_y"][0])
    dash = (np.arange(max(c1 - c0, r1 - r0)) // 15) % 2 == 0
    cols, rows = c0 + np.flatnonzero(dash[:c1 - c0]), r0 + np.flatnonzero(dash[:r1 - r0])
    for k in range(3):
        img[r0 + k, cols] = img[r1 - 1 - k, cols] = 60
        img[rows, c0 + k] = img[rows, c1 - 1 - k] = 60
    return np.ascontiguousarray(img[:, ::-1])


def write_mat(sc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = mat_image(sc)
    plt.imsave(out / "mat_texture.png", img)
    W, D = sc.cfg["counter"]["mat_size"]
    fig = plt.figure(figsize=(W / 0.0254, D / 0.0254))   # true scale
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(W * 100, 0, 0, D * 100), interpolation="nearest")  # x grows to the left
    g = sc.cfg["counter"]["grid_cm"]
    for x in range(0, int(W * 100) + 1, 10):
        ax.text(x - 0.3, 0.4, f"{x}", ha="right" if x == 0 else "left", fontsize=9, color="0.25")
    for y in range(10, int(D * 100) + 1, 10):
        ax.text(0.5, y + 0.3, f"{y}", ha="right", fontsize=9, color="0.25")
    for name, (cx, cy) in sc.cfg["bins"]["centers"].items():
        ax.text(cx * 100, cy * 100, f"{name.upper()} BIN", ha="center", va="center", fontsize=12, color="0.3")
    ax.text(W * 50, 1.5, f"ROBOT SIDE (robot faces up the page)  <-  x towards robot's left (cm), y away from robot (cm), grid {g} cm",
            ha="center", fontsize=10, color="0.25")
    ax.set_axis_off()
    fig.savefig(out / "mat_print.pdf")
    plt.close(fig)


def scene_sheet(sc):
    c, b, L = sc.cfg["counter"], sc.cfg["bins"], sc.cfg["layout"]
    o = c["mat_origin_in_pelvis"]
    lines = [
        "AALTONOMOUS SCENE SHEET (generated from aaltonomous/scene/scene.yaml)",
        "",
        f"Mat: {c['mat_size'][0] * 100:.0f} x {c['mat_size'][1] * 100:.0f} cm, printed at 100 %, taped flat, "
        f"grid {c['grid_cm']} cm.",
        f"Counter top: {c['counter_size'][0] * 100:.0f} x {c['counter_size'][1] * 100:.0f} cm, mat centred on it.",
        "Mat origin = mat corner on the ROBOT'S RIGHT at the edge NEAREST the robot.",
        f"  measure: mat origin in the pelvis frame (x fwd, y left, z up): ({o[0] * 100:.0f}, {o[1] * 100:.0f}, "
        f"{o[2] * 100:.0f}) cm  (default -- measure and write back into scene.yaml)",
        f"  => mat near edge {o[0] * 100:.0f} cm in front of the pelvis, counter surface "
        f"{-o[2] * 100:.0f} cm below the pelvis.",
        "",
        f"Bins ({b['outer_size'][0] * 100:.0f} x {b['outer_size'][1] * 100:.0f} cm, walls "
        f"{b['wall_height'] * 100:.0f} cm), centred on the printed outlines:",
    ]
    lines += [f"  {n:<7} x={cx * 100:5.1f} cm  y={cy * 100:5.1f} cm  (identical, RGB {tuple(b['rgb'])})"
              for n, (cx, cy) in b["centers"].items()]
    o, lid = sc.cfg["objects"], sc.cfg.get("lid")
    lines += ["",
              f"Objects: {len(o['shapes'])} shapes x {len(o['colors'])} colours ({', '.join(o['shapes'])}; "
              f"{', '.join(o['colors'])}), {o['copies_per_type']} per type. size parameter {sc.size * 1000:.0f} mm "
              f"(default -- measure the printed parts and write back into objects.size), height "
              f"{sc.object_height() * 1000:.0f} mm, fillet {o['fillet'] * 1000:.1f} mm; grasp width = size for all "
              f"shapes (triangle side {sc.object_dims('triangle')['side'] * 1000:.0f} mm).",
              (f"Rung-3 lid on the {lid['bin']} bin: {lid['thickness'] * 1000:.0f} mm plate on the rim, holes = object "
               f"+ {lid['hole_clearance'] * 1000:.0f} mm clearance (circle, square, triangle)." if lid else ""),
              f"Layout region (dashed): x {L['region_x'][0] * 100:.0f}-"
              f"{L['region_x'][1] * 100:.0f} cm, y {L['region_y'][0] * 100:.0f}-{L['region_y'][1] * 100:.0f} cm.",
              "", "Success rule: " + sc.rule_text()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "assets"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scene.load()
    for shape in sc.cfg["objects"]["shapes"]:
        write_stl(out / f"object_{shape}.stl", object_tris(sc, shape), shape)
    if sc.cfg.get("lid"):
        write_stl(out / "lid_shapes.stl", lid_tris(sc), "lid")
    write_stl(out / "bin.stl", bin_tris(sc), "bin")
    write_mat(sc, out)
    (out / "scene_sheet.txt").write_text(scene_sheet(sc))
    print(f"wrote {sorted(p.name for p in out.iterdir())} to {out}")


if __name__ == "__main__":
    main()
