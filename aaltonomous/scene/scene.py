"""Scene geometry, success checker and scoring, shared by sim, real resets, eval and the safety box (spec §2-§3).

    from aaltonomous.scene.scene import Scene
    sc = Scene.load()                              # scene.yaml next to this file
    sc.mat_to_pelvis([0.4, 0.33, 0.0])             # mat coords -> pelvis frame
    sc.classify_cube(pos_pelvis, speed, "red", mapping, shape="cube")   # -> "correct" | "wrong" | "counter" | "off"
    sc.score("sort_language", outcomes)            # spec §3 points

The checker is written once here; Isaac Lab (WP3), the eval (WP4) and the staff scoring sheet all call it.
On the real robot, staff decide the outcome per cube by eye using the same rule (see rule_text()).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
OUTCOMES = ("correct", "wrong", "counter", "off")
INSERT_OUTCOMES = ("inserted", "forced", "dropped", "wrong_hole", "counter", "off")
SHAPES = ("cube", "cylinder", "triangle")
N_CIRCLE = 64                                      # polygon vertices used for the circle outline


def shape_polygon(shape, apothem, rot=0.0):
    """Convex polygon (N, 2) CCW, centred on the centroid, with the given inradius (apothem).
    cube: square, sides axis-aligned. triangle: equilateral, apex towards +y (rot rotates it). cylinder: N_CIRCLE-gon."""
    n, a0 = {"cube": (4, np.pi / 4), "triangle": (3, np.pi / 2), "cylinder": (N_CIRCLE, 0.0)}[shape]
    ang = a0 + rot + 2 * np.pi * np.arange(n) / n
    R = apothem / np.cos(np.pi / n)
    return R * np.stack([np.cos(ang), np.sin(ang)], -1)


def in_convex(poly, pt):
    """pt (2,) inside CCW convex polygon (boundary counts as inside)."""
    e = np.roll(poly, -1, axis=0) - poly
    d = np.asarray(pt, float) - poly
    return bool(np.all(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0] >= -1e-12))


def resolve_bin(mapping, color, shape=None):
    """Target bin for an object. mapping keys may be a colour, a shape, or (colour, shape); the most specific wins,
    then the optional "default" key ('everything else'). Rung 1/2-by-colour mappings need no shape."""
    for k in ((color, shape), color, shape, "default"):
        if k is not None and k in mapping:
            return mapping[k]
    return None


@dataclass
class Scene:
    cfg: dict

    def __post_init__(self):
        c = self.cfg
        if "objects" not in c and "cube" in c:      # old scene files (scene_v1_proposal.yaml): single cube section
            q = c["cube"]
            c["objects"] = {"size": q["size"], "height_ratio": 1.0, "fillet": 0.0025, "shapes": ["cube"],
                            "colors": q["colors"], "copies_per_type": q.get("copies_per_color", 2)}

    @classmethod
    def load(cls, path=None):
        with open(path or HERE / "scene.yaml") as f:
            return cls(yaml.safe_load(f))

    # ------------------------------------------------------------ frames
    @property
    def origin(self):
        return np.asarray(self.cfg["counter"]["mat_origin_in_pelvis"], float)

    def mat_to_pelvis(self, p):
        """(..., 2 or 3) mat coords -> (..., 3) pelvis frame. Mat x -> pelvis y, mat y -> pelvis x."""
        p = np.asarray(p, float)
        z = p[..., 2] if p.shape[-1] == 3 else np.zeros(p.shape[:-1])
        return self.origin + np.stack([p[..., 1], p[..., 0], z], axis=-1)

    def pelvis_to_mat(self, p):
        d = np.asarray(p, float) - self.origin
        return np.stack([d[..., 1], d[..., 0], d[..., 2]], axis=-1)

    @property
    def counter_z(self):
        """Counter (mat) surface height in the pelvis frame."""
        return float(self.origin[2])

    # ------------------------------------------------------------ geometry
    @property
    def cube_size(self):
        """Object size parameter (m): grasp width of every shape. Alias kept for older callers."""
        return float(self.cfg["objects"]["size"])

    size = cube_size

    @property
    def colors(self):
        return self.cfg["objects"]["colors"]

    @property
    def bin_rgb(self):
        """RGB (0-255) of every bin: all bins are identical, named only by position."""
        return list(self.cfg["bins"]["rgb"])

    def object_height(self, shape=None):
        o = self.cfg["objects"]
        return float(o["size"]) * float(o.get("height_ratio", 1.0))

    def object_dims(self, shape):
        """Per-shape dims derived from `size` (m). apothem = centre-to-flat distance; circumradius = centre to
        farthest point (before rounding); width = grasp width across flats (= size for every shape)."""
        s = self.size
        a = {"cube": s / 2, "cylinder": s / 2, "triangle": s / 3}[shape]   # triangle: altitude = size
        n = {"cube": 4, "cylinder": None, "triangle": 3}[shape]
        return {"apothem": a, "circumradius": a / np.cos(np.pi / n) if n else a, "width": s,
                "side": {"cube": s, "cylinder": s, "triangle": 2 * s / np.sqrt(3)}[shape],
                "height": self.object_height(shape), "fillet": float(self.cfg["objects"]["fillet"])}

    @property
    def bin_names(self):
        return tuple(self.cfg["bins"]["centers"])

    def bin_inner_box(self, name):
        """Inner volume of a bin in MAT coords: (lo xyz, hi xyz). z from the bin floor to the rim."""
        b = self.cfg["bins"]
        c = np.asarray(b["centers"][name], float)
        half = np.asarray(b["outer_size"], float) / 2 - b["wall_thickness"]
        lo = np.r_[c - half, b["floor_thickness"]]
        hi = np.r_[c + half, b["wall_height"]]
        return lo, hi

    def cube_spawn_mat(self, cube):
        """Layout entry {color,shape,x,y,yaw} -> object centre in mat coords (resting on the mat)."""
        return np.array([cube["x"], cube["y"], self.object_height(cube.get("shape")) / 2])

    # ------------------------------------------------------------ rung 3 lid
    @property
    def lid_bin(self):
        return self.cfg["lid"]["bin"]

    @property
    def lid_z(self):
        """(underside, top) of the lid plate in mat coords."""
        z = float(self.cfg["bins"]["wall_height"])
        return z, z + float(self.cfg["lid"]["thickness"])

    def lid_holes(self):
        """[{shape, center (mat xy), poly (N,2 mat xy)}]: hole = shape outline grown by hole_clearance/2 per side.
        Layout (mat axes): circle and square side by side on the bin's far (+y) half, triangle (apex towards the robot,
        -y) in front of them; asserted to fit the lid and not to overlap."""
        b, c = self.cfg["bins"], self.cfg["lid"]
        cx, cy = b["centers"][c["bin"]]
        off = float(c["hole_clearance"]) / 2
        a = {k: self.object_dims(k)["apothem"] + off for k in SHAPES}
        H = min(b["outer_size"]) / 2 - b["wall_thickness"]          # inner half-width
        m, g = 0.005, 0.005                                         # edge margin, gap between holes
        yu = H - m - a["cube"]                                      # upper row centre
        xu = (H - m - a["cube"], H - m - a["cylinder"])
        xs = min(xu)
        t_base = yu - a["cube"] - g                                 # triangle base (+y side) must stay below this
        yt = t_base - a["triangle"]
        apex = yt - 2 * a["triangle"]
        holes = [("cylinder", (cx - xs, cy + yu), 0.0), ("cube", (cx + xs, cy + yu), 0.0),
                 ("triangle", (cx, cy + yt), np.pi)]                # rot pi: apex towards -y
        if apex < -H + m or 2 * xs - a["cube"] - a["cylinder"] < g:
            raise ValueError("lid holes do not fit: reduce objects.size or hole_clearance")
        return [{"shape": k, "center": np.array(p, float),
                 "poly": np.array(p, float) + shape_polygon(k, a[k], rot)} for k, p, rot in holes]

    def lid_hole_at(self, xy):
        """Name of the shape whose hole contains mat point xy, else None."""
        for h in self.lid_holes():
            if in_convex(h["poly"], xy):
                return h["shape"]
        return None

    def classify_insert(self, pos_pelvis, speed, shape, track_pelvis=None, beside_margin=0.05):
        """Rung 3 outcome for one object at episode end: one of INSERT_OUTCOMES.
          inserted   at rest below the lid, inside the lid bin, and it entered through the hole of its own shape
          wrong_hole same but it entered through another shape's hole (e.g. a cylinder through the square hole)
          forced     resting on / jammed in the lid (centre above the lid underside, over the lid footprint)
          dropped    on the counter or off it within `beside_margin` of the lid footprint
          counter    anything else on the counter; off = knocked off elsewhere
        track_pelvis (N,3 object centre positions over the episode, optional) identifies the hole used: the first
        sample after which the centre is below the lid underside inside the footprint. Without it an object below the
        lid is assumed to have used the matching hole (it can only enter through a hole, and the others rarely fit)."""
        p = self.pelvis_to_mat(pos_pelvis)
        b = self.cfg["bins"]
        cx, cy = b["centers"][self.lid_bin]
        hx, hy = np.asarray(b["outer_size"], float) / 2
        z_under, z_top = self.lid_z
        over = abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy
        near = abs(p[0] - cx) <= hx + beside_margin and abs(p[1] - cy) <= hy + beside_margin
        if not self.on_counter(p):
            return "dropped" if near else "off"
        resting = speed <= float(self.cfg["success"]["at_rest_speed"])
        lo, hi = self.bin_inner_box(self.lid_bin)
        if over and resting and lo[2] <= p[2] < z_under - 1e-9 and np.all(p[:2] >= lo[:2]) and np.all(p[:2] <= hi[:2]):
            used = shape
            if track_pelvis is not None:
                t = self.pelvis_to_mat(np.asarray(track_pelvis, float))
                below = np.flatnonzero(t[:, 2] < z_under)
                used = None
                if len(below):
                    k = max(below[0] - 1, 0) if below[0] > 0 else 0
                    used = self.lid_hole_at(t[k, :2]) or self.lid_hole_at(t[below[0], :2])
            return "inserted" if used == shape else "wrong_hole"
        if over:
            return "forced" if p[2] >= z_under - self.object_height(shape) / 2 else "counter"
        if near and p[2] < z_under:
            return "dropped"
        return "counter"

    # ------------------------------------------------------------ success / scoring
    def bin_of(self, p_mat):
        """Name of the bin whose inner volume contains the point (mat coords), else None."""
        m = float(self.cfg["success"]["inside_margin"])
        for name in self.bin_names:
            lo, hi = self.bin_inner_box(name)
            if np.all(p_mat[:2] >= lo[:2] + m) and np.all(p_mat[:2] <= hi[:2] - m) and lo[2] <= p_mat[2] <= hi[2]:
                return name
        return None

    def on_counter(self, p_mat):
        c = self.cfg["counter"]
        mat = np.asarray(c["mat_size"], float)
        size = np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2  # counter is centred on the mat
        inside = np.all(p_mat[:2] >= lo) and np.all(p_mat[:2] <= hi)
        return bool(inside and p_mat[2] > -float(self.cfg["success"]["off_counter_drop"]))

    def classify_cube(self, pos_pelvis, speed, color, mapping, shape=None):
        """Outcome for one object at episode end (rungs 1-2). pos in the pelvis frame (m), speed (m/s) of its centre,
        mapping = colour -> bin name (tasks.make_instruction), or keyed by shape / (colour, shape) / "default" for
        shape-based instructions (see resolve_bin). Rung 1 ignores shape (its mapping is by colour)."""
        p = self.pelvis_to_mat(pos_pelvis)
        if not self.on_counter(p):
            return "off"
        b = self.bin_of(p)
        if b is None or speed > float(self.cfg["success"]["at_rest_speed"]):
            return "counter"
        return "correct" if resolve_bin(mapping, color, shape) == b else "wrong"

    def score(self, task, outcomes):
        """Episode score (spec §3). outcomes: iterable of OUTCOMES, one per cube."""
        table = self.cfg["scoring"][task]
        outcomes = list(outcomes)
        bad = set(outcomes) - set(INSERT_OUTCOMES if task == "insert_shapes" else OUTCOMES)
        if bad:
            raise ValueError(f"unknown outcomes {bad}")
        return sum(table[o] for o in outcomes)

    def rule_text(self):
        s = self.cfg["success"]
        return (f"An object counts as IN a bin if its centre is inside the bin's inner walls, below the rim, and it is "
                f"at rest (< {s['at_rest_speed'] * 100:.0f} cm/s) at episode end. Off the counter (or more than "
                f"{s['off_counter_drop'] * 100:.0f} cm below its surface) = knocked off. Anything else = left on counter. "
                f"Rung 3: +3 if it rests inside the lid bin having gone through the hole of its own shape; -1 if it "
                f"is forced/jammed on the lid or dropped beside it; 0 otherwise.")

    # ------------------------------------------------------------ layout + safety helpers
    def layout_kwargs(self):
        """Arguments for tasks.sample_layout() matching this scene."""
        L = self.cfg["layout"]
        return dict(n_min=L["n_cubes"][0], n_max=L["n_cubes"][1],
                    region=(tuple(L["region_x"]), tuple(L["region_y"])), min_spacing=L["min_spacing"],
                    yaw_range=tuple(np.radians(L["yaw_range_deg"])))

    def suggested_workspace(self, z_margin=0.02, xy_margin=0.10):
        """Pelvis-frame box for SafetyConfig.workspace_{min,max}: over the counter, wrist/palm frames kept
        z_margin above the mat. Palm-frame-to-fingertip offset is NOT included; tune z_margin on the robot."""
        c = self.cfg["counter"]
        mat = np.asarray(c["mat_size"], float)
        lo = self.mat_to_pelvis([0.0, 0.0, 0.0])
        hi = self.mat_to_pelvis([mat[0], mat[1], 0.0])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        wmin = (0.0, lo[1] - xy_margin, self.counter_z + z_margin)
        wmax = (hi[0] + xy_margin, hi[1] + xy_margin, 0.60)
        return tuple(float(v) for v in wmin), tuple(float(v) for v in wmax)
