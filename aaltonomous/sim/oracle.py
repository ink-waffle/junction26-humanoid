"""Scripted oracle for the Aalto-* sim (spec WP3): privileged object poses from the scene service (:55580) instead
of perception, the classical baseline's IK + grasp planner (aaltonomous/baselines/classical, imported, not edited),
commanded through AaltoEnv(backend="sim") + the safety filter like any policy (30 Hz absolute joint targets).

    from aaltonomous.sim.oracle import OraclePolicy
    pol = OraclePolicy(SceneClient())               # threaded: holds while it plans, never trips the watchdog
    pol.reset({"instruction": obs["instruction"]})  # rung 2: mapping parsed from the instruction (training templates)
    act = pol.act(obs)

What it adds over the baseline (all privileged-state only, no camera):
  * detections = active objects resting on the counter (not in a bin), exact centre + shape,
  * per-object target bin via scene.resolve_bin (colour, shape or (colour, shape) keys),
  * relay: an object one arm can grasp but cannot carry to its bin (cross-body) is put down at a free relay
    point near the centre line that the other arm can reach; it is sorted on the next look.
  * faster, slightly looser motion (ORACLE_OVERRIDES), still pre-checked against the safety filter.

Run episodes / scores with aaltonomous/sim/evaluate.py (--policy oracle). Rung 2: --task sort_language.
"""
from __future__ import annotations

import copy
import itertools
import threading
import time

import numpy as np

from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.planner import Plan, Planner
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.interface import tasks
from aaltonomous.interface.safety import JointSpec
from aaltonomous.sim.aalto_task.layout import ARM_HOME
from aaltonomous.scene.scene import Scene, resolve_bin

# Planner overrides (classical config.yaml keys). Defaults leave ~25 % of layout objects unplannable in offline
# tests (3 cm transit clearance, 0.4 rad IK step); these give ~75 % direct + relay for most of the rest.
ORACLE_OVERRIDES = {
    "motion.counter_clearance": 0.02,
    "motion.max_joint_step": 0.6,
    "ik.max_tilt_deg": 35,
    "motion.arm_max_vel": 0.97,    # safety filter allows 1.0
    "motion.arm_max_acc": 8.0,
    "motion.hand_max_vel": 2.5,    # filter allows 3.0
    "motion.close_hold_s": 0.25,
    "motion.settle_s": 0.3,
    "ik.random_restarts": 4,
    # Pinch thumb vs ONE finger: the Dex3 index/middle tips are 57 mm apart (URDF), so with the object centred
    # between them (classical `auto` grasp point) a 45 mm object wedges between the two fingers and is never released
    # (carried back home; seen with cylinders). Shift the grasp point 15 mm sideways in the palm (palm z).
    "grasp.point_in_palm": [0.099, -0.084, 0.015],
    "hand.open": [0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4],    # release = preshape (thumb back + up)
    "motion.open_hold_s": 0.4,
    "grasp.arm": "by_y",          # first valid arm (preferred side first) instead of planning both: ~2x faster looks
    # Sim home = the task's arm home pose (layout.ARM_HOME, spec "fixed home pose"). The classical home_arm_q
    # (wrist pitch 1.0, arms out to the sides) collides hand-vs-body in the sim: wrist pitch stalls at ~0.6 and
    # wrist yaw is deflected by ~0.7 rad, which ruins every grasp. The oracle needs no camera view, so no conflict.
    "home_arm_q": [ARM_HOME[n] for n in JointSpec().arm_names],
}
# Relay candidates (pelvis xy, m), tried in order; must be >= RELAY_CLEAR from every other object centre.
RELAY_POINTS = [(0.30, 0.0), (0.30, 0.05), (0.30, -0.05), (0.27, 0.0), (0.34, 0.0), (0.27, 0.06), (0.27, -0.06)]
RELAY_CLEAR = 0.075
MAX_RELAYS_PER_OBJECT = 1
MAX_RELAY_POINTS_TRIED = 3        # per object and look (each try = 2 full plans, ~1-3 s)
RELAY_BUDGET_S = 10.0             # per look
# Drop points inside a bin (pelvis xy offsets from its centre), cycled by how many objects are already in it, so
# the next object does not land on top of the previous one (a stacked object pokes above the rim = not "in").
DROP_OFFSETS = [(0.0, 0.0), (0.03, 0.03), (-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03)]


# Face-aligned grasp: object yaw (pelvis) -> outward face normals at yaw + offset + k * 2pi/n (mesh frame of
# scene/export.py: cube corners at 45 deg, triangle apex along +y so its faces point at -90, +30, +150 deg).
import os as _os
TRI_PHI_DEG = float(_os.environ.get("AALTO_TRI_PHI", "105"))  # finger direction vs apex (deg); grasp_test: 105 best
# (4/6), 90 3/6 + 1/6, 60 = fingers on a face (worst), 30 1/6, 0 0/6 -- noisy, see ORGANISER_NOTES.md
FACE_ALIGN = {"cube": (0.0, 4), "triangle": (np.pi / 2 + np.radians(TRI_PHI_DEG), 3)}
FINGER_REF_DEG = {"left": -25.0, "right": 25.0}   # natural finger direction (forward, slightly inward)
FACE_MAX_DEV_DEG = 75.0           # faces further than this from the natural direction are not tried
FACE_HINT_WEIGHT = 3.0            # IK weight of the finger direction (stock 0.3 = a weak preference)
FACE_TOL_DEG = 12.0               # achieved finger direction must be this close to the face normal
BIN_MARGIN = 0.0                  # m around a bin's footprint where hand points must stay above the rim (stock 0.01:
                                  # the far object row is only ~3 cm from the centre bin wall, so a forward-pointing
                                  # hand was rejected for a few mm and those objects needed a 20 s relay)
SHORTCUT_TOL = 0.01               # m: a Cartesian segment may become one joint-space move if the grasp point stays this
                                  # close to the straight line (the per-waypoint IK lets wrist yaw wander: ~3x longer
                                  # joint paths, i.e. ~3x slower moves at the same joint speed limit)


def _tri_shift(phi, r=0.030):
    """Midpoint (m, along f) of the triangle's extent in direction f, f at angle phi from the apex direction; vertices
    at r from the centroid (apex 30 mm, faces 15 mm). phi=0 -> +7.5 mm (towards the apex), phi=60 deg -> -7.5 mm."""
    v = np.radians([0.0, 120.0, 240.0])
    return 0.5 * (r * np.cos(phi - v).max() - r * np.cos(phi + np.pi - v).max())


class _OraclePlanner(Planner):
    """Stock planner whose bin target can be offset inside the bin (drop_offsets: bin -> (dx, dy)), with shorter
    joint paths: carry = joint-space first (Cartesian as fallback), other Cartesian segments shortcut to one
    joint-space move when the grasp point stays within SHORTCUT_TOL of the line. All of the stock planner's
    clearance + safety dry-run checks still run on the final trajectory."""
    drop_offsets: dict = {}

    def bin_xy(self, name):
        return super().bin_xy(name) + np.asarray(self.drop_offsets.get(name, (0.0, 0.0)), float)

    def _cartesian(self, side, p0, p1, q0, name, plan):
        if name == "carry":
            wall = float(self.scene.cfg["bins"]["wall_height"])
            z_min = self.scene.counter_z + wall + self.height / 2 + self.cfg["grasp"]["drop_above_rim"]
            reason = plan.reason
            out = self._joint_carry(side, q0, p0, p1, z_min, "carry", plan)
            if out is not None:
                return out
            plan.reason = reason
        qs = super()._cartesian(side, p0, p1, q0, name, plan)
        if not qs or len(qs) < 2:
            return qs
        q0, q1 = np.asarray(q0, float), qs[-1]
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L2 = float(d @ d)
        for f in np.linspace(0, 1, 12)[1:-1]:
            p = self.kin.grasp_pose(side, q0 + f * (q1 - q0))[0]
            t = np.clip((p - p0) @ d / L2, 0, 1) if L2 > 0 else 0.0
            if np.linalg.norm(p - (p0 + t * d)) > SHORTCUT_TOL:
                return qs
        return [q1]

    # -------------------------------------------------------------- face-aligned grasp (cube, triangle)
    def bin_clearance(self, side, arm_q, hand_q, margin=None):
        return super().bin_clearance(side, arm_q, hand_q, margin=BIN_MARGIN if margin is None else margin)

    def _reaches_bin(self, side, bin_name):
        """Cheap reject (cached): can this arm put its grasp point above the bin at carry height at all? The stock
        planner otherwise spends 10-30 s per cross-body attempt in the joint-space carry search."""
        cache = self.__dict__.setdefault("_reach", {})
        key = (side, bin_name)
        if key not in cache:
            g, sc = self.cfg["grasp"], self.scene
            z = sc.counter_z + max(g["carry_height"], float(sc.cfg["bins"]["wall_height"]) + self.height + 0.03)
            xy = super().bin_xy(bin_name)
            cache[key] = bool(self.kin.solve(side, np.r_[xy, z], self.home)[1])
        return cache[key]

    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        if kw.get("place_xy") is None and bin_name in self.scene.cfg["bins"]["centers"] \
                and not self._reaches_bin(side, bin_name):
            return Plan(False, side=side, color=det.color, reason=f"{side} arm cannot reach the {bin_name} bin")
        return self._plan_side_faces(det, side, bin_name, arm_q, hand_q, **kw)

    def _plan_side_faces(self, det, side, bin_name, arm_q, hand_q, **kw):
        """Cubes / triangles: point the fingers along the outward normal of one flat face (thumb on the opposite face
        / the triangle's apex), i.e. pinch across the object's grasp width, not across a corner. Faces tried nearest
        the side's natural finger direction first; the stock finger hints are the fallback."""
        shape, yaw = getattr(det, "shape", None), getattr(det, "yaw", None)
        if shape in FACE_ALIGN and yaw is not None:
            off, n = FACE_ALIGN[shape]
            ref = np.radians(FINGER_REF_DEG[side])
            cands = []
            for k in range(n):
                a = yaw + off + 2 * np.pi * k / n
                da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
                if da <= np.radians(FACE_MAX_DEV_DEG):
                    cands.append((da, a))
            saved = (self.kin.hint, self.kin.w_hint)
            try:
                for _, a in sorted(cands):
                    f = np.array([np.cos(a), np.sin(a), 0.0])
                    d2 = copy.copy(det)
                    if shape == "triangle":      # pinch midpoint along f is not the centroid (support function)
                        d2.top = det.top + _tri_shift(a - (yaw + np.pi / 2)) * f
                    self.kin.hint, self.kin.w_hint = f, FACE_HINT_WEIGHT
                    p = self._plan_side(d2, side, bin_name, arm_q, hand_q, **kw)
                    if p.ok:
                        err = self._finger_err(p, side, f)
                        if err <= FACE_TOL_DEG:
                            p.reason = f"face-aligned ({np.degrees(a):.0f} deg, err {err:.0f})"
                            return p
            finally:
                self.kin.hint, self.kin.w_hint = saved
        return super()._plan_side_hints(det, side, bin_name, arm_q, hand_q, **kw)

    def _finger_err(self, plan, side, f):
        """Angle (deg) between the palm x axis (finger direction) at the grasp and f, in the horizontal plane."""
        e = next(e for name, s, e in plan.phases if name == "descend")
        self.kin._fk(plan.arm[e - 1])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        a = np.arctan2(x[1], x[0]) - np.arctan2(f[1], f[0])
        return float(np.degrees(abs((a + np.pi) % (2 * np.pi) - np.pi)))


# ---------------------------------------------------------------------- rung-2 instruction parsing
_PARSE = None


def _parse_table():
    """Every training-template instruction (tasks.TEMPLATES x colour/bin permutations x bin phrasing) -> mapping."""
    table = {}
    for tpl in tasks.TEMPLATES:
        for colors in itertools.permutations(tasks.COLORS):
            for bins in itertools.permutations(tasks.BINS):
                for by_color in itertools.product((False, True), repeat=3):
                    w = {f"c{i}": c for i, c in enumerate(colors)}
                    w.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
                    w.update({f"b{i}": (tasks.BIN_COLOR[b] if by_color[i] else b) for i, b in enumerate(bins)})
                    table[tpl.format(**w)] = dict(zip(colors, bins))
    return table


def parse_instruction(text):
    """colour -> bin for the fixed sentence or any training-template sentence; None if it doesn't match
    (e.g. held-out templates: then the oracle falls back to the privileged mapping)."""
    global _PARSE
    if text == tasks.FIXED_INSTRUCTION:
        return {c: b for b, c in tasks.BIN_COLOR.items()}
    if _PARSE is None:
        _PARSE = _parse_table()
    m = _PARSE.get(text)
    return dict(m) if m else None


# ---------------------------------------------------------------------- privileged "perception"
class PrivilegedPerception:
    """Drop-in for classical Perception.detect(): Detections from the scene service, not the camera."""

    def __init__(self, scene: Scene, client):
        self.scene, self.client = scene, client

    def detect(self, rgb=None, depth=None, debug=False):
        dets = []
        for o in self.client.state()["objects"]:
            if not o["active"]:
                continue
            p = np.asarray(o["pos"], float)
            m = self.scene.pelvis_to_mat(p)
            if not self.scene.on_counter(m) or self.scene.bin_of(m) is not None:
                continue
            if p[2] > self.scene.counter_z + self.scene.object_height(o["shape"]) * 1.5:
                continue                                   # in a hand / in the air: not a pick candidate
            top = p + np.array([0.0, 0.0, self.scene.object_height(o["shape"]) / 2])
            q = o.get("quat_xyzw") or [0, 0, 0, 1]
            yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
            d = Detection(o["color"], top, 1000, 1000, "oracle", yaw=yaw)
            d.shape, d.name = o["shape"], o["name"]
            dets.append(d)
        return (dets, []) if debug else dets

    def masks(self, rgb):
        return {}


def _relay_scene(scene: Scene, cfg):
    """Scene copy whose 'bins' are the relay points at counter level, so the stock planner places (not drops) there:
    wall_height 0 -> release height = counter + object/2 + drop_above_rim."""
    cfg2 = copy.deepcopy(scene.cfg)
    b = cfg2["bins"]
    b["wall_height"] = 0.0
    b["centers"] = {}
    for i, (x, y) in enumerate(RELAY_POINTS):
        m = scene.pelvis_to_mat([x, y, 0.0])
        b["centers"][f"relay{i}"] = [float(m[0]), float(m[1])]
    return Scene(cfg2)


class OraclePolicy(ClassicalPolicy):
    def __init__(self, client, scene=None, overrides=None, privileged_mapping=None, threaded=True, log=print,
                 prefetch=True, **kw):
        ov = dict(ORACLE_OVERRIDES)
        ov.update(overrides or {})
        self.client = client
        self.prefetch = prefetch
        self._pre = None
        self._plan_lock = threading.Lock()     # pinocchio data + planner state are not thread-safe
        self.privileged_mapping = privileged_mapping     # callable -> mapping (e.g. lambda: env.mapping)
        super().__init__(scene=scene, overrides=ov, threaded=threaded, log=log, **kw)
        self.perception = PrivilegedPerception(self.scene, client)
        self.planner = _OraclePlanner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec,
                                      mapping=self.mapping)
        self.planner.drop_offsets = {}
        rcfg = copy.deepcopy(self.cfg)
        rcfg["grasp"]["drop_above_rim"] = 0.012
        self.relay = _OraclePlanner(rcfg, _relay_scene(self.scene, self.cfg), self.post, self.home, self.kin, self.safety,
                             self.spec)
        self.relays = {}
        self.mapping_source = "fixed"

    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = dict(info or {})
        if not info.get("mapping"):
            m = parse_instruction(info.get("instruction", tasks.FIXED_INSTRUCTION))
            self.mapping_source = "parsed"
            if m is None and self.privileged_mapping is not None:
                m, self.mapping_source = self.privileged_mapping(), "privileged"
            if m is None:
                m, self.mapping_source = {c: b for b, c in tasks.BIN_COLOR.items()}, "default"
            info["mapping"] = m
        else:
            self.mapping_source = "given"
        if getattr(self, "_pre", None) is not None:   # a prefetch from the previous episode may still run: drop it
            self._pre["cancel"] = True                  # (don't wait here: the env watchdog is already running;
                                                        # the next look waits for the plan lock in its thread)
        super().reset(info)
        self.relays = {}
        self._pre = None

    # ------------------------------------------------------------------ look + plan (privileged)
    def _start(self, plan, t):
        """Execute `plan` and, meanwhile, plan the next object from the pose this plan ends in (its last tick)."""
        tb = str(plan.target_bin or "")
        self._exec_bin = None if tb.startswith("relay") else tb
        out = super()._start(plan, t)
        if self.prefetch and plan.arm is not None and len(plan.arm):
            job = {"arm": plan.arm[-1].copy(), "hand": plan.hand[-1].copy(), "exclude": getattr(plan, "obj", None),
                   "result": None}
            job["thread"] = threading.Thread(target=self._prefetch_job, args=(job,), daemon=True)
            self._pre = job
            job["thread"].start()
        return out

    def _prefetch_job(self, job):
        try:
            with self._plan_lock:
                if job.get("cancel"):
                    return
                job["result"] = self._plan_next(job["arm"], job["hand"], exclude=job["exclude"], commit=False)
        except Exception as e:  # fall back to a normal look
            job["result"] = ("error", repr(e))

    def _look(self, obs):
        with self._plan_lock:
            return self._look_locked()

    def _look_locked(self):
        job, self._pre = getattr(self, "_pre", None), None
        if job is not None:
            res = job["result"]          # the job ran (or was skipped) before we got the lock
            if (res and res[0] == "plan" and np.allclose(job["arm"], self.arm_cmd, atol=1e-6)
                    and np.allclose(job["hand"], self.hand_cmd, atol=1e-6)):
                p = res[1]
                now = {d.name: d for d in self.perception.detect()}
                d = now.get(p.obj)
                if d is not None and np.linalg.norm(d.top[:2] - p.obj_xy) < 0.01:   # object still where planned
                    self._commit(p)
                    return ("plan", p, len(now))
        return self._plan_next(self.arm_cmd, self.hand_cmd)

    def _commit(self, p):
        if p.relay is not None:
            self.relays[p.obj] = self.relays.get(p.obj, 0) + 1

    def _update_drop_offsets(self, exclude=None):
        n = {b: 0 for b in self.scene.bin_names}
        for o in self.client.state()["objects"]:
            if o["active"]:
                b = self.scene.bin_of(self.scene.pelvis_to_mat(np.asarray(o["pos"], float)))
                if b is not None:
                    n[b] += 1
        if exclude is not None and getattr(self, "_exec_bin", None) in n:   # the object being carried now
            n[self._exec_bin] += 1
        self.planner.drop_offsets = {b: DROP_OFFSETS[k % len(DROP_OFFSETS)] for b, k in n.items()}

    def _plan_next(self, arm, hand, exclude=None, commit=True):
        self._update_drop_offsets(exclude)
        dets = [d for d in self.perception.detect() if d.name != exclude]
        self.last_detections = dets
        r = self.cfg["policy"]["same_object_radius"]
        cands = []
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < r for f in self.failed):
                continue
            if sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < r) >= self.cfg["policy"]["max_attempts"]:
                continue
            d.bin = resolve_bin(self.mapping, d.color, getattr(d, "shape", None))
            if d.bin is not None:
                cands.append(d)
        cands.sort(key=lambda d: np.linalg.norm(d.top[:2] - self.planner.bin_xy(d.bin)))
        unplanned = []
        for d in cands:
            p = self.planner.plan(d, arm, hand, bin_name=d.bin, relocate=False)   # the oracle relays instead
            if p.ok:
                p.obj, p.obj_xy, p.relay = d.name, d.top[:2].copy(), None
                return ("plan", p, len(dets))
            unplanned.append(d)
        t0 = time.monotonic()
        for d in unplanned:
            if time.monotonic() - t0 > RELAY_BUDGET_S:
                break
            p = self._plan_relay(d, dets, arm, hand)
            if p is not None:
                if commit:
                    self._commit(p)
                return ("plan", p, len(dets))
            if commit:
                self.failed.append(d.top[:2])
                self.log(f"[oracle] skip {d.name}: no direct or relay plan")
        return ("none", None, len(dets))

    def _plan_relay(self, d, dets, arm, hand):
        if self.relays.get(d.name, 0) >= MAX_RELAYS_PER_OBJECT:
            return None
        others = [o.top[:2] for o in dets if o.name != d.name]
        tried = 0
        for i, xy in enumerate(RELAY_POINTS):
            if tried >= MAX_RELAY_POINTS_TRIED:
                break
            xy = np.asarray(xy, float)
            if np.linalg.norm(xy - d.top[:2]) < 0.04 or any(np.linalg.norm(xy - o) < RELAY_CLEAR for o in others):
                continue
            # second leg first (cheap reject): from the relay point to the bin, starting at home
            fake = Detection(d.color, np.r_[xy, d.top[2]], 1000, 1000, "relay")
            tried += 1
            p2 = self.planner.plan(fake, self.home, self.post["open"], bin_name=d.bin, relocate=False)
            if not p2.ok:
                continue
            p1 = self.relay.plan(d, arm, hand, bin_name=f"relay{i}", relocate=False)
            if p1.ok:
                p1.target_bin = f"relay -> {d.bin}"
                p1.obj, p1.obj_xy, p1.relay = d.name, d.top[:2].copy(), i
                self.log(f"[oracle] relay {d.name} via ({xy[0]:.2f},{xy[1]:+.2f}): {p1.side} then {p2.side}")
                return p1
        return None
