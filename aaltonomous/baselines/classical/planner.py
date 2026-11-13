"""Pick-and-place planning for one detected object: Cartesian waypoints -> IK -> 30 Hz joint trajectories.

Sequence (one arm; the other holds its current target):
  home -> pregrasp at carry height (joint space, hand to preshape) -> descend (Cartesian) -> close -> lift -> carry
  above the bin (Cartesian, else joint space direct / via a raised midpoint) -> lower to drop height -> release
  (hand.release) -> retreat up -> home (hand back to open)
Each side is tried with every grasp.finger_hints direction (fingers forward can touch the centre bin's wall).
Cross-body (neither arm can do object -> bin): a RELOCATION plan picks with one arm and puts the object down on
the counter at a spot from which the bin-side arm has a full plan; the policy's next look picks it up from there.
Every planned tick is checked before it is used:
  * per-joint speed <= motion.arm_max_vel (trapezoidal time scaling on the inf-norm path length),
  * hand points (palm, grasp point, finger links) over the counter footprint stay above the counter
    (counter_clearance in transit; fingertip_clearance during the grasp itself),
  * hand points over a bin footprint (+1 cm) stay above its rim, except while releasing,
  * a dry run through a fresh SafetyFilter (same config as the env) produces 0 clipped ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter
from aaltonomous.interface.tasks import FIXED_MAPPING

from .kinematics import ArmKinematics


@dataclass
class Plan:
    ok: bool
    side: str | None = None
    color: str = ""
    reason: str = ""
    target_bin: str | None = None
    waypoints: dict = field(default_factory=dict)      # name -> grasp point (pelvis)
    ik: dict = field(default_factory=dict)             # name -> (ok, pos_err, tilt_deg)
    arm: np.ndarray | None = None                      # (T, 14)
    hand: np.ndarray | None = None                     # (T, 14)
    phases: list = field(default_factory=list)         # (name, start tick, end tick)
    safety_clips: int | None = None

    @property
    def duration(self):
        return 0.0 if self.arm is None else len(self.arm) / 30.0

    def summary(self):
        if not self.ok:
            return f"{self.color:<6} SKIP ({self.reason})"
        wp = " ".join(f"{k}=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for k, v in self.waypoints.items()
                      if k in ("grasp", "drop"))
        tilt = max(t for _, _, t in self.ik.values())
        err = max(e for _, e, _ in self.ik.values()) * 1000
        return (f"{self.color:<6} {self.side:<5} arm -> {self.target_bin} bin  {wp}  IK ok max_err={err:.1f}mm "
                f"max_tilt={tilt:.0f}deg  {self.duration:.1f}s  safety_clips={self.safety_clips}")


def time_scale(Qs, vmax, amax, dt):
    """Piecewise-linear joint path through Qs (n, d) -> samples (T, d) at dt, EXCLUDING Qs[0], with a trapezoidal
    profile on the inf-norm arc length: no joint ever moves faster than vmax."""
    Qs = np.asarray(Qs, float)
    seg = np.abs(np.diff(Qs, axis=0)).max(axis=1)
    S = float(seg.sum())
    if S < 1e-9:
        return np.empty((0, Qs.shape[1]))
    ta = vmax / amax
    if amax * ta * ta > S:                       # triangle profile
        ta = np.sqrt(S / amax)
        vpk = amax * ta
        T = 2 * ta
    else:
        vpk = vmax
        T = 2 * ta + (S - amax * ta * ta) / vmax
    n = max(1, int(np.ceil(T / dt)))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * amax * t ** 2,
                 np.where(t <= T - ta, 0.5 * amax * ta ** 2 + vpk * (t - ta), S - 0.5 * amax * (T - t) ** 2))
    s = np.clip(s, 0, S)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((n, Qs.shape[1]))
    for k, sk in enumerate(s):
        i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-12 else (sk - cum[i]) / seg[i]
        out[k] = Qs[i] + f * (Qs[i + 1] - Qs[i])
    return out


def ramp(a, b, vmax, dt):
    """Linear move a -> b (excluding a) with every component at most vmax."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.ceil(np.abs(b - a).max() / (vmax * dt) + 1e-9))
    if n == 0:
        return np.empty((0, len(a)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


class Planner:
    def __init__(self, cfg, scene, postures, home, kin: ArmKinematics | None = None, safety: SafetyConfig | None = None,
                 spec: JointSpec | None = None, mapping=None, dt=1.0 / 30.0):
        self.cfg, self.scene, self.post, self.home = cfg, scene, postures, np.asarray(home, float)
        self.spec = spec or JointSpec()
        self.kin = kin or ArmKinematics(cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=dt)
        self.dt = dt
        self.height = float(cfg["perception"].get("object_height") or scene.object_height())
        self.mapping = mapping or dict(FIXED_MAPPING)  # sort_fixed: fixed colour -> bin position

    # ------------------------------------------------------------------ geometry helpers
    def bin_xy(self, name):
        return self.scene.mat_to_pelvis(np.r_[self.scene.cfg["bins"]["centers"][name], 0.0])[:2]

    def drop_xy(self, name):
        """Release point (pelvis xy) in bin `name`: the point of the inner footprint nearest the pelvis, `grasp.drop_inset`
        inside the inner walls (the published box row puts the box centres past the arm's reach). None -> centre."""
        inset = self.cfg["grasp"].get("drop_inset")
        c = self.bin_xy(name)
        if inset is None:
            return c
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        lo2 = np.minimum(np.minimum(a, b) + inset, c)
        hi2 = np.maximum(np.maximum(a, b) - inset, c)
        return np.clip(np.zeros(2), lo2, hi2)

    def _over_counter(self, p):
        pm = self.scene.pelvis_to_mat(p)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        return np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)

    def bin_clearance(self, side, arm_q, hand_q, margin=0.01):
        """Lowest height (m) above the bin rims of the hand points over a bin's outer footprint (+margin), inf if none.
        Bins are static boxes in the sim and on the table: only the drop itself may go below a rim."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + margin
        rim = self.scene.counter_z + float(b["wall_height"])
        out = np.inf
        for c in b["centers"].values():
            over = np.all(np.abs(pm[:, :2] - np.asarray(c, float)) <= half, axis=1)
            if over.any():
                out = min(out, float((pts[over, 2] - rim).min()))
        return out

    def clearance(self, side, arm_q, hand_q, fingers_only=False):
        """Lowest height (m) above the counter surface of the hand points that are over the counter (inf if none)."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        if fingers_only:
            pts = pts[2:]
        over = self._over_counter(pts)
        if not over.any():
            return np.inf
        return float((pts[over, 2] - self.scene.counter_z).min())

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, side, p, seed, name, plan):
        q, ok, err, tilt = self.kin.solve(side, p, seed, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        plan.ik[name] = (ok, err, float(np.degrees(tilt)))
        plan.waypoints[name] = np.asarray(p, float)
        return q, ok

    def _cartesian(self, side, p0, p1, q0, name, plan):
        """IK waypoints along the straight line p0 -> p1 (excluding p0), seeded continuously. A step whose joints
        jump more than `max_joint_step` is bisected (up to 3 times) before it counts as a failure."""
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / self.cfg["motion"]["cart_step"])))
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qs, q, info = [], q0, [True, 0.0, 0.0]
        jmax = float(self.cfg["motion"].get("max_joint_step", 0.4))

        def seg(pa, pb, q, depth):
            qn, ok, err, tilt = self.kin.solve(side, pb, q, top_down=True, restarts=False)
            if ok and np.abs(qn - q).max() <= jmax:
                info[1], info[2] = max(info[1], err), max(info[2], float(np.degrees(tilt)))
                return [qn]
            if depth >= 3:
                plan.reason = plan.reason or (f"IK jump on {name}" if ok else "")
                return None
            pm = 0.5 * (pa + pb)
            a = seg(pa, pm, q, depth + 1)
            b = a and seg(pm, pb, a[-1], depth + 1)
            return a + b if b else None

        for k in range(1, n + 1):
            out = seg(p0 + (p1 - p0) * (k - 1) / n, p0 + (p1 - p0) * k / n, q, 0)
            if out is None:
                plan.ik[name] = (False, info[1], info[2])
                return None
            qs += out
            q = out[-1]
        plan.ik[name] = (True, info[1], info[2])
        plan.waypoints[name] = p1
        return qs

    def _joint_carry(self, side, q_from, p_from, p_to, z_min, name, plan):
        """Joint-space move to an IK solution at p_to (direct, else via a raised midpoint), accepted if the grasp
        point never dips below z_min on the way."""
        def low(qa, qb):
            return any(self.kin.grasp_pose(side, qa + f * (qb - qa))[0][2] < z_min for f in np.linspace(0, 1, 15))

        q_to, ok = self._ik(side, p_to, q_from, name, plan)
        if ok and not low(q_from, q_to):
            return [q_to]
        for dz in (0.06, 0.12):
            mid = 0.5 * (np.asarray(p_from, float) + p_to) + [0, 0, dz]
            q_mid, ok1 = self._ik(side, mid, q_from, name + "_mid", plan)
            if not ok1 or low(q_from, q_mid):
                continue
            q_to, ok2 = self._ik(side, p_to, q_mid, name, plan)
            if ok2 and not low(q_mid, q_to):
                plan.reason = ""
                return [q_mid, q_to]
        plan.ik[name] = (False, 0.0, 0.0)
        plan.reason = f"{name}: no joint-space path above the bin rim"
        return None

    # ------------------------------------------------------------------ main
    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        """Plan for Detection `det` starting from the commanded arm_q/hand_q (14,). Tries the preferred arm (the
        target bin's side; for the centre bin the object's side), then the other one. If neither arm can do the
        whole pick -> bin (cross-body: the arm that reaches the object cannot reach the bin), plans a RELOCATION
        instead (`grasp.relocate`): pick + place on the counter at a spot from which an arm can finish the job;
        the next look re-detects the object there. obstacles: (n, 2) pelvis xy of the other detected objects.
        Returns the first Plan that passes every check, else the last failure."""
        g = self.cfg["grasp"]
        bin_name = bin_name or self.mapping.get(det.color)
        by = self.bin_xy(bin_name)[1] if bin_name is not None else 0.0
        ref = by if abs(by) > 0.1 else det.top[1]
        side0 = "left" if ref > g["y_split"] else "right"
        last = Plan(False, color=det.color, reason=f"no bin for colour {det.color!r}")
        if bin_name is None:
            return last
        reasons, good = [], []
        for side in (side0, "right" if side0 == "left" else "left"):
            p = self._plan_side_hints(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float))
            if p.ok:
                if g.get("arm", "best") != "best":
                    return p
                good.append(p)
            else:
                reasons.append(f"{side}: {p.reason}")
                last = p
        if good:
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons)
        if relocate and g.get("relocate", {}).get("enabled", True):
            r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles)
            if r.ok:
                return r
            last.reason += f"; relocate: {r.reason}"
        return last

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Candidate counter spots (pelvis xy) for a relocation, nearest the target bin first."""
        rc = self.cfg["grasp"].get("relocate", {})
        sc = self.scene
        L = sc.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 4)
        pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        return keep[: int(rc.get("max_spots", 6))]

    def plan_relocation(self, det, bin_name, arm_q, hand_q, obstacles=()):
        """Cross-body fallback: pick `det` with an arm that reaches it and put it down at a counter spot from which
        some arm has a full plan to `bin_name` (checked from home). Returns a Plan with target_bin 'counter'."""
        from dataclasses import replace
        last = Plan(False, color=det.color, reason="no relocation spot")
        H = self.height
        for spot in self.relocation_spots(det, bin_name, obstacles):
            d2 = replace(det, top=np.r_[spot, self.scene.counter_z + H])
            by = self.bin_xy(bin_name)[1]
            fin_side = "left" if (by if abs(by) > 0.1 else spot[1]) > 0 else "right"
            fin = self._plan_side_hints(d2, fin_side, bin_name, self.home.copy(), self.post["open"].copy())
            if not fin.ok:
                last.reason = f"no {fin_side} plan from spot ({spot[0]:+.2f},{spot[1]:+.2f}): {fin.reason}"
                continue
            side = "right" if fin_side == "left" else "left"
            p = self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=spot)
            if p.ok:
                p.reason = f"relocate for the {fin_side} arm"
                return p
            last = p
            last.reason = f"{side} cannot place at ({spot[0]:+.2f},{spot[1]:+.2f}): {p.reason}"
        return last
    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        """_plan_side with each finger direction of grasp.finger_hints in turn (fingers forward can reach over the
        centre bin's wall when the object is close to it; then fingers inward/sideways)."""
        hints = self.cfg["grasp"].get("finger_hints") or [self.cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0])]
        saved, first = self.kin.hint, None
        try:
            for h in hints:
                h = np.asarray(h, float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
                self.kin.hint = h / np.linalg.norm(h)
                p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    return p
                first = first or p
            return first
        finally:
            self.kin.hint = saved

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter")
        sl = ArmKinematics.sl(side)
        H, post = self.height, self.post
        hand_pre = hand_q.copy(); hand_pre[sl] = post["preshape"][sl]
        hand_cls = hand_q.copy(); hand_cls[sl] = post["closed"][sl]
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]

        # grasp height: object mid-height, raised until the preshaped fingers clear the counter
        xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
        z = zc + g["height_frac"] * H
        for _ in range(4):
            qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
            if not ok:
                plan.reason = f"IK grasp {side}"
                return plan
            c = self.clearance(side, qg, hand_pre, fingers_only=True)
            if c >= g["fingertip_clearance"] - 1e-3:
                break
            z += g["fingertip_clearance"] - c
        else:
            plan.reason = "fingers hit the counter"
            return plan
        if z > zc + H - 0.008:
            plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
            return plan

        wall = float(sc.cfg["bins"]["wall_height"])
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        z_drop = zc + wall + H / 2 + g["drop_above_rim"]
        bxy = self.drop_xy(bin_name)
        if place_xy is not None:  # relocation: put it down on the counter (release just above the grasp height)
            bxy, z_drop = np.asarray(place_xy, float), z + 0.005
        P = {"pregrasp": np.r_[xy, max(z + g["approach_height"], z_carry)], "grasp": np.r_[xy, z], "lift": np.r_[xy, z_carry],
             "above_bin": np.r_[bxy, z_carry], "drop": np.r_[bxy, z_drop], "retreat": np.r_[bxy, z_drop + g["retreat_up"]]}

        q_pre, ok = self._ik(side, P["pregrasp"], qg, "pregrasp", plan)  # same IK branch as the grasp
        if not ok:
            plan.reason = f"IK pregrasp {side}"
            return plan
        plan.waypoints["grasp"] = np.r_[xy, z]
        down = self._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
        up = down and self._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
        carry = up and (self._cartesian(side, P["lift"], P["above_bin"], up[-1], "carry", plan)
                        or self._joint_carry(side, up[-1], P["lift"], P["above_bin"], z_drop, "carry", plan))
        lower = carry and self._cartesian(side, P["above_bin"], P["drop"], carry[-1], "lower", plan)
        retreat = lower and self._cartesian(side, P["drop"], P["retreat"], lower[-1], "retreat", plan)
        if not retreat:
            plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            return plan
        plan.waypoints["drop"] = P["drop"]

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def move(name, Qs, hand_to=None):
            nonlocal cur_a, cur_h
            a = time_scale([cur_a] + list(Qs), vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:  # hand moves during the arm move (ends no later than the arm)
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            h = ramp(cur_h, to, hv, self.dt)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        move("to_pregrasp", [q_pre], hand_to=hand_pre)
        move("descend", down)
        hand("close", hand_cls, mo["close_hold_s"])
        move("lift", up)
        move("carry", carry + lower)
        hand("open", hand_rel, mo["open_hold_s"])  # widest posture: `open` alone left the cube wedged in the sim
        move("retreat", retreat)
        move("home", [home], hand_to=hand_open)
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        grasp_ticks = np.zeros(len(plan.arm), bool)
        near = ("descend", "close", "lift") + (("carry", "open", "retreat") if place_xy is not None else ())
        for name, s, e in phases:
            if name in near:
                grasp_ticks[s:e] = True
        closing = np.zeros(len(plan.arm), bool)
        for name, s, e in phases:
            closing[s:e] |= name == "close" or (place_xy is not None and name == "open")
        for k in range(0, len(plan.arm), 3):
            # closing: the thumb swings through its lowest point; touching the mat lightly is acceptable
            need = 0.003 if closing[k] else g["fingertip_clearance"] if grasp_ticks[k] else mo["counter_clearance"]
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                # the object itself sits between the fingers during descend/lift; only the counter matters here
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
                return plan
        drop = np.zeros(len(plan.arm), bool)
        for name, s_, e in phases:
            drop[s_:e] |= name in ("open",)
        for k in range(0, len(plan.arm), 3):
            if drop[k]:
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                name = next(n for n, s_, e in phases if s_ <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan

    def dry_run_safety(self, arm_q, hand_q, arm, hand):
        f = SafetyFilter(self.spec, self.safety)
        f.reset(arm_q, hand_q, 0.0)
        for k in range(len(arm)):
            f.filter({"arm_q_target": arm[k], "hand_q_target": hand[k]}, (k + 1) * self.dt)
        return f.clipped_steps

    def go_to(self, arm_from, hand_from, arm_to, hand_to):
        """Joint-space move (both arms + hands) at the configured speeds, e.g. start pose -> home."""
        mo = self.cfg["motion"]
        a = time_scale([arm_from, arm_to], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = ramp(hand_from, hand_to, mo["hand_max_vel"], self.dt)
        n = max(len(a), len(h))
        a = np.vstack([a, np.repeat(np.asarray(arm_to, float)[None], n - len(a), 0)]) if len(a) < n else a
        h = np.vstack([h, np.repeat(np.asarray(hand_to, float)[None], n - len(h), 0)]) if len(h) < n else h
        return a, h
