"""Arm kinematics for the classical baseline: FK of a grasp point on each Dex3 + damped-least-squares IK.

Model: the full G1 URDF (the one the safety filter uses), fixed base at the pelvis, legs/waist at 0 (the stiff
pose the safety filter and calibration assume). Each arm is solved alone (7 joints); the other arm is untouched.

Grasp point = a point fixed in `<side>_hand_palm_link` (config `grasp.point_in_palm`, right side mirrored in y),
where the object centre should sit between thumb and fingers. `auto` = midway between the index/middle tips and the thumb tip in
the `closed` posture (from the URDF): `closed` leaves a gap a bit smaller than the object, so it is squeezed there.
Orientation: palm-frame x = finger direction; the fingers curl and the thumb points along palm -y (left; +y right).
Default grasp = "claw from above": that curl axis points straight down (palm faces the counter), fingers roughly
forward. IK: position (3) + approach axis down (2, hard-ish) + finger direction hint (weak); yaw is otherwise free.
"""
from __future__ import annotations

import numpy as np

from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec

SIDES = ("left", "right")
DOWN = np.array([0.0, 0.0, -1.0])
# Distal finger links and the fingertip in each link frame (from the Dex3 STL extents: 52 mm along the link;
# the thumb tip points along -y on the left hand, +y on the right).
FINGER_LINKS = ("hand_index_1_link", "hand_middle_1_link", "hand_thumb_2_link")
TIP = {"left": ([0.052, 0, 0], [0.052, 0, 0], [0, -0.052, 0]), "right": ([0.052, 0, 0], [0.052, 0, 0], [0, 0.052, 0])}


class ArmKinematics:
    def __init__(self, cfg, spec: JointSpec | None = None, urdf=DEFAULT_URDF, lower_body_q=None):
        import pinocchio as pin
        self.pin = pin
        self.cfg = cfg
        self.spec = spec or JointSpec()
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        for name, val in (lower_body_q or {}).items():
            self.q0[self._iq(name)] = val
        self.arm_iq = np.array([self._iq(n) for n in self.spec.arm_names])
        self.arm_iv = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in self.spec.arm_names])
        self.hand_iq = np.array([self._iq(n) for n in self.spec.hand_names])
        self.palm = {s: self.model.getFrameId(f"{s}_hand_palm_link") for s in SIDES}
        self.fingers = {s: [self.model.getFrameId(f"{s}_{f}") for f in FINGER_LINKS] for s in SIDES}
        self.offset = {"left": np.zeros(3), "right": np.zeros(3)}
        m = float(cfg["ik"]["limit_margin"])
        lo, hi = self.spec.arm_lower + m, self.spec.arm_upper - m
        self.lo, self.hi = lo, hi
        p = cfg["grasp"]["point_in_palm"]
        if isinstance(p, str) and p == "auto":
            from .config import hand_postures
            closed = hand_postures(cfg, self.spec)["closed"]
            t = self.tips_in_palm("left", closed)
            p = 0.5 * (t[:2].mean(0) + t[2])  # midway between the finger pads and the thumb tip
        p = np.asarray(p, float)
        mirror = np.array([1.0, -1.0, 1.0])
        self.offset = {"left": p, "right": p * mirror}
        a = np.asarray(cfg["grasp"]["approach_axis_in_palm"], float)
        self.axis = {"left": a / np.linalg.norm(a), "right": a * mirror / np.linalg.norm(a)}
        self.hint = np.asarray(cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0]), float)
        self.w_hint = float(cfg["grasp"].get("finger_hint_weight", 0.0))

    def tips_in_palm(self, side, hand_q):
        """Fingertip positions (3, 3) in the palm frame for a hand posture (arm pose irrelevant)."""
        self._fk(np.zeros(14), hand_q)
        M = self.data.oMf[self.palm[side]]
        tips = self._tips(side)
        return (tips - M.translation) @ M.rotation

    def _tips(self, side):
        return np.array([self.data.oMf[f].translation + self.data.oMf[f].rotation @ np.asarray(t, float)
                         for f, t in zip(self.fingers[side], TIP[side])])

    def _iq(self, name):
        return self.model.joints[self.model.getJointId(name)].idx_q

    @staticmethod
    def sl(side):
        return slice(0, 7) if side == "left" else slice(7, 14)

    # ------------------------------------------------------------------ FK
    def _fk(self, arm_q, hand_q=None):
        q = self.q0.copy()
        q[self.arm_iq] = arm_q
        if hand_q is not None:
            q[self.hand_iq] = hand_q
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return q

    def grasp_pose(self, side, arm_q):
        """(grasp point position (3,), approach axis (3,)) in the pelvis frame for the full arm_q (14,)."""
        self._fk(arm_q)
        M = self.data.oMf[self.palm[side]]
        return M.translation + M.rotation @ self.offset[side], M.rotation @ self.axis[side]

    def hand_points(self, side, arm_q, hand_q):
        """Pelvis-frame points of one hand: palm, grasp point, distal finger link origins, fingertips (8, 3)."""
        self._fk(arm_q, hand_q)
        M = self.data.oMf[self.palm[side]]
        pts = [M.translation.copy(), M.translation + M.rotation @ self.offset[side]]
        pts += [self.data.oMf[f].translation.copy() for f in self.fingers[side]]
        return np.vstack([np.array(pts), self._tips(side)])

    # ------------------------------------------------------------------ IK
    def solve(self, side, target, arm_q_seed, top_down=True, extra_seeds=(), restarts=True):
        """IK for one arm. target: grasp-point position (pelvis). arm_q_seed: (14,) full arm vector; the other
        arm's values are kept. Returns (arm_q (14,), ok, pos_err m, tilt rad)."""
        ik = self.cfg["ik"]
        seeds = [np.asarray(arm_q_seed, float)[self.sl(side)]] + [np.asarray(s, float) for s in extra_seeds]
        rng = np.random.default_rng(0)
        if restarts:
            seeds += [rng.uniform(self.lo[self.sl(side)], self.hi[self.sl(side)]) for _ in range(int(ik["random_restarts"]))]
        weights = ik["orient_weight"] if isinstance(ik["orient_weight"], (list, tuple)) else [ik["orient_weight"]]
        best = None
        for w in (weights if top_down else [0.0]):
            for k, s in enumerate(seeds):
                q7, perr, tilt = self._lsq(side, np.asarray(target, float), s, arm_q_seed, w)
                ok = perr <= ik["pos_tol"] and (not top_down or tilt <= np.radians(ik["max_tilt_deg"]))
                cost = perr + 0.01 * tilt + (0.0 if ok else 10.0)
                cost += 0.001 * np.abs(q7 - seeds[0]).sum()  # prefer solutions near the seed (continuity)
                if best is None or cost < best[0]:
                    best = (cost, q7, ok, perr, tilt)
                if ok and k == 0:
                    break  # the continuity seed converged: keep it
            if best[2]:
                break
        out = np.asarray(arm_q_seed, float).copy()
        out[self.sl(side)] = best[1]
        return out, best[2], best[3], best[4]

    def _lsq(self, side, target, q7, arm_q_full, w_o):
        """Bounded least squares (scipy TRF) on the same residuals as _dls: much more reliable than clipped DLS near
        the joint limits (a top-down grasp low in front of the body needs wrist pitch/yaw close to their limits)."""
        from scipy.optimize import least_squares
        pin, ik = self.pin, self.cfg["ik"]
        sl = self.sl(side)
        lo, hi = self.lo[sl], self.hi[sl]
        arm = np.asarray(arm_q_full, float).copy()
        q_start = np.clip(np.asarray(q7, float), lo + 1e-6, hi - 1e-6)
        w_reg = float(ik.get("seed_weight", 0.0))
        iv, fid = self.arm_iv[sl], self.palm[side]
        cache = {}

        def evaluate(x):
            key = x.tobytes()
            if key not in cache:
                arm[sl] = x
                q = self._fk(arm)
                M = self.data.oMf[fid]
                r = M.rotation @ self.offset[side]
                a = M.rotation @ self.axis[side]
                f = M.rotation[:, 0]
                J6 = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, iv]
                Jv, Jw = J6[:3], J6[3:]
                res = [M.translation + r - target]
                jac = [Jv - _skew(r) @ Jw]
                if w_o > 0:
                    res += [w_o * (a - DOWN), self.w_hint * w_o * np.cross(f, self.hint)]
                    jac += [-w_o * _skew(a) @ Jw, self.w_hint * w_o * _skew(self.hint) @ _skew(f) @ Jw]
                if w_reg > 0:
                    res.append(w_reg * (x - q_start))
                    jac.append(w_reg * np.eye(7))
                cache.clear()
                cache[key] = (np.concatenate(res), np.vstack(jac))
            return cache[key]

        def run(x0):
            return least_squares(lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], bounds=(lo, hi),
                                 method="trf", max_nfev=int(ik["max_iters"]), xtol=1e-8, ftol=1e-10, gtol=1e-10).x
        q7 = run(q_start)
        if w_reg > 0:  # the seed pull picked the branch; now polish without it biasing the pose (keep a tiny one)
            q_start, w_reg = q7.copy(), 1e-3
            cache.clear()
            q7 = run(q7)
        arm[sl] = q7
        p, a = self.grasp_pose(side, arm)
        tilt = float(np.arccos(np.clip(a @ DOWN, -1.0, 1.0)))
        return q7, float(np.linalg.norm(target - p)), tilt

    def _dls(self, side, target, q7, arm_q_full, w_o):
        pin, ik = self.pin, self.cfg["ik"]
        sl = self.sl(side)
        lo, hi = self.lo[sl], self.hi[sl]
        arm = np.asarray(arm_q_full, float).copy()
        q7 = np.clip(q7, lo, hi)
        q_start = q7.copy()
        w_reg = float(ik.get("seed_weight", 0.0))
        iv = self.arm_iv[sl]
        fid = self.palm[side]
        lam2 = float(ik["damping"]) ** 2
        for _ in range(int(ik["max_iters"])):
            arm[sl] = q7
            q = self._fk(arm)
            M = self.data.oMf[fid]
            r = M.rotation @ self.offset[side]
            p = M.translation + r
            a = M.rotation @ self.axis[side]
            e_p = target - p
            e_o = np.cross(a, DOWN)
            J6 = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, iv]
            Jv, Jw = J6[:3], J6[3:]
            Jp = Jv - _skew(r) @ Jw
            if w_o > 0:
                f = M.rotation[:, 0]
                e_h = np.cross(f, self.hint)  # weak: fingers towards the hint direction
                J = np.vstack([Jp, w_o * Jw, self.w_hint * w_o * Jw])
                e = np.r_[e_p, w_o * e_o, self.w_hint * w_o * e_h]
            else:
                J, e = Jp, e_p
            if w_reg > 0:  # stay near the seed: no null-space drift between neighbouring Cartesian waypoints
                J = np.vstack([J, w_reg * np.eye(7)])
                e = np.r_[e, w_reg * (q_start - q7)]
            if np.linalg.norm(e_p) < 0.2 * ik["pos_tol"] and (w_o == 0 or np.linalg.norm(e_o) < 0.01):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(len(e)), e)
            n = np.abs(dq).max()
            if n > ik["max_step"]:
                dq *= ik["max_step"] / n
            q7 = np.clip(q7 + dq, lo, hi)
        arm[sl] = q7
        p, a = self.grasp_pose(side, arm)
        tilt = float(np.arccos(np.clip(a @ DOWN, -1.0, 1.0)))
        return q7, float(np.linalg.norm(target - p)), tilt


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
