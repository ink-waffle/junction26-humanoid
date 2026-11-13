"""Safety filter between every policy action and the robot (challenge spec §5, TODO M6 / WP2).

Pure Python + numpy + Pinocchio (FK only). No DDS, no robot: the runner owns the robot connection and
calls this filter once per 30 Hz tick. The same filter runs in sim and on the real robot.

    f = SafetyFilter.from_files()                       # joint_spec.yaml + defaults
    f.reset(arm_q_measured, hand_q_measured, t=0.0)
    res = f.filter(action, t, arm_q=..., hand_q=...)    # one action (14,) per tick; runner unpacks chunks
    res = f.tick(t)                                     # tick with no new action (policy still computing)
    send(res.arm_q_cmd, res.hand_q_cmd)

Per tick, in order:
  1. validate   -- missing key, wrong shape, NaN/inf  -> that part holds its last command
  2. position   -- URDF limits shrunk by a margin
  3. lead cap   -- optional: command at most `max_lead` rad from the MEASURED joint (caps finger squeeze force)
  4. velocity   -- command moves at most max_vel * dt from the LAST COMMAND (not from measured)
  5. workspace  -- wrist + palm frames (Pinocchio FK, legs/waist at the stiff pose) must stay in a box in the
                   pelvis frame. A step that would leave the box (or go further out of it) is shortened by
                   bisection, so the arm slides up to the boundary and can always move back in.
Watchdog: no action for `hold_after_s` -> stale (hold); for `stop_after_s` -> safe stop (latched until reset).
Abort: more than `max_clipped_steps` clipped ticks -> aborted (latched), output holds.
Every clip is logged with its reason; `events` keeps them for the episode log.

Not covered (yet): self-collision (arm vs torso, arm vs arm), table collision of fingers below the palm frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("aaltonomous.safety")

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "joint_spec.yaml"
DEFAULT_URDF = HERE.parents[1] / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"

N = 14  # arm and hand vector length


@dataclass
class SafetyConfig:
    dt: float = 1.0 / 30.0
    arm_pos_margin: float = 0.05          # rad inside URDF limits
    hand_pos_margin: float = 0.02
    arm_max_vel: float = 1.0              # rad/s of COMMAND change (finals default; tune on robot)
    hand_max_vel: float = 3.0             # (default) fingers must close in well under a second
    arm_max_lead: float | None = None     # rad from measured; None = off
    hand_max_lead: float | None = None    # e.g. 0.3 caps how hard a finger presses a cube
    workspace: bool = True
    workspace_frames: tuple = ("left_wrist_yaw_link", "left_hand_palm_link",
                               "right_wrist_yaw_link", "right_hand_palm_link")
    # Box in the pelvis frame (m). (default) placeholders until the counter is measured:
    # z_min must become counter surface height (rel. pelvis) + a grasp margin.
    workspace_min: tuple = (0.0, -0.70, -0.25)
    workspace_max: tuple = (0.80, 0.70, 0.60)
    # Counter keep-out (pelvis frame, m): wrist/palm frames may not enter this box (the table volume + a 2 cm
    # margin above it). v1 scene 2026-11-10: 80 cm table, surface +0.01 m rel. pelvis (pelvis ~0.79 m assumed),
    # near edge 0.14 m in front of the pelvis, 1.10 m wide. Keep in sync with scene.yaml. Fingertips not covered.
    counter_keepout_min: tuple | None = (0.14, -0.55, -1.0)
    counter_keepout_max: tuple | None = (1.20, 0.55, 0.03)
    workspace_tol: float = 1e-4           # m of extra violation tolerated per step
    workspace_bisect_iters: int = 8
    lower_body_q: dict = field(default_factory=dict)  # joint name -> rad; missing = 0 (stiff pose for FK)
    hold_after_s: float = 0.2
    stop_after_s: float = 1.0
    safe_stop: str = "hold"               # "hold" | "home" (home sweeps the arms: only with a clear table)
    safe_stop_vel: float = 0.3            # rad/s when safe_stop == "home"
    home_arm_q: tuple = (0.0,) * N
    home_hand_q: tuple = (0.0,) * N
    max_clipped_steps: int = 30           # abort when exceeded (spec N, to be decided)
    eps: float = 1e-6                     # numerical slack before a change counts as a clip

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


@dataclass
class FilterResult:
    arm_q_cmd: np.ndarray
    hand_q_cmd: np.ndarray
    mode: str                 # "ok" | "stale" | "safe_stop" | "aborted"
    clipped: bool
    reasons: list
    clipped_steps: int


class JointSpec:
    def __init__(self, path=DEFAULT_SPEC):
        spec = yaml.safe_load(Path(path).read_text())
        self.arm_names = [j["name"] for j in spec["arm"]]
        self.hand_names = [j["name"] for j in spec["hand"]]
        self.locked_names = [j["name"] for j in spec["locked"]]
        self.arm_lower = np.array([j["lower"] for j in spec["arm"]])
        self.arm_upper = np.array([j["upper"] for j in spec["arm"]])
        self.hand_lower = np.array([j["lower"] for j in spec["hand"]])
        self.hand_upper = np.array([j["upper"] for j in spec["hand"]])
        assert len(self.arm_names) == N and len(self.hand_names) == N


def _shrink(lo, hi, margin):
    m = np.minimum(margin, 0.25 * (hi - lo))  # never invert a narrow range
    return lo + m, hi - m


class WorkspaceChecker:
    """Distance (m) by which the configured frames lie outside the box, via Pinocchio FK."""

    def __init__(self, spec: JointSpec, cfg: SafetyConfig, urdf=DEFAULT_URDF):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))  # fixed base at the pelvis
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        for name, val in cfg.lower_body_q.items():
            self.q[self.model.joints[self.model.getJointId(name)].idx_q] = val
        self.arm_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in spec.arm_names])
        self.frames = [self.model.getFrameId(f) for f in cfg.workspace_frames]
        self.lo = np.asarray(cfg.workspace_min, float)
        self.hi = np.asarray(cfg.workspace_max, float)
        self.ko = None if cfg.counter_keepout_min is None else (np.asarray(cfg.counter_keepout_min, float),
                                                                 np.asarray(cfg.counter_keepout_max, float))

    def positions(self, arm_q):
        self.q[self.arm_idx] = arm_q
        self.pin.forwardKinematics(self.model, self.data, self.q)
        return np.array([self.pin.updateFramePlacement(self.model, self.data, f).translation.copy()
                         for f in self.frames])

    def violation(self, arm_q):
        p = self.positions(arm_q)
        out = np.maximum(self.lo - p, 0.0) + np.maximum(p - self.hi, 0.0)
        v = float(np.linalg.norm(out, axis=1).sum())
        if self.ko is not None:  # penetration depth into the counter box (distance to its nearest face)
            depth = np.minimum(p - self.ko[0], self.ko[1] - p).min(axis=1)
            v += float(np.maximum(depth, 0.0).sum())
        return v


class SafetyFilter:
    def __init__(self, spec: JointSpec, cfg: SafetyConfig | None = None):
        self.spec = spec
        self.cfg = cfg or SafetyConfig()
        c = self.cfg
        self.arm_lo, self.arm_hi = _shrink(spec.arm_lower, spec.arm_upper, c.arm_pos_margin)
        self.hand_lo, self.hand_hi = _shrink(spec.hand_lower, spec.hand_upper, c.hand_pos_margin)
        self.ws = WorkspaceChecker(spec, c) if c.workspace else None
        self.home_arm = np.clip(np.asarray(c.home_arm_q, float), self.arm_lo, self.arm_hi)
        self.home_hand = np.clip(np.asarray(c.home_hand_q, float), self.hand_lo, self.hand_hi)
        self.reset(self.home_arm, self.home_hand, 0.0)

    @classmethod
    def from_files(cls, spec_path=DEFAULT_SPEC, config_path=None):
        cfg = SafetyConfig.from_yaml(config_path) if config_path else SafetyConfig()
        return cls(JointSpec(spec_path), cfg)

    # ---------------------------------------------------------------- episode state
    def reset(self, arm_q, hand_q, t):
        """Start an episode from the MEASURED pose: the first command equals where the robot is."""
        self.arm_cmd = np.asarray(arm_q, float).copy()
        self.hand_cmd = np.asarray(hand_q, float).copy()
        self.last_action_t = t
        self.t = t
        self.step_i = 0
        self.clipped_steps = 0
        self.mode = "ok"
        self.events = []

    def _log(self, t, reasons, details):
        ev = {"t": round(float(t), 4), "step": self.step_i, "reasons": reasons, "details": details}
        self.events.append(ev)
        log.warning("safety clip t=%.3f step=%d %s %s", t, self.step_i, reasons, details)

    def _result(self, clipped, reasons):
        return FilterResult(self.arm_cmd.copy(), self.hand_cmd.copy(), self.mode, clipped, reasons,
                            self.clipped_steps)

    # ---------------------------------------------------------------- per tick
    def tick(self, t):
        """A control tick with no new action. Holds, or runs the safe stop after `stop_after_s`."""
        self.t = t
        c = self.cfg
        if self.mode in ("ok", "stale"):
            idle = t - self.last_action_t
            if idle > c.stop_after_s:
                self.mode = "safe_stop"
                self._log(t, ["watchdog_stop"], {"idle_s": round(idle, 3)})
            elif idle > c.hold_after_s and self.mode == "ok":
                self.mode = "stale"
                self._log(t, ["watchdog_hold"], {"idle_s": round(idle, 3)})
        if self.mode == "safe_stop" and c.safe_stop == "home":
            d = c.safe_stop_vel * c.dt
            arm = self.arm_cmd + np.clip(self.home_arm - self.arm_cmd, -d, d)
            self.arm_cmd = self._workspace(arm, [], {})
            self.hand_cmd = self.hand_cmd + np.clip(self.home_hand - self.hand_cmd, -d, d)
        return self._result(False, [])

    def filter(self, action, t, arm_q=None, hand_q=None):
        """Filter one action dict {arm_q_target: (14,), hand_q_target: (14,)} for this tick.

        arm_q / hand_q: measured joints, needed only for the lead caps."""
        self.t = t
        if self.mode in ("safe_stop", "aborted"):
            return self.tick(t)
        self.step_i += 1
        self.last_action_t = t
        self.mode = "ok"
        c = self.cfg
        reasons, details = [], {}

        arm = self._validate(action, "arm_q_target", self.arm_cmd, reasons)
        hand = self._validate(action, "hand_q_target", self.hand_cmd, reasons)

        # Beyond the URDF limit = a clip. Inside it but within the margin = silently clamped (e.g. open Dex3 = 0).
        sp = self.spec
        arm = np.clip(self._clamp(arm, sp.arm_lower, sp.arm_upper, sp.arm_names, "arm_position", reasons, details),
                      self.arm_lo, self.arm_hi)
        hand = np.clip(self._clamp(hand, sp.hand_lower, sp.hand_upper, sp.hand_names, "hand_position", reasons,
                                   details), self.hand_lo, self.hand_hi)
        if c.arm_max_lead is not None and arm_q is not None:
            arm = self._clamp(arm, np.asarray(arm_q) - c.arm_max_lead, np.asarray(arm_q) + c.arm_max_lead,
                              self.spec.arm_names, "arm_lead", reasons, details)
        if c.hand_max_lead is not None and hand_q is not None:
            hand = self._clamp(hand, np.asarray(hand_q) - c.hand_max_lead, np.asarray(hand_q) + c.hand_max_lead,
                               self.spec.hand_names, "hand_lead", reasons, details)
        da, dh = c.arm_max_vel * c.dt, c.hand_max_vel * c.dt
        arm = self._clamp(arm, self.arm_cmd - da, self.arm_cmd + da, self.spec.arm_names, "arm_velocity",
                          reasons, details)
        hand = self._clamp(hand, self.hand_cmd - dh, self.hand_cmd + dh, self.spec.hand_names, "hand_velocity",
                           reasons, details)
        arm = self._workspace(arm, reasons, details)

        self.arm_cmd, self.hand_cmd = arm, hand
        clipped = bool(reasons)
        if clipped:
            self.clipped_steps += 1
            self._log(t, reasons, details)
            if self.clipped_steps > c.max_clipped_steps:
                self.mode = "aborted"
                self._log(t, ["abort"], {"clipped_steps": self.clipped_steps})
        return self._result(clipped, reasons)

    # ---------------------------------------------------------------- stages
    @staticmethod
    def _validate(action, key, last, reasons):
        v = action.get(key) if isinstance(action, dict) else None
        if v is None:
            reasons.append(f"invalid:{key}:missing")
            return last.copy()
        v = np.asarray(v, dtype=float)
        if v.shape != (N,):
            reasons.append(f"invalid:{key}:shape{tuple(v.shape)}")
            return last.copy()
        if not np.all(np.isfinite(v)):
            reasons.append(f"invalid:{key}:nonfinite")
            return last.copy()
        return v

    def _clamp(self, v, lo, hi, names, reason, reasons, details):
        out = np.clip(v, lo, hi)
        hit = np.abs(out - v) > self.cfg.eps
        if hit.any():
            reasons.append(reason)
            details[reason] = [names[i] for i in np.flatnonzero(hit)]
        return out

    def _workspace(self, arm, reasons, details):
        if self.ws is None:
            return arm
        tol = self.cfg.workspace_tol
        allowed = max(self.ws.violation(self.arm_cmd), tol)  # no accumulation
        if self.ws.violation(arm) <= allowed:
            return arm
        lo_s, hi_s = 0.0, 1.0  # largest fraction of the step that stays allowed
        step = arm - self.arm_cmd
        for _ in range(self.cfg.workspace_bisect_iters):
            mid = 0.5 * (lo_s + hi_s)
            if self.ws.violation(self.arm_cmd + mid * step) <= allowed:
                lo_s = mid
            else:
                hi_s = mid
        reasons.append("workspace")
        details["workspace"] = {"step_fraction": round(lo_s, 3)}
        return self.arm_cmd + lo_s * step

    def summary(self):
        counts = {}
        for ev in self.events:
            for r in ev["reasons"]:
                counts[r.split(":")[0]] = counts.get(r.split(":")[0], 0) + 1
        return {"steps": self.step_i, "clipped_steps": self.clipped_steps, "mode": self.mode, "counts": counts}
