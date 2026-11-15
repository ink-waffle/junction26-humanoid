"""Arm sag: how much more each arm sinks under gravity than the reference (sim) arm, and the arm commands that undo it (logs/sag.md)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

import numpy as np
import pinocchio as pin

from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

log = logging.getLogger(__name__)
SIDES = ("left", "right")
KAPPA_LOG_STEP = 0.05   # a committed kappa is logged when it moves this far


def compliance(arm: Mapping[str, Any]) -> np.ndarray:
    """Steady error per gravity torque (rad/Nm) of the 14 arm joints: torque scale over PD stiffness, both arms alike."""
    return np.tile(np.asarray(arm["alpha"], float) / np.asarray(arm["kp"], float), 2)


def boxes_near_x(scene: Scene) -> float:
    """Pelvis x of the nearest inner wall of any box."""
    return min(float(scene.mat_to_pelvis(np.array(scene.bin_inner_box(name)))[:, 0].min()) for name in scene.bin_names)


class ArmModel:
    """URDF gravity torques on the arm joints and hand link positions (legs and waist at 0), one pinocchio workspace per thread."""

    def __init__(self, spec: JointSpec, urdf: Any = DEFAULT_URDF) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf))
        joints = [self.model.joints[self.model.getJointId(name)] for name in spec.arm_names]
        self.arm_q, self.arm_v = np.array([j.idx_q for j in joints]), np.array([j.idx_v for j in joints])
        self.hand_q = np.array([self.model.joints[self.model.getJointId(name)].idx_q for name in spec.hand_names])
        self.hand_frames = [[i for i, f in enumerate(self.model.frames) if f.name.startswith(f"{side}_hand")] for side in SIDES]
        self._local = threading.local()

    def _posed(self, arm_q: np.ndarray, hand_q: np.ndarray) -> tuple[Any, np.ndarray]:
        local = self._local
        if not hasattr(local, "data"):
            local.data, local.q = self.model.createData(), pin.neutral(self.model)
        local.q[self.arm_q], local.q[self.hand_q] = arm_q, hand_q
        return local.data, local.q

    def gravity(self, arm: np.ndarray, hand: np.ndarray) -> np.ndarray:
        """Gravity torques (n, 14) for (n, 14) arm and hand rows."""
        out = np.empty((len(arm), 14))
        for k in range(len(arm)):
            data, q = self._posed(arm[k], hand[k])
            out[k] = pin.computeGeneralizedGravity(self.model, data, q)[self.arm_v]
        return out

    def hand_points(self, arm_q: np.ndarray, hand_q: np.ndarray) -> list[np.ndarray]:
        """Origins of every hand link of each side (pelvis frame)."""
        data, q = self._posed(arm_q, hand_q)
        pin.framesForwardKinematics(self.model, data, q)
        return [np.array([data.oMf[i].translation for i in frames]) for frames in self.hand_frames]


class Sag:
    """Per-episode fit of kappa (0 = sags like the reference arm, 1 = like the real arm measured on 15.11) and the compensated commands."""

    def __init__(self, model: ArmModel, cfg: Mapping[str, Any], spec: JointSpec, safety: SafetyConfig, dt: float) -> None:
        self.model, self.cfg = model, cfg
        self.reference = compliance(cfg["reference"])
        self.excess = compliance(cfg["real"]) - self.reference
        self.lag = np.tile(np.asarray(cfg["real"]["kd"], float) / np.asarray(cfg["real"]["kp"], float), 2)   # s: damping error per rad/s
        self.dt = dt
        margin = np.minimum(safety.arm_pos_margin, 0.25 * (spec.arm_upper - spec.arm_lower))
        self.lower, self.upper = spec.arm_lower + margin, spec.arm_upper - margin
        self.step, self.offset_step = cfg["max_speed"] * dt, cfg["max_offset_speed"] * dt
        self.reset()

    def reset(self) -> None:
        """A new episode: nothing measured, nothing compensated."""
        self.kappa = np.zeros(2)
        self.offset = np.zeros(14)   # sent minus plan-space arm command of the row last sent
        self._xr, self._xx, self._samples = np.zeros(2), np.zeros(2), np.zeros(2, int)
        self._previous: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def idle(self) -> bool:
        return not self.kappa.any() and not self.offset.any()

    def observe(self, sent: np.ndarray, held: np.ndarray, arm_q: np.ndarray, hand_q: np.ndarray, scene: Scene) -> None:
        """One observation: every arm still since the last one, with its hand clear of counter and boxes, adds a sample."""
        sent, arm_q, hand_q = (np.asarray(v, float) for v in (sent, arm_q, hand_q))
        previous, self._previous = self._previous, (sent.copy(), arm_q.copy())
        still = [previous is not None and self._still(k, sent, held, arm_q, previous) for k in range(2)]
        if not any(still):
            return
        error = sent - arm_q
        effective = error * min(1.0, self.cfg["clip"] / max(float(np.abs(error).max()), 1e-12))
        tau = self.model.gravity(arm_q[None], hand_q[None])[0]
        x, r = self.excess * tau, effective - self.reference * tau
        points = self.model.hand_points(arm_q, hand_q)
        for k in range(2):
            arm = slice(7 * k, 7 * k + 7)
            if still[k] and self._clear(points[k], scene):
                self._xr[k] += float(x[arm] @ r[arm])
                self._xx[k] += float(x[arm] @ x[arm])
                self._samples[k] += 1

    def _still(self, k: int, sent: np.ndarray, held: np.ndarray, arm_q: np.ndarray, previous: tuple[np.ndarray, np.ndarray]) -> bool:
        arm = slice(7 * k, 7 * k + 7)
        same = np.array_equal(sent[arm], previous[0][arm])
        return bool(held[k] >= self.cfg["still_ticks"] and same and np.abs(arm_q[arm] - previous[1][arm]).max() <= self.cfg["still_motion"])

    def _clear(self, points: np.ndarray, scene: Scene) -> bool:
        """The hand is off the counter and in front of the boxes: nothing but the arm holds it up."""
        high = points[:, 2].min() - scene.counter_z >= self.cfg["clearance"]
        return bool(high and points[:, 0].max() <= boxes_near_x(scene) - self.cfg["box_margin"])

    def commit(self, t: float) -> None:
        """The estimate so far becomes the one plans and commands use; an arm with too few samples or under the dead band gets none."""
        fit = np.clip(np.divide(self._xr, self._xx, out=np.zeros(2), where=self._xx > 0), 0.0, self.cfg["max_kappa"])
        kappa = np.where((self._samples >= self.cfg["min_samples"]) & (fit >= self.cfg["dead_band"]), fit, 0.0)
        if np.abs(kappa - self.kappa).max() >= KAPPA_LOG_STEP or (kappa > 0).tolist() != (self.kappa > 0).tolist():
            log.info("t=%.1f arm sag: kappa left %.2f, right %.2f (fit %.2f / %.2f on %d / %d still samples)", t, *kappa, *fit,
                     *self._samples)
        self.kappa = kappa

    def estimated(self) -> np.ndarray:
        """Compliance (rad/Nm) of the 14 arm joints at the committed kappa."""
        return self.reference + np.repeat(self.kappa, 7) * self.excess

    def static_use(self, arm_q: np.ndarray, hand_q: np.ndarray) -> float:
        """Share of the clip that the sag at this pose takes on its worst joint."""
        tau = self.model.gravity(np.asarray(arm_q, float)[None], np.asarray(hand_q, float)[None])[0]
        return float(np.abs(self.estimated() * tau).max() / self.cfg["clip"])

    def holds(self, arm_q: np.ndarray, hand_q: np.ndarray) -> bool:
        """Whether the arm can stand still at this pose within the static budget (always at kappa 0)."""
        return not self.kappa.any() or self.static_use(arm_q, hand_q) <= self.cfg["budget"]["static"]

    def timed(self, arm_from: np.ndarray, hand_from: np.ndarray, arm: np.ndarray, hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rows after (`arm_from`, `hand_from`) slowed wherever sag plus damping lag would pass the moving budget; the same rows where they do not."""
        if not self.kappa.any() or not len(arm):
            return arm, hand
        arm_rows, hand_rows = np.vstack([arm_from, arm]), np.vstack([hand_from, hand])
        velocity = np.diff(arm_rows, axis=0) / self.dt
        sag = self.estimated() * self.model.gravity(arm, hand)
        room = self.cfg["budget"]["moving"] * self.cfg["clip"] - np.sign(velocity) * sag   # moving with gravity frees room
        stretch = np.maximum(1.0, (np.abs(velocity) * self.lag / np.maximum(room, self.cfg["budget"]["min_room"])).max(1))
        if np.all(stretch <= 1.0):
            return arm, hand
        times = np.r_[0.0, np.cumsum(stretch)] * self.dt
        at = np.minimum(np.arange(1, int(np.ceil(times[-1] / self.dt - 1e-9)) + 1) * self.dt, times[-1])
        at[-1] = times[-1]
        rows = np.hstack([arm_rows, hand_rows])
        out = np.column_stack([np.interp(at, times, rows[:, j]) for j in range(rows.shape[1])])
        return out[:, :14], out[:, 14:]

    def targets(self, arm: np.ndarray, hand: np.ndarray) -> np.ndarray:
        """Offsets (n, 14) that undo the excess sag at each planned pose."""
        return np.repeat(self.kappa, 7) * self.excess * self.model.gravity(arm, hand)

    def commands(self, start: np.ndarray, arm: np.ndarray, hand: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sent rows for plan-space rows that follow `start` (sent with `offset`), and each row's offset; no row moves faster than `max_speed`."""
        if not self.kappa.any() and not offset.any():
            return arm, np.zeros_like(arm)
        targets, offsets, previous = self.targets(arm, hand), np.empty_like(arm), np.asarray(start, float)
        for k in range(len(arm)):
            room = np.minimum(self.offset_step, np.maximum(self.step - np.abs(arm[k] - previous), 0.0))
            offset = offset + np.clip(targets[k] - offset, -room, room)
            offsets[k], previous = offset, arm[k]
        return np.clip(arm + offsets, self.lower, self.upper), offsets

    def sent_at(self, arm_q: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """The command sent for plan-space pose `arm_q` carrying `offset`."""
        return np.clip(np.asarray(arm_q, float) + offset, self.lower, self.upper)
