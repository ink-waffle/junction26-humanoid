"""Forward-exit release: the hand turned about the pelvis y axis through the grasp point, open side towards the box (logs/sag.md)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aaltonomous.baselines.classical.kinematics import ArmKinematics, _skew

SEED_WEIGHT = 1e-3   # pull towards the untilted pose: picks the nearest of the arm's redundant solutions
MAX_EVALUATIONS = 200


def turned_pose(kin: ArmKinematics, side: str, arm_q: np.ndarray, angle: float) -> tuple[np.ndarray, float, float]:
    """Arm pose (14,) holding the grasp point of `arm_q` with the hand turned `angle` rad about pelvis y (approach axis towards +x); its position error (m) and orientation error (rad)."""
    cols, start = ArmKinematics.sl(side), np.asarray(arm_q, float)
    point, approach, fingers = _hand_frame(kin, side, start)
    turn = Rotation.from_rotvec([0.0, -angle, 0.0]).as_matrix()
    goal = (point, turn @ approach, turn @ fingers)
    solved = least_squares(lambda x: _residual(kin, side, start, x, goal), start[cols],
                           jac=lambda x: _jacobian(kin, side, start, x), bounds=(kin.lo[cols], kin.hi[cols]),
                           method="trf", xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=MAX_EVALUATIONS).x
    pose = start.copy()
    pose[cols] = solved
    reached = _hand_frame(kin, side, pose)
    error = max(_angle(reached[1], goal[1]), _angle(reached[2], goal[2]))
    return pose, float(np.linalg.norm(reached[0] - point)), error


def _hand_frame(kin: ArmKinematics, side: str, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grasp point, approach axis and finger direction of one hand (pelvis frame)."""
    kin._fk(arm_q)
    palm = kin.data.oMf[kin.palm[side]]
    return palm.translation + palm.rotation @ kin.offset[side], palm.rotation @ kin.axis[side], palm.rotation[:, 0].copy()


def _residual(kin: ArmKinematics, side: str, start: np.ndarray, x: np.ndarray, goal: tuple[np.ndarray, ...]) -> np.ndarray:
    pose = start.copy()
    pose[ArmKinematics.sl(side)] = x
    frame = _hand_frame(kin, side, pose)
    return np.concatenate([a - b for a, b in zip(frame, goal)] + [SEED_WEIGHT * (x - start[ArmKinematics.sl(side)])])


def _jacobian(kin: ArmKinematics, side: str, start: np.ndarray, x: np.ndarray) -> np.ndarray:
    cols = ArmKinematics.sl(side)
    pose = start.copy()
    pose[cols] = x
    q = kin._fk(pose)
    palm = kin.data.oMf[kin.palm[side]]
    lever, approach, fingers = palm.rotation @ kin.offset[side], palm.rotation @ kin.axis[side], palm.rotation[:, 0]
    frame = kin.pin.computeFrameJacobian(kin.model, kin.data, q, kin.palm[side], kin.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    linear, angular = frame[:3, kin.arm_iv[cols]], frame[3:, kin.arm_iv[cols]]
    return np.vstack([linear - _skew(lever) @ angular, -_skew(approach) @ angular, -_skew(fingers) @ angular, SEED_WEIGHT * np.eye(7)])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
