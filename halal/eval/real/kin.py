"""Forward kinematics over recorded joint trajectories: palm poses and fingertips of both hands in the pelvis frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aaltonomous.baselines.classical.kinematics import SIDES, ArmKinematics
from halal import config
from halal.eval.real import dataset

TIP_NAMES = ("index", "middle", "thumb")


@dataclass(frozen=True)
class HandTrack:
    """Per-frame palm origin (T,3), palm rotation (T,3,3) and fingertips (T,3,3: index, middle, thumb) of one hand."""
    palm: np.ndarray
    rotation: np.ndarray
    tips: np.ndarray


def kinematics() -> ArmKinematics:
    """The baseline's Pinocchio model of the G1 with the legs and waist at zero."""
    return ArmKinematics(config.baseline_config())


def track(kin: ArmKinematics, joints: np.ndarray) -> dict[str, HandTrack]:
    """FK of every row of `joints` (T,28: arm_q ++ hand_q) for both hands."""
    out = {side: ([], [], []) for side in SIDES}
    for row in joints:
        kin._fk(row[:dataset.N_ARM], row[dataset.N_ARM:])
        for side in SIDES:
            pose = kin.data.oMf[kin.palm[side]]
            out[side][0].append(pose.translation.copy())
            out[side][1].append(pose.rotation.copy())
            out[side][2].append(kin._tips(side))
    return {side: HandTrack(*(np.array(part) for part in parts)) for side, parts in out.items()}


def tips_in_palm(hand: HandTrack) -> np.ndarray:
    """Fingertips (T,3,3) expressed in the palm frame."""
    return np.einsum("tfi,tij->tfj", hand.tips - hand.palm[:, None, :], hand.rotation)
