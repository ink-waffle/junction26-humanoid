"""Dev-only sim main: the organisers' aalto_sim_main.py with the Dex3 gains from $HAND_GAINS ("kp,kd[,armature]") and the arm gains from $ARM_GAINS.

    HAND_GAINS="1.5,0.2" bash halal/ops/sims.sh start b     (a sim started without HAND_GAINS or ARM_GAINS never loads this file)
    ARM_GAINS="80,3,40,1.5,9.6,4.8" ...                     (shoulder/elbow kp, kd, wrist kp, kd, then torque limits: the real runner's PD and its clip)
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parents[2]
SIM_MAIN = REPO / "aaltonomous/sim/aalto_sim_main.py"
ROBOT_MODULE = "robots.unitree"
ROBOT_CONFIG = "G129_CFG_WITH_DEX3_BASE_FIX"


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Runs `patch(module)` right after the normal import machinery has loaded one named module."""

    def __init__(self, name: str, patch: Callable[[ModuleType], None]) -> None:
        self.name, self.patch = name, patch

    def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        load = spec.loader.exec_module

        def load_and_patch(module: ModuleType) -> None:
            load(module)
            self.patch(module)

        spec.loader.exec_module = load_and_patch
        return spec


def soften_hands(module: ModuleType) -> None:
    """Replace the hand actuators' gains in the upstream robot config and print what was applied."""
    kp, kd, *armature = (float(value) for value in os.environ["HAND_GAINS"].split(","))
    hands = getattr(module, ROBOT_CONFIG).actuators["hands"]
    hands.stiffness, hands.damping = {".*": kp}, {".*": kd}
    if armature:
        hands.armature = {".*": armature[0]}
    print(f"[soft_hand_sim] hand actuators: stiffness {hands.stiffness}, damping {hands.damping}, armature {hands.armature}", flush=True)


def soften_arms(module: ModuleType) -> None:
    """Replace the arm actuators' gains (and torque limits, when given) in the upstream robot config and print what was applied."""
    kp, kd, kp_wrist, kd_wrist, *limits = (float(value) for value in os.environ["ARM_GAINS"].split(","))
    arms = getattr(module, ROBOT_CONFIG).actuators["arms"]
    groups = (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint")
    arms.stiffness, arms.damping = dict(zip(groups, (kp, kp, kp_wrist))), dict(zip(groups, (kd, kd, kd_wrist)))
    if limits:
        arms.joint_effort_limit = dict(zip(groups, (limits[0], limits[0], limits[1])))
    print(f"[soft_hand_sim] arm actuators: stiffness {arms.stiffness}, damping {arms.damping}, limit {getattr(arms, 'joint_effort_limit', None)}", flush=True)


def patch_robot(module: ModuleType) -> None:
    """The gains asked for by $HAND_GAINS and $ARM_GAINS."""
    if os.environ.get("HAND_GAINS"):
        soften_hands(module)
    if os.environ.get("ARM_GAINS"):
        soften_arms(module)


def main() -> None:
    sys.meta_path.insert(0, PatchOnImport(ROBOT_MODULE, patch_robot))
    sys.argv[0] = str(SIM_MAIN)
    runpy.run_path(str(SIM_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()
