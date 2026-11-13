#!/usr/bin/env python3
"""Generate joint_spec.yaml: the single source of truth for the challenge action/observation joints.

Reads joint limits from Unitree's G1 29-DoF + Dex3 URDF (xr_teleoperate assets) and combines them
with the DDS motor indices used on the real robot:
  * arms: rt/lowcmd motor indices 15..28 (G1_29_JointArmIndex in xr_teleoperate robot_arm.py)
  * Dex3: rt/dex3/{left,right}/cmd motor indices 0..6 (Dex3_1_{Left,Right}_JointIndex).
    NOTE: the two hands do NOT share a motor order: left is thumb, middle, index;
    right is thumb, index, middle. Their limits are also mirrored (opposite signs).

`arm_q` / `arm_q_target` = left arm 7 then right arm 7, in DDS motor order.
`hand_q` / `hand_q_target` = left Dex3 7 then right Dex3 7, each in its own DDS motor order.

Usage:  python aaltonomous/interface/gen_joint_spec.py   (rewrites joint_spec.yaml next to this file)
"""
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
URDF = REPO / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
OUT = HERE / "joint_spec.yaml"

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
ARM_MOTOR_BASE = {"left": 15, "right": 22}
# DDS motor order per hand (robot_hand_unitree.py Dex3_1_*_JointIndex)
DEX3_ORDER = {
    "left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
    "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
}
# Lower body + waist: held stiff, not part of the action space. Indices 0..14 of rt/lowcmd.
LOCKED_JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
]


def urdf_limits(path):
    text = path.read_text()
    out = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', text, re.S):
        lim = re.search(r"<limit([^>]*)/>", m.group(2))
        attrs = dict(re.findall(r'(\w+)="([^"]+)"', lim.group(1)))
        out[m.group(1)] = {k: float(attrs[k]) for k in ("lower", "upper", "velocity", "effort")}
    return out


def entry(name, lim, **extra):
    return {"name": name, **extra,
            "lower": round(lim["lower"], 6), "upper": round(lim["upper"], 6),
            "velocity": lim["velocity"], "effort": lim["effort"]}


def main():
    lims = urdf_limits(URDF)
    try:
        rev = subprocess.check_output(["git", "-C", str(URDF.parents[3]), "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:
        rev = "unknown"

    arm = []
    for side in ("left", "right"):
        for i, j in enumerate(ARM_JOINTS):
            name = f"{side}_{j}_joint"
            arm.append(entry(name, lims[name], dds_topic="rt/lowcmd", dds_index=ARM_MOTOR_BASE[side] + i))
    hand = []
    for side in ("left", "right"):
        for i, j in enumerate(DEX3_ORDER[side]):
            name = f"{side}_hand_{j}_joint"
            hand.append(entry(name, lims[name], dds_topic=f"rt/dex3/{side}/cmd", dds_index=i))
    locked = [{"name": f"{j}_joint", "dds_topic": "rt/lowcmd", "dds_index": i}
              for i, j in enumerate(LOCKED_JOINTS)]

    spec = {
        "version": 1,
        "generated_by": "aaltonomous/interface/gen_joint_spec.py",
        "source_urdf": f"xr_teleoperate@{rev}:assets/g1/g1_body29_hand14.urdf",
        "units": "rad, rad/s, Nm (URDF values, no safety margin applied)",
        "notes": [
            "arm_q / arm_q_target: index 0-6 left arm, 7-13 right arm, DDS rt/lowcmd motor order.",
            "hand_q / hand_q_target: index 0-6 left Dex3, 7-13 right Dex3, each in its own DDS motor order.",
            "Left Dex3 order is thumb,middle,index; right Dex3 order is thumb,index,middle.",
            "Left and right Dex3 limits are mirrored: the same physical motion has opposite signs.",
            "locked joints (legs, waist) are held stiff by the runner and are not in the action space.",
        ],
        "arm": arm,
        "hand": hand,
        "locked": locked,
    }
    OUT.write_text(yaml.safe_dump(spec, sort_keys=False, width=120))
    print(f"wrote {OUT.relative_to(REPO)}: {len(arm)} arm, {len(hand)} hand, {len(locked)} locked joints")


if __name__ == "__main__":
    main()
