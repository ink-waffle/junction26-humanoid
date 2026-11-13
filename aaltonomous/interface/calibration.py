"""Camera calibration bundle (spec §4.2 "intrinsics and extrinsics as a calibration file", M7).

    from aaltonomous.interface.calibration import Calibration
    cal = Calibration.load()                       # calibration.yaml next to this file
    p_pelvis = cal.pixel_to_pelvis("head", u, v, depth_mm)
    uv = cal.project("head", p_pelvis)             # pelvis-frame points -> pixels

Frames: `pelvis` = robot pelvis (x fwd, y left, z up). Each camera has `T_pelvis_optical` (4x4), the pose of its
OPTICAL frame (z forward, x right, y down; OpenCV/RealSense convention) in the pelvis frame. Images are 640x480.

Regenerate the head extrinsics from the URDF (waist held at the stiff pose) with
    python -m aaltonomous.interface.calibration --regen
M7 replaces the (default) values with measured ones (RealSense factory intrinsics, AprilTag check, wrist mounts).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
URDF = HERE.parents[1] / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
# optical frame axes expressed in a ROS body frame (x fwd, y left, z up): x_opt = -y, y_opt = -z, z_opt = x
R_LINK_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray            # 3x3
    dist: np.ndarray         # (5,) plumb_bob / brown-conrady
    T_pelvis_optical: np.ndarray | None   # 4x4, None = not calibrated yet
    depth_scale: float = 0.001


class Calibration:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cams = {}
        for name, c in cfg["cameras"].items():
            i = c["intrinsics"]
            K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
            T = None if c.get("T_pelvis_optical") is None else np.asarray(c["T_pelvis_optical"], float)
            self.cams[name] = Camera(i["width"], i["height"], K, np.asarray(i.get("dist", [0] * 5), float), T,
                                     float(c.get("depth_scale", 0.001)))

    @classmethod
    def load(cls, path=None):
        with open(path or HERE / "calibration.yaml") as f:
            return cls(yaml.safe_load(f))

    def _T(self, cam):
        T = self.cams[cam].T_pelvis_optical
        if T is None:
            raise ValueError(f"camera {cam!r} has no extrinsics yet (M7)")
        return T

    def deproject(self, cam, u, v, depth_mm):
        """Pixels + depth (mm, aligned to colour) -> optical-frame points (m). Distortion ignored (D435 colour ~0)."""
        c = self.cams[cam]
        z = np.asarray(depth_mm, float) * c.depth_scale
        x = (np.asarray(u, float) - c.K[0, 2]) / c.K[0, 0] * z
        y = (np.asarray(v, float) - c.K[1, 2]) / c.K[1, 1] * z
        return np.stack([x, y, z], axis=-1)

    def pixel_to_pelvis(self, cam, u, v, depth_mm):
        p = self.deproject(cam, u, v, depth_mm)
        T = self._T(cam)
        return p @ T[:3, :3].T + T[:3, 3]

    def project(self, cam, p_pelvis):
        """Pelvis-frame points (...,3) -> (u, v, z_optical). z <= 0 means behind the camera."""
        T = self._T(cam)
        p = (np.asarray(p_pelvis, float) - T[:3, 3]) @ T[:3, :3]
        c = self.cams[cam]
        z = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = c.K[0, 0] * p[..., 0] / z + c.K[0, 2]
            v = c.K[1, 1] * p[..., 1] / z + c.K[1, 2]
        return np.stack([u, v, z], axis=-1)

    def visible(self, cam, p_pelvis):
        uvz = self.project(cam, p_pelvis)
        c = self.cams[cam]
        return (uvz[..., 2] > 0) & (uvz[..., 0] >= 0) & (uvz[..., 0] < c.width) & (uvz[..., 1] >= 0) & (uvz[..., 1] < c.height)


def head_extrinsics_from_urdf(waist_q=(0.0, 0.0, 0.0), urdf=URDF):
    """T_pelvis_optical of the head D435 from the URDF (d435_link on torso_link), waist at `waist_q`
    (yaw, roll, pitch). Uses Pinocchio with a fixed base at the pelvis, all other joints at 0."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q = pin.neutral(model)
    for name, val in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), waist_q):
        jid = model.getJointId(name)
        q[model.joints[jid].idx_q] = val
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[model.getFrameId("d435_link")]
    T = np.eye(4)
    T[:3, :3] = M.rotation @ R_LINK_OPTICAL
    T[:3, 3] = M.translation
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true", help="recompute head T_pelvis_optical from the URDF")
    ap.add_argument("--waist", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("YAW", "ROLL", "PITCH"))
    a = ap.parse_args(argv)
    path = HERE / "calibration.yaml"
    if a.regen:
        T = head_extrinsics_from_urdf(a.waist)
        text = path.read_text()
        start = text.index("    T_pelvis_optical:  # head")
        end = text.index("\n\n", start)
        rows = "\n".join(f"      - [{', '.join(f'{v:.6f}' for v in r)}]" for r in T)
        path.write_text(text[:start] + "    T_pelvis_optical:  # head: from URDF d435_link, waist "
                        f"{list(a.waist)} (default)\n" + rows + text[end:])
        print(f"updated head extrinsics in {path}")
    cal = Calibration.load()
    for n, c in cal.cams.items():
        print(n, "K fx/fy/cx/cy:", c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2],
              "extrinsics:", "yes" if c.T_pelvis_optical is not None else "MISSING")


if __name__ == "__main__":
    main()
