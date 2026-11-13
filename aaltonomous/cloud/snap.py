"""Save one frame per sim camera as JPEG/PNG and print mean brightness (all ~0 = shader cache still cold).

    python aaltonomous/cloud/snap.py [--out DIR] [--host 127.0.0.1] [--secs 5]
Ports follow AALTO_CAM_PORT_BASE (head mono, +1 left wrist, +2 right wrist) and AALTO_RGBD_PORT (head RGB-D).
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/aalto_snap")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--secs", type=float, default=5.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = int(os.environ.get("AALTO_CAM_PORT_BASE", "55555"))
    rgbd_port = int(os.environ.get("AALTO_RGBD_PORT", "55560"))

    from teleimager.client import TeleImageClient
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    cams = {t: TeleImageClient(t, server_host=a.host, zmq_port=base + i, request_bgr=True)
            for i, t in enumerate(("head_camera", "left_wrist_camera", "right_wrist_camera"))}
    rgbd = HeadRGBDClient(a.host, rgbd_port)
    got, t0 = {}, time.time()
    while time.time() - t0 < a.secs and len(got) < 4:
        for k, c in cams.items():
            fr = c.get_frame()
            if fr.bgr is not None:
                got[k] = fr.bgr
        f = rgbd.get()
        if f is not None:
            got["head_rgbd"] = f
        time.sleep(0.05)
    rgbd.close()
    for k in ("head_camera", "left_wrist_camera", "right_wrist_camera", "head_rgbd"):
        if k not in got:
            print(f"{k:20s} NO FRAME")
            continue
        if k == "head_rgbd":
            rgb, depth = got[k][0], got[k][1]
            cv2.imwrite(str(out / "head_rgbd_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "head_rgbd_depth.png"), depth)
            print(f"{k:20s} {rgb.shape} mean {rgb.mean():5.1f}  depth valid {np.mean(depth > 0) * 100:4.1f} %")
        else:
            cv2.imwrite(str(out / f"{k}.jpg"), got[k])
            print(f"{k:20s} {got[k].shape} mean {got[k].mean():5.1f}")
    print(f"saved to {out}")
    os._exit(0)   # teleimager client threads do not stop on their own


if __name__ == "__main__":
    main()
