"""Probe a running unitree_sim_isaaclab: DDS topic rates (domain 1) + teleimager camera frames.

    conda activate isaac
    python aaltonomous/sim/probe_sim.py --secs 10 --save /tmp/probe   # sim already running

Prints msgs/s per topic, joint count and a sample of q, and camera frame shape + receive rate.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--iface", default=None, help="DDS interface (default: CycloneDDS picks; use 'lo' for loopback)")
    ap.add_argument("--host", default="127.0.0.1", help="teleimager host")
    ap.add_argument("--save", default=None, help="dir to save one JPEG per camera")
    ap.add_argument("--no-cams", action="store_true")
    a = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

    ChannelFactoryInitialize(a.domain, a.iface) if a.iface else ChannelFactoryInitialize(a.domain)
    stats = {}

    def cb(name):
        def f(msg):
            s = stats.setdefault(name, {"n": 0, "last": None})
            s["n"] += 1
            s["last"] = msg
        return f

    subs = []
    for name, typ in (("rt/lowstate", LowState_), ("rt/dex3/left/state", HandState_), ("rt/dex3/right/state", HandState_)):
        sub = ChannelSubscriber(name, typ)
        sub.Init(cb(name), 10)
        subs.append(sub)

    cams = {}
    if not a.no_cams:
        from teleimager.client import TeleImageClient
        for topic, port in (("head_camera", 55555), ("left_wrist_camera", 55556), ("right_wrist_camera", 55557)):
            cams[topic] = TeleImageClient(topic, server_host=a.host, zmq_port=port, request_bgr=True)
    cam_n = {k: 0 for k in cams}
    cam_last = {}
    t0 = time.time()
    last_ts = {}
    while time.time() - t0 < a.secs:
        for k, c in cams.items():
            fr = c.get_frame()
            if fr.jpg is not None and id(fr.jpg) != last_ts.get(k):
                last_ts[k] = id(fr.jpg)
                cam_n[k] += 1
                cam_last[k] = fr
        time.sleep(0.005)
    dt = time.time() - t0

    print(f"\n=== DDS domain {a.domain}, {dt:.1f} s ===")
    for name in ("rt/lowstate", "rt/dex3/left/state", "rt/dex3/right/state"):
        s = stats.get(name)
        if not s:
            print(f"  {name:22s} NO MESSAGES")
            continue
        m = s["last"]
        q = [m.motor_state[i].q for i in range(len(m.motor_state))]
        print(f"  {name:22s} {s['n'] / dt:7.1f} msg/s  motors={len(q)}  q[:4]={[round(x, 3) for x in q[:4]]}")
    if cams:
        print("=== cameras (teleimager ZMQ) ===")
        for k in cams:
            fr = cam_last.get(k)
            shape = None if fr is None or fr.bgr is None else fr.bgr.shape
            print(f"  {k:20s} {cam_n[k] / dt:6.1f} new frames/s  shape={shape}")
            if a.save and fr is not None and fr.jpg is not None:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                (Path(a.save) / f"{k}.jpg").write_bytes(bytes(fr.jpg))
        for c in cams.values():
            c.close()


if __name__ == "__main__":
    main()
