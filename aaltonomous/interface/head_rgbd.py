"""Laptop-side client for pc2/head_rgbd_server.py (see its docstring for the wire format)."""
import json
import struct
import threading
import time
import zlib

import numpy as np
import zmq


class HeadRGBDClient:
    """Background thread keeps only the newest frame; get() never blocks.

    get() -> (rgb uint8 HxWx3, depth uint16 HxW mm, stamp float, frame_id int) or None if nothing yet.
    age() -> seconds since the newest frame ARRIVED (laptop monotonic clock; inf before the first frame).
    `stamp` is PC2's clock, so it cannot say whether a frame is fresh here; age() can.
    """

    def __init__(self, host="192.168.123.164", port=55560, ctx=None):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 100)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect("tcp://%s:%d" % (host, port))
        self._latest, self._lock, self._stop = None, threading.Lock(), threading.Event()
        self._t_recv = None
        self.intrinsics = None  # factory colour intrinsics from the server header (None on old servers)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                parts = self.sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            try:
                f = self._decode(parts)
                self.intrinsics = json.loads(parts[12:12 + struct.unpack_from("<I", parts)[0]]).get("intr")
            except Exception:
                continue
            with self._lock:
                self._latest, self._t_recv = f, time.monotonic()

    @staticmethod
    def _decode(msg):
        a, b, c = struct.unpack_from("<III", msg)
        o = 12
        parts = [msg[o:o + a], msg[o + a:o + a + b], msg[o + a + b:o + a + b + c]]
        hdr = json.loads(parts[0].decode())
        h, w = hdr["h"], hdr["w"]
        if hdr["rgb_enc"] == "jpeg":
            import cv2
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            rgb = np.frombuffer(parts[1], np.uint8).reshape(h, w, 3).copy()
        depth = np.frombuffer(zlib.decompress(parts[2]), "<u2").reshape(h, w).astype(np.uint16)
        return rgb, depth, hdr["stamp"], hdr["frame_id"]

    def get(self):
        with self._lock:
            return self._latest

    def age(self):
        with self._lock:
            return float("inf") if self._t_recv is None else time.monotonic() - self._t_recv

    def close(self):
        self._stop.set()
        self._t.join(1.0)
        self.sock.close(0)


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Print one head RGB-D frame's info and the camera's factory intrinsics.")
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--port", type=int, default=55560)
    ap.add_argument("--intrinsics", action="store_true", help="print the calibration.yaml intrinsics line")
    a = ap.parse_args()
    cli = HeadRGBDClient(a.host, a.port)
    t0 = time.time()
    while cli.get() is None and time.time() - t0 < 5:
        time.sleep(0.05)
    f = cli.get()
    if f is None:
        raise SystemExit("no frame from %s:%d" % (a.host, a.port))
    rgb, depth, stamp, fid = f
    print("frame %d rgb %s depth %s valid %.0f%% median %d mm" % (fid, rgb.shape, depth.shape,
          100.0 * (depth > 0).mean(), int(np.median(depth[depth > 0])) if (depth > 0).any() else 0))
    i = cli.intrinsics
    if i is None:
        print("server sends no intrinsics (old head_rgbd_server.py; redeploy and restart it)")
    elif a.intrinsics:
        print("intrinsics: {width: %d, height: %d, fx: %.3f, fy: %.3f, cx: %.3f, cy: %.3f, dist: [%s]}  # factory, %s"
              % (depth.shape[1], depth.shape[0], i["fx"], i["fy"], i["cx"], i["cy"],
                 ", ".join("%.6g" % c for c in i["dist"]), i["model"]))
    else:
        print("intr", i)
    cli.close()
