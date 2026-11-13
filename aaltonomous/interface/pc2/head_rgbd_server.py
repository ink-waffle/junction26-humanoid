"""Head RGB-D publisher for PC2 (Jetson, Ubuntu 20.04, Python 3.8 OK). Deps: pyrealsense2, numpy, pyzmq
(cv2 optional: JPEG colour, else raw RGB).

Design choice: a standalone server that OWNS the head D435i, used INSTEAD of teleimager's head camera
during challenge runs. Only one process can open the RealSense, and teleimager (type uvc/realsense in
cam_config_server.yaml) only streams colour, so patching it was rejected as more invasive. Teleimager
is configured per camera (head_camera / left_wrist_camera / right_wrist_camera, ports 55555-55557), so
it keeps serving the wrist cams: in the server yaml set head_camera enable_zmq: false and
enable_webrtc: false ("camera will not start"), leave the wrist cameras on.

Wire format: zmq PUB (CONFLATE, latest frame only; CONFLATE forbids multipart, so ONE message), tcp
port 55560: struct "<III" lengths (header, colour, depth) followed by the three blobs:
  [0] JSON header {"v":1,"frame_id":int,"stamp":float (time.time() at capture, PC2 clock),
      "h":480,"w":640,"rgb_enc":"jpeg"|"raw","depth_enc":"zlib-u16",
      "intr": {"fx","fy","cx","cy","model","dist"[5]} factory colour intrinsics (depth is aligned to colour,
      so it shares them) or null; read them with `python -m aaltonomous.interface.head_rgbd --intrinsics`}
  [1] colour (RGB order; JPEG or raw H*W*3 uint8)
  [2] depth, zlib(uint16 millimetres, little endian, H*W, aligned to colour, 0 = invalid)
Note the PC2 and laptop clocks may differ; use frame_id/stamp deltas, not absolute comparison.

Run on PC2 (ssh g1; with teleimager's head camera disabled / stopped):
    python3 head_rgbd_server.py [--port 55560] [--fps 30] [--serial SN]
Copy: scp aaltonomous/interface/pc2/head_rgbd_server.py g1:~/  (scripts/deploy_pc2.sh is the SONIC
bundle installer and is not used for this).
"""
import argparse
import json
import struct
import threading
import time
import zlib

import numpy as np
import zmq

H, W = 480, 640


class RealSenseSource:
    """Yields (rgb uint8 HxWx3, depth uint16 HxW mm aligned to colour, stamp). Blocks until a frame."""

    def __init__(self, serial=None, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        # z16 units are device-dependent (D435i default 1 mm); convert to mm explicitly.
        self.scale_mm = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        self.align = rs.align(rs.stream.color)
        # Factory colour intrinsics (only readable by the process that owns the camera, so publish them).
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {"fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy, "model": str(i.model),
                           "dist": [float(c) for c in i.coeffs]}

    def read(self):
        fs = self.align.process(self.pipe.wait_for_frames(5000))
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        rgb = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data())
        if abs(self.scale_mm - 1.0) > 1e-3:
            depth = np.clip(depth * self.scale_mm, 0, 65535)
        return rgb, depth.astype(np.uint16), time.time()

    def close(self):
        self.pipe.stop()


class HeadRGBDServer:
    def __init__(self, source, port=55560, host="*", jpeg_quality=85, ctx=None):
        self.source, self.jpeg_quality = source, jpeg_quality
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = port
        if port == 0:
            self.port = self.sock.bind_to_random_port("tcp://" + host)
        else:
            self.sock.bind("tcp://%s:%d" % (host, port))
        self.frame_id = 0
        self._stop = threading.Event()
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def publish_one(self):
        rgb, depth, stamp = self.source.read()
        rgb = np.ascontiguousarray(rgb, np.uint8)
        depth = np.ascontiguousarray(depth, "<u2")
        if self.cv2 is not None:
            ok, buf = self.cv2.imencode(".jpg", rgb[:, :, ::-1], [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            col, enc = buf.tobytes(), "jpeg"
        else:
            col, enc = rgb.tobytes(), "raw"
        hdr = {"v": 1, "frame_id": self.frame_id, "stamp": stamp, "h": depth.shape[0], "w": depth.shape[1],
               "rgb_enc": enc, "depth_enc": "zlib-u16", "intr": getattr(self.source, "intrinsics", None)}
        hb, db = json.dumps(hdr).encode(), zlib.compress(depth.tobytes(), 1)
        self.sock.send(struct.pack("<III", len(hb), len(col), len(db)) + hb + col + db)
        self.frame_id += 1

    def serve_forever(self):
        while not self._stop.is_set():
            self.publish_one()

    def start(self):
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()
        time.sleep(0.05)
        self.sock.close(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=55560)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default=None)
    a = ap.parse_args()
    src = RealSenseSource(a.serial, a.fps)
    srv = HeadRGBDServer(src, a.port)
    print("head RGB-D publishing on tcp://*:%d" % srv.port)
    print("colour intrinsics:", json.dumps(src.intrinsics))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    main()
