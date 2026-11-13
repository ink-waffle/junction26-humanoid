import time
import unittest

import numpy as np

from aaltonomous.interface.head_rgbd import HeadRGBDClient
from aaltonomous.interface.pc2.head_rgbd_server import HeadRGBDServer


class FakeSource:
    def __init__(self):
        self.n = 0

    def read(self):
        time.sleep(0.01)
        self.n += 1
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[:, :] = (200, 100, 50)  # flat colour survives JPEG closely
        depth = np.full((480, 640), 1234, np.uint16)
        depth[0, 0] = 0
        depth[10, 10] = 65000
        return rgb, depth, time.time()


class TestHeadRGBD(unittest.TestCase):
    def test_roundtrip(self):
        srv = HeadRGBDServer(FakeSource(), port=0, host="127.0.0.1")
        cli = HeadRGBDClient("127.0.0.1", srv.port)
        srv.start()
        try:
            t0 = time.time()
            f = None
            while f is None and time.time() - t0 < 5:
                f = cli.get()
                time.sleep(0.01)
            self.assertIsNotNone(f)
            rgb, depth, stamp, fid = f
            self.assertEqual(rgb.shape, (480, 640, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertLess(abs(int(rgb[240, 320, 0]) - 200), 6)
            self.assertLess(abs(int(rgb[240, 320, 2]) - 50), 6)
            self.assertEqual(depth.shape, (480, 640))
            self.assertEqual(depth.dtype, np.uint16)
            self.assertEqual((depth[0, 0], depth[10, 10], depth[100, 100]), (0, 65000, 1234))
            self.assertLess(abs(time.time() - stamp), 5)
            time.sleep(0.2)
            self.assertGreater(cli.get()[3], fid)  # newer frames arrive
            self.assertLess(cli.age(), 0.2)        # laptop-side arrival age while the server streams
            srv.close()
            time.sleep(0.4)
            self.assertGreater(cli.age(), 0.3)     # server gone: age grows, so the recorder can refuse
        finally:
            cli.close()
            srv.close()

    def test_no_server_returns_none(self):
        cli = HeadRGBDClient("127.0.0.1", 59999)
        try:
            time.sleep(0.2)
            self.assertIsNone(cli.get())
            self.assertEqual(cli.age(), float("inf"))
        finally:
            cli.close()


if __name__ == "__main__":
    unittest.main()
