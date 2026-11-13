# Run check_interface --backend sim with a shim mapping xr_teleoperate's teleimager 1.x ImageClient API
# onto the sim's teleimager 2.x TeleImageClient (no edits to aaltonomous/interface).
import sys, types
from teleimager.client import TeleImageClient

class ImageClient:
    def __init__(self, host, request_bgr=True, **_):
        self.c = {k: TeleImageClient(k, server_host=host, zmq_port=p, request_bgr=request_bgr)
                  for k, p in (("head_camera", 55555), ("left_wrist_camera", 55556), ("right_wrist_camera", 55557))}
    def get_head_frame(self):
        f = self.c["head_camera"].get_frame()
        if f.bgr is not None and f.bgr.shape[1] == 1280:   # stereo side-by-side -> left eye
            f = types.SimpleNamespace(bgr=f.bgr[:, :640])
        return f
    def get_left_wrist_frame(self): return self.c["left_wrist_camera"].get_frame()
    def get_right_wrist_frame(self): return self.c["right_wrist_camera"].get_frame()
    def close(self):
        for c in self.c.values(): c.close()

m = types.ModuleType("teleimager.image_client"); m.ImageClient = ImageClient
sys.modules["teleimager.image_client"] = m
sys.path.insert(0, "/home/erkka/DIGITWIN/UNITREE-G1-RESEARCH")
from aaltonomous.interface.check_interface import main
sys.exit(main(sys.argv[1:]))
