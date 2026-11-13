"""Robot backends behind AaltoEnv. Each one moves arms + Dex3 to joint targets and returns observations.

  MockBackend -- offline kinematic stand-in (joints follow the command with a first-order lag, blank images).
                 No sim physics, no cubes: for testing interface, runner and policy plumbing on any laptop.
  DDSBackend  -- Unitree DDS. The same code drives Isaac (unitree_sim_isaaclab, DDS domain 1) and the
                 real robot (domain 0, wired interface). Arms + stiff legs/waist go through upstream
                 xr_teleoperate's G1_29_ArmController in debug mode (rt/lowcmd), i.e. exactly the mode the
                 VR teleop uses; Dex3 through rt/dex3/{left,right}/cmd at 100 Hz. Cameras via teleimager.
                 Real robot: the robot must be in debug mode (locomotion service off), on the gantry.
                 Sim: `clock` = SimClock (the sim's step clock, aalto_task/hooks.py), so AaltoEnv paces its 30 Hz
                 on sim time; None on the real robot or if the sim publishes no clock (wall-time pacing then).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

from .safety import N

H, W = 480, 640
XR = Path(__file__).resolve().parents[2] / "third_party/xr_teleoperate"


def blank_images():
    return {"head_rgb": np.zeros((H, W, 3), np.uint8), "head_depth": np.zeros((H, W), np.uint16),
            "left_wrist_rgb": np.zeros((H, W, 3), np.uint8), "right_wrist_rgb": np.zeros((H, W, 3), np.uint8)}


class MockBackend:
    name = "mock"
    has_success = False

    def __init__(self, tau=0.05, arm_q0=None, hand_q0=None, **_):
        self.tau = tau
        self.arm_q = np.zeros(N) if arm_q0 is None else np.asarray(arm_q0, float).copy()
        self.hand_q = np.zeros(N) if hand_q0 is None else np.asarray(hand_q0, float).copy()
        self.arm_dq = np.zeros(N)
        self.arm_cmd, self.hand_cmd = self.arm_q.copy(), self.hand_q.copy()
        self.layout = None

    def reset_scene(self, layout, instruction):
        self.layout = layout

    def joints(self):
        return self.arm_q.copy(), self.arm_dq.copy(), self.hand_q.copy()

    def command(self, arm_q, hand_q):
        self.arm_cmd, self.hand_cmd = np.asarray(arm_q, float), np.asarray(hand_q, float)

    def advance(self, dt):
        """Called once per tick by the env (only meaningful for the mock)."""
        a = 1.0 - np.exp(-dt / self.tau)
        new = self.arm_q + a * (self.arm_cmd - self.arm_q)
        self.arm_dq = (new - self.arm_q) / dt
        self.arm_q = new
        self.hand_q = self.hand_q + a * (self.hand_cmd - self.hand_q)

    def images(self):
        return blank_images()

    def close(self):
        pass


def make_safe_arm_controller(base):
    """Subclass of upstream G1_29_ArmController whose target is the MEASURED pose until arm() is called.

    Upstream starts q_target = 0 and its 250 Hz thread drives the arms toward zero from the constructor,
    past our safety filter. Here the getter returns the measured pose while unarmed, so nothing moves."""

    class SafeArm(base):
        _armed = False
        _qt = np.zeros(14)

        @property
        def q_target(self):
            return self._qt if self._armed else np.asarray(self.get_current_dual_arm_q(), float)

        @q_target.setter
        def q_target(self, v):
            if self._armed:
                self._qt = np.asarray(v, float)

        def arm(self):
            self._qt = np.asarray(self.get_current_dual_arm_q(), float).copy()
            self._armed = True

    return SafeArm


def enter_debug_mode(switcher):
    """Release Unitree's locomotion service so it stops writing rt/lowcmd. Raises if it did not work."""
    status, result = switcher.Enter_Debug_Mode()
    if status != 0 or (result or {}).get("name"):
        raise RuntimeError(f"could not enter debug mode (status={status}, result={result}); "
                           "refusing to start: the locomotion service would fight our rt/lowcmd")


class _V2ImageClient:
    """teleimager 2.x (`teleimager.client.TeleImageClient`, one ZMQ client per camera) behind the 1.x
    ImageClient API (get_{head,left_wrist,right_wrist}_frame -> object with .bgr)."""
    PORTS = (("head_camera", 55555), ("left_wrist_camera", 55556), ("right_wrist_camera", 55557))

    def __init__(self, cls, host):
        # 2nd sim instance (AALTO_INSTANCE): cameras move to AALTO_CAM_PORT_BASE .. +2 (sim/README "2nd sim")
        off = int(os.environ.get("AALTO_CAM_PORT_BASE", self.PORTS[0][1])) - self.PORTS[0][1]
        self.c = {k: cls(k, server_host=host, zmq_port=p + off, request_bgr=True) for k, p in self.PORTS}

    def get_head_frame(self):
        f = self.c["head_camera"].get_frame()
        if f is not None and f.bgr is not None and f.bgr.shape[1] == 1280:  # stereo side-by-side -> left eye
            f = types.SimpleNamespace(bgr=f.bgr[:, :640])
        return f

    def get_left_wrist_frame(self):
        return self.c["left_wrist_camera"].get_frame()

    def get_right_wrist_frame(self):
        return self.c["right_wrist_camera"].get_frame()

    def close(self):
        for c in self.c.values():
            c.close()


def make_image_client(host):
    """Pick the teleimager API: 2.x `teleimager.client` if importable (sim env), else 1.x
    `teleimager.image_client` (xr_teleoperate checkout; real robot / laptop). Override: AALTO_TELEIMAGER=1|2."""
    import os
    want = os.environ.get("AALTO_TELEIMAGER", "")
    if want != "1":
        try:
            from teleimager.client import TeleImageClient
            return _V2ImageClient(TeleImageClient, host)
        except ImportError:
            if want == "2":
                raise
    try:
        from teleimager.image_client import ImageClient
    except ImportError:
        sys.path.insert(0, str(XR / "teleop/teleimager/src"))
        from teleimager.image_client import ImageClient
    return ImageClient(host=host, request_bgr=True)


class SimClock:
    """Sim time from the Aalto sim (aalto_task/hooks.py publishes <sim_t, wall> per env step on zmq PUB
    :AALTO_CLOCK_PORT, default scene port + 1). The sim runs below real time when loaded (~0.5-0.65x with cameras),
    so the sim backend paces on this clock: the physics then sees 30 Hz commands at 30 Hz of sim time."""

    STALL_S = 5.0     # no clock message for this long (wall) -> the sim is dead or paused: raise

    def __init__(self, host, port=None, wait_s=3.0):
        import zmq
        port = port or int(os.environ.get("AALTO_CLOCK_PORT", int(os.environ.get("AALTO_SCENE_PORT", "55580")) + 1))
        self.zmq, self.addr = zmq, f"tcp://{host}:{port}"
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(self.addr)
        self.t, self._wall = None, time.monotonic()
        if not self.sock.poll(int(wait_s * 1000)):
            self.sock.close(0)
            raise TimeoutError(f"no sim clock on {self.addr}")
        self._recv()

    def _recv(self):
        import struct
        self.t = struct.unpack("<dd", self.sock.recv())[0]
        self._wall = time.monotonic()

    def now(self):
        """Latest sim time [s] (non-blocking)."""
        while self.sock.poll(0):
            self._recv()
        return self.t

    def wait_until(self, t):
        """Block until sim time >= t. Raises RuntimeError if the clock stalls for STALL_S."""
        while self.now() < t - 1e-6:
            if self.sock.poll(int(self.STALL_S * 1000)):
                self._recv()
            elif time.monotonic() - self._wall > self.STALL_S:
                raise RuntimeError(f"sim clock {self.addr} stalled for {self.STALL_S:.0f} s (sim dead or paused?)")
        return self.t

    def close(self):
        self.sock.close(0)


class DDSBackend:
    has_success = False  # sim outcome: aaltonomous.sim.scene_client.SceneClient(...).evaluate(task, mapping)

    def __init__(self, sim, network_interface=None, img_server_ip="192.168.123.164", cameras=True,
                 hand_kp=1.5, hand_kd=0.2, hand_hz=100.0, confirm=None, head_rgbd=False, head_rgbd_port=55560,
                 sim_clock=None, **_):
        self.name = "sim" if sim else "real"
        if sim_clock is None:   # AALTO_SIM_CLOCK=0: old wall-time pacing (comparison / escape hatch)
            sim_clock = os.environ.get("AALTO_SIM_CLOCK", "1") != "0"
        self.confirm = confirm or (lambda card: input(f"\n{card}\nPlace the cubes, then press Enter... "))
        if str(XR) not in sys.path:
            sys.path.insert(0, str(XR))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        from teleop.robot_control.robot_arm import G1_29_ArmController

        ChannelFactoryInitialize(1 if sim else 0, networkInterface=network_interface)
        if not sim:  # real robot: Unitree's own service must be off before we write rt/lowcmd
            from teleop.utils.motion_switcher import MotionSwitcher
            enter_debug_mode(MotionSwitcher())
        # Locks legs + waist at their CURRENT pose (kp 300) and streams arm targets at 250 Hz.
        # Target = measured pose until arm() below (upstream would otherwise drive the arms to zero).
        self.arm = make_safe_arm_controller(G1_29_ArmController)(motion_mode=False, simulation_mode=sim)
        self.arm.arm()

        self.hand_pub, self.hand_sub, self.hand_msg = {}, {}, {}
        self.hand_state = np.zeros(N)
        for k, side in enumerate(("left", "right")):
            pub = ChannelPublisher(f"rt/dex3/{side}/cmd", HandCmd_)
            pub.Init()
            sub = ChannelSubscriber(f"rt/dex3/{side}/state", HandState_)
            sub.Init(lambda m, k=k: self._on_hand(k, m), 10)
            msg = unitree_hg_msg_dds__HandCmd_()
            for i in range(7):
                msg.motor_cmd[i].mode = (i & 0x0F) | (0x01 << 4)  # RIS mode: id, status=1, no timeout
                msg.motor_cmd[i].kp, msg.motor_cmd[i].kd = hand_kp, hand_kd
            self.hand_pub[side], self.hand_sub[side], self.hand_msg[side] = pub, sub, msg
        t0 = time.time()
        while not self.hand_state.any() and time.time() - t0 < 5.0:
            time.sleep(0.01)
        self.hand_target = self.hand_state.copy()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._hand_loop, args=(1.0 / hand_hz,), daemon=True).start()

        self.img = None
        if cameras:
            self.img = make_image_client(img_server_ip)
        self.clock = None
        if sim and sim_clock:
            try:
                self.clock = SimClock(img_server_ip)
                print(f"[DDSBackend] pacing on the sim clock ({self.clock.addr}, sim t {self.clock.t:.2f} s)")
            except Exception as e:
                print(f"[DDSBackend] WARNING no sim clock ({e}); pacing on WALL time - if the sim runs below real "
                      f"time the physics sees motions faster than commanded (restart the sim to get the clock)")
        self.scene = None
        if sim:  # Aalto-* tasks place layouts through their scene service; upstream tasks have none
            try:
                from aaltonomous.sim.scene_client import SceneClient
                self.scene = SceneClient(img_server_ip, timeout_ms=3000)
                self.scene.ping()
            except Exception as e:
                print(f"[DDSBackend] no sim scene service ({e}); reset_scene will not place objects")
                self.scene = None
        self.rgbd = None  # head RGB-D from pc2/head_rgbd_server.py (needs teleimager head cam off)
        if head_rgbd:
            from aaltonomous.interface.head_rgbd import HeadRGBDClient
            self.rgbd = HeadRGBDClient(img_server_ip, head_rgbd_port)

    def _on_hand(self, k, msg):
        self.hand_state[7 * k: 7 * k + 7] = [msg.motor_state[i].q for i in range(7)]

    def _hand_loop(self, dt):
        while not self._stop.is_set():
            with self._lock:
                tgt = self.hand_target.copy()
            for k, side in enumerate(("left", "right")):
                msg = self.hand_msg[side]
                for i in range(7):
                    msg.motor_cmd[i].q = float(tgt[7 * k + i])
                self.hand_pub[side].Write(msg)
            time.sleep(dt)

    def reset_scene(self, layout, instruction):
        if self.name == "real":
            from .tasks import layout_card
            self.confirm(layout_card(layout, instruction))
        elif self.scene is not None:  # sim: Aalto-* task scene service (aalto_task/hooks.py)
            self.scene.reset(layout)

    def joints(self):
        return (np.asarray(self.arm.get_current_dual_arm_q(), float),
                np.asarray(self.arm.get_current_dual_arm_dq(), float), self.hand_state.copy())

    def command(self, arm_q, hand_q):
        self.arm.ctrl_dual_arm(np.asarray(arm_q, float), np.zeros(N))
        with self._lock:
            self.hand_target = np.asarray(hand_q, float).copy()

    def advance(self, dt):
        pass

    def images(self):
        out = blank_images()
        rgbd = self.rgbd.get() if self.rgbd is not None else None
        if rgbd is not None:  # else fall back to teleimager colour + zero depth
            out["head_rgb"], out["head_depth"] = rgbd[0], rgbd[1]
        if self.img is None:
            return out
        import cv2
        cams = [("left_wrist_rgb", self.img.get_left_wrist_frame), ("right_wrist_rgb", self.img.get_right_wrist_frame)]
        if rgbd is None:
            cams.insert(0, ("head_rgb", self.img.get_head_frame))
        for key, get in cams:
            try:
                f = get()
                if f is not None and f.bgr is not None:
                    out[key] = cv2.resize(cv2.cvtColor(f.bgr, cv2.COLOR_BGR2RGB), (W, H))
            except Exception:  # camera disabled in the server config
                pass
        return out

    def close(self):
        self._stop.set()
        if self.clock is not None:
            self.clock.close()
        if self.rgbd is not None:
            self.rgbd.close()
        if self.img is not None:
            self.img.close()


def make_backend(name, **kw):
    if name == "mock":
        return MockBackend(**kw)
    if name in ("sim", "real"):
        return DDSBackend(sim=(name == "sim"), **kw)
    raise ValueError(f"unknown backend {name!r}")
