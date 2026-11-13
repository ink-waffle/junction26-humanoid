"""Upstream sim_main.py with our tasks registered: runs third_party/unitree_sim_isaaclab/sim_main.py unchanged
except that `import tasks` is followed by `import aaltonomous.sim.aalto_task` (registers Aalto-* gym ids).
third_party stays unedited. Use via run_sim.sh (TASK=Aalto-Sort-G129-Dex3-Joint), same arguments as sim_main."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UP = REPO / "third_party/unitree_sim_isaaclab"
MAIN = UP / "sim_main.py"

src = MAIN.read_text()
hook = "\nimport tasks\n"
if hook not in src:
    sys.exit("aalto_sim_main: upstream sim_main.py no longer has a top-level `import tasks`; update this shim")
src = src.replace(hook, hook + "import aaltonomous.sim.aalto_task.register  # noqa: E402  (registers Aalto-* tasks)\n", 1)

for p in (str(UP), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(UP)

# Second, isolated instance on the same machine (AALTO_INSTANCE=<tag>, see run_sim.sh): upstream hard-codes DDS
# domain 1 and fixed shared-memory names (DDS bridge + camera images), so two sims would cross-talk. Suffix every
# shm name with the tag and move DDS domain 1 -> $AALTO_DDS_DOMAIN. Ports: TELEIMAGER_CONFIG / AALTO_*_PORT.
_INST = os.environ.get("AALTO_INSTANCE", "")
if _INST:
    # dds.dds_master builds its DDSManager singleton (ChannelFactoryInitialize(1)) at IMPORT time, so the domain
    # patch must be on unitree_sdk2py before that import (patching _dm afterwards left both sims on domain 1).
    _dom = int(os.environ.get("AALTO_DDS_DOMAIN", "1"))
    import unitree_sdk2py.core.channel as _ch
    _orig_cfi = _ch.ChannelFactoryInitialize
    _ch.ChannelFactoryInitialize = lambda id=0, *a, **k: _orig_cfi(_dom if id == 1 else id, *a, **k)
    import dds.dds_master as _dm
    import dds.sharedmemorymanager as _smm
    import tools.shared_memory_utils as _smu
    _orig_smm_init = _smm.SharedMemoryManager.__init__

    def _smm_init(self, name=None, size=512):
        return _orig_smm_init(self, f"{name}_{_INST}" if name else None, size)
    _smm.SharedMemoryManager.__init__ = _smm_init
    _smu.get_shm_name = lambda image_name: f"isaac_{_INST}_{image_name}_image_shm"
    os.environ["TELEIMAGER_SIM_SHM_PREFIX"] = f"isaac_{_INST}_"       # teleimager reader side (subprocess)
    _dm.ChannelFactoryInitialize = _ch.ChannelFactoryInitialize
    # teleimager's config/request responder has a fixed port 60000 and runs in a child interpreter
    # (tools/image_server_process.py): start that child through a -c stub that moves the port first.
    os.environ.setdefault("AALTO_TELEIMAGER_REQ_PORT", "61000")
    import types as _types
    import tools.image_server_process as _isp
    _stub = ("import os, runpy, teleimager.transport.zmq_request as z; o = z.ZMQ_Responser.__init__; "
             "p = int(os.environ['AALTO_TELEIMAGER_REQ_PORT']); "
             "z.ZMQ_Responser.__init__ = lambda s, c, host='0.0.0.0', port=60000: o(s, c, host, p); "
             "runpy.run_module('tools.image_server_process', run_name='__main__')")
    _real_sub = _isp.subprocess

    def _popen(args, *a, **k):
        if list(args[1:3]) == ["-m", "tools.image_server_process"]:
            args = [args[0], "-c", _stub]
        return _real_sub.Popen(args, *a, **k)
    _isp.subprocess = _types.SimpleNamespace(Popen=_popen, PIPE=_real_sub.PIPE, STDOUT=_real_sub.STDOUT,
                                             TimeoutExpired=_real_sub.TimeoutExpired)
    print(f"[aalto] instance '{_INST}': DDS domain {_dom}, shm suffix _{_INST}, "
          f"scene :{os.environ.get('AALTO_SCENE_PORT', 55580)}, rgbd :{os.environ.get('AALTO_RGBD_PORT', 55560)}")

# Upstream publishes the whole scene state as JSON on rt/sim_state every step through a 4 KB shared-memory slot;
# with our 18 objects it overflows and prints a warning per step (and costs ~10 Hz of loop rate). Aalto tasks
# serve object state through the scene service instead (aalto_task/hooks.py), so turn that path off.
if any(a.startswith("Aalto-") for a in sys.argv):
    import dds.sim_state_dds as _ssd
    import tools.data_json_load as _djl
    for _c in vars(_ssd).values():
        if isinstance(_c, type) and hasattr(_c, "write_sim_state_data"):
            _c.write_sim_state_data = lambda self, sim_state_data=None: None
    _djl.sim_state_to_json = lambda state: {}

    # Absolute joint targets. Upstream's DDS action provider writes ABSOLUTE positions for the commanded joints
    # into a zero-filled vector and the action term adds the default pose (use_default_offset=True). That is only
    # right while every default is 0 (upstream arms); our arm home pose is not, and upstream's leg defaults are
    # doubled the same way. Aalto tasks use use_default_offset=False (env_cfg.py) and this wrapper fills every joint
    # that has NOT been commanded (legs, waist, and the arms/hands before the first rt/lowcmd / dex3 cmd) with
    # the default (home) pose instead of 0.
    import torch as _torch
    from action_provider.action_provider_dds import DDSActionProvider as _DAP
    _orig_get_action = _DAP.get_action

    def _get_action_abs(self, env):
        seen = {}

        def _spy(obj, meth, key):
            if obj is None or not hasattr(obj, meth):
                return None
            f = getattr(obj, meth)

            def g(*a, **k):
                r = f(*a, **k)
                seen[key] = r
                return r
            setattr(obj, meth, g)
            return f
        o1 = _spy(getattr(self, "robot_dds", None), "get_robot_command", "arm")
        o2 = _spy(getattr(self, "dex3_dds", None), "get_hand_commands", "hand")
        try:
            out = _orig_get_action(self, env)
        finally:
            if o1 is not None:
                self.robot_dds.get_robot_command = o1
            if o2 is not None:
                self.dex3_dds.get_hand_commands = o2
        if out is None:
            return None
        d = env.scene["robot"].data.default_joint_pos
        d = (d.torch if hasattr(d, "torch") else d)[0].to(out.device)
        keep = _torch.zeros(out.shape[1], dtype=_torch.bool, device=out.device)
        arm = seen.get("arm")
        if arm and "motor_cmd" in arm and len(arm["motor_cmd"]["positions"]) >= 29:
            keep[self._arm_target_idx_t] = True
        hand = seen.get("hand")
        if hand and hand.get("left_hand_cmd") and hand.get("right_hand_cmd") and hasattr(self, "_left_hand_target_idx_t"):
            keep[self._left_hand_target_idx_t] = True
            keep[self._right_hand_target_idx_t] = True
        out[0, ~keep] = d[~keep]
        return out
    _DAP.get_action = _get_action_abs

# AALTO_PROFILE=1: per-step wall-time breakdown (physics / render / observations / scene update / whole step),
# printed every 10 s. Off by default (a few perf_counter calls per step).
if os.environ.get("AALTO_PROFILE") and "_DAP" in globals():   # Aalto tasks only
    import time as _time
    from layeredcontrol.robot_control_system import RobotController as _RC
    _prof = {"n": 0, "t0": _time.perf_counter()}

    def _timed(cls, meth, key):
        f = getattr(cls, meth)

        def g(*a, **k):
            t = _time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                _prof[key] = _prof.get(key, 0.0) + _time.perf_counter() - t
                _prof[key + "#"] = _prof.get(key + "#", 0) + 1
        setattr(cls, meth, g)
    _rc_init, _rc_step = _RC.__init__, _RC.step

    def _rc_init_prof(self, env, *a, **k):   # Isaac Lab classes may only be imported after the app launched
        from isaaclab.sim import SimulationContext as _SC
        from isaaclab.managers import ObservationManager as _OM
        from isaaclab.scene import InteractiveScene as _IS
        _timed(_SC, "step", "physics")
        _timed(_SC, "render", "render")
        _timed(_OM, "compute", "obs")
        _timed(_IS, "update", "scene_update")
        _timed(_DAP, "get_action", "action")
        _rc_init(self, env, *a, **k)
        om = env.observation_manager
        for name, cfg in zip(om._group_obs_term_names["policy"], om._group_obs_term_cfgs["policy"]):
            f = cfg.func

            def g(*a, _f=f, _k="obs:" + name, **k):
                t = _time.perf_counter()
                try:
                    return _f(*a, **k)
                finally:
                    _prof[_k] = _prof.get(_k, 0.0) + _time.perf_counter() - t
                    _prof[_k + "#"] = _prof.get(_k + "#", 0) + 1
            cfg.func = g
            _prof_keys.append("obs:" + name)
    _prof_keys = ["physics", "render", "obs", "scene_update", "action"]
    _RC.__init__ = _rc_init_prof

    def _rc_step_prof(self):
        t = _time.perf_counter()
        _rc_step(self)
        _prof["n"] += 1
        _prof["loop"] = _prof.get("loop", 0.0) + _time.perf_counter() - t
        wall = _time.perf_counter() - _prof["t0"]
        if wall >= 10.0:
            n = _prof["n"]
            parts = " ".join(f"{k}={_prof[k] / n * 1e3:.1f}ms(x{_prof[k + '#'] / n:.2f})"
                             for k in _prof_keys if k in _prof)
            print(f"[aalto-prof] {n / wall:.1f} steps/s, RTF {n * self.env.step_dt / wall:.2f}, per step: "
                  f"controller.step {_prof['loop'] / n * 1e3:.1f}ms = {parts}", flush=True)
            _prof.clear()
            _prof.update(n=0, t0=_time.perf_counter())
    _RC.step = _rc_step_prof

sys.argv[0] = str(MAIN)
exec(compile(src, str(MAIN), "exec"), {"__name__": "__main__", "__file__": str(MAIN)})
