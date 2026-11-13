"""Runtime side of the Aalto task, called once per env step from an observation term (main thread, so it may
write sim state safely).

1. Head RGB-D: the head pinhole camera's rgb + distance_to_image_plane are published on tcp :55560 in the exact
   wire format of interface/pc2/head_rgbd_server.py (uint16 mm depth, 0 = invalid), so AaltoEnv reads sim and
   real depth through the same HeadRGBDClient (DDSBackend head_rgbd=True).
2. Scene service: zmq REP on tcp :55580, JSON in / JSON out (client: aaltonomous/sim/scene_client.py):
     {"cmd": "reset", "layout": [{color, shape, x, y, yaw}, ...]}   place a layout (mat coords), park the rest
     {"cmd": "reset_random", "seed": int|null}                      sample + place (scene.yaml layout knobs)
     {"cmd": "state"}                                               objects: name, color, shape, active, pos
                                                                    (pelvis frame), quat_xyzw, speed (m/s)
     {"cmd": "evaluate", "task": "sort_fixed"|"sort_language"|"insert_shapes", "mapping": {...}|null}
                                                                    spec §3 outcomes + score (Scene checker)
     {"cmd": "ping"}
     {"cmd": "info"}                                                DR config (null = off), ports
     {"cmd": "randomize", "seed": int|null}                         DR: sample + apply one parameter set (dr.py);
                                                                    no-op returning params null when DR is off
   Ports via env AALTO_RGBD_PORT / AALTO_SCENE_PORT; 0 disables.
   ping / info also return the sim clock: sim_t (s), rtf (sim/wall over the last ~10 s window), rtf_mean.
4. Sim clock (real-time governor): zmq PUB on tcp :AALTO_CLOCK_PORT (default scene port + 1 = 55581), one
   b"<sim_t float64 LE><wall float64 LE>" message per env step. The sim runs slower than real time (rendering);
   AaltoEnv's sim backend paces its 30 Hz ticks on this clock instead of wall time (interface/backends.py SimClock),
   so the physics sees commands at the intended rate whatever the machine load. Logs a warning when RTF < 0.95.
3. Domain randomization (dr.py), only if AALTO_DR names a config: lights, mat/counter/bin/object colours,
   object mass/friction, layout pose jitter, head camera pose + RGB-D noise (noise on :55560 only; the teleimager
   mono head stream and the wrist cameras get the lighting/colour DR but no pixel noise).
"""
from __future__ import annotations

import json
import os
import struct
import time

import numpy as np
import torch

from aaltonomous.scene.scene import Scene
from . import dr
from .layout import PELVIS_W, assign, evaluate, object_names, parse_name, random_layout

_RT = None


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class AaltoRuntime:
    def __init__(self, env):
        import zmq
        self.sc = Scene.load()
        self.names = object_names(self.sc)
        self.layout, self.active = [], []
        self.ctx = zmq.Context.instance()
        self.rep = self.rgbd = None
        sp = int(os.environ.get("AALTO_SCENE_PORT", "55580"))
        if sp:
            self.rep = self.ctx.socket(zmq.REP)
            self.rep.setsockopt(zmq.LINGER, 0)
            self.rep.bind(f"tcp://*:{sp}")
            print(f"[aalto] scene service on tcp :{sp}")
        rp = int(os.environ.get("AALTO_RGBD_PORT", "55560"))
        if rp and "front_camera" in env.scene.keys():
            from aaltonomous.interface.pc2.head_rgbd_server import HeadRGBDServer
            self._latest = None
            self.rgbd = HeadRGBDServer(self, port=rp, ctx=self.ctx)
            print(f"[aalto] head RGB-D on tcp :{rp}")
        self.last_frame = None
        self.zmq = zmq
        self.dr_cfg = dr.load()
        self.dr_params, self._dr_base, self._cam_rng = None, None, None
        cp = int(os.environ.get("AALTO_CLOCK_PORT", str(sp + 1 if sp else 0)))
        self.clock = None
        if cp:
            self.clock = self.ctx.socket(zmq.PUB)
            self.clock.setsockopt(zmq.LINGER, 0)
            self.clock.setsockopt(zmq.SNDHWM, 10)
            self.clock.bind(f"tcp://*:{cp}")
            print(f"[aalto] sim clock on tcp :{cp}")
        self.step_dt = float(env.step_dt)
        self.sim_t = 0.0
        self.rtf = self.rtf_mean = None
        self._win = self._start = None            # (wall, sim_t) at window / clock start
        self.ports = {"scene": sp, "rgbd": rp, "clock": cp}
        if self.dr_cfg:
            print(f"[aalto] domain randomization ON: {self.dr_cfg['_path']}")

    # HeadRGBDServer source interface
    def read(self):
        return self._latest

    # ------------------------------------------------------------------ scene
    def place(self, env, layout):
        if self.dr_params:
            layout = dr.jitter_layout(layout, self.dr_params)
        poses, active = assign(self.sc, layout)
        origin = _t(env.scene.env_origins)[0]
        dev = origin.device
        for n in self.names:
            p, q = poses[n]
            pose = torch.tensor([[*(np.asarray(p) + PELVIS_W), *q]], dtype=torch.float32, device=dev)
            pose[:, :3] += origin
            obj = env.scene[n]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev),
                                                 env_ids=torch.tensor([0], device=dev))
        self.layout, self.active = list(layout), active
        return active

    def state(self, env):
        origin = _t(env.scene.env_origins)[0].cpu().numpy()
        out = []
        for n in self.names:
            d = env.scene[n].data
            p = _t(d.root_pos_w)[0].cpu().numpy() - origin - PELVIS_W
            q = _t(d.root_quat_w)[0].cpu().numpy()
            v = _t(d.root_lin_vel_w)[0].cpu().numpy()
            color, shape, _ = parse_name(n)
            out.append({"name": n, "color": color, "shape": shape, "active": n in self.active,
                        "pos": [round(float(x), 5) for x in p], "quat_xyzw": [round(float(x), 5) for x in q],
                        "speed": round(float(np.linalg.norm(v)), 5)})
        return out

    def handle(self, env, req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "t": time.time(), **self.clock_info()}
        if cmd == "reset":
            return {"ok": True, "active": self.place(env, req["layout"]), "layout": req["layout"]}
        if cmd == "info":
            d = None if not self.dr_cfg else {"name": self.dr_cfg.get("name"), "path": self.dr_cfg["_path"]}
            return {"ok": True, "dr": d, "ports": self.ports, "n_objects": len(self.names), **self.clock_info()}
        if cmd == "randomize":
            return {"ok": True, "params": self.randomize(env, req.get("seed"))}
        if cmd == "reset_random":
            if self.dr_cfg:
                self.randomize(env, req.get("seed"))
            lay = random_layout(self.sc, req.get("seed"))
            return {"ok": True, "active": self.place(env, lay), "layout": lay}
        if cmd == "state":
            return {"ok": True, "objects": self.state(env), "layout": self.layout}
        if cmd == "evaluate":
            r = evaluate(self.sc, req.get("task", "sort_fixed"), self.state(env), req.get("mapping"))
            return {"ok": True, **r}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    # ------------------------------------------------------------------ sim clock
    def clock_info(self):
        r = lambda x: None if x is None else round(x, 3)
        return {"sim_t": round(self.sim_t, 4), "step_dt": self.step_dt, "rtf": r(self.rtf), "rtf_mean": r(self.rtf_mean)}

    def _tick_clock(self, env):
        self.sim_t = int(env.common_step_counter) * self.step_dt
        now = time.monotonic()
        if self.clock is not None:
            try:
                self.clock.send(struct.pack("<dd", self.sim_t, time.time()), self.zmq.NOBLOCK)
            except self.zmq.Again:
                pass
        if self._win is None:
            self._win = self._start = (now, self.sim_t)
            return
        if now - self._win[0] >= 10.0:
            self.rtf = (self.sim_t - self._win[1]) / (now - self._win[0])
            self.rtf_mean = (self.sim_t - self._start[1]) / (now - self._start[0])
            self._win = (now, self.sim_t)
            if self.rtf < 0.95:
                print(f"[aalto] WARNING real-time factor {self.rtf:.2f} (< 0.95; mean {self.rtf_mean:.2f}): the sim runs "
                      f"slower than real time; AaltoEnv paces on the sim clock", flush=True)

    # ------------------------------------------------------------------ per step
    def step(self, env):
        self._tick_clock(env)
        if self.rep is not None:
            for _ in range(4):
                try:
                    msg = self.rep.recv(self.zmq.NOBLOCK)
                except self.zmq.Again:
                    break
                try:
                    resp = self.handle(env, json.loads(msg))
                except Exception as e:  # never kill the sim over a bad request
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.rep.send(json.dumps(resp).encode())
        if self.rgbd is not None:
            self._publish_rgbd(env)

    def _publish_rgbd(self, env):
        from tasks.common_observations.camera_state import _camera_frame
        cam = env.scene["front_camera"]
        out = cam.data.output
        frame = _camera_frame(cam)
        if frame is not None and frame == self.last_frame:
            return
        self.last_frame = frame
        rgb = _t(out["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
        dist = _t(out["distance_to_image_plane"])[0, ..., 0].cpu().numpy()
        dist = np.where(np.isfinite(dist), dist, 0.0)
        depth = np.clip(np.round(dist * 1000.0), 0, 65535).astype(np.uint16)
        depth[(dist < 0.1) | (dist > 10.0)] = 0     # D435 has no valid depth this close / far
        if self.dr_params and self._cam_rng is not None:
            rgb, depth = dr.corrupt(rgb, depth, self.dr_params.get("camera"), self._cam_rng)
        self._latest = (rgb, depth, time.time())
        self.rgbd.publish_one()


    # ------------------------------------------------------------------ domain randomization
    def randomize(self, env, seed):
        if not self.dr_cfg:
            return None
        params = dr.sample(self.dr_cfg, seed, self.names, self.sc.bin_names)
        if self._dr_base is None:
            self._dr_base = _capture_base(env, self)
        params["applied"] = _apply_dr(env, self, params)
        self.dr_params = params
        self._cam_rng = np.random.default_rng(params["camera"]["stream_seed"])
        return params


def _env0(path):
    return path.replace("env_.*", "env_0")


def _stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


def _shaders(stage, root, name):
    """UsdPreviewSurface shaders under root with input `name`."""
    from pxr import Usd, UsdShade
    out = []
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return out
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(p)
            if sh.GetInput(name):
                out.append(sh)
    return out


def _capture_base(env, rt):
    """Nominal values, captured once before the first randomization (DR is always relative to these)."""
    stage = _stage()
    b = {"lights": {}, "colors": {}, "mass": {}, "material": {}, "cam": None}
    for key, path in (("key", "/World/KeyLight"), ("dome", "/World/DomeLight")):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            t = p.GetAttribute("xformOp:translate")
            b["lights"][key] = {"intensity": p.GetAttribute("inputs:intensity").Get(),
                                "color": tuple(p.GetAttribute("inputs:color").Get() or (1, 1, 1)),
                                "translate": tuple(t.Get()) if t and t.Get() is not None else None}
    roots = {"Counter": "/World/envs/env_0/Counter"}
    for bn in rt.sc.bin_names:
        for part in ("floor", "wall_xlo", "wall_xhi", "wall_ylo", "wall_yhi"):
            roots[f"bin:{bn}:{part}"] = f"/World/envs/env_0/Bin_{bn}_{part}"
    for n in rt.names:
        roots[f"obj:{n}"] = _env0(env.scene[n].cfg.prim_path)
    for k, r in roots.items():
        b["colors"][k] = [(sh, tuple(sh.GetInput("diffuseColor").Get())) for sh in _shaders(stage, r, "diffuseColor")]
    for n in rt.names:
        obj = env.scene[n]
        try:
            b["mass"][n] = _t(obj.data.body_mass).clone()
        except Exception as e:
            print(f"[aalto] DR: no mass access for {n}: {e}")
        try:
            import warp as wp
            b["material"][n] = wp.to_torch(obj.root_view.get_material_properties()).clone()
        except Exception as e:
            print(f"[aalto] DR: no material access for {n}: {e}")
    if "front_camera" in env.scene.keys():
        cam = env.scene["front_camera"]
        try:
            b["cam"] = (_t(cam.data.pos_w).clone(), _t(cam.data.quat_w_ros).clone())
        except Exception as e:
            print(f"[aalto] DR: no camera pose access: {e}")
    return b


def _apply_dr(env, rt, P):
    """Apply one parameter set. Returns which groups were applied (False = API missing, logged)."""
    from pxr import Gf, Sdf
    stage, base, ok = _stage(), rt._dr_base, {}
    try:
        L = P["lighting"]
        for key in ("key", "dome"):
            lb = base["lights"].get(key)
            if not lb:
                continue
            p = stage.GetPrimAtPath("/World/KeyLight" if key == "key" else "/World/DomeLight")
            p.GetAttribute("inputs:intensity").Set(float(lb["intensity"]) * L[f"{key}_intensity_scale"])
            p.GetAttribute("inputs:color").Set(Gf.Vec3f(*[float(c) * j for c, j in zip(lb["color"], L[f"{key}_rgb"])]))
            if key == "key" and lb["translate"] is not None:
                p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(a) + d for a, d in zip(lb["translate"], L["key_pos_offset"])]))
        ok["lighting"] = True
    except Exception as e:
        ok["lighting"] = f"{type(e).__name__}: {e}"
    try:
        T, O = P["textures"], P["objects"]
        for k, lst in base["colors"].items():
            for sh, rgb in lst:
                if k == "Counter":
                    new = [c * j for c, j in zip(rgb, T["counter_rgb"])]
                elif k.startswith("bin:"):
                    new = [c * j for c, j in zip(rgb, T["bin_rgb"][k.split(":")[1]])]
                else:
                    o = O[k[4:]]
                    new = dr.jitter_color(rgb, o["dh"], o["s"], o["v"])
                sh.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(min(max(c, 0.0), 1.0)) for c in new]))
        tex = stage.GetPrimAtPath("/World/envs/env_0/Mat/material/texture")
        if tex.IsValid():
            from pxr import UsdShade
            sh = UsdShade.Shader(tex)
            s, bias = T["mat_scale"], T["mat_bias"]
            (sh.GetInput("scale") or sh.CreateInput("scale", Sdf.ValueTypeNames.Float4)).Set(Gf.Vec4f(*s, 1.0))
            (sh.GetInput("bias") or sh.CreateInput("bias", Sdf.ValueTypeNames.Float4)).Set(Gf.Vec4f(bias, bias, bias, 0.0))
        ok["colors"] = True
    except Exception as e:
        ok["colors"] = f"{type(e).__name__}: {e}"
    try:
        ids = torch.tensor([0], dtype=torch.int32)
        for n in rt.names:
            o, obj = P["objects"][n], env.scene[n]
            if n in base["mass"]:
                m = base["mass"][n] * o["mass_scale"]
                obj.set_masses_index(masses=m.to(m.device), env_ids=ids.to(m.device))
            if n in base["material"]:
                import warp as wp
                mat = base["material"][n].clone()
                mat[..., 0:2] *= o["friction_scale"]
                obj.root_view.set_material_properties(wp.from_torch(mat.contiguous(), dtype=wp.float32),
                                                      wp.from_torch(ids, dtype=wp.int32))
        ok["physics"] = True
    except Exception as e:
        ok["physics"] = f"{type(e).__name__}: {e}"
    try:
        if base["cam"] is not None:
            from scipy.spatial.transform import Rotation as R
            pos0, q0 = base["cam"]
            C = P["camera"]
            q = (R.from_rotvec(C["rot_offset_rad"]) * R.from_quat(q0[0].cpu().numpy())).as_quat()   # xyzw
            pos = pos0[0].cpu().numpy() + np.asarray(C["pos_offset"])
            env.scene["front_camera"].set_world_poses(torch.tensor([pos], dtype=torch.float32),
                                                      torch.tensor([q], dtype=torch.float32), convention="ros")
            ok["camera_pose"] = True
    except Exception as e:
        ok["camera_pose"] = f"{type(e).__name__}: {e}"
    return ok


def runtime(env):
    global _RT
    if _RT is None:
        _RT = AaltoRuntime(env)
    return _RT


# ---------------------------------------------------------------------- Isaac Lab hooks
_placeholder = None


def aalto_step_obs(env) -> torch.Tensor:
    """ObsTerm: runs the runtime once per step; returns a dummy (num_envs, 1) tensor."""
    global _placeholder
    runtime(env).step(env)
    if _placeholder is None:
        _placeholder = torch.zeros((env.num_envs, 1), device=env.device)
    return _placeholder


def reset_random_layout(env, env_ids, seed=None):
    """Event (startup / DDS reset category 1): a fresh random layout from scene.yaml."""
    rt = runtime(env)
    rt.place(env, random_layout(rt.sc, seed))
