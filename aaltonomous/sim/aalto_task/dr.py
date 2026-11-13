"""Domain randomization (spec §6 knobs) for the Aalto-* tasks: config loading + seeded sampling (pure numpy, tested
offline in test_layout.py); hooks.py applies the sampled params inside Isaac.

Off by default: with no AALTO_DR the sim is exactly the nominal scene (existing checks unchanged).
    AALTO_DR=train  TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh   # aaltonomous/sim/dr/train.yaml
    AALTO_DR=/path/to/my.yaml ...                                                      # any file, same schema
Per episode: scene service {"cmd": "randomize", "seed": s} samples + applies one parameter set (evaluate.py does
this with the trial seed before each reset); "reset_random" also re-randomizes when DR is on. Same seed + same
config = same parameters. Every knob is optional; a missing section or a 0 / [1, 1] range means "off".

Schema (all ranges [lo, hi], sampled uniformly; *_scale are multiplicative on the nominal value):
  name: str
  lighting: {key_intensity_scale, key_pos_jitter: [dx, dy, dz] m, key_rgb_jitter: frac per channel,
             dome_intensity_scale, dome_rgb_jitter}
  textures: {mat_scale: [lo, hi] (texture * s per channel: brightness/tint), mat_bias: [lo, hi] (+ b),
             mat_rgb_jitter: frac per channel extra on mat_scale, counter_rgb_jitter: frac, bin_rgb_jitter: frac}
  objects:  {hue_jitter: max |dH| (0..1 turn), sat_scale, val_scale, mass_scale, friction_scale}
  pose:     {xy_jitter: m (per object, uniform in a disc), yaw_jitter_deg}
  camera:   {rgb_noise_std: 0-255 units, rgb_gain: [lo, hi], depth_noise_mm, depth_noise_rel (x depth),
             depth_hole_frac, pose_jitter_m, pose_jitter_deg}
"""
from __future__ import annotations

import colorsys
import os
from pathlib import Path

import numpy as np
import yaml

DR_DIR = Path(__file__).resolve().parents[1] / "dr"


def load(ref=None):
    """ref: None -> $AALTO_DR; '' / None / 'off' -> None (DR off); a name in aaltonomous/sim/dr/ or a yaml path."""
    ref = os.environ.get("AALTO_DR", "") if ref is None else ref
    if not ref or ref == "off":
        return None
    p = Path(ref)
    if not p.suffix:
        p = DR_DIR / f"{ref}.yaml"
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg.setdefault("name", p.stem)
    cfg["_path"] = str(p)
    return cfg


def _u(rng, r, default=1.0):
    if r is None:
        return default
    if np.isscalar(r):
        return float(r)
    lo, hi = float(r[0]), float(r[1])
    return float(rng.uniform(lo, hi)) if hi > lo else lo


def _rgb_jit(rng, frac):
    frac = float(frac or 0.0)
    return [float(v) for v in (1.0 + rng.uniform(-frac, frac, 3))] if frac > 0 else [1.0, 1.0, 1.0]


def jitter_color(rgb, dh=0.0, s_scale=1.0, v_scale=1.0):
    """rgb 0..1 -> rgb 0..1 with hue shift dh (turns) and saturation / value scales."""
    h, s, v = colorsys.rgb_to_hsv(*[float(c) for c in rgb])
    r = colorsys.hsv_to_rgb((h + dh) % 1.0, float(np.clip(s * s_scale, 0, 1)), float(np.clip(v * v_scale, 0, 1)))
    return [float(c) for c in r]


def sample(cfg, seed, object_names=(), bin_names=()):
    """One JSON-serialisable parameter set. Separate child streams per section so adding a knob to one section
    doesn't change the others' draws for the same seed."""
    if not cfg:
        return None
    ss = np.random.SeedSequence(int(seed) if seed is not None else None)
    rl, rt, ro, rp, rc = [np.random.default_rng(s) for s in ss.spawn(5)]
    L, T, O, P, C = (cfg.get(k) or {} for k in ("lighting", "textures", "objects", "pose", "camera"))
    out = {"name": cfg.get("name"), "seed": None if seed is None else int(seed)}
    out["lighting"] = {
        "key_intensity_scale": _u(rl, L.get("key_intensity_scale")),
        "key_pos_offset": [float(rl.uniform(-d, d)) if d else 0.0 for d in (L.get("key_pos_jitter") or [0, 0, 0])],
        "key_rgb": _rgb_jit(rl, L.get("key_rgb_jitter")),
        "dome_intensity_scale": _u(rl, L.get("dome_intensity_scale")),
        "dome_rgb": _rgb_jit(rl, L.get("dome_rgb_jitter")),
    }
    ms = _u(rt, T.get("mat_scale"))
    out["textures"] = {
        "mat_scale": [ms * j for j in _rgb_jit(rt, T.get("mat_rgb_jitter"))],
        "mat_bias": _u(rt, T.get("mat_bias"), 0.0),
        "counter_rgb": _rgb_jit(rt, T.get("counter_rgb_jitter")),
        "bin_rgb": {b: _rgb_jit(rt, T.get("bin_rgb_jitter")) for b in bin_names},
    }
    hj = float(O.get("hue_jitter") or 0.0)
    out["objects"] = {n: {"dh": float(ro.uniform(-hj, hj)) if hj else 0.0, "s": _u(ro, O.get("sat_scale")),
                          "v": _u(ro, O.get("val_scale")), "mass_scale": _u(ro, O.get("mass_scale")),
                          "friction_scale": _u(ro, O.get("friction_scale"))} for n in object_names}
    out["pose"] = {"xy_jitter": float(P.get("xy_jitter") or 0.0), "yaw_jitter": float(np.radians(P.get("yaw_jitter_deg") or 0.0)),
                   "stream_seed": int(rp.integers(2 ** 31))}
    pj, aj = float(C.get("pose_jitter_m") or 0.0), float(np.radians(C.get("pose_jitter_deg") or 0.0))
    out["camera"] = {
        "rgb_noise_std": float(C.get("rgb_noise_std") or 0.0), "rgb_gain": _u(rc, C.get("rgb_gain")),
        "depth_noise_mm": float(C.get("depth_noise_mm") or 0.0), "depth_noise_rel": float(C.get("depth_noise_rel") or 0.0),
        "depth_hole_frac": float(C.get("depth_hole_frac") or 0.0),
        "pos_offset": [float(rc.uniform(-pj, pj)) if pj else 0.0 for _ in range(3)],
        "rot_offset_rad": [float(rc.uniform(-aj, aj)) if aj else 0.0 for _ in range(3)],
        "stream_seed": int(rc.integers(2 ** 31)),
    }
    return out


def jitter_layout(layout, params):
    """Apply pose jitter (mat coords) to a layout copy. Deterministic for the params' stream seed."""
    P = (params or {}).get("pose") or {}
    xy, yaw = P.get("xy_jitter", 0.0), P.get("yaw_jitter", 0.0)
    if not (xy or yaw):
        return [dict(o) for o in layout]
    rng = np.random.default_rng(P.get("stream_seed", 0))
    out = []
    for o in layout:
        o = dict(o)
        r, a = xy * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        o["x"] = float(o["x"] + r * np.cos(a))
        o["y"] = float(o["y"] + r * np.sin(a))
        o["yaw"] = float(o.get("yaw", 0.0) + rng.uniform(-yaw, yaw))
        out.append(o)
    return out


def corrupt(rgb, depth, cam, rng):
    """Camera noise on one head frame: rgb uint8 (H,W,3), depth uint16 mm (0 = invalid)."""
    if not cam:
        return rgb, depth
    g, sd = cam.get("rgb_gain", 1.0), cam.get("rgb_noise_std", 0.0)
    if g != 1.0 or sd > 0:
        f = rgb.astype(np.float32) * g
        if sd > 0:
            f += rng.normal(0.0, sd, f.shape).astype(np.float32)
        rgb = np.clip(f, 0, 255).astype(np.uint8)
    dn, dr, hole = cam.get("depth_noise_mm", 0.0), cam.get("depth_noise_rel", 0.0), cam.get("depth_hole_frac", 0.0)
    if dn > 0 or dr > 0 or hole > 0:
        valid = depth > 0
        d = depth.astype(np.float32)
        if dn > 0 or dr > 0:
            d += rng.normal(0.0, 1.0, d.shape).astype(np.float32) * (dn + dr * d)
        if hole > 0:
            valid &= rng.uniform(size=d.shape) >= hole
        depth = np.where(valid, np.clip(np.round(d), 1, 65535), 0).astype(np.uint16)
    return rgb, depth
