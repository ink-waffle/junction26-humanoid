"""Dev-only sim main: the organisers' sim patched in memory to look and behave like the real setup (real_sim.yaml).

    REAL_SIM=lying,colours,boxes bash halal/ops/sims.sh start c     (REAL_SIM=all: every feature; composes with HAND_GAINS)
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import yaml

from soft_hand_sim import REPO, ROBOT_MODULE, SIM_MAIN, PatchOnImport, soften_hands

FEATURES = ("lying", "colours", "boxes", "table", "offsets", "camera")
CONFIG = Path(os.environ.get("REAL_SIM_CONFIG") or Path(__file__).with_name("real_sim.yaml"))
TASK_PACKAGE = "aaltonomous.sim.aalto_task"
WALLS = {"xlo": (-1, 0), "xhi": (1, 0), "ylo": (0, -1), "yhi": (0, 1)}  # wall name -> side of the box in mat x, y

Layout = list[dict[str, Any]]


def enabled() -> set[str]:
    """Features named in $REAL_SIM; `all` means every one."""
    asked = set(filter(None, os.environ["REAL_SIM"].split(",")))
    unknown = asked - {*FEATURES, "all"}
    if unknown:
        sys.exit(f"real_sim: unknown feature(s) {sorted(unknown)}; known: {', '.join(FEATURES)}, all")
    return set(FEATURES) if "all" in asked else asked


def lay_prisms(layout: Layout, fraction: float, height: float) -> Layout:
    """Copy of a layout with a share of its triangles marked as lying: `ridge_yaw` (rad) and the centre height `z`."""
    out = []
    for ob in layout:
        rng = np.random.default_rng(zlib.crc32(json.dumps([ob["x"], ob["y"], ob["yaw"]]).encode()))
        lying = ob.get("shape") == "triangle" and rng.uniform() < fraction
        out.append({**ob, "ridge_yaw": float(rng.uniform(0.0, np.pi)), "z": height} if lying else dict(ob))
    return out


def lying_quat(ridge_yaw: float) -> np.ndarray:
    """Quaternion (xyzw) of a prism resting on the face opposite its apex, ridge turned by `ridge_yaw` about z."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("ZX", [ridge_yaw, np.pi / 2]).as_quat()


def patch_layout(hooks: ModuleType, cfg: dict) -> None:
    """Scene service: every placed layout gets its lying prisms, and their poses the lying orientation."""
    place, assign = hooks.AaltoRuntime.place, hooks.assign

    def place_lying(self: Any, env: Any, layout: Layout) -> list[str]:
        marked = lay_prisms(layout, cfg["fraction"], self.sc.object_dims("triangle")["apothem"])
        print(f"[real_sim] lying prisms: {[(o['color'], o['x'], o['y']) for o in marked if 'ridge_yaw' in o]}", flush=True)
        return place(self, env, marked)

    def assign_lying(sc: Any, layout: Layout) -> tuple[dict, list[str]]:
        poses, active = assign(sc, layout)
        for name, ob in zip(active, layout):
            if "ridge_yaw" in ob:
                poses[name] = (poses[name][0], lying_quat(ob["ridge_yaw"]))
        return poses, active

    hooks.AaltoRuntime.place, hooks.assign = place_lying, assign_lying


def override_scene(scene: dict, cfg: dict, features: set[str]) -> None:
    """Edit a loaded scene.yaml dict in place: object colours, box walls and tape, placement offsets."""
    if "colours" in features:
        scene["objects"]["colors"] = {name: list(cfg["colours"][name]) for name in scene["objects"]["colors"]}
    if "boxes" in features:
        scene["bins"].update(wall_height=cfg["boxes"]["wall_height"], rgb=list(cfg["boxes"]["tape_rgb"]))
    if "offsets" in features:
        counter, (bx, by) = cfg["offsets"]["counter"], cfg["offsets"]["bins"]
        scene["counter"]["mat_origin_in_pelvis"] = [a + b for a, b in zip(scene["counter"]["mat_origin_in_pelvis"], counter)]
        scene["bins"]["centers"] = {name: [x + by, y + bx] for name, (x, y) in scene["bins"]["centers"].items()}


def patch_scene_file(cfg: dict, features: set[str]) -> None:
    """Every Scene.load() of this process (sim geometry and the scene service's scoring) returns the overridden scene."""
    from aaltonomous.scene.scene import Scene
    load = Scene.load

    def load_real(path: str | None = None) -> Scene:
        scene = load(path)
        override_scene(scene.cfg, cfg, features)
        return scene

    Scene.load = staticmethod(load_real)


def liner_boxes(scene_cfg: ModuleType, cfg: dict) -> dict[str, Any]:
    """Thin cardboard panels against the inside of every box wall and on every box floor."""
    sc, bins, thin = scene_cfg.SC, scene_cfg.SC.cfg["bins"], cfg["liner_thickness"]
    rgb = tuple(c / 255.0 for c in cfg["cardboard_rgb"])
    out = {}
    for name in sc.bin_names:
        lo, hi = sc.bin_inner_box(name)
        mid, span = (lo + hi) / 2, hi - lo
        panels = {"floor": ((mid[0], mid[1], bins["floor_thickness"] + thin / 2), (span[0], span[1], thin))}
        for wall, (sx, sy) in WALLS.items():
            centre = (mid[0] + sx * (span[0] - thin) / 2, mid[1] + sy * (span[1] - thin) / 2, bins["wall_height"] / 2)
            panels[wall] = (centre, (thin if sx else span[0], thin if sy else span[1], bins["wall_height"]))
        for part, (centre, size) in panels.items():
            out[f"bin_{name}_liner_{part}"] = scene_cfg._static_box(
                f"Bin_{name}_liner_{part}", sc.mat_to_pelvis(centre), (size[1], size[0], size[2]), rgb)
    return out


def bare_table(fixed: dict[str, Any], cfg: dict) -> dict[str, Any]:
    """The fixed scene entities without the printed mat, the counter painted as the real table."""
    material = fixed["counter"].spawn.visual_material
    material.diffuse_color, material.roughness = tuple(c / 255.0 for c in cfg["rgb"]), cfg["roughness"]
    return {name: entity for name, entity in fixed.items() if name != "mat"}


def moved_camera(camera: Any, cfg: dict) -> Any:
    """The head camera config with its pose moved by a rotation vector (deg) and a translation (m) in the optical frame."""
    from scipy.spatial.transform import Rotation
    mount = Rotation.from_quat(camera.offset.rot)
    camera.offset.pos = tuple(float(v) for v in np.asarray(camera.offset.pos) + mount.apply(cfg["translation"]))
    camera.offset.rot = tuple(float(v) for v in (mount * Rotation.from_rotvec(np.radians(cfg["rotation_deg"]))).as_quat())
    return camera


def patch_look(scene_cfg: ModuleType, cfg: dict, features: set[str]) -> None:
    """Rebuild the two scene classes with the cardboard liners, the bare table and the moved head camera."""
    bins, fixed, camera = scene_cfg.bin_cfgs, scene_cfg._fixed_cfgs, scene_cfg.head_camera_cfg
    if "camera" in features:
        scene_cfg.head_camera_cfg = lambda: moved_camera(camera(), cfg["camera"])
    if "boxes" in features:
        scene_cfg.bin_cfgs = lambda: {**bins(), **liner_boxes(scene_cfg, cfg["boxes"])}
    if "table" in features:
        scene_cfg._fixed_cfgs = lambda: bare_table(fixed(), cfg["table"])
    scene_cfg.AaltoSceneCfg = scene_cfg.make_scene_cfg_class()
    scene_cfg.AaltoLidSceneCfg = scene_cfg.make_scene_cfg_class("AaltoLidSceneCfg", lid=True)


def install(features: set[str], cfg: dict) -> None:
    """Register every patch the chosen features need; nothing is touched before the organisers' modules load."""
    patches: list[tuple[str, Callable[[ModuleType], None]]] = []
    if os.environ.get("HAND_GAINS"):
        patches.append((ROBOT_MODULE, soften_hands))
    if "lying" in features:
        patches.append((f"{TASK_PACKAGE}.hooks", lambda module: patch_layout(module, cfg["lying"])))
    if features & {"boxes", "table", "camera"}:
        patches.append((f"{TASK_PACKAGE}.scene_cfg", lambda module: patch_look(module, cfg, features)))
    if features & {"colours", "boxes", "offsets"}:
        patch_scene_file(cfg, features)
    for name, patch in patches:
        sys.meta_path.insert(0, PatchOnImport(name, patch))


def main() -> None:
    features, cfg = enabled(), yaml.safe_load(CONFIG.read_text())
    sys.path.insert(0, str(REPO))
    install(features, cfg)
    print(f"[real_sim] features {sorted(features)} from {CONFIG}", flush=True)
    sys.argv[0] = str(SIM_MAIN)
    runpy.run_path(str(SIM_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()
