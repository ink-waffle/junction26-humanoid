"""Isaac Lab scene for the Aaltonomous challenge, generated from aaltonomous/scene/scene.yaml + interface/calibration.yaml.

World = pelvis frame + PELVIS_W (robot at the origin facing +x). Contents:
  robot            G1 29-DoF + Dex3, fixed base (upstream preset), pelvis at PELVIS_W
  counter          static slab, top at scene.counter_z, footprint counter_size centred on the mat
  mat              visual quad with scene/assets/mat_texture.png (the printed mat), 0.5 mm above the counter
  bin_<name>_*     static floor + 4 walls per bin (fixed + known), colour from scene.yaml bins.color
  obj_<c>_<s>_<k>  18 rigid objects from the printed STL meshes (scene/assets/object_<shape>.stl), convex hulls
  lid              rung 3 only (Aalto-Insert-*): static triangle-mesh plate with the 3 holes on the lid bin
  front_camera     head RealSense colour, ONE pinhole cam at calibration.yaml head.T_pelvis_optical, rgb + depth
  left/right_wrist_camera  upstream presets (no measured mount yet, M7)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from tasks.common_config import CameraPresets, G1RobotPresets  # upstream (unitree_sim_isaaclab on sys.path)

from aaltonomous.scene.scene import Scene
from .layout import ARM_HOME, PELVIS_W, assign, object_names, parse_name

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "aaltonomous/scene/assets"
CALIB = REPO / "aaltonomous/interface/calibration.yaml"
SC = Scene.load()
ENV = "/World/envs/env_.*"

# Physics of the printed PLA parts (DR knobs later, spec §6). Friction is on the high side: Dex3 fingertips
# are rubber; the sim tends to need more grip than reality for the same grasp.
OBJ_MASS = 0.05            # kg, 45 mm PLA part at ~20 % infill
OBJ_STATIC_FRICTION = 1.2
OBJ_DYNAMIC_FRICTION = 1.0
COUNTER_THICKNESS = 0.04


def _w(p_pelvis):
    return tuple(float(v) for v in np.asarray(p_pelvis, float) + PELVIS_W)


def _rgb(name):
    return tuple(c / 255.0 for c in SC.colors[name])


def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return tuple(float(v) for v in q / np.linalg.norm(q))


# ------------------------------------------------------------------ custom spawners
@configclass
class StlMeshCfg(sim_utils.MeshCfg):
    """Rigid/static body from an STL (mm), recentred so the origin is the object centroid (bbox centre in z).
    approximation: PhysX MeshCollisionAPI approximation ("convexHull" for dynamic, "none" = exact for static)."""
    func: object = None
    path: str = ""
    center_z: bool = True
    approximation: str = "convexHull"
    swap_xy: bool = False      # mesh authored in MAT axes -> pelvis axes (x<->y mirror, faces re-wound)

    def __post_init__(self):
        self.func = spawn_stl_mesh


@clone
def spawn_stl_mesh(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import trimesh
    from pxr import UsdPhysics
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh

    mesh = trimesh.load(cfg.path, force="mesh")
    mesh.apply_scale(0.001)
    if cfg.swap_xy:
        mesh = trimesh.Trimesh(mesh.vertices[:, [1, 0, 2]], mesh.faces[:, ::-1], process=False)
    if cfg.center_z:
        lo, hi = mesh.bounds
        mesh.apply_translation([0.0, 0.0, -(lo[2] + hi[2]) / 2])
    prim = _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, **kwargs)
    if cfg.collision_props is not None and cfg.approximation != "convexHull":
        mp = prim.GetStage().GetPrimAtPath(prim_path + "/geometry/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(mp).GetApproximationAttr().Set(cfg.approximation)
    return prim


@configclass
class TexturedQuadCfg(sim_utils.SpawnerCfg):
    """Visual-only quad (size x, y) in the xy plane, UV-mapped 0..1 with a texture image."""
    func: object = None
    size: tuple = (1.0, 1.0)
    texture: str = ""
    uvs: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))   # for vertices (-x,-y) (x,-y) (x,y) (-x,y)

    def __post_init__(self):
        self.func = spawn_textured_quad


@clone
def spawn_textured_quad(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    from isaaclab.sim.utils import create_prim, get_current_stage

    stage = get_current_stage()
    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    hx, hy = cfg.size[0] / 2, cfg.size[1] / 2
    mesh = UsdGeom.Mesh.Define(stage, prim_path + "/quad")
    mesh.CreatePointsAttr([Gf.Vec3f(-hx, -hy, 0), Gf.Vec3f(hx, -hy, 0), Gf.Vec3f(hx, hy, 0), Gf.Vec3f(-hx, hy, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in cfg.uvs]))

    mat = UsdShade.Material.Define(stage, prim_path + "/material")
    sh = UsdShade.Shader.Define(stage, prim_path + "/material/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, prim_path + "/material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, prim_path + "/material/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(cfg.texture)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return prim


# ------------------------------------------------------------------ scene pieces
def _static_box(name, center_pelvis, size, rgb, roughness=0.8):
    return AssetBaseCfg(
        prim_path=f"{ENV}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(center_pelvis)),
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(s) for s in size),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005,
                                                             rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=roughness, metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6,
                                                            restitution=0.0),
        ),
    )


def counter_cfg():
    c = SC.cfg["counter"]
    mat = np.asarray(c["mat_size"], float)
    size = np.asarray(c["counter_size"], float)
    centre = SC.mat_to_pelvis([mat[0] / 2, mat[1] / 2, -COUNTER_THICKNESS / 2])
    # mat x runs along pelvis y: footprint (pelvis x, pelvis y) = (size[1], size[0])
    return _static_box("Counter", centre, (size[1], size[0], COUNTER_THICKNESS), (0.55, 0.45, 0.35))


def mat_cfg():
    """Printed mat texture on the counter. export.mat_image: row 0 = far edge (pelvis +x), column 0 = robot's LEFT
    (pelvis +y). USD UVs: v = 0 at the image bottom, u = 0 at the left column."""
    c = SC.cfg["counter"]
    mat = np.asarray(c["mat_size"], float)
    centre = SC.mat_to_pelvis([mat[0] / 2, mat[1] / 2, 0.0005])
    return AssetBaseCfg(
        prim_path=f"{ENV}/Mat",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(centre)),
        spawn=TexturedQuadCfg(size=(float(mat[1]), float(mat[0])), texture=str(ASSETS / c["mat_texture"]),
                              uvs=((1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))),
    )


def bin_cfgs():
    b = SC.cfg["bins"]
    ox, oy = b["outer_size"]
    t, ft, h = b["wall_thickness"], b["floor_thickness"], b["wall_height"]
    out = {}
    for name, (cx, cy) in b["centers"].items():
        rgb = _rgb(b["color"][name])
        parts = {  # (centre mat xyz, size mat xyz)
            "floor": ((cx, cy, ft / 2), (ox, oy, ft)),
            "wall_xlo": ((cx - ox / 2 + t / 2, cy, h / 2), (t, oy, h)),
            "wall_xhi": ((cx + ox / 2 - t / 2, cy, h / 2), (t, oy, h)),
            "wall_ylo": ((cx, cy - oy / 2 + t / 2, h / 2), (ox - 2 * t, t, h)),
            "wall_yhi": ((cx, cy + oy / 2 - t / 2, h / 2), (ox - 2 * t, t, h)),
        }
        for part, (c, s) in parts.items():
            # mat (x, y, z) sizes -> pelvis (y, x, z)
            out[f"bin_{name}_{part}"] = _static_box(f"Bin_{name}_{part}", SC.mat_to_pelvis(c), (s[1], s[0], s[2]), rgb)
    return out


def object_cfgs():
    poses, _ = assign(SC, [])  # everything parked; the reset event places the layout
    out = {}
    for n in object_names(SC):
        color, shape, _ = parse_name(n)
        p, q = poses[n]
        out[n] = RigidObjectCfg(
            prim_path=f"{ENV}/{n}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_w(p), rot=tuple(float(v) for v in q)),
            spawn=StlMeshCfg(
                path=str(ASSETS / f"object_{shape}.stl"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=OBJ_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.004,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_rgb(color), roughness=0.7, metallic=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max", restitution_combine_mode="min",
                    static_friction=OBJ_STATIC_FRICTION, dynamic_friction=OBJ_DYNAMIC_FRICTION, restitution=0.0),
            ),
        )
    return out


def lid_cfg():
    """Rung-3 lid plate on the lid bin rim. The STL is in mat axes around the lid centre -> swap_xy."""
    cx, cy = SC.cfg["bins"]["centers"][SC.lid_bin]
    return AssetBaseCfg(
        prim_path=f"{ENV}/Lid",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_w(SC.mat_to_pelvis([cx, cy, SC.lid_z[0]]))),
        spawn=StlMeshCfg(path=str(ASSETS / "lid_shapes.stl"), center_z=False, swap_xy=True, approximation="none",
                         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002,
                                                                          rest_offset=0.0),
                         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85), roughness=0.8)),
    )


def head_camera_cfg():
    cal = yaml.safe_load(open(CALIB))["cameras"]["head"]
    K = cal["intrinsics"]
    T = np.asarray(cal["T_pelvis_optical"], float)
    return CameraCfg(
        prim_path=f"{ENV}/Robot/pelvis/front_cam",
        update_period=1.0 / 30,
        height=int(K["height"]), width=int(K["width"]),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[K["fx"], 0, K["cx"], 0, K["fy"], K["cy"], 0, 0, 1],
            width=int(K["width"]), height=int(K["height"]), clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in T[:3, 3]), rot=rotmat_to_quat_xyzw(T[:3, :3]),
                                   convention="ros"),
    )


def _robot_cfg():
    cfg = G1RobotPresets.g1_29dof_dex3_base_fix(init_pos=tuple(float(v) for v in PELVIS_W),
                                                init_rot=(0.0, 0.0, 0.0, 1.0))
    cfg.init_state.joint_pos = {**cfg.init_state.joint_pos, **ARM_HOME}
    return cfg


def _fixed_cfgs():
    return {
        "robot": _robot_cfg(),
        "floor": AssetBaseCfg(
            prim_path=f"{ENV}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            spawn=sim_utils.CuboidCfg(size=(8.0, 8.0, 0.1),
                                      collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)))),
        "dome_light": AssetBaseCfg(prim_path="/World/DomeLight",
                                   spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0)),
        # "our own lights at the counter": one overhead area-ish light slightly in front of the robot
        "key_light": AssetBaseCfg(prim_path="/World/KeyLight",
                                  init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 2.2)),
                                  spawn=sim_utils.SphereLightCfg(color=(1.0, 0.97, 0.92), intensity=60000.0,
                                                                 radius=0.15)),
        "counter": counter_cfg(),
        "mat": mat_cfg(),
        "front_camera": head_camera_cfg(),
        "left_wrist_camera": CameraPresets.left_dex3_wrist_camera(),
        "right_wrist_camera": CameraPresets.right_dex3_wrist_camera(),
    }


def make_scene_cfg_class(name="AaltoSceneCfg", lid=False):
    """InteractiveSceneCfg subclass with every entity as a dataclass field (InteractiveScene reads fields only)."""
    attrs = {**_fixed_cfgs(), **bin_cfgs(), **object_cfgs()}
    if lid:
        attrs["lid"] = lid_cfg()
    ns = dict(attrs)
    ns["__annotations__"] = {k: type(v) for k, v in attrs.items()}
    ns["__module__"] = __name__
    return configclass(type(name, (InteractiveSceneCfg,), ns))


AaltoSceneCfg = make_scene_cfg_class()
AaltoLidSceneCfg = make_scene_cfg_class("AaltoLidSceneCfg", lid=True)
