"""Pure-numpy side of the Isaac task (no Isaac imports, unit-tested offline in test_layout.py).

Frames: the sim world is the pelvis frame shifted up by PELVIS_W[2] (robot at the origin, facing +x, no yaw),
so   p_world = p_pelvis + PELVIS_W.   All geometry comes from aaltonomous/scene/scene.yaml via Scene.

Objects: scene.yaml has 3 shapes x 3 colours x copies_per_type = 18 rigid bodies, named obj_<colour>_<shape>_<k>.
A layout (tasks.sample_layout format: [{color, shape, x, y, yaw}], mat coords) activates some of them on the mat;
the rest are parked on the floor behind the robot (out of every camera view, outside the counter = never scored).
"""
from __future__ import annotations

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.scene.scene import Scene

PELVIS_W = np.array([0.0, 0.0, 0.79])   # pelvis in the sim world; z = scene.yaml pelvis height assumption (0.79 m)
# Sim arm home pose (rad). Arm zero puts the G1 hands ~6-12 cm above the pelvis, i.e. just above the counter
# (top = pelvis +1 cm) and over the object band. NB on the G1 a NEGATIVE elbow raises the forearm. This pose
# (pinocchio FK on g1_body29_hand14.urdf) holds the hand/wrist frames 15-30 cm above the pelvis, ~10 cm above
# the object tops. (default) -- replace with the real robot's home pose once defined (spec §2 "fixed home pose").
ARM_HOME = {
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.35, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.35, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}
PARK_X = -0.60                          # park row behind the robot (pelvis frame)


def object_names(sc: Scene):
    o = sc.cfg["objects"]
    return [f"obj_{c}_{s}_{k}" for s in o["shapes"] for c in o["colors"] for k in range(int(o["copies_per_type"]))]


def parse_name(name):
    _, color, shape, k = name.split("_")
    return color, shape, int(k)


def yaw_quat_xyzw(theta):
    return np.array([0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)])


def mat_yaw_to_pelvis(yaw):
    """Layout yaw (mat frame, CCW from mat +x, mesh apex along mesh +y) -> yaw about pelvis z.
    The mat -> pelvis map swaps x/y (a mirror), so the triangle apex direction (-sin y, cos y)_mat becomes
    (cos y, -sin y)_pelvis, i.e. a pelvis rotation of -pi/2 - yaw (the shapes are mirror-symmetric, so a
    rotation reproduces the card exactly)."""
    return -np.pi / 2 - float(yaw)


def park_pose(sc: Scene, i, n):
    """Pelvis-frame pose for parked object i of n: on the floor, in a row behind the robot."""
    y = (i - (n - 1) / 2) * 0.10
    z = -PELVIS_W[2] + sc.object_height() / 2 + 0.002
    return np.r_[PARK_X, y, z], yaw_quat_xyzw(0.0)


def assign(sc: Scene, layout):
    """Layout -> {name: (pos_pelvis (3,), quat_xyzw (4,))} for every object (active ones on the mat, rest parked),
    plus the list of active names in layout order. Raises if a type is requested more often than it exists."""
    names = object_names(sc)
    free = {n: True for n in names}
    poses, active = {}, []
    for ob in layout:
        shape = ob.get("shape", "cube")
        pick = next((n for n in names if free[n] and parse_name(n)[:2] == (ob["color"], shape)), None)
        if pick is None:
            raise ValueError(f"layout wants more {ob['color']} {shape} objects than the scene has")
        free[pick] = False
        m = sc.cube_spawn_mat(ob)
        if "z" in ob:                                    # optional centre height (mat coords), e.g. drop tests
            m[2] = float(ob["z"])
        p = sc.mat_to_pelvis(m)
        p[2] += 0.001                                    # 1 mm drop so nothing starts interpenetrating the mat
        poses[pick] = (p, yaw_quat_xyzw(mat_yaw_to_pelvis(ob.get("yaw", 0.0))))
        active.append(pick)
    parked = [n for n in names if free[n]]
    for i, n in enumerate(parked):
        poses[n] = park_pose(sc, names.index(n), len(names))
    return poses, active


def random_layout(sc: Scene, seed=None):
    rng = np.random.default_rng(seed)
    return tasks.sample_layout(rng, shapes=tuple(sc.cfg["objects"]["shapes"]), **sc.layout_kwargs())


def parse_mapping(mapping):
    """JSON mapping -> Scene mapping. Keys: colour, shape, 'default', or 'colour/shape' for a (colour, shape) key."""
    out = {}
    for k, v in (mapping or {}).items():
        out[tuple(k.split("/")) if "/" in k else k] = v
    return out


def evaluate(sc: Scene, task, objects, mapping=None):
    """Spec §3 outcomes + score for the ACTIVE objects. objects: [{name, color, shape, active, pos, speed}]
    (pos = pelvis frame). For sort tasks a None mapping means the rung-1 colour -> bin mapping."""
    if task == "sort_fixed" and not mapping:
        mapping = {c: b for b, c in tasks.BIN_COLOR.items()}
    mapping = parse_mapping(mapping)
    res = []
    for o in objects:
        if not o["active"]:
            continue
        pos, v = np.asarray(o["pos"], float), float(o["speed"])
        if task == "insert_shapes":
            out = sc.classify_insert(pos, v, o["shape"])
        else:
            out = sc.classify_cube(pos, v, o["color"], mapping, shape=o["shape"])
        res.append({"name": o["name"], "outcome": out})
    return {"task": task, "outcomes": res, "score": sc.score(task, [r["outcome"] for r in res]),
            "all_correct": bool(res) and all(r["outcome"] in ("correct", "inserted") for r in res)}
