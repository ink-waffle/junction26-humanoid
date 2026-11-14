"""Grasp strategies: how a hand takes an object of a given shape and yaw; candidates come best first."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from halal import config

POSTURES = ("preshape", "closed", "release")
INWARD = {"left": -1.0, "right": 1.0}


@dataclass(frozen=True)
class Grasp:
    """Overrides of the planner's default top-down pinch; unset fields keep the config values."""
    name: str
    finger_yaw: float | None = None                            # pelvis-frame heading of the fingers; None = default hints
    center_shift: tuple[float, float] = (0.0, 0.0)             # grasp point minus detected top centre, pelvis xy
    point_in_palm: tuple[float, float, float] | None = None    # object centre in the LEFT palm frame
    postures: Mapping[str, tuple[float, ...]] = field(default_factory=dict)   # LEFT-order preshape / closed / release
    height_frac: float | None = None                           # grasp height as a fraction of the object height


def candidate_grasps(shape: str | None, yaw: float | None, side: str) -> list[Grasp]:
    """Grasps worth trying for this object with this arm, best first."""
    motion = config.load()["motion"]
    names = motion["candidates"][shape or "unknown"]
    grasps = [build_grasp(name, motion["grasps"][name], shape, yaw, side) for name in names]
    return [grasp for grasp in grasps if grasp is not None]


def build_grasp(name: str, params: Mapping[str, Any], shape: str | None, yaw: float | None, side: str) -> Grasp | None:
    """One configured strategy applied to one object; None when it needs a finger heading it cannot get."""
    offsets = (params.get("headings_deg") or {}).get(shape)
    finger_yaw = None
    if offsets is not None:
        finger_yaw = None if yaw is None else nearest_heading(yaw + np.radians(offsets), side)
        if finger_yaw is None:
            return None
    return Grasp(name=name, finger_yaw=finger_yaw, center_shift=_shift(params, finger_yaw),
                 point_in_palm=_optional_tuple(params.get("point_in_palm")),
                 postures={k: tuple(params[k]) for k in POSTURES if k in params},
                 height_frac=params.get("height_frac"))


def nearest_heading(headings: np.ndarray, side: str) -> float | None:
    """The heading closest to the arm's natural finger direction, None when all are out of its range."""
    motion = config.load()["motion"]
    natural = INWARD[side] * np.radians(motion["natural_heading_deg"])
    deviation = np.abs((np.asarray(headings, float) - natural + np.pi) % (2 * np.pi) - np.pi)
    best = int(np.argmin(deviation))
    if deviation[best] > np.radians(motion["max_heading_deviation_deg"]):
        return None
    return float(natural + (headings[best] - natural + np.pi) % (2 * np.pi) - np.pi)


def _shift(params: Mapping[str, Any], finger_yaw: float | None) -> tuple[float, float]:
    """Grasp point offset along the finger heading for strategies whose contacts are not centred on the object."""
    distance = params.get("shift_along_fingers", 0.0)
    if finger_yaw is None or not distance:
        return (0.0, 0.0)
    return (distance * float(np.cos(finger_yaw)), distance * float(np.sin(finger_yaw)))


def _optional_tuple(values: Any) -> tuple[float, float, float] | None:
    return None if values is None else tuple(float(v) for v in values)
