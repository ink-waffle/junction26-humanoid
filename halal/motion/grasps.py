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
    squeeze_axis: tuple[float, ...] = ()                       # LEFT-order joint weights of a deeper squeeze; () = config default
    squeeze: float = 0.0                                       # default depth (rad along the squeeze axis past `closed`)


def right_hand(left: np.ndarray) -> np.ndarray:
    """LEFT-order hand vector (7) as the right hand's mirror image: thumb_0 turns about palm y and keeps its sign, the rest flip."""
    left = np.asarray(left, float)
    return np.array([left[0], -left[1], -left[2], -left[5], -left[6], -left[3], -left[4]])


def candidate_grasps(shape: str | None, yaw: float | None, side: str, hand: str = "stiff") -> list[Grasp]:
    """Grasps worth trying for this object with this arm, best first, closed as the hand class (stiff | soft) needs."""
    motion = config.load()["motion"]
    names = motion["candidates"][shape or "unknown"]
    return [grasp for name in names
            for grasp in strategy_grasps(name, for_hand(motion["grasps"][name], hand), shape, yaw, side)]


def for_hand(params: Mapping[str, Any], hand: str) -> Mapping[str, Any]:
    """A strategy's parameters with its `soft` overrides applied for a soft hand."""
    return {**params, **params.get("soft", {})} if hand == "soft" else params


def strategy_grasps(name: str, params: Mapping[str, Any], shape: str | None, yaw: float | None, side: str) -> list[Grasp]:
    """One configured strategy on one object: a grasp per reachable finger heading, the most natural first."""
    offsets = (params.get("headings_deg") or {}).get(shape)
    if offsets is None:
        return [_grasp(name, params, shape, None)]
    if yaw is None:
        return []
    return [_grasp(name, params, shape, heading) for heading in reachable_headings(yaw + np.radians(offsets), side)]


def reachable_headings(headings: np.ndarray, side: str) -> list[float]:
    """Headings within the arm's range of its natural finger direction, nearest first, as angles around it."""
    motion = config.load()["motion"]
    natural = INWARD[side] * np.radians(motion["natural_heading_deg"])
    offsets = (np.asarray(headings, float) - natural + np.pi) % (2 * np.pi) - np.pi
    near = sorted(offsets[np.abs(offsets) <= np.radians(motion["max_heading_deviation_deg"])], key=abs)
    return [float(natural + offset) for offset in near]


def _grasp(name: str, params: Mapping[str, Any], shape: str | None, finger_yaw: float | None) -> Grasp:
    return Grasp(name=name, finger_yaw=finger_yaw, center_shift=_shift(params, finger_yaw),
                 point_in_palm=_optional_tuple(params.get("point_in_palm")),
                 postures={k: tuple(params[k]) for k in POSTURES if k in params},
                 height_frac=params.get("height_frac"), squeeze_axis=tuple(params.get("squeeze_axis", ())),
                 squeeze=_depth(params.get("squeeze", 0.0), shape))


def _depth(squeeze: float | Mapping[str, float], shape: str | None) -> float:
    """A strategy's squeeze depth: one number, or one per shape with `unknown` for unclassified objects."""
    return float(squeeze[shape or "unknown"] if isinstance(squeeze, Mapping) else squeeze)


def _shift(params: Mapping[str, Any], finger_yaw: float | None) -> tuple[float, float]:
    """Grasp point offset along the finger heading for strategies whose contacts are not centred on the object."""
    distance = params.get("shift_along_fingers", 0.0)
    if finger_yaw is None or not distance:
        return (0.0, 0.0)
    return (distance * float(np.cos(finger_yaw)), distance * float(np.sin(finger_yaw)))


def _optional_tuple(values: Any) -> tuple[float, float, float] | None:
    return None if values is None else tuple(float(v) for v in values)
