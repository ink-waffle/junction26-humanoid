"""Grasp strategies: how a hand takes an object of a given shape and yaw; candidates come best first."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


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
    return [Grasp(name="pinch")]
