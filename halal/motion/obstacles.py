"""Other objects seen from above: the outline the hand has to stay off, from a detection's shape and yaw or from a bare top point."""
from __future__ import annotations

from typing import Any

import numpy as np

SQUARE = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) / 2
TRIANGLE = np.array([[2.0, 0.0], [-1.0, np.sqrt(3.0)], [-1.0, -np.sqrt(3.0)]]) / 3   # apex along +x, altitude 1


def top_of(obstacle: Any) -> np.ndarray:
    """Top centre (pelvis xyz) of an obstacle given as a detection or as that point itself."""
    return np.asarray(getattr(obstacle, "top", obstacle), float)


def outline(obstacle: Any, size: float, unknown_radius: float) -> tuple[np.ndarray, float]:
    """Corners (n, 2) and the radius around them that cover an obstacle: square, triangle or circle by shape and yaw, else a circle of `unknown_radius`."""
    centre = top_of(obstacle)[:2]
    shape, yaw = getattr(obstacle, "shape", None), getattr(obstacle, "yaw", None)
    size = getattr(obstacle, "size", None) or size
    if shape == "cylinder":
        return centre[None], size / 2
    corners = {"cube": SQUARE, "triangle": TRIANGLE}.get(shape)
    if corners is None or yaw is None or getattr(obstacle, "lying", False):
        return centre[None], unknown_radius
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return centre + size * corners @ turn.T, 0.0


def distance(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Horizontal distance (m) of each xy point to a convex outline, 0 inside it; a single corner is a point."""
    if len(corners) == 1:
        return np.linalg.norm(points - corners[0], axis=1)
    edges, offsets = np.roll(corners, -1, axis=0) - corners, points[:, None] - corners[None]
    along = np.clip(np.einsum("pej,ej->pe", offsets, edges) / np.einsum("ej,ej->e", edges, edges), 0.0, 1.0)
    outside = np.linalg.norm(offsets - along[..., None] * edges[None], axis=2).min(axis=1)
    turn = edges[None, :, 0] * offsets[..., 1] - edges[None, :, 1] * offsets[..., 0]
    inside = np.all(turn >= 0, axis=1) | np.all(turn <= 0, axis=1)
    return np.where(inside, 0.0, outside)
