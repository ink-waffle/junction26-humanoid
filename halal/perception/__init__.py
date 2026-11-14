"""Perception seam: one observation in, detected objects in the pelvis frame out; detectors are adapters behind `Detector`."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping, Protocol

import numpy as np

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")


@dataclass(frozen=True, eq=False)
class ObjectDetection:
    """One object on the counter; `top` is the centre of its top face in the pelvis frame (m)."""
    color: str | None           # None = an object of no colour family (black, grey, a hue between families): an obstacle, never a target
    top: np.ndarray
    shape: str | None = None    # None = not classified
    yaw: float | None = None    # rad about pelvis z: cube = a face normal, triangle = apex direction; None = unknown
    confidence: float = 1.0     # how sure the colour is (0..1); the pose rests on `visible`, the shape is None when unsure
    pixels: int = 0
    visible: float = 1.0        # fraction of the top face the camera saw; low = centre and yaw rest on little evidence
    size: float | None = None   # height = edge length (m) the depth image shows; None = not measured, take the scene file's
    lying: bool = False         # a prism on a rectangular side: `top` = middle of its ridge, `yaw` = ridge direction (mod pi), `size` = ridge height


@dataclass(frozen=True, eq=False)
class SceneLayout:
    """Where the fixed scene parts are this episode (pelvis frame, m); scene-file values until perception knows better."""
    counter_z: float                    # counter height under the middle of the object band
    bins: Mapping[str, np.ndarray]      # "left" | "center" | "right" -> centre xy of the bin footprint
    source: str                         # "scene": the file's, camera anchored to them | "hands": as seen | "nominal": the file's, unchecked
    rim_z: float | None = None          # highest bin rim seen (pelvis z); None = not measured, counter_z + wall height


class Detector(Protocol):
    name: str

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]: ...

    def layout(self) -> SceneLayout: ...


DETECTORS: dict[str, str] = {"baseline": "halal.perception.detectors.baseline:BaselineDetector",
                             "topface": "halal.perception.detectors.topface:TopFaceDetector"}


def scene_layout(scene: Any) -> SceneLayout:
    """The layout exactly as the scene file has it."""
    bins = {name: scene.mat_to_pelvis([*centre, 0.0])[:2] for name, centre in scene.cfg["bins"]["centers"].items()}
    return SceneLayout(scene.counter_z, bins, "scene")


def make_detector(name: str) -> Detector:
    module, cls = DETECTORS[name].split(":")
    return getattr(import_module(module), cls)()
