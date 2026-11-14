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
    color: str
    top: np.ndarray
    shape: str | None = None    # None = not classified
    yaw: float | None = None    # rad about pelvis z: cube = a face normal, triangle = apex direction; None = unknown
    confidence: float = 1.0     # probability that this is a real object with this colour and shape
    pixels: int = 0
    visible: float = 1.0        # fraction of the top face the camera saw; low = centre and yaw rest on little evidence


class Detector(Protocol):
    name: str

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]: ...


DETECTORS: dict[str, str] = {"baseline": "halal.perception.detectors.baseline:BaselineDetector",
                             "topface": "halal.perception.detectors.topface:TopFaceDetector"}


def make_detector(name: str) -> Detector:
    module, cls = DETECTORS[name].split(":")
    return getattr(import_module(module), cls)()
