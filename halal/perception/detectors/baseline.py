"""The organisers' HSV + depth blob detector behind our interface: colour and top centre, no shape."""
from __future__ import annotations

from typing import Any, Mapping

from aaltonomous.baselines.classical.perception import Perception
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from halal import config
from halal.perception import ObjectDetection


class BaselineDetector:
    name = "baseline"

    def __init__(self) -> None:
        self.perception = Perception(config.baseline_config(), Scene.load(), Calibration.load())

    def detect(self, obs: Mapping[str, Any]) -> list[ObjectDetection]:
        found = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        return [ObjectDetection(color=d.color, top=d.top, pixels=d.n_px) for d in found]
