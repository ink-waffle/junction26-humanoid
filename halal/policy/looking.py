"""One look: detections -> sortable targets -> the next pick, with the scene parts where perception (or the scene file) has them."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

from halal.motion import PickPlanner
from halal.perception import Detector, ObjectDetection, SceneLayout, scene_layout
from halal.policy.area import ObjectArea
from halal.policy.background import Cancelled
from halal.policy.hands import HandCalibrator
from halal.policy.memory import Attempts, Shapes
from halal.policy.picking import Pick, PickChooser, Target
from halal.policy.release import ReleaseSlots

log = logging.getLogger(__name__)
BinFor = Callable[[str, str | None], str | None]


@dataclass(frozen=True)
class Look:
    """Result of one look: what was seen, what cannot be an object, the next pick and the targets without a plan."""
    detections: list[ObjectDetection]
    outside: list[ObjectDetection]
    pick: Pick | None
    unplannable: list[Target]
    unclear: int                # sortable objects seen too poorly to grasp from this view
    layout_source: str
    layout_rejected: bool       # the perceived layout allowed no plan at all and the scene file's was used instead
    primer: bool                # nothing was planned: the frame only went to the detector


@dataclass(frozen=True)
class Situation:
    """What a look plans from: the pose the pick starts in, the view, and the episode's mapping, memory and layout choice."""
    arm_q: np.ndarray
    hand_q: np.ndarray
    clear_view: bool
    bin_for: BinFor
    attempts: Attempts
    shapes: Shapes
    released: Mapping[str, int]
    arms: tuple[str, ...]       # arms still in use: one whose hand could not let go of an object is retired for the episode
    file_layout: bool
    hands_pending: bool         # a calibration close is running: plan only once the hands are known
    primer: bool                # only show the frame to the detector: its arm-based camera fit wants two still looks
    missed: Target | None = None   # the last pick came up empty on this target: the object has probably been pushed aside


class Looker:
    def __init__(self, detector: Detector, planner: PickPlanner, calibrator: HandCalibrator, settings: Mapping[str, Any]) -> None:
        self.detector, self.planner, self.calibrator, self.settings = detector, planner, calibrator, settings
        self.area = ObjectArea(planner.scene, settings["counter_edge_margin"], settings["bin_clearance"])
        self.file_layout = scene_layout(planner.scene)
        self.chooser = PickChooser(planner, ReleaseSlots(settings["release_shifts"]), settings["retry_offsets"])

    def look(self, obs: Mapping[str, Any], situation: Situation, cancelled: Cancelled) -> Look | None:
        """Background job: detect and plan the next pick; None when a head image has not arrived."""
        if not (np.any(obs["head_rgb"]) and np.any(obs["head_depth"])):
            return None
        detections = self.detector.detect(obs)
        if situation.primer:
            return Look(detections, [], None, [], 0, self.detector.layout().source, False, True)
        if situation.hands_pending:
            self.calibrator.finished.wait(self.settings["calibration_timeout_s"])
        seen = self.file_layout if situation.file_layout else self.detector.layout()
        look = self._plan(seen, detections, situation, cancelled)
        rejected = len(look.unplannable) + len(look.outside)
        if look.pick is None and rejected and seen.source != self.file_layout.source:
            log.warning("the perceived layout leaves no pick among %d detections: trying the scene file's", rejected)
            return replace(self._plan(self.file_layout, detections, situation, cancelled), layout_rejected=True)
        return look

    def _plan(self, layout: SceneLayout, detections: list[ObjectDetection], situation: Situation,
              cancelled: Cancelled) -> Look:
        """The sortable targets and the next pick with counter and bins where `layout` has them."""
        self.planner.set_layout(layout)
        inside = [det for det in detections if self.area.contains(det.top[:2], layout.bins)]
        outside = [det for det in detections if not self.area.contains(det.top[:2], layout.bins)]
        self._follow_missed(inside, situation)
        targets = self._targets(inside, situation)
        clear = [target for target in targets if situation.clear_view or target.detection.visible >= self.settings["min_visible"]]
        hands = {side: self.calibrator.grasp_class(side) for side in situation.arms}
        pick, unplannable = self.chooser.choose(clear, situation.released, hands, situation.arm_q, situation.hand_q,
                                                cancelled)
        return Look(detections, outside, pick, unplannable, len(targets) - len(clear), layout.source, False, False)

    def _follow_missed(self, detections: list[ObjectDetection], situation: Situation) -> None:
        """A failed grasp pushes its object a few centimetres: the nearest untried detection of that colour inherits its attempts."""
        missed, attempts = situation.missed, situation.attempts
        if missed is None:
            return
        moved = [det.top[:2] for det in detections if det.color == missed.detection.color and not attempts.picks(det.top[:2])]
        nearest = min(moved, key=lambda xy: float(np.linalg.norm(xy - missed.xy)), default=None)
        if nearest is not None and float(np.linalg.norm(nearest - missed.xy)) < self.settings["displaced_radius"]:
            attempts.follow(missed.xy, nearest)

    def _targets(self, detections: list[ObjectDetection], situation: Situation) -> list[Target]:
        """Confident detections with their clearest-seen shape that the instruction sends to a bin and that have picks left."""
        targets = []
        for seen in detections:
            det = replace(seen, shape=situation.shapes.clearest(seen.top[:2], seen.shape, seen.visible))
            sure = det.confidence >= self.settings["min_detection_confidence"]
            bin_name = situation.bin_for(det.color, det.shape) if sure else None
            limit = self.settings["max_attempts"] if det.visible > 0 else self.settings["blob_attempts"]
            if bin_name is not None and situation.attempts.picks(det.top[:2]) < limit:
                targets.append(Target(det, bin_name, situation.attempts.count(det.top[:2])))
        return targets
