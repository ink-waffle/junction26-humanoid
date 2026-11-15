"""Per episode, where each real hand stands from its FK through the episode's camera: the median of still looks' hand-only shifts."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from halal.perception.geometry import Camera, PlaneFit
from halal.perception.hover import HandShift

LOG = logging.getLogger(__name__)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Sample:
    """One still look's believed hand shift and the camera it was measured through."""
    camera: Camera
    centre: np.ndarray      # (3,) FK centre of the fitted hand points (pelvis)
    offset: np.ndarray      # (3,) seen hand minus FK through `camera` (pelvis xyz, m)


class ArmOffsets:
    """Collects hand-only shifts per arm on still looks and gives each arm's offset through the camera in use now."""

    def __init__(self, shifts: HandShift, cfg: Mapping[str, Any]) -> None:
        self.shifts, self.cfg = shifts, cfg
        self.reset()

    def reset(self) -> None:
        """Forget the samples: a new episode has its own camera and posture."""
        self.samples: dict[str, list[Sample]] = {side: [] for side in SIDES}
        self.logged: dict[str, tuple[str, tuple[float, ...] | None]] = {}

    def measure(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, joints: tuple[np.ndarray, np.ndarray],
                bins: Mapping[str, np.ndarray]) -> None:
        """Fit each arm's hand that still wants samples at this still look (fingers open); keep the shift if every guard passes."""
        if not self.cfg["enabled"] or not self.shifts.open(joints[1]):
            return
        for side in SIDES:
            if len(self.samples[side]) < self.cfg["samples"]:
                self._sample(depth, camera, counter, joints, bins, side)

    def _sample(self, depth: np.ndarray, camera: Camera, counter: PlaneFit, joints: tuple[np.ndarray, np.ndarray],
                bins: Mapping[str, np.ndarray], side: str) -> None:
        """One arm's hand shift at this look, logged, kept as a sample when believed."""
        fit = self.shifts.fit(depth, camera, counter, side, *joints, bins)
        if fit is None:
            LOG.debug("hand offset (%s): too little of the hand in view", side)
            return
        failed = [name for name, ok in self.shifts.checks(fit).items() if not ok]
        LOG.info("hand offset (%s): seen minus FK (%+.1f, %+.1f, %+.1f) mm, %d px, rms %.1f mm, coverage %.2f, inliers %.2f, spread %.1f mm: %s",
                 side, *(-1e3 * fit.shift), fit.points, 1e3 * fit.rms, fit.coverage, fit.inliers, 1e3 * fit.spread,
                 f"rejected {failed}" if failed else f"sample {len(self.samples[side]) + 1}")
        if not failed:
            self.samples[side].append(Sample(camera, fit.centre, -fit.shift))

    def offsets(self, camera: Camera) -> dict[str, np.ndarray]:
        """Real hand minus FK per arm through `camera`: the median of the samples; no entry with too few, inside the dead band or beyond belief."""
        medians = {side: self._median(side, camera) for side in SIDES}
        verdicts = {side: self._verdict(median) for side, median in medians.items()}
        self._log(medians, verdicts)
        return {side: medians[side] for side, verdict in verdicts.items() if verdict == "applied"}

    def _median(self, side: str, camera: Camera) -> np.ndarray | None:
        """The median of one arm's samples as `camera` sees them, z only with `correct_z`; None with fewer than `min_samples`."""
        samples = self.samples[side]
        if len(samples) < self.cfg["min_samples"]:
            return None
        return np.median([self._through(sample, camera) for sample in samples], axis=0) * np.r_[1.0, 1.0, float(self.cfg["correct_z"])]

    def _verdict(self, median: np.ndarray | None) -> str:
        """Whether a median is applied, or why not."""
        if median is None:
            return "too few samples"
        size = float(np.linalg.norm(median))
        return "dead band" if size < self.cfg["dead_band"] else "too large" if size > self.cfg["max_offset"] else "applied"

    @staticmethod
    def _through(sample: Sample, camera: Camera) -> np.ndarray:
        """A sample's offset as `camera` sees the same depth pixels: the seen point moves rigidly with the camera."""
        seen = camera.T @ np.linalg.solve(sample.camera.T, np.r_[sample.centre + sample.offset, 1.0])
        return seen[:3] - sample.centre

    def _log(self, medians: Mapping[str, np.ndarray | None], verdicts: Mapping[str, str]) -> None:
        """Log each arm's verdict and median whenever one changes."""
        state = {side: (verdict, None if medians[side] is None else tuple(np.round(1e3 * medians[side], 1))) for side, verdict in verdicts.items()}
        if state == self.logged:
            return
        self.logged = state
        LOG.info("hand offsets (real minus FK through the episode's camera): %s", "; ".join(
            f"{side} {verdict} ({len(self.samples[side])} samples)" + ("" if median is None else " median ({:+.1f}, {:+.1f}, {:+.1f}) mm".format(*median))
            for side, (verdict, median) in state.items()))
