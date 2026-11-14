"""Sequential sorter: home -> look -> plan one pick -> execute -> look again, until nothing sortable is left."""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.motion import PickPlanner
from halal.perception import Detector, ObjectDetection
from halal.policy.background import Background, Cancelled
from halal.policy.commands import Action, CommandStream
from halal.policy.memory import Attempts
from halal.policy.picking import Pick, PickChooser, Target

log = logging.getLogger(__name__)
HZ = 30.0
BinFor = Callable[[str, str | None], str | None]


class Stage(enum.Enum):
    MOVING = enum.auto()     # a trajectory runs; a look follows when it ends
    LOOKING = enum.auto()    # perception and planning run in the background
    WAITING = enum.auto()    # the next pick does not fit into the remaining time
    DONE = enum.auto()


@dataclass(frozen=True)
class Look:
    """Result of one look: what was seen, the pick to run next and the targets no plan was found for."""
    detections: list[ObjectDetection]
    pick: Pick | None
    unplannable: list[Target]


def leave_everything(color: str, shape: str | None) -> None:
    """The mapping of a policy that must not touch anything."""
    return None


class SortPolicy:
    def __init__(self, detector: Detector, planner: PickPlanner, settings: Mapping[str, Any], time_limit: float) -> None:
        self.detector, self.planner, self.settings = detector, planner, settings
        self.episode_s = time_limit
        self.chooser = PickChooser(planner, settings["relocate"])
        self.jobs = Background(settings["background_nice"])
        self.reset(leave_everything)

    def reset(self, bin_for: BinFor) -> None:
        """Start an episode with its colour/shape -> bin mapping; a look still running from the last one is abandoned."""
        s = self.settings
        self.jobs.cancel()
        self.bin_for = bin_for
        self.time_limit = self.episode_s
        self.stream = CommandStream(self.planner.spec, s["chunk_ticks"], s["hold_ticks"])
        self.attempts = Attempts(s["same_object_radius"], s["max_attempts"])
        self.stage = Stage.MOVING
        self.empty_looks = 0

    def act(self, obs: Mapping[str, Any]) -> Action:
        """One reply: the next chunk of the running trajectory, or a hold while the background job works."""
        if not self.stream.started:
            self.stream.start(obs["arm_q"], obs["hand_q"])
            self._load_homing()
        try:
            self._advance(obs)
        except Exception:
            log.exception("policy step failed, holding")
        if self.stream.busy:
            return self.stream.next_chunk()
        return self.stream.hold(done=self.stage is Stage.DONE)

    def _advance(self, obs: Mapping[str, Any]) -> None:
        """Move the state machine one step once the running trajectory has been handed out."""
        t = float(obs["t"])
        if self.stream.busy:
            return
        if self.stage is Stage.MOVING:
            self._start_look(obs)
        elif self.stage is Stage.LOOKING:
            self._finish_look(t)
        elif self.stage is Stage.WAITING and t > self.time_limit:
            log.info("t=%.1f the episode outlasts the assumed %.0f s limit: sorting on", t, self.time_limit)
            self.time_limit, self.stage = math.inf, Stage.MOVING

    def _load_homing(self) -> None:
        """Queue the move from the start pose to home with open hands; hands only if the arm move would be clipped."""
        stream, planner = self.stream, self.planner
        arm, hand = planner.go_to(stream.arm_cmd, stream.hand_cmd, planner.home, planner.post["open"])
        if planner.dry_run_safety(stream.arm_cmd, stream.hand_cmd, arm, hand):
            log.warning("the move home would be clipped: looking from the start pose")
            arm, hand = planner.go_to(stream.arm_cmd, stream.hand_cmd, stream.arm_cmd, planner.post["open"])
        stream.load(arm, hand, self._settle_ticks)

    def _start_look(self, obs: Mapping[str, Any]) -> None:
        arm_q, hand_q = self.stream.arm_cmd.copy(), self.stream.hand_cmd.copy()
        bin_for, attempts = self.bin_for, self.attempts
        self.jobs.submit(lambda cancelled: self._look(obs, arm_q, hand_q, bin_for, attempts, cancelled))
        self.stage = Stage.LOOKING

    def _look(self, obs: Mapping[str, Any], arm_q: np.ndarray, hand_q: np.ndarray, bin_for: BinFor,
              attempts: Attempts, cancelled: Cancelled) -> Look | None:
        """Background job: detect, keep what is sortable, plan the next pick; None when the head frame is blank."""
        if not np.any(obs["head_rgb"]):
            return None
        detections = self.detector.detect(obs)
        targets = self._targets(detections, bin_for, attempts)
        pick, unplannable = self.chooser.choose(targets, detections, arm_q, hand_q, cancelled)
        return Look(detections, pick, unplannable)

    def _targets(self, detections: list[ObjectDetection], bin_for: BinFor, attempts: Attempts) -> list[Target]:
        """Confident detections that the instruction sends to a bin and that still have attempts left."""
        targets = []
        for det in detections:
            sure = det.confidence >= self.settings["min_detection_confidence"]
            bin_name = bin_for(det.color, det.shape) if sure else None
            if bin_name is not None and not attempts.exhausted(det.top[:2]):
                targets.append(Target(det, bin_name))
        return targets

    def _finish_look(self, t: float) -> None:
        job = self.jobs.poll()
        if job is None:
            return
        look: Look | None = job.value
        if look is None:
            if job.error is not None:
                log.error("t=%.1f look failed: %r", t, job.error)
            self.stage = Stage.MOVING   # blank frame or failure: look again at the next frame
            return
        for target in look.unplannable:
            self.attempts.abandon(target.xy)
        if look.pick is None:
            self._nothing_to_pick(t, look)
        elif t + _release_s(look.pick.plan) + self.settings["time_margin_s"] > self.time_limit:
            log.info("t=%.1f no time left for %s (%.1f s)", t, look.pick.target, look.pick.plan.duration)
            self.stage = Stage.WAITING
        else:
            self._start_pick(look.pick, t)

    def _nothing_to_pick(self, t: float, look: Look) -> None:
        """Finish only after several looks in a row agree that nothing sortable is left."""
        self.empty_looks += 1
        log.info("t=%.1f nothing to pick among %d detections (look %d)", t, len(look.detections), self.empty_looks)
        if self.empty_looks >= self.settings["empty_looks_before_done"]:
            self.stage = Stage.DONE
        else:
            self.stream.load(np.empty((0, 14)), np.empty((0, 14)), self._settle_ticks)
            self.stage = Stage.MOVING

    def _start_pick(self, pick: Pick, t: float) -> None:
        self.empty_looks = 0
        self.attempts.record(pick.target.xy)
        self.stream.load(pick.plan.arm, pick.plan.hand, self._settle_ticks)
        self.stage = Stage.MOVING
        log.info("t=%.1f pick %s: %s arm, grasp %s, %.1f s%s", t, pick.target, pick.side,
                 pick.grasp.name if pick.grasp else "default", pick.plan.duration,
                 " (relocation)" if pick.relocation else "")

    @property
    def _settle_ticks(self) -> int:
        return round(self.settings["settle_s"] * HZ)


def _release_s(plan: Plan) -> float:
    """Seconds from the start of the plan until the hand has let go of the object."""
    return next(end for name, _, end in plan.phases if name == "open") / HZ
