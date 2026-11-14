"""Sequential sorter: go to the look pose -> look -> plan one pick -> execute -> look again, until nothing is left."""
from __future__ import annotations

import enum
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from halal.perception import Detector, ObjectDetection
from halal.policy.area import ObjectArea
from halal.policy.background import Background, Cancelled
from halal.policy.commands import Action, CommandStream
from halal.policy.memory import Attempts
from halal.policy.picking import Pick, PickChooser, Target
from halal.policy.release import ReleaseSlots, ShiftedReleasePlanner

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
    """Result of one look: what was seen, what cannot be an object, the next pick and the targets without a plan."""
    detections: list[ObjectDetection]
    outside: list[ObjectDetection]
    pick: Pick | None
    unplannable: list[Target]


def leave_everything(color: str, shape: str | None) -> None:
    """The mapping of a policy that must not touch anything."""
    return None


class SortPolicy:
    def __init__(self, detector: Detector, planner: ShiftedReleasePlanner, look_pose: np.ndarray,
                 settings: Mapping[str, Any], time_limit: float) -> None:
        self.detector, self.planner, self.look_pose, self.settings = detector, planner, look_pose, settings
        self.episode_s = time_limit
        self.area = ObjectArea(planner.scene, settings["band_margin"], settings["bin_clearance"])
        self.chooser = PickChooser(planner, ReleaseSlots(settings["release_shifts"]), settings["relocate"])
        self.jobs = Background(settings["background_nice"])
        self.tally: Counter[str] = Counter()
        self.reset(leave_everything)

    def reset(self, bin_for: BinFor) -> None:
        """Start an episode with its colour/shape -> bin mapping; a look still running from the last one is abandoned."""
        s = self.settings
        self._log_tally()
        self.jobs.cancel()
        self.bin_for = bin_for
        self.time_limit = self.episode_s
        self.stream = CommandStream(self.planner.spec, s["chunk_ticks"], s["hold_ticks"])
        self.attempts = Attempts(s["same_object_radius"], s["max_attempts"])
        self.released: Counter[str] = Counter()
        self.stage = Stage.MOVING
        self.at_look_pose = False
        self.empty_looks = 0

    def act(self, obs: Mapping[str, Any]) -> Action:
        """One reply: the next chunk of the running trajectory, or a hold while the background job works."""
        if not self.stream.started:
            self.stream.start(obs["arm_q"], obs["hand_q"])
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
        if self.stage is Stage.MOVING and not self.at_look_pose:
            self._load_look_pose_move()
        elif self.stage is Stage.MOVING:
            self._start_look(obs)
        elif self.stage is Stage.LOOKING:
            self._finish_look(t)
        elif self.stage is Stage.WAITING and t > self.time_limit:
            log.info("t=%.1f the episode outlasts the assumed %.0f s limit: sorting on", t, self.time_limit)
            self.time_limit, self.stage = math.inf, Stage.MOVING

    def _load_look_pose_move(self) -> None:
        """Queue the move to the look pose with open hands, then a settle; hands only if the arm move would be clipped."""
        stream, planner = self.stream, self.planner
        arm, hand = planner.go_to(stream.arm_cmd, stream.hand_cmd, self.look_pose, planner.post["open"])
        if planner.dry_run_safety(stream.arm_cmd, stream.hand_cmd, arm, hand):
            log.warning("the move to the look pose would be clipped: looking from here")
            arm, hand = planner.go_to(stream.arm_cmd, stream.hand_cmd, stream.arm_cmd, planner.post["open"])
        stream.load(arm, hand, round(self.settings["settle_s"] * HZ))
        self.at_look_pose = True

    def _start_look(self, obs: Mapping[str, Any]) -> None:
        arm_q, hand_q = self.stream.arm_cmd.copy(), self.stream.hand_cmd.copy()
        bin_for, attempts, released = self.bin_for, self.attempts, dict(self.released)
        self.jobs.submit(lambda cancelled: self._look(obs, arm_q, hand_q, bin_for, attempts, released, cancelled))
        self.stage = Stage.LOOKING

    def _look(self, obs: Mapping[str, Any], arm_q: np.ndarray, hand_q: np.ndarray, bin_for: BinFor,
              attempts: Attempts, released: Mapping[str, int], cancelled: Cancelled) -> Look | None:
        """Background job: detect, keep what is sortable, plan the next pick; None when the head frame is blank."""
        if not np.any(obs["head_rgb"]):
            return None
        detections = self.detector.detect(obs)
        inside = [det for det in detections if self.area.contains(det.top[:2])]
        outside = [det for det in detections if not self.area.contains(det.top[:2])]
        targets = self._targets(inside, bin_for, attempts)
        pick, unplannable = self.chooser.choose(targets, inside, released, arm_q, hand_q, cancelled)
        return Look(detections, outside, pick, unplannable)

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
        self._note_look(t, look)
        if look.pick is None:
            self._nothing_to_pick(t)
        elif t + _release_s(look.pick.plan) + self.settings["time_margin_s"] > self.time_limit:
            log.info("t=%.1f no time left for %s (%.1f s)", t, look.pick.target, look.pick.plan.duration)
            self.stage = Stage.WAITING
        else:
            self._start_pick(look.pick, t)

    def _note_look(self, t: float, look: Look) -> None:
        for target in look.unplannable:
            self.attempts.abandon(target.xy)
        self.tally.update(looks=1, detections=len(look.detections), outside_area=len(look.outside),
                          unplannable=len(look.unplannable))
        spots = " ".join(f"{d.color}({d.top[0]:+.3f},{d.top[1]:+.3f})" for d in look.outside)
        log.info("t=%.1f look: %d detections, %d outside the object area %s", t, len(look.detections),
                 len(look.outside), spots)

    def _nothing_to_pick(self, t: float) -> None:
        """Finish only after several looks in a row agree that nothing sortable is left."""
        self.empty_looks += 1
        log.info("t=%.1f nothing to pick (look %d in a row)", t, self.empty_looks)
        if self.empty_looks >= self.settings["empty_looks_before_done"]:
            self.stage = Stage.DONE
            self._log_tally()
        else:
            self.stage, self.at_look_pose = Stage.MOVING, False   # settles again, so the next look sees a later frame

    def _start_pick(self, pick: Pick, t: float) -> None:
        self.empty_looks = 0
        self.attempts.record(pick.target.xy)
        self.tally.update(picks=1, relocations=int(pick.relocation))
        self.released[pick.plan.target_bin] += 1
        self.stream.load(pick.plan.arm, pick.plan.hand)
        self.stage, self.at_look_pose = Stage.MOVING, False
        drop = pick.plan.waypoints["drop"]
        log.info("t=%.1f pick %s: %s arm, grasp %s, release at (%+.3f,%+.3f), %.1f s%s", t, pick.target, pick.side,
                 pick.grasp.name if pick.grasp else "default", drop[0], drop[1], pick.plan.duration,
                 " (relocation)" if pick.relocation else "")

    def _log_tally(self) -> None:
        if self.tally:
            log.info("episode tally: %s", dict(self.tally))
        self.tally.clear()


def _release_s(plan: Plan) -> float:
    """Seconds from the start of the plan until the hand has let go of the object."""
    return next(end for name, _, end in plan.phases if name == "open") / HZ
