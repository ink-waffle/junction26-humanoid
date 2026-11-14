"""Sequential sorter: look pose -> calibrate the hands once -> look -> plan one pick -> execute and verify -> look again."""
from __future__ import annotations

import enum
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from halal.motion import PickPlanner
from halal.perception import Detector, ObjectDetection
from halal.policy.area import ObjectArea
from halal.policy.background import Background, Cancelled
from halal.policy.commands import OBS_LAG_TICKS, Action, CommandStream
from halal.policy.grip import CalibrationTrace, GripWindow, HandResponse, calibration_close, phase
from halal.policy.memory import Attempts
from halal.policy.picking import Pick, PickChooser, Target
from halal.policy.release import ReleaseSlots
from halal.policy.transit import checked_move

log = logging.getLogger(__name__)
HZ = 30.0
BinFor = Callable[[str, str | None], str | None]


class Stage(enum.Enum):
    MOVING = enum.auto()        # a trajectory runs; a look follows when it ends
    CALIBRATING = enum.auto()   # both hands close on nothing once per episode
    LOOKING = enum.auto()       # perception and planning run in the background
    WAITING = enum.auto()       # the next pick does not fit into the remaining time
    DONE = enum.auto()


@dataclass(frozen=True)
class Look:
    """Result of one look: what was seen, what cannot be an object, the next pick and the targets without a plan."""
    detections: list[ObjectDetection]
    outside: list[ObjectDetection]
    pick: Pick | None
    unplannable: list[Target]


@dataclass
class ActivePick:
    """The pick being executed and what its grip check has seen so far."""
    pick: Pick
    window: GripWindow | None   # None = not watched (check disabled, or no clear way back after a failed grip)
    lowest_score: float = math.inf


def leave_everything(color: str, shape: str | None) -> None:
    """The mapping of a policy that must not touch anything."""
    return None


class SortPolicy:
    def __init__(self, detector: Detector, planner: PickPlanner, look_pose: np.ndarray, settings: Mapping[str, Any],
                 time_limit: float) -> None:
        self.detector, self.planner, self.look_pose, self.settings = detector, planner, look_pose, settings
        self.episode_s = time_limit
        self.area = ObjectArea(planner.scene, settings["counter_edge_margin"], settings["bin_clearance"])
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
        self.hands: dict[str, HandResponse] = {}
        self.trace: CalibrationTrace | None = None
        self.active: ActivePick | None = None
        self.grip_trusted, self.grip_proven, self.empty_verdicts = bool(s["grip_check"]), False, 0
        self.stage = Stage.MOVING
        self.at_look_pose = self.hands_clear = False
        self.empty_looks = 0

    def act(self, obs: Mapping[str, Any]) -> Action:
        """One reply: the next chunk of the running trajectory, or a hold while the background job works."""
        if not self.stream.started:
            self.stream.start(obs["arm_q"], obs["hand_q"])
        try:
            self._observe(obs)
            self._advance(obs)
        except Exception:
            log.exception("policy step failed, holding")
        if self.stream.busy:
            return self.stream.next_chunk()
        return self.stream.hold(done=self.stage is Stage.DONE)

    def _observe(self, obs: Mapping[str, Any]) -> None:
        """Use the measured hand joints: calibration samples, and the grip check of a running pick."""
        if self.stage is Stage.CALIBRATING:
            self.trace.add_tick(float(obs["t"]), self.stream.observed_tick, obs["hand_q"])
        elif self.active is not None and self.active.window is not None:
            self._check_grip(obs)

    def _advance(self, obs: Mapping[str, Any]) -> None:
        """Move the state machine one step once the running trajectory has been handed out."""
        t = float(obs["t"])
        if self.stream.busy:
            return
        if self.active is not None:
            self._finish_pick(t)
        if self.stage is Stage.CALIBRATING:
            self._finish_calibration(t)
        if self.stage is Stage.MOVING and not self.at_look_pose:
            self._load_look_pose_move()
        elif self.stage is Stage.MOVING and self._calibration_due():
            self._start_calibration(obs)
        elif self.stage is Stage.MOVING:
            self._start_look(obs)
        elif self.stage is Stage.LOOKING:
            self._finish_look(t)
        elif self.stage is Stage.WAITING and t > self.time_limit:
            log.info("t=%.1f the episode outlasts the assumed %.0f s limit: sorting on", t, self.time_limit)
            self.time_limit, self.stage = math.inf, Stage.MOVING

    def _load_look_pose_move(self) -> None:
        """Queue the move to the look pose with open hands, then a settle; hands only if the arm move is not safe."""
        stream, planner = self.stream, self.planner
        move = checked_move(planner, stream.arm_cmd, stream.hand_cmd, self.look_pose, planner.post["open"],
                            self.settings["transit_clearance"])
        self.hands_clear = move is not None
        if move is None:
            log.warning("no safe move to the look pose: looking from here")
            move = planner.go_to(stream.arm_cmd, stream.hand_cmd, stream.arm_cmd, planner.post["open"])
        stream.load(*move, settle_ticks=round(self.settings["settle_s"] * HZ))
        self.at_look_pose = True

    def _calibration_due(self) -> bool:
        return self.grip_trusted and self.hands_clear and not self.hands and self.trace is None

    def _start_calibration(self, obs: Mapping[str, Any]) -> None:
        """At the look pose, close both hands on nothing and record how they follow the command."""
        stream, planner, every = self.stream, self.planner, self.settings["calibration_sample_ticks"]
        hand, closed_until = calibration_close(stream.hand_cmd, planner.post["closed"], self._grip_settle_ticks,
                                               planner.cfg["motion"]["hand_max_vel"], 1.0 / HZ)
        self.trace = CalibrationTrace(hand, closed_until)
        self.trace.add(float(obs["t"]), stream.hand_cmd, obs["hand_q"])
        stream.load(np.repeat(stream.arm_cmd[None], len(hand), 0), hand,
                    cuts=range(OBS_LAG_TICKS + every, closed_until + OBS_LAG_TICKS + 1, every))
        self.stage = Stage.CALIBRATING

    def _finish_calibration(self, t: float) -> None:
        self.hands = self.trace.responses(self.settings["stiff_hand_lag_s"])
        for side, hand in self.hands.items():
            log.info("t=%.1f %s hand: lag %.3f s (%s), empty-hand error %s", t, side, hand.lag_s,
                     "stiff" if hand.stiff else "soft", np.round(hand.empty_error, 3).tolist())
        self.stage, self.at_look_pose = Stage.MOVING, False   # settle with open hands before the look

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
        elif t + phase(look.pick.plan, "open")[1] / HZ + self.settings["time_margin_s"] > self.time_limit:
            log.info("t=%.1f no time left for %s (%.1f s)", t, look.pick.target, look.pick.plan.duration)
            self.stage = Stage.WAITING
        else:
            self._start_pick(look.pick, t)

    def _note_look(self, t: float, look: Look) -> None:
        for target in look.unplannable:
            self.attempts.abandon(target.xy)
        self.tally.update(looks=1, detections=len(look.detections), outside_area=len(look.outside),
                          unplannable=len(look.unplannable))
        seen = " ".join(f"{d.color}/{d.shape}({d.top[0]:+.3f},{d.top[1]:+.3f})" for d in look.detections)
        log.info("t=%.1f look: %d detections, %d outside the object area: %s", t, len(look.detections),
                 len(look.outside), seen)

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
        watch = self.grip_trusted
        window = GripWindow.of(pick.plan, pick.side, self._grip_settle_ticks) if watch else None
        self.stream.load(pick.plan.arm, pick.plan.hand, cuts=[window.first_tick + OBS_LAG_TICKS] if watch else ())
        self.active = ActivePick(pick, window)
        self.stage, self.at_look_pose = Stage.MOVING, False
        drop = pick.plan.waypoints["drop"]
        log.info("t=%.1f pick %s: %s arm, grasp %s, release at (%+.3f,%+.3f), %.1f s%s", t, pick.target, pick.side,
                 pick.grasp.name if pick.grasp else "default", drop[0], drop[1], pick.plan.duration,
                 " (relocation)" if pick.relocation else "")

    def _check_grip(self, obs: Mapping[str, Any]) -> None:
        """While the object should be in the hand: when the fingers say it is not, give the pick up."""
        active, tick, t = self.active, self.stream.observed_tick, float(obs["t"])
        window = active.window
        if not window.first_tick <= tick <= window.last_tick:
            return
        hand = self.hands.get(window.side)
        score = window.score(obs["hand_q"], hand.empty_error if hand else 0.0)
        if math.isinf(active.lowest_score):
            log.info("t=%.1f grip score after the lift %.3f (empty below %.3f)", t, score, self._empty_below(window.side))
        active.lowest_score = min(active.lowest_score, score)
        if score < self._empty_below(window.side):
            self._abandon_pick(t, score)
        else:
            self.grip_proven = True

    def _empty_below(self, side: str) -> float:
        """Grip score under which this hand holds nothing; an uncalibrated hand gets the value that never aborts wrongly."""
        hand = self.hands.get(side)
        return self.settings["grip_threshold"]["soft" if hand and not hand.stiff else "stiff"]

    def _abandon_pick(self, t: float, score: float) -> None:
        """The hand is empty: go back to the look pose if the way is clear, else let the plan run out unwatched."""
        stream, planner = self.stream, self.planner
        move = checked_move(planner, stream.arm_cmd, stream.hand_cmd, self.look_pose, planner.post["open"],
                            self.settings["transit_clearance"])
        if move is None:
            log.info("t=%.1f hand empty (grip score %.3f) but no clear way back: finishing the plan", t, score)
            self.active.window = None
            return
        log.info("t=%.1f hand empty (grip score %.3f): back to the look pose", t, score)
        self.tally.update(empty_hand=1)
        stream.load(*move, settle_ticks=round(self.settings["settle_s"] * HZ))
        self.active, self.at_look_pose = None, True
        self._doubt_grip_signal(t)

    def _doubt_grip_signal(self, t: float) -> None:
        """A signal that has only ever said `empty` may be no signal at all: stop acting on it after a few picks."""
        self.empty_verdicts += 1
        if not self.grip_proven and self.empty_verdicts >= self.settings["grip_check_patience"]:
            log.warning("t=%.1f %d picks in a row read as empty and none as held: grip check off for this episode", t,
                        self.empty_verdicts)
            self.grip_trusted = False

    def _finish_pick(self, t: float) -> None:
        active, self.active = self.active, None
        if active.lowest_score < self._empty_below(active.pick.side):
            return
        self.released[active.pick.plan.target_bin] += 1
        self.tally.update(released=1)
        log.info("t=%.1f released %s, lowest grip score %.3f", t, active.pick.target, active.lowest_score)

    def _log_tally(self) -> None:
        if self.tally:
            log.info("episode tally: %s", dict(self.tally))
        self.tally.clear()

    @property
    def _grip_settle_ticks(self) -> int:
        return round(self.settings["grip_settle_s"] * HZ)
