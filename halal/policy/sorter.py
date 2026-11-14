"""Sequential sorter: look (calibrating the hands meanwhile, once) -> plan one pick -> execute and verify -> home -> look again; arms out of view when a look needs it."""
from __future__ import annotations

import enum
import logging
import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

from halal.motion import PickPlanner
from halal.perception import Detector, ObjectDetection, SceneLayout, scene_layout
from halal.policy.area import ObjectArea
from halal.policy.background import Background, Cancelled
from halal.policy.commands import OBS_LAG_TICKS, Action, CommandStream
from halal.policy.grip import SIDES, GripWindow, phase
from halal.policy.hands import HandCalibrator
from halal.policy.memory import Attempts, Shapes
from halal.policy.picking import Pick, PickChooser, Target
from halal.policy.release import ReleaseSlots
from halal.policy.transit import checked_move

log = logging.getLogger(__name__)
HZ = 30.0
BinFor = Callable[[str, str | None], str | None]


class View(enum.Enum):
    START = enum.auto()         # where the episode began, hands untouched: the camera fit wants the arms in view
    HOME = enum.auto()          # the pose picks end in
    CLEAR = enum.auto()         # arms swung out of the head camera's view of the counter


class Stage(enum.Enum):
    MOVING = enum.auto()        # a trajectory runs; a look follows when it ends
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
    unclear: int                # sortable objects seen too poorly to grasp from this view
    layout_source: str
    layout_rejected: bool       # the perceived layout allowed no plan at all and the scene file's was used instead


@dataclass(frozen=True)
class Situation:
    """What a look plans from: the pose the pick will start in, the view, the episode's mapping and memory, release counts, layout choice."""
    arm_q: np.ndarray
    hand_q: np.ndarray
    clear_view: bool
    bin_for: BinFor
    attempts: Attempts
    shapes: Shapes
    released: Mapping[str, int]
    file_layout: bool
    hands_pending: bool         # a calibration close is running: plan only once the hands are known


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
    def __init__(self, detector: Detector, planner: PickPlanner, clear_pose: np.ndarray, settings: Mapping[str, Any],
                 time_limit: float) -> None:
        self.detector, self.planner, self.settings = detector, planner, settings
        self.view_poses = {View.HOME: planner.home, View.CLEAR: np.asarray(clear_pose, float)}
        self.episode_s = time_limit
        self.area = ObjectArea(planner.scene, settings["counter_edge_margin"], settings["bin_clearance"])
        self.file_layout = scene_layout(planner.scene)
        self.chooser = PickChooser(planner, ReleaseSlots(settings["release_shifts"]))
        self.calibrator = HandCalibrator(planner, settings)
        self.jobs = Background(settings["background_nice"])
        self.tally: Counter[str] = Counter()
        self.reset(leave_everything)

    def reset(self, bin_for: BinFor) -> None:
        """Start an episode with its colour/shape -> bin mapping; a look still running from the last one is abandoned."""
        s = self.settings
        self._log_tally()
        self.jobs.cancel()
        self.calibrator.reset(enabled=bool(s["grip_check"]))
        self.bin_for = bin_for
        self.time_limit = self.episode_s
        self.stream = CommandStream(self.planner.spec, s["chunk_ticks"], s["hold_ticks"])
        self.attempts = Attempts(s["same_object_radius"], s["max_attempts"])
        self.shapes = Shapes(s["same_object_radius"])
        self.layout_source, self.use_file_layout = self.file_layout.source, False
        self.released: Counter[str] = Counter()
        self.active: ActivePick | None = None
        self.grip_trusted, self.grip_proven, self.empty_verdicts = bool(s["grip_check"]), False, 0
        self.stage = Stage.MOVING
        self.view, self.next_view = View.START, None
        self.empty_looks, self.empty_since = 0, math.inf

    def act(self, obs: Mapping[str, Any]) -> Action:
        """One reply: the next chunk of the running trajectory, or a hold while the background job works."""
        if not self.stream.started:
            self.stream.start(obs["arm_q"], obs["hand_q"])
            self._settle()   # the very first frame may still show the scene before the reset
        try:
            self._observe(obs)
        except Exception:
            log.exception("reading the hands failed: grip check off for this episode")
            self._stop_grip_check()
        try:
            self._advance(obs)
        except Exception:
            log.exception("policy step failed: starting over from a look")
            self._start_over()
        if self.stream.busy:
            return self.stream.next_chunk()
        return self.stream.hold(done=self.stage is Stage.DONE)

    def _observe(self, obs: Mapping[str, Any]) -> None:
        """Use the measured hand joints: calibration samples, and the grip check of a running pick."""
        if self.calibrator.running:
            self.calibrator.sample(self.stream, obs)
        elif self.active is not None and self.active.window is not None:
            self._check_grip(obs)

    def _advance(self, obs: Mapping[str, Any]) -> None:
        """Move the state machine one step once the running trajectory has been handed out."""
        t = float(obs["t"])
        if self.stream.busy:
            return
        if self.active is not None:
            self._finish_pick(t)
        if self.calibrator.running:
            self.calibrator.finish(t)
        if self.stage is Stage.MOVING and not self.jobs.idle:
            return   # a look abandoned at reset still uses the planner: wait for it
        if self.stage is Stage.MOVING and self.next_view is not None:
            self._load_view_move()
        elif self.stage is Stage.MOVING:
            self._start_look(obs)
        elif self.stage is Stage.LOOKING:
            self._finish_look(t)
        elif self.stage is Stage.WAITING and t > self.time_limit:
            log.info("t=%.1f the episode outlasts the assumed %.0f s limit: sorting on", t, self.time_limit)
            self.time_limit, self.stage = math.inf, Stage.MOVING

    def _stop_grip_check(self) -> None:
        """Run the rest of the episode without calibration and without watching the grip."""
        self.grip_trusted = False
        self.calibrator.stop()
        if self.active is not None:
            self.active.window = None

    def _start_over(self) -> None:
        """Drop whatever was in progress and continue with a pause and a fresh look."""
        self.jobs.cancel()
        self.active = None
        self._stop_grip_check()
        self._pause(self.settings["replan_pause_s"])

    def _load_view_move(self) -> None:
        """Queue the move of both arms to the pose of the next view with open hands, then a settle; hands only if the arm move is not safe."""
        stream, planner = self.stream, self.planner
        move = self._move_to(self.next_view)
        if move is None and self.layout_source != self.file_layout.source:
            log.warning("no safe move in the perceived layout: scene-file layout from now on")
            self.use_file_layout, self.layout_source = True, self.file_layout.source
            planner.set_layout(self.file_layout)
            move = self._move_to(self.next_view)
        if move is None:
            log.warning("no safe move to the %s view: looking from here", self.next_view.name.lower())
            move = planner.go_to(stream.arm_cmd, stream.hand_cmd, stream.arm_cmd, planner.post["open"])
        stream.load(*move, settle_ticks=round(self.settings["settle_s"] * HZ))
        self.view, self.next_view = self.next_view, None

    def _move_to(self, view: View) -> tuple[np.ndarray, np.ndarray] | None:
        return checked_move(self.planner, self.stream.arm_cmd, self.stream.hand_cmd, self.view_poses[view],
                            self.planner.post["open"], self.settings["transit_clearance"])

    def _settle(self) -> None:
        """Hold still for the settle time; the next look uses a frame from after it."""
        self.stream.load(np.empty((0, 14)), np.empty((0, 14)), settle_ticks=round(self.settings["settle_s"] * HZ))

    def _start_look(self, obs: Mapping[str, Any]) -> None:
        """Look at this frame; if the hands are not calibrated yet they close on nothing while it is processed."""
        stream, calibrate = self.stream, self.calibrator.due(self.stream.arm_cmd)
        hand_q = self.planner.post["open"] if calibrate else stream.hand_cmd.copy()
        situation = Situation(stream.arm_cmd.copy(), hand_q, self.view is View.CLEAR, self.bin_for, self.attempts,
                              self.shapes, dict(self.released), self.use_file_layout, calibrate)
        self.jobs.submit(lambda cancelled: self._look(obs, situation, cancelled))
        if calibrate:
            self.calibrator.start(stream)
        self.stage = Stage.LOOKING

    def _look(self, obs: Mapping[str, Any], situation: Situation, cancelled: Cancelled) -> Look | None:
        """Background job: detect and plan the next pick; None when a head image has not arrived."""
        if not (np.any(obs["head_rgb"]) and np.any(obs["head_depth"])):
            return None
        detections = self.detector.detect(obs)
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
        targets = self._targets(inside, situation)
        clear = [target for target in targets if situation.clear_view or target.detection.visible >= self.settings["min_visible"]]
        squeeze = {side: self.settings["squeeze"][self.calibrator.hand_class(side)] for side in SIDES}
        pick, unplannable = self.chooser.choose(clear, situation.released, squeeze, situation.arm_q, situation.hand_q,
                                                cancelled)
        return Look(detections, outside, pick, unplannable, len(targets) - len(clear), layout.source, False)

    def _targets(self, detections: list[ObjectDetection], situation: Situation) -> list[Target]:
        """Confident detections, each with its clearest-seen shape, that the instruction sends to a bin and that have picks left."""
        targets = []
        for seen in detections:
            det = replace(seen, shape=situation.shapes.clearest(seen.top[:2], seen.shape, seen.visible))
            sure = det.confidence >= self.settings["min_detection_confidence"]
            bin_name = situation.bin_for(det.color, det.shape) if sure else None
            if bin_name is not None and not situation.attempts.exhausted(det.top[:2]):
                targets.append(Target(det, bin_name, situation.attempts.count(det.top[:2])))
        return targets

    def _finish_look(self, t: float) -> None:
        job = self.jobs.poll()
        if job is None:
            return
        look: Look | None = job.value
        if look is None:
            self._look_again(t, job.error)
            return
        self._note_look(t, look)
        if look.pick is None and look.unclear:
            log.info("t=%.1f %d targets seen too poorly to grasp: looking with the arms out of the way", t, look.unclear)
            self.stage, self.next_view = Stage.MOVING, View.CLEAR
        elif look.pick is None and look.unplannable:
            log.info("t=%.1f no plan for any of %d targets: looking again", t, len(look.unplannable))
            self._pause(self.settings["replan_pause_s"])
        elif look.pick is None:
            self._nothing_to_pick(t)
        elif t + phase(look.pick.plan, "open")[1] / HZ + self.settings["time_margin_s"] > self.time_limit:
            log.info("t=%.1f no time left for %s (%.1f s)", t, look.pick.target, look.pick.plan.duration)
            self.stage = Stage.WAITING
        else:
            self._start_pick(look.pick, t)

    def _look_again(self, t: float, error: BaseException | None) -> None:
        """No look result: at once after a blank frame, after a pause when the detector or planner raised."""
        self.stage = Stage.MOVING
        if error is not None:
            log.error("t=%.1f look failed: %r", t, error)
            self._pause(self.settings["replan_pause_s"])

    def _pause(self, seconds: float) -> None:
        """Hold for a while, then go (back) home and look again."""
        self.stream.load(np.empty((0, 14)), np.empty((0, 14)), settle_ticks=round(seconds * HZ))
        self.stage, self.next_view = Stage.MOVING, View.HOME

    def _note_look(self, t: float, look: Look) -> None:
        for target in look.unplannable:
            self.attempts.record_plan_failure(target.xy)
        self.layout_source, self.use_file_layout = look.layout_source, self.use_file_layout or look.layout_rejected
        self.tally.update(looks=1, detections=len(look.detections), outside_area=len(look.outside),
                          unplannable=len(look.unplannable))
        seen = " ".join(f"{d.color}/{d.shape}({d.top[0]:+.3f},{d.top[1]:+.3f};c{d.confidence:.2f};v{d.visible:.2f})"
                        for d in look.detections)
        log.info("t=%.1f look from %s: %d detections, %d outside the object area, layout from %s: %s", t,
                 self.view.name.lower(), len(look.detections), len(look.outside), look.layout_source, seen)

    def _nothing_to_pick(self, t: float) -> None:
        """Finish only after several looks with the arms out of the way, spread over some time, show nothing sortable."""
        self.stage = Stage.MOVING
        if self.view is not View.CLEAR:
            self.next_view = View.CLEAR   # an arm in the image can hide an object: only a clear look may say it is gone
            return
        self.empty_looks += 1
        self.empty_since = min(self.empty_since, t)
        log.info("t=%.1f nothing to pick (clear look %d in a row)", t, self.empty_looks)
        s = self.settings
        if self.empty_looks >= s["empty_looks_before_done"] and t - self.empty_since >= s["empty_span_before_done_s"]:
            self.stage = Stage.DONE
            self._log_tally()
        else:
            self._settle()

    def _start_pick(self, pick: Pick, t: float) -> None:
        self.empty_looks, self.empty_since = 0, math.inf
        self.attempts.record_pick(pick.target.xy)
        self.tally.update(picks=1)
        window = GripWindow.of(pick.plan, pick.side, round(self.settings["grip_settle_s"] * HZ)) if self.grip_trusted else None
        cuts = (window.first_tick + OBS_LAG_TICKS, window.release_tick + OBS_LAG_TICKS) if window else ()
        self.stream.load(pick.plan.arm, pick.plan.hand, cuts=cuts)
        self.active = ActivePick(pick, window)
        self.stage, self.next_view = Stage.MOVING, View.HOME
        drop = pick.plan.waypoints["drop"]
        log.info("t=%.1f pick %s (try %d): %s arm, grasp %s, release at (%+.3f,%+.3f), %.1f s", t, pick.target,
                 pick.target.attempts + 1, pick.side, pick.grasp.name, drop[0], drop[1], pick.plan.duration)

    def _check_grip(self, obs: Mapping[str, Any]) -> None:
        """While the object should be in the hand: when the fingers say it is not, give the pick up."""
        active, tick, t = self.active, self.stream.observed_tick, float(obs["t"])
        window = active.window
        if tick == window.release_tick:
            log.info("t=%.1f fingers short of the released posture by %.3f after letting go", t, window.stuck(obs["hand_q"]))
        if not window.first_tick <= tick <= window.last_tick:
            return
        score = window.score(obs["hand_q"], self.calibrator.empty_error(window.side))
        if math.isinf(active.lowest_score):
            log.info("t=%.1f grip score after the lift %.3f (empty below %.3f)", t, score, self._empty_below(window.side))
        active.lowest_score = min(active.lowest_score, score)
        if score < self._empty_below(window.side):
            self._abandon_pick(t, score)
        else:
            self.grip_proven = True

    def _empty_below(self, side: str) -> float:
        """Grip score under which this hand holds nothing."""
        return self.settings["grip_threshold"][self.calibrator.hand_class(side)]

    def _abandon_pick(self, t: float, score: float) -> None:
        """The hand is empty: go home at once if the way is clear, else let the plan run out unwatched."""
        if not self.jobs.idle:
            return   # the planner is still busy with an abandoned look: decide at the next reply
        move = self._move_to(View.HOME)
        if move is None:
            log.info("t=%.1f hand empty (grip score %.3f) but no clear way home: finishing the plan", t, score)
            self.active.window = None
            return
        log.info("t=%.1f hand empty (grip score %.3f): going home", t, score)
        self.tally.update(empty_hand=1)
        self.stream.load(*move, settle_ticks=round(self.settings["settle_s"] * HZ))
        self.active, self.view, self.next_view = None, View.HOME, None
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
