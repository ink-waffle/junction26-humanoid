"""Sequential sorter: look -> plan one pick -> execute and verify -> home -> look again, until nothing is left."""
from __future__ import annotations

import enum
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aaltonomous.baselines.classical.planner import Plan, ramp

from halal.motion import PickPlanner
from halal.perception import Detector
from halal.policy.background import Background
from halal.policy.commands import OBS_LAG_TICKS, Action, CommandStream
from halal.policy.grip import SIDES, GripWindow, ReleaseCheck, catching_up, hand_slice, opening_in_turn, phase
from halal.policy.hands import HandCalibrator
from halal.policy.looking import BinFor, Look, Looker, Situation
from halal.policy.memory import Attempts, Shapes
from halal.policy.picking import Pick, Target
from halal.policy.transit import checked_move, lowest_clearance

log = logging.getLogger(__name__)
HZ = 30.0


class View(enum.Enum):
    START = enum.auto()         # where the episode began, hands untouched: the camera fit wants the arms in view
    HOME = enum.auto()          # the pose picks end in
    CLEAR = enum.auto()         # arms swung out of the head camera's view of the counter


class Stage(enum.Enum):
    MOVING = enum.auto()        # a trajectory runs; a look follows when it ends
    LOOKING = enum.auto()       # perception and planning run in the background
    WAITING = enum.auto()       # the next pick does not fit into the remaining time
    DONE = enum.auto()


@dataclass
class ActivePick:
    """The pick being executed and what its grip checks have seen so far."""
    pick: Pick
    window: GripWindow | None   # None = not watched (check disabled, or no clear way back after a failed grip)
    lowest_score: float = math.inf
    carried: bool = False       # the carry is over: the window's ticks no longer apply
    check_tick: int | None = None   # trajectory tick at which the hand, closed once more after letting go, is read
    checks_left: int = 0
    stuck: bool = False         # a release check found the hand still holding something
    tail: tuple[np.ndarray, np.ndarray] | None = None   # the way home from a release check: the closed hand opens on it


def leave_everything(color: str, shape: str | None) -> None:
    """The mapping of a policy that must not touch anything."""
    return None


class SortPolicy:
    def __init__(self, detector: Detector, planner: PickPlanner, clear_pose: np.ndarray, settings: Mapping[str, Any],
                 time_limit: float) -> None:
        self.planner, self.settings = planner, settings
        self.view_poses = {View.HOME: planner.home, View.CLEAR: np.asarray(clear_pose, float)}
        self.episode_s = time_limit
        self.calibrator = HandCalibrator(planner, settings)
        self.looker = Looker(detector, planner, self.calibrator, settings)
        self.file_layout = self.looker.file_layout
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
        self.attempts = Attempts(s["same_object_radius"])
        self.shapes = Shapes(s["same_object_radius"])
        self.layout_source, self.use_file_layout = self.file_layout.source, False
        self.released: Counter[str] = Counter()
        self.blocked: set[str] = set()
        self.active: ActivePick | None = None
        self.grip_trusted, self.grip_proven, self.empty_verdicts = bool(s["grip_check"]), False, 0
        self.release_checked, self.unconfirmed_releases = True, 0
        self.missed: Target | None = None
        self.stage = Stage.MOVING
        self.view, self.next_view, self.primed = View.START, None, False
        self.start_looks, self.empty_looks, self.empty_since = 0, 0, math.inf

    def act(self, obs: Mapping[str, Any]) -> Action:
        """One reply: the next chunk of the running trajectory, or a hold while the background job works."""
        if not self.stream.started:
            self.stream.start(obs["arm_q"], obs["hand_q"])
            self._settle(self.settings["start_settle_s"])   # the first frames can predate the reset and its camera pose
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
        """Queue the move of both arms to the next view's pose with open hands, then a settle; hands only if it is not safe."""
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

    def _settle(self, seconds: float | None = None) -> None:
        """Hold still (for the settle time unless told otherwise); the next look uses a frame from after it."""
        seconds = self.settings["settle_s"] if seconds is None else seconds
        self.stream.load(np.empty((0, 14)), np.empty((0, 14)), settle_ticks=round(seconds * HZ))

    def _start_look(self, obs: Mapping[str, Any]) -> None:
        """Look at this frame; if the hands are not calibrated yet they close on nothing while it is processed."""
        stream, primer = self.stream, self.view is View.START and not self.primed
        calibrate = not primer and self.calibrator.due(stream.arm_cmd)
        hand_q = self.planner.post["open"] if calibrate else stream.hand_cmd.copy()
        situation = Situation(stream.arm_cmd.copy(), hand_q, self.view is View.CLEAR, self.bin_for, self.attempts,
                              self.shapes, dict(self.released), tuple(side for side in SIDES if side not in self.blocked),
                              self.use_file_layout, calibrate, primer, self.missed)
        self.jobs.submit(lambda cancelled: self.looker.look(obs, situation, cancelled))
        if calibrate:
            self.calibrator.start(stream)
        self.stage = Stage.LOOKING

    def _finish_look(self, t: float) -> None:
        job = self.jobs.poll()
        if job is None:
            return
        look: Look | None = job.value
        if look is None:
            self._look_again(t, job.error)
            return
        if look.primer:
            self.primed, self.stage = True, Stage.MOVING   # the next reply looks again, from the same pose
            return
        self._note_look(t, look)
        if look.pick is None and look.unclear:
            log.info("t=%.1f %d targets seen too poorly to grasp: looking with the arms out of the way", t, look.unclear)
            self.stage, self.next_view = Stage.MOVING, View.CLEAR
        elif look.pick is None and look.unplannable:
            log.info("t=%.1f no plan for any of %d targets: looking again", t, len(look.unplannable))
            self._pause(self.settings["replan_pause_s"])
        elif look.pick is None:
            self._nothing_to_pick(t, look)
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
        self._settle(seconds)
        self.stage, self.next_view = Stage.MOVING, View.HOME

    def _note_look(self, t: float, look: Look) -> None:
        self.missed = None
        for target in look.unplannable:
            self.attempts.record_plan_failure(target.xy)
        self.layout_source, self.use_file_layout = look.layout_source, self.use_file_layout or look.layout_rejected
        self.tally.update(looks=1, detections=len(look.detections), outside_area=len(look.outside),
                          unplannable=len(look.unplannable))
        seen = " ".join(f"{d.color}/{d.shape}({d.top[0]:+.3f},{d.top[1]:+.3f};c{d.confidence:.2f};v{d.visible:.2f})"
                        for d in look.detections)
        log.info("t=%.1f look from %s: %d detections, %d outside the object area, layout from %s: %s", t,
                 self.view.name.lower(), len(look.detections), len(look.outside), look.layout_source, seen)

    def _nothing_to_pick(self, t: float, look: Look) -> None:
        """Finish only after several looks with the arms out of the way, spread over some time, show nothing sortable."""
        self.stage = Stage.MOVING
        self.start_looks += self.view is View.START
        if self.view is View.START and not look.detections and self.start_looks < self.settings["start_looks"]:
            self._settle()   # an empty counter at the very start is a late frame: look again before the arms move
        elif self.view is not View.CLEAR:
            self.next_view = View.CLEAR   # an arm can hide an object: only a clear look may say it is gone
        else:
            self._count_empty_look(t)

    def _count_empty_look(self, t: float) -> None:
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
        window = GripWindow.of(pick.plan, pick.side, self._grip_settle_ticks) if self.grip_trusted else None
        self.active = ActivePick(pick, window)
        if window is not None and self._checks_release(pick):
            self._load_with_release_check(pick.plan, window)
        else:
            self.stream.load(pick.plan.arm, pick.plan.hand)
        self.stage, self.next_view = Stage.MOVING, View.HOME
        drop = pick.plan.waypoints["drop"]
        log.info("t=%.1f pick %s (try %d): %s arm, grasp %s, release at (%+.3f,%+.3f), %.1f s", t, pick.target,
                 pick.target.attempts + 1, pick.side, pick.grasp.name, drop[0], drop[1], pick.plan.duration)

    def _checks_release(self, pick: Pick) -> bool:
        """Whether the hand is closed once more after letting go: configured per shape, and only a calibrated hand has an empty reading."""
        after = (pick.target.detection.shape or "unknown") in self.settings["release_check_after"]
        return after and self.release_checked and self.calibrator.hand_class(pick.side) != "unknown"

    def _load_with_release_check(self, plan: Plan, window: GripWindow) -> None:
        """Queue the pick with one more close of the hand above the bin; it starts while the arm retreats."""
        active, stream, (start, end) = self.active, self.stream, phase(plan, "retreat")
        check = self._release_check(plan.hand[start - 1], window, not_before=end - start)
        arm = np.vstack([plan.arm[start:end], np.repeat(plan.arm[end - 1][None], len(check.hand) - (end - start), 0)])
        active.check_tick, active.checks_left = start + check.read_tick, self.settings["release_checks"] - 1
        active.tail = self._way_home(plan, check.hand[-1])
        stream.load(np.vstack([plan.arm[:start], arm, active.tail[0]]), np.vstack([plan.hand[:start], check.hand, active.tail[1]]),
                    cuts=[window.first_tick + OBS_LAG_TICKS, active.check_tick + OBS_LAG_TICKS])

    def _release_check(self, hand_q: np.ndarray, window: GripWindow, not_before: int = 0) -> ReleaseCheck:
        return ReleaseCheck.closing(hand_q, window, self._grip_settle_ticks, not_before, self._hand_vel, 1.0 / HZ)

    def _way_home(self, plan: Plan, hand_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The plan's rows after the retreat for a hand closed at `hand_q`: it opens on the way if that stays clear, else first."""
        end = phase(plan, "retreat")[1]
        arm, hand = plan.arm[end:], plan.hand[end:]
        on_the_way = catching_up(hand_q, hand, self._hand_vel, 1.0 / HZ)
        floor = min(self.settings["transit_clearance"], lowest_clearance(self.planner, arm, hand))
        if len(hand) and np.allclose(on_the_way[-1], hand[-1]) and lowest_clearance(self.planner, arm, on_the_way) >= floor:
            return arm, on_the_way
        opening = ramp(hand_q, plan.hand[end - 1], self._hand_vel, 1.0 / HZ)
        return np.vstack([np.repeat(plan.arm[end - 1][None], len(opening), 0), arm]), np.vstack([opening, hand])

    def _check_release(self, obs: Mapping[str, Any], t: float) -> None:
        """The hand has closed again after letting go: read whether it still holds something."""
        active, side = self.active, self.active.window.side
        score = self._grip_score(obs)
        active.check_tick, active.carried = None, True
        if score >= self._empty_below(side):
            self._open_again(t, score)
        elif active.stuck:
            self.tally.update(freed=1)
            log.info("t=%.1f the %s hand is empty after opening finger by finger (grip score %.3f)", t, side, score)
            if active.checks_left < 0:
                self._doubt_release_check(t)
        else:
            log.info("t=%.1f the %s hand is empty after letting go (grip score %.3f)", t, side, score)

    def _doubt_release_check(self, t: float) -> None:
        """Reads above the bin that the read back home did not confirm may be the grasp posture, not an object: stop after a few."""
        self.unconfirmed_releases += 1
        if self.unconfirmed_releases >= self.settings["release_check_patience"]:
            log.warning("t=%.1f %d release checks not confirmed back home: no more of them in this episode", t,
                        self.unconfirmed_releases)
            self.release_checked = False

    def _open_again(self, t: float, score: float) -> None:
        """Something is still in the hand: open it finger by finger and close once more, above the bin, the last time back home."""
        active, side = self.active, self.active.window.side
        self.tally.update(stuck_in_hand=int(not active.stuck), still_stuck=int(active.stuck))
        active.stuck = True
        if active.checks_left < 0:
            log.warning("t=%.1f the %s hand still holds something back home (grip score %.3f): arm not used again", t, side, score)
            self.blocked.add(side)
            self.stream.load(*self._opening_at_rest(self.stream.hand_cmd))
            return
        log.warning("t=%.1f the %s hand still holds something after letting go (grip score %.3f): opening finger by finger%s",
                    t, side, score, "" if active.checks_left else ", next check back home")
        self._load_reopening(home=active.checks_left == 0)

    def _load_reopening(self, home: bool) -> None:
        """Queue: open finger by finger where the arm is, stay open (or go home so), close and read the hand once more."""
        active, stream, window, plan = self.active, self.stream, self.active.window, self.active.pick.plan
        opening = opening_in_turn(stream.hand_cmd, window.both_hands(stream.hand_cmd, window.released), window.side,
                                  round(self.settings["reopen_pause_s"] * HZ), self._hand_vel, 1.0 / HZ)
        opened = np.repeat(opening[-1][None], round(self.planner.cfg["motion"]["open_hold_s"] * HZ), 0)
        end = phase(plan, "retreat")[1] if home else len(plan.arm)
        hand = np.vstack([opening, opened, plan.hand[end:]])
        if home:
            closed, opened = (self.planner.post[name][hand_slice(window.side)] for name in ("closed", "open"))
            active.window = window = window.closing_to(closed, opened)   # the posture the empty-hand error was measured in
        arm = np.vstack([np.repeat(stream.arm_cmd[None], len(opening) + len(opened), 0), plan.arm[end:]])
        check = self._release_check(hand[-1], window)
        after = self._opening_at_rest(check.hand[-1]) if home else active.tail
        active.check_tick, active.checks_left = len(hand) + check.read_tick, active.checks_left - 1
        stream.load(np.vstack([arm, np.repeat(arm[-1][None], len(check.hand), 0), after[0]]),
                    np.vstack([hand, check.hand, after[1]]), cuts=[active.check_tick + OBS_LAG_TICKS])

    def _opening_at_rest(self, hand_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows that open the hands from `hand_q` to where the running pick's plan leaves them, arms still at its end."""
        plan = self.active.pick.plan
        hand = ramp(hand_q, plan.hand[-1], self._hand_vel, 1.0 / HZ)
        return np.repeat(plan.arm[-1][None], len(hand), 0), hand

    def _check_grip(self, obs: Mapping[str, Any]) -> None:
        """While the object should be in the hand: when the fingers say it is not, give the pick up."""
        active, tick, t = self.active, self.stream.observed_tick, float(obs["t"])
        window = active.window
        if active.check_tick is not None and tick >= active.check_tick:
            self._check_release(obs, t)
        if active.carried or not window.first_tick <= tick <= window.last_tick:
            return
        score = self._grip_score(obs)
        if math.isinf(active.lowest_score):
            log.info("t=%.1f grip score after the lift %.3f (empty below %.3f)", t, score, self._empty_below(window.side))
        active.lowest_score = min(active.lowest_score, score)
        if score < self._empty_below(window.side):
            self._abandon_pick(t, score)
        else:
            self.grip_proven = True

    def _grip_score(self, obs: Mapping[str, Any]) -> float:
        """How far the picking hand's load-bearing joints stay short of the closed posture, beyond what its empty close showed."""
        window = self.active.window
        return window.score(obs["hand_q"], self.calibrator.empty_error(window.side), self.settings["grip_joints"])

    def _empty_below(self, side: str) -> float:
        """Grip score under which this hand holds nothing."""
        return self.settings["grip_threshold"][self.calibrator.hand_class(side)]

    def _abandon_pick(self, t: float, score: float) -> None:
        """The hand is empty: go home at once if the way is clear, else let the plan run out unwatched."""
        if not self.jobs.idle:
            return   # the planner is still busy with an abandoned look: decide at the next reply
        self.missed = self.active.pick.target
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
        if active.lowest_score < self._empty_below(active.pick.side) or active.pick.side in self.blocked:
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

    @property
    def _hand_vel(self) -> float:
        return self.planner.cfg["motion"]["hand_max_vel"]
