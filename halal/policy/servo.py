"""Grasp feedback: the hand hovers beside its target and one look at both corrects the grasp point (`grasp_feedback`: open | measure | servo)."""
from __future__ import annotations

import enum
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from halal.motion import PickPlanner
from halal.motion.hover import Hover, grasp_point, plan_hover
from halal.perception import Detector
from halal.perception.geometry import Camera
from halal.perception.hover import HoverLook, HoverMeasurer
from halal.policy.background import Background, Cancelled
from halal.policy.commands import CommandStream
from halal.policy.grip import phase
from halal.policy.looking import Look, Situation
from halal.policy.picking import Pick, Target

log = logging.getLogger(__name__)
HZ = 30.0
MODES = ("open", "measure", "servo")
MODE_ENV = "HALAL_GRASP_FEEDBACK"


def feedback_mode(settings: Mapping[str, Any]) -> str:
    """The configured grasp feedback; $HALAL_GRASP_FEEDBACK overrides it (one image per mode, the runner passes no arguments)."""
    mode = os.environ.get(MODE_ENV) or settings["grasp_feedback"]
    if mode not in MODES:
        raise ValueError(f"grasp feedback {mode!r} is none of {MODES}")
    return mode


class Next(enum.Enum):
    WAIT = enum.auto()      # hold: the hover is settling or a look and plan run in the background
    PICK = enum.auto()      # `GraspServo.final` is the pick to execute from the hover
    HOME = enum.auto()      # nothing plannable from the hover: go home and look again


@dataclass
class Hovering:
    """One servoed pick: the open-loop pick, where the hand hovers, and what the looks there have found so far."""
    pick: Pick
    hover: Hover
    source: str                     # what fixed the episode's head camera at the look the pick came from
    started: float = math.nan       # episode time the hover move began
    correction: np.ndarray = field(default_factory=lambda: np.zeros(3))   # added to the target's top so far
    looks: int = 0
    measuring: bool = False
    held_ticks: int = 0
    applied: bool = False
    failed: str = ""                # why no correction is applied: the first look's refusal
    measured: np.ndarray | None = None   # the last believable total correction, applied or not (measure mode never applies it)


@dataclass(frozen=True)
class Measured:
    """One hover look turned into a correction of the target's top centre."""
    look: HoverLook
    vector: np.ndarray | None       # hand -> object, both in the arm frame (object top minus the measured grasp point)
    planned: np.ndarray             # hand -> object as planned (target top minus the commanded grasp point)
    tracking: np.ndarray            # commanded minus measured grasp point at the hover
    step: np.ndarray | None         # change of the target's top this look asks for
    refused: str                    # why the step is not applied ("" when it is)
    parts: tuple[np.ndarray | None, ...] = (None, None, None)   # object minus target: the look's arm-fit turn, the top re-fit, the hand shift


class GraspServo:
    def __init__(self, planner: PickPlanner, detector: Detector, settings: Mapping[str, Any]) -> None:
        self.planner, self.detector, self.cfg = planner, detector, settings["servo"]
        self.mode = feedback_mode(settings)
        self.measurer = HoverMeasurer() if self.mode != "open" else None
        self.nominal = Camera.head()
        self.pending: Hovering | None = None
        self.state: Hovering | None = None
        self.final: Pick | None = None
        self.stranded: Target | None = None   # the target of a hover with no plan onward
        self.picks: dict[int, Hovering] = {}
        self.tally: Counter[str] = Counter()
        self.corrections: list[float] = []

    @property
    def on(self) -> bool:
        return self.mode != "open"

    def reset(self) -> None:
        """New episode: log the last one's summary and forget its picks."""
        if self.on and self.tally:
            sizes = np.array(self.corrections) if self.corrections else np.zeros(1)
            log.info("servo episode (%s): %s; believable corrections (%d) median %.1f mm, max %.1f mm", self.mode, dict(self.tally),
                     len(self.corrections), 1e3 * float(np.median(sizes)), 1e3 * float(sizes.max()))
        self.pending = self.state = self.final = None
        self.picks.clear()
        self.tally.clear()
        self.corrections.clear()

    def after_look(self, look: Look | None, situation: Situation, cancelled: Cancelled) -> Look | None:
        """Look job, last step: plan the hover beside the chosen target while the planner is still ours; the look itself is unchanged."""
        self.pending = None
        if not self.on or look is None or look.primer or look.pick is None or cancelled():
            return look
        det = look.pick.target.detection
        if det.lying or det.visible <= 0.0:
            return look
        hover = plan_hover(self.planner, look.pick.plan, look.pick.grasp, look.pick.side, det, self._camera(),
                           situation.arm_q, situation.hand_q)
        if hover is None:
            log.info("servo: no hover beside %s with the %s arm: open-loop pick", look.pick.target, look.pick.side)
            self.tally.update(no_hover=1)
        else:
            self.pending = Hovering(look.pick, hover, look.layout_source)
        return look

    def hovers(self, pick: Pick, t: float, time_limit: float) -> bool:
        """Whether this pick goes by way of its hover: one was planned for it and the detour still fits into the episode."""
        if self.pending is None or self.pending.pick is not pick:
            return False
        detour = (len(self.pending.hover.arm) + self._hold_ticks) / HZ + self.cfg["overhead_s"]
        if t + detour + phase(pick.plan, "open")[1] / HZ > time_limit:
            log.info("t=%.1f servo: no time for the hover beside %s: open-loop pick", t, pick.target)
            self.tally.update(no_time=1)
            return False
        return True

    def start(self, stream: CommandStream, t: float) -> None:
        """Queue the move to the hover pose and the hold before its look."""
        state, self.pending = self.pending, None
        state.started = t
        self.state, self.final = state, None
        stream.load(state.hover.arm, state.hover.hand, settle_ticks=self._hold_ticks)
        self.tally.update(hovered=1)
        log.info("t=%.1f servo: %s arm hovers beside %s, offset (%+.0f,%+.0f,%+.0f) mm, %.0f %% of the top hidden by the model, "
                 "hand %.1f cm above the counter", t, state.pick.side, state.pick.target, *(1e3 * state.hover.offset),
                 100 * state.hover.hidden, 100 * state.hover.low)

    def step(self, obs: Mapping[str, Any], t: float, stream: CommandStream, jobs: Background) -> Next:
        """The hover move has run out: wait for still arms, look, then replan from here (or hover again)."""
        state = self.state
        if state.measuring:
            return self._collect(t, stream, jobs)
        moving = float(np.abs(np.asarray(obs.get("arm_dq", 0.0))).max()) > self.cfg["still_speed"]
        if moving and state.held_ticks < round(self.cfg["max_hold_s"] * HZ):
            state.held_ticks += self._hold_ticks
            stream.load(np.empty((0, 14)), np.empty((0, 14)), settle_ticks=self._hold_ticks)
            return Next.WAIT
        arm_q, hand_q = stream.arm_cmd.copy(), stream.hand_cmd.copy()
        state.measuring, state.held_ticks = True, 0
        jobs.submit(lambda cancelled: self._iterate(obs, t, state, arm_q, hand_q, cancelled))
        return Next.WAIT

    def outcome(self, pick: Pick, score: float, released: bool) -> None:
        """A pick has ended: log how it went next to what its hover looks did."""
        state = self.picks.pop(id(pick), None)
        if state is None:
            return
        held = "held" if released else "empty"
        self.tally.update({f"{'corrected' if state.applied else 'open_loop'}_{held}": 1})
        shown = state.correction if state.applied else state.measured
        size = "n/a" if shown is None else "({:+.1f},{:+.1f},{:+.1f}) mm".format(*(1e3 * shown))
        log.info("servo outcome: %s, %s arm, camera %s, correction %s %s after %d looks, lowest grip score %.3f: %s", pick.target, pick.side,
                 state.source, size, "applied" if state.applied else f"not applied ({state.failed})", state.looks, score,
                 "released in the bin" if released else "hand empty")

    def _collect(self, t: float, stream: CommandStream, jobs: Background) -> Next:
        """The background look has finished: its pick, or the move to the next hover."""
        job = jobs.poll()
        if job is None:
            return Next.WAIT
        state = self.state
        state.measuring = False
        result = job.value if job.error is None else None
        if isinstance(result, Hover):
            state.hover = result
            stream.load(result.arm, result.hand, settle_ticks=self._hold_ticks)
            log.info("t=%.1f servo: hovering again beside the corrected target, offset (%+.0f,%+.0f,%+.0f) mm, %.0f %% of the top hidden by the model",
                     t, *(1e3 * result.offset), 100 * result.hidden)
            return Next.WAIT
        self.state, self.final, self.stranded = None, result, state.pick.target
        if result is None:
            log.warning("t=%.1f servo: no plan from the hover beside %s%s: going home", t, state.pick.target,
                        "" if job.error is None else f" ({job.error!r})")
            self.tally.update(no_plan=1)
            return Next.HOME
        self.picks[id(result)] = state
        self.tally.update(overhead_ms=round(1e3 * (t - state.started)))
        return Next.PICK

    def _iterate(self, obs: Mapping[str, Any], t: float, state: Hovering, arm_q: np.ndarray, hand_q: np.ndarray,
                 cancelled: Cancelled) -> Pick | Hover | None:
        """Background: measure at the hover, correct the target, then hover again (servo, step not small yet) or plan the pick from here."""
        began = time.monotonic()
        measured = self._safely_measured(obs, state, arm_q)
        state.looks += 1
        if measured.step is not None and measured.refused in ("", "measure only"):
            state.measured = state.correction + measured.step
        if measured.step is not None and not measured.refused:
            state.correction = state.correction + measured.step
            state.applied, state.failed = True, ""
        elif not state.applied:
            state.failed = measured.refused
        self._log(t, state, measured, time.monotonic() - began)
        again = (self.mode == "servo" and not measured.refused and state.looks < self.cfg["max_iterations"]
                 and float(np.linalg.norm(measured.step)) >= self.cfg["converged"])
        hover = self._hover_again(state, arm_q, hand_q) if again and not cancelled() else None
        return hover if hover is not None else self._final_pick(state, arm_q, hand_q)

    def _safely_measured(self, obs: Mapping[str, Any], state: Hovering, arm_q: np.ndarray) -> Measured:
        """`_measure`; a look that raises counts as a failed one, so the pick still goes on from the hover."""
        try:
            return self._measure(obs, state, arm_q)
        except Exception:
            log.exception("servo: the hover look failed")
            nothing = HoverLook(None, None, None, 0.0, math.nan, 0.0, 0.0, 0, math.nan, "error", 0.0)
            return Measured(nothing, None, np.full(3, math.nan), np.zeros(3), None, "error")

    def _measure(self, obs: Mapping[str, Any], state: Hovering, arm_q: np.ndarray) -> Measured:
        """Hand and object at the hover in the arm frame: the step that puts the commanded grasp on the object, if it is believable."""
        pick, target = state.pick, state.pick.target.detection
        top = target.top + state.correction
        camera = self._camera()
        look = self.measurer.measure(obs, camera, pick.side, target, self._bins())   # the object stays where the camera saw it
        turned = target.top if look.camera is None else look.camera.deproject(camera.project(target.top[None]))[0]
        parts = (turned - target.top, None if look.seen_top is None else look.seen_top - turned, look.hand_shift)
        commanded = grasp_point(self.planner, pick.grasp, pick.side, arm_q)
        measured_hand = grasp_point(self.planner, pick.grasp, pick.side, np.asarray(obs["arm_q"], float))
        tracking = (commanded - measured_hand) * bool(self.cfg["track_offset"])
        planned = top - commanded
        if look.object_top is None or look.failed:
            return Measured(look, None, planned, tracking, None, look.failed or "not measured", parts)
        vector = look.object_top - measured_hand
        step = (look.object_top + tracking - top) * np.r_[1.0, 1.0, float(self.cfg["correct_z"])]
        total = state.correction + step
        refused = ("too large" if np.linalg.norm(total) > self.cfg["max_correction"] or abs(total[2]) > self.cfg["max_z_correction"]
                   else "measure only" if self.mode == "measure" else "")
        return Measured(look, vector, planned, tracking, step, refused, parts)

    def _hover_again(self, state: Hovering, arm_q: np.ndarray, hand_q: np.ndarray) -> Hover | None:
        """The hover re-centred on the corrected target, reached from where the hand hovers now."""
        pick = state.pick
        target = pick.plan.waypoints["grasp"] + state.correction
        return plan_hover(self.planner, pick.plan, pick.grasp, pick.side, pick.target.detection, self._camera(), arm_q, hand_q, target)

    def _final_pick(self, state: Hovering, arm_q: np.ndarray, hand_q: np.ndarray) -> Pick | None:
        """The pick planned from the hover onto the corrected target; onto the open-loop one if that has no plan; None if neither."""
        pick = state.pick
        tops = {"corrected": pick.target.detection.top + state.correction} if state.applied else {}
        for name, top in {**tops, "open-loop": pick.target.detection.top}.items():
            det = replace(pick.target.detection, top=top)
            plan = self.planner.plan_pick(det, pick.grasp, pick.side, pick.target.bin_name, arm_q, hand_q, drop_shift=pick.shift,
                                          obstacles=pick.obstacles)
            if plan.ok:
                return replace(pick, plan=plan)
            log.info("servo: no plan from the hover onto the %s target: %s", name, plan.reason)
            state.applied, state.failed = False, f"no plan onto the {name} target"
        return None

    def _log(self, t: float, state: Hovering, measured: Measured, seconds: float) -> None:
        """One line per hover look: what was measured, what is applied, the fit's support and the cost."""
        look, mm = measured.look, lambda v: "(n/a)" if v is None else "({:+.1f},{:+.1f},{:+.1f})".format(*(1e3 * np.asarray(v)))
        verdict = "applied" if not measured.refused else f"not applied: {measured.refused}"
        log.info("t=%.1f servo look %d (%s, camera %s, at the hover %s) %s, %s arm: target top %s mm, hand->object measured %s planned %s mm, tracking %s mm, "
                 "object - target = turn %s + top re-fit %s + hand shift %s mm, step %s mm %s, total %s mm; fit rms %.1f mm, coverage %.2f, "
                 "inliers %.2f, %d points, spread %.1f mm, top visible %.2f; look %.2f s, %.2f s since the hover began", t, state.looks, self.mode,
                 state.source, "the episode's" if look.camera is None else "its own arm fit", state.pick.target, state.pick.side,
                 mm(state.pick.target.detection.top), mm(measured.vector), mm(measured.planned), mm(measured.tracking), *map(mm, measured.parts),
                 mm(measured.step), verdict, mm(state.correction), 1e3 * look.rms, look.coverage, look.inliers, look.points, 1e3 * look.spread,
                 look.visible, seconds, t - state.started)
        believable = measured.step is not None and measured.refused in ("", "measure only")
        self.tally.update(["looks", "believable" if believable else f"refused: {measured.refused}"])
        if state.measured is not None and measured.step is not None:
            self.corrections.append(float(np.linalg.norm(state.measured)))

    def _bins(self) -> dict[str, np.ndarray]:
        """Bin centres (pelvis xy) the planner works against this look."""
        return {name: self.planner.bin_xy(name) for name in self.planner.scene.cfg["bins"]["centers"]}

    def _camera(self) -> Camera:
        """The head camera the detector used at its last look (nominal for a detector that does not say)."""
        extrinsics = getattr(self.detector, "extrinsics", None)
        return self.nominal if extrinsics is None else Camera(self.nominal.K, extrinsics)

    @property
    def _hold_ticks(self) -> int:
        return round(self.cfg["hold_s"] * HZ)
