"""Grasp bench on a running sim: one explicit Grasp per arm on single objects, traced with privileged object poses.

    python -m halal.eval.grasp_bench --spec runs/grasp/first.yaml --out runs/grasp/first     (sim env vars select the instance)

Spec (yaml): seed, repeats, dr (any | hard), bin (side | center), arms (2 | 1 at a time), end (home | retreat), empty (bool), shapes, yaws_deg
(layout yaw of the left-arm object; the right arm gets the mirror image), spots (pelvis xy, left arm), noise (perceived
minus true pose, [{pos: m in a random direction, dx: m forward, dy: m outward, yaw_deg, yaw_known}], all optional),
variants ({name: grasp params as in config `motion.grasps`, plus `base`, its own `shapes` / `spots` / `yaws_deg` / `noise` /
`repeats`, `squeeze`, `hand` (stiff | soft), `candidates`, `overrides`
(organiser config keys), `motion` (our `motion` config keys),
`progressive` ({firm, step, max}: squeeze one step deeper after each close until the largest grip joint error reaches `firm`)}).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import ssl  # noqa: F401  must load before DDS, as in the organisers' evaluator
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

from aaltonomous.baselines.classical import config as baseline
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.sim import evaluate
from aaltonomous.sim.aalto_task import dr as dr_module
from aaltonomous.sim.aalto_task.layout import object_names
from aaltonomous.sim.scene_client import SceneClient

from halal import config
from halal.motion import Grasp, PickPlanner, candidate_grasps
from halal.motion.grasps import for_hand, right_hand, strategy_grasps
from halal.perception import ObjectDetection

SIDES = ("left", "right")
BIN_COLOR = {bin_name: color for color, bin_name in FIXED_MAPPING.items()}
MARKS = {"close": ("close", 1.0), "grip": ("lift", 0.0), "lift": ("lift", 1.0), "carry": ("carry", 0.5), "open": ("open", 1.0)}
GRIP_JOINTS = [1, 3, 5]
STUCK_ABOVE = 0.15


@dataclass(frozen=True)
class Case:
    """One pick: a grasp variant on one object pose with one arm, a bin, a perception error and the object's presence."""
    variant: str
    shape: str
    yaw_deg: float
    spot: tuple[float, float]
    side: str
    bin_name: str
    pos_noise: float
    offset: tuple[float, float]
    yaw_noise_deg: float
    yaw_known: bool
    present: bool


@dataclass
class Pick:
    """A case inside a running episode: the sim object, what the planner was told and what it planned."""
    case: Case
    name: str | None
    true_pos: np.ndarray
    true_yaw: float | None
    detection: ObjectDetection
    grasp: Grasp | None
    plan: Plan
    squeeze: float


def list_cases(spec: Mapping[str, Any]) -> list[Case]:
    """Every variant on every shape, spot, yaw and noise level in a seeded random order, arms alternating."""
    grid = [(name, *rest) for name, variant in spec["variants"].items()
            for rest in itertools.product(*(variant.get(key, spec[key]) for key in ("shapes", "spots", "yaws_deg", "noise")),
                                          range(variant.get("repeats", spec["repeats"])))]
    order = np.random.default_rng(spec["seed"]).permutation(len(grid))
    return [_case(spec, *grid[k], SIDES[i % 2]) for i, k in enumerate(order)]


def _case(spec: Mapping[str, Any], variant: str, shape: str, spot: list[float], yaw: float, noise: Mapping[str, Any],
          _repeat: int, side: str) -> Case:
    return Case(variant=variant, shape=shape, yaw_deg=float(yaw), spot=(float(spot[0]), float(spot[1])), side=side,
                bin_name=side if spec["bin"] == "side" else spec["bin"], pos_noise=float(noise.get("pos", 0.0)),
                offset=(float(noise.get("dx", 0.0)), float(noise.get("dy", 0.0))),
                yaw_noise_deg=float(noise.get("yaw_deg", 0.0)), yaw_known=bool(noise.get("yaw_known", True)),
                present=not spec["empty"])


def mirrored(case: Case) -> tuple[np.ndarray, float]:
    """Object centre (pelvis xy) and layout yaw (rad) of a case: the right arm works on the mirror image."""
    sign = 1.0 if case.side == "left" else -1.0
    return np.array([case.spot[0], sign * case.spot[1]]), sign * np.radians(case.yaw_deg)


def quat_yaw(quat_xyzw: list[float]) -> float:
    x, y, z, w = quat_xyzw
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_tilt(quat_xyzw: list[float]) -> float:
    """Angle (rad) between the object's axis and the vertical."""
    x, y, _, _ = quat_xyzw
    return float(2 * np.arcsin(min(1.0, float(np.hypot(x, y)))))


def perceived_yaw(shape: str, quat_xyzw: list[float]) -> float | None:
    """Sim pose -> the perception convention: cube = a face normal, triangle = apex direction, cylinder = none."""
    offset = {"cube": 0.0, "triangle": np.pi / 2}.get(shape)
    return None if offset is None else quat_yaw(quat_xyzw) + offset


class DrSeeds:
    """Seeds for the sim's DR config; `hard` keeps those that give every named object low friction and high mass."""

    def __init__(self, scene_client: SceneClient, scene: Any, first: int, hard: bool) -> None:
        info = scene_client.info().get("dr")
        self.cfg = dr_module.load(info["path"]) if info else None
        self.names, self.bins = object_names(scene), scene.bin_names
        self.next, self.hard = first, hard and self.cfg is not None

    def take(self, names: list[str]) -> int:
        limits = config.load()["grasp_bench"]["hard_dr"]
        while True:
            seed, self.next = self.next, self.next + 1
            if not self.hard:
                return seed
            objects = dr_module.sample(self.cfg, seed, self.names, self.bins)["objects"]
            if all(objects[n]["friction_scale"] <= limits["friction_max"] and objects[n]["mass_scale"] >= limits["mass_min"]
                   for n in names):
                return seed


class Bench:
    def __init__(self, spec: Mapping[str, Any], out: Path, host: str) -> None:
        evaluate.apply_instance_env()
        self.spec, self.out, self.settings = spec, out, config.load()["grasp_bench"]
        self.scene_client = SceneClient(host)
        self.env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=1e9, realtime=True, img_server_ip=host,
                            head_rgbd=False, cameras=False, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
        self.scene = self.env.scene
        self.planners = {name: self._planner(variant) for name, variant in spec["variants"].items()}
        self.seeds = DrSeeds(self.scene_client, self.scene, int(spec["seed"]), spec["dr"] == "hard")
        self.rest_z = self.scene.counter_z + self.scene.object_height() / 2

    def _planner(self, variant: Mapping[str, Any]) -> PickPlanner:
        """A variant's own planner: organiser config with its `overrides`, our motion settings with its `motion`."""
        overrides = {**config.load()["baseline_overrides"], **variant.get("overrides", {})}
        planner = PickPlanner.create(self.env.filter.cfg, baseline.load_config(profile="sim", overrides=overrides))
        planner.motion.update(variant.get("motion", {}))
        return planner

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------ one episode
    def run(self, index: int, cases: list[Case]) -> list[dict[str, Any]]:
        """Home, randomise, place the objects, plan one pick per arm, execute both, score and summarise."""
        evaluate.home_arms(self.env, self.scene_client)
        names = [f"obj_{BIN_COLOR[c.bin_name]}_{c.shape}_0" for c in cases if c.present]
        seed = self.seeds.take(names)
        dr = self.scene_client.randomize(seed).get("params") or {}
        self.scene_client.reset([self._layout_entry(c) for c in cases if c.present])
        self._hold(self.settings["settle_ticks"])
        truth = {o["name"]: o for o in self.scene_client.state()["objects"]}
        rng = np.random.default_rng(seed)
        picks = [self._prepare(c, truth, rng) for c in cases]
        trace = self._execute(picks)
        evaluate.settle(self.scene_client, 8.0, float(self.scene.cfg["success"]["at_rest_speed"]))
        outcomes = {o["name"]: o["outcome"] for o in self.scene_client.evaluate("sort_fixed", dict(FIXED_MAPPING))["outcomes"]}
        final = {o["name"]: o for o in self.scene_client.state()["objects"]}
        np.savez_compressed(self.out / "traces" / f"episode_{index:04d}.npz", **trace)
        return [self._record(index, seed, p, k, trace, outcomes, final, dr) for k, p in enumerate(picks)]

    def _layout_entry(self, case: Case) -> dict[str, Any]:
        xy, yaw = mirrored(case)
        mat = self.scene.pelvis_to_mat(np.r_[xy, 0.0])
        return {"color": BIN_COLOR[case.bin_name], "shape": case.shape, "x": float(mat[0]), "y": float(mat[1]), "yaw": float(yaw)}

    def _hold(self, ticks: int) -> None:
        action = {"arm_q_target": self.env.filter.arm_cmd.copy(), "hand_q_target": self.env.filter.hand_cmd.copy()}
        for _ in range(ticks):
            self.env.step(action)

    def _prepare(self, case: Case, truth: Mapping[str, Any], rng: np.random.Generator) -> Pick:
        """Detection with the case's perception error, the variant's grasp for it, and the plan."""
        name = f"obj_{BIN_COLOR[case.bin_name]}_{case.shape}_0" if case.present else None
        pos = np.asarray(truth[name]["pos"], float) if name else np.r_[mirrored(case)[0], self.rest_z]
        yaw = perceived_yaw(case.shape, truth[name]["quat_xyzw"]) if name else None
        angle, sign = rng.uniform(0, 2 * np.pi), rng.choice([-1.0, 1.0])
        outward = 1.0 if case.side == "left" else -1.0
        error = case.pos_noise * np.r_[np.cos(angle), np.sin(angle)] + [case.offset[0], outward * case.offset[1]]
        top = pos + np.r_[error, self.scene.object_height() / 2]
        seen_yaw = yaw + sign * np.radians(case.yaw_noise_deg) if case.yaw_known and yaw is not None else None
        detection = ObjectDetection(color=BIN_COLOR[case.bin_name], top=top, shape=case.shape, yaw=seen_yaw)
        grasp, plan = self._plan(case, detection)
        depth = self._squeeze_override(case.variant)
        return Pick(case, name, pos, yaw, detection, grasp, plan, (grasp.squeeze if grasp else 0.0) if depth is None else depth)

    def _squeeze_override(self, variant: str) -> float | None:
        """A variant's `squeeze` when it is one number for every shape; None leaves the depth to the grasp."""
        depth = self.spec["variants"][variant].get("squeeze")
        return float(depth) if isinstance(depth, (int, float)) else None

    def _plan(self, case: Case, detection: ObjectDetection) -> tuple[Grasp | None, Plan]:
        """First of the variant's grasps (one, or the policy's candidate list) that plans."""
        variant, planner = self.spec["variants"][case.variant], self.planners[case.variant]
        hand = variant.get("hand", "stiff")
        if variant.get("candidates"):
            grasps = candidate_grasps(detection.shape, detection.yaw, case.side, hand)
        else:
            params = {**for_hand(config.load()["motion"]["grasps"].get(variant.get("base"), {}), hand), **variant}
            grasps = strategy_grasps(case.variant, params, detection.shape, detection.yaw, case.side)
        plan = Plan(False, side=case.side, reason="no grasp for this object")
        for grasp in grasps:
            plan = planner.plan_pick(detection, grasp, case.side, case.bin_name, self.env.filter.arm_cmd.copy(),
                                     self.env.filter.hand_cmd.copy(), end=self.spec["end"],
                                     squeeze=self._squeeze_override(case.variant))
            if plan.ok:
                return grasp, plan
        return (grasps[-1] if grasps else None), plan

    def _execute(self, picks: list[Pick]) -> dict[str, np.ndarray]:
        """Run both arms' plans side by side; per tick commanded and measured joints, every few ticks the objects' poses."""
        every, names = self.settings["trace_every"], [p.name for p in picks]
        rows: dict[str, list] = {key: [] for key in ("arm_cmd", "hand_cmd", "arm_q", "hand_q", "poses")}
        tick = 0
        while any(p.plan.ok and tick < len(p.plan.arm) for p in picks):
            arm, hand = self._targets(picks, tick)
            obs, info = self.env.step({"arm_q_target": arm, "hand_q_target": hand})
            if info["clipped"]:
                raise RuntimeError(f"safety clip during a dry-run-checked plan: {info['clip_reasons']}")
            for key, value in (("arm_cmd", arm), ("hand_cmd", hand), ("arm_q", obs["arm_q"]), ("hand_q", obs["hand_q"])):
                rows[key].append(value)
            if tick % every == 0:
                state = {o["name"]: o for o in self.scene_client.state()["objects"]}
                rows["poses"].append([state[n]["pos"] + state[n]["quat_xyzw"] if n else [np.nan] * 7 for n in names])
            for pick in picks:
                self._deepen_if_weak(pick, tick, obs["hand_q"] - hand)
            tick += 1
        trace = {key: np.array(value, float) for key, value in rows.items()}
        return {**trace, "poses": trace["poses"].reshape(-1, len(names), 7)}

    def _targets(self, picks: list[Pick], tick: int) -> tuple[np.ndarray, np.ndarray]:
        """Arm and hand targets of this tick: each arm follows its own plan and holds once it is done or has none."""
        arm, hand = self.env.filter.arm_cmd.copy(), self.env.filter.hand_cmd.copy()
        for pick in picks:
            if pick.plan.ok:
                cols, row = ArmKinematics.sl(pick.case.side), min(tick, len(pick.plan.arm) - 1)
                arm[cols], hand[cols] = pick.plan.arm[row, cols], pick.plan.hand[row, cols]
        return arm, hand

    def _deepen_if_weak(self, pick: Pick, tick: int, error: np.ndarray) -> None:
        """Progressive variants: at the end of a close or squeeze phase, squeeze one step deeper unless the grip is firm."""
        rule = self.spec["variants"][pick.case.variant].get("progressive")
        if not rule or not pick.plan.ok:
            return
        last = max(stop for name, _, stop in pick.plan.phases if name in ("close", "squeeze")) - 1
        grip = np.abs(error[ArmKinematics.sl(pick.case.side)][GRIP_JOINTS]).max()
        if tick == last and grip < rule["firm"] and pick.squeeze + rule["step"] <= rule["max"] + 1e-9:
            pick.squeeze += rule["step"]
            pick.plan = self.planners[pick.case.variant].resqueezed(pick.plan, pick.grasp, pick.case.side, pick.squeeze)

    # ------------------------------------------------------------------ summary of one pick
    def _record(self, index: int, seed: int, pick: Pick, slot: int, trace: Mapping[str, np.ndarray],
                outcomes: Mapping[str, str], final: Mapping[str, Mapping[str, Any]], dr: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"episode": index, "seed": seed, **asdict(pick.case), "object": pick.name,
                                  "true_pos": pick.true_pos.round(4).tolist(), "true_yaw": pick.true_yaw,
                                  "seen_yaw": pick.detection.yaw, "grasp": pick.grasp.name if pick.grasp else None,
                                  "finger_yaw": pick.grasp.finger_yaw if pick.grasp else None,
                                  "planned": bool(pick.plan.ok), "plan_reason": pick.plan.reason, "squeeze": pick.squeeze,
                                  "dr": (dr.get("objects") or {}).get(pick.name)}
        if not pick.plan.ok:
            return {**record, "outcome": outcomes.get(pick.name), "failure": "no_plan"}
        marks = {key: self._tick(pick.plan, phase, fraction) for key, (phase, fraction) in MARKS.items()}
        cols = ArmKinematics.sl(pick.case.side)
        error = (trace["hand_q"] - trace["hand_cmd"])[:, cols]
        left_order = (lambda e: e) if pick.case.side == "left" else right_hand
        record["finger_error"] = {key: np.round(left_order(error[t]), 4).tolist() for key, t in marks.items()}
        record["phases_s"] = {name: round((end - start) / 30.0, 2) for name, start, end in pick.plan.phases}
        record["release_xy"] = np.round(pick.plan.waypoints["drop"][:2], 4).tolist()
        if pick.name is None:
            return {**record, "outcome": None, "failure": "empty"}
        record.update(self._object_metrics(pick, slot, marks, trace))
        record["outcome"], record["final_pos"] = outcomes[pick.name], np.round(final[pick.name]["pos"], 4).tolist()
        record["final_tilt_deg"] = round(float(np.degrees(quat_tilt(final[pick.name]["quat_xyzw"]))), 1)
        record["failure"] = self._failure(record)
        return record

    @staticmethod
    def _tick(plan: Plan, phase: str, fraction: float) -> int:
        """Tick index at a fraction of a plan phase (1.0 = its last tick)."""
        start, end = next((s, e) for name, s, e in plan.phases if name == phase)
        return start + int(round(fraction * (end - 1 - start)))

    def _object_metrics(self, pick: Pick, slot: int, marks: Mapping[str, int], trace: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """Where the object was relative to the hand at the marks, and how it turned between close and lift."""
        every, planner = self.settings["trace_every"], self.planners[pick.case.variant]
        poses = trace["poses"][:, slot]
        marks = {**marks, "retreat": self._tick(pick.plan, "retreat", 1.0), "start": self._tick(pick.plan, "descend", 1.0)}
        in_palm, held, height = {}, {}, {}
        for key, tick in marks.items():
            pose = poses[min(tick // every, len(poses) - 1)]
            planner.kin._fk(trace["arm_q"][tick])
            palm = planner.kin.data.oMf[planner.kin.palm[pick.case.side]]
            in_palm[key] = np.round((pose[:3] - palm.translation) @ palm.rotation, 4).tolist()
            grasp_point = palm.translation + palm.rotation @ planner.kin.offset[pick.case.side]
            height[key] = round(float(pose[2] - self.rest_z), 4)
            held[key] = bool(np.linalg.norm(pose[:3] - grasp_point) < self.settings["held_radius"]
                             and height[key] > self.settings["lifted_height"])
        start, lifted = poses[marks["start"] // every], poses[min(marks["lift"] // every, len(poses) - 1)]
        spin = (quat_yaw(lifted[3:]) - quat_yaw(start[3:]) + np.pi) % (2 * np.pi) - np.pi
        slip = np.linalg.norm(np.subtract(in_palm["carry"], in_palm["close"]))
        return {"in_palm": in_palm, "held": held, "height": height, "spin_deg": round(float(np.degrees(spin)), 1),
                "tilt_deg": round(float(np.degrees(quat_tilt(lifted[3:]))), 1), "slip_mm": round(1000 * float(slip), 1),
                "max_height": round(float(np.nanmax(poses[:, 2]) - self.rest_z), 4)}

    @staticmethod
    def _failure(record: Mapping[str, Any]) -> str:
        """Where a pick went wrong, from the object's presence in the hand at the phase marks."""
        held = record["held"]
        if record["outcome"] == "correct":
            return "ok"
        if not held["lift"]:
            return "lost_in_lift" if record["max_height"] > 0.02 else "not_lifted"
        if not held["carry"] or not held["open"] and record["height"]["open"] < 0.02:
            return "lost_in_carry"
        if held["retreat"] or record["final_pos"][2] > STUCK_ABOVE:
            return "stuck_in_hand"
        return "off_counter" if record["outcome"] == "off" else "missed_bin"


def episodes(cases: list[Case], arms: int) -> Iterator[list[Case]]:
    """Cases in groups of `arms`: 2 = one pick per arm at once, 1 = a single arm working (centre bin, cross-body spots)."""
    for k in range(0, len(cases), arms):
        yield cases[k:k + arms]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    spec = yaml.safe_load(args.spec.read_text())
    (args.out / "traces").mkdir(parents=True, exist_ok=True)
    (args.out / "spec.yaml").write_text(args.spec.read_text())
    bench = Bench(spec, args.out, args.host)
    try:
        with open(args.out / "picks.jsonl", "a") as log:
            for index, cases in enumerate(episodes(list_cases(spec), spec.get("arms", 2))):
                for record in bench.run(index, cases):
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{index:4d} {record['variant']:<14} {record['shape']:<8} {record['side']:<5} yaw {record['yaw_deg']:+4.0f} "
                          f"-> {record['failure']}", flush=True)
    finally:
        bench.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
