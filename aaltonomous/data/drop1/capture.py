"""S4 / Drop 1: guided, READ-ONLY capture of the venue conditions (empty scene, our objects, lighting, noise).

Walks the steps of checklist.yaml one by one: prints what to set up, waits for Enter, grabs N frames from every
camera (head RGB + aligned depth, both wrists; lossless PNG) plus joint states, and saves them with camera metadata
and timestamps. It NEVER commands the robot: DDS is subscribe-only (record_demos.DDSTap), cameras are clients.
Arm poses are only captured while a human holds them with teleop. In sim, objects are placed through the scene
service (a stand-in for the human placing them; it does not touch the robot).

    # laptop, env uni3, robot on the gantry, PC2 cameras in Junction mode (scripts/deploy_rgbd_pc2.sh)
    python -m aaltonomous.data.drop1.capture --backend real --network-interface enx00e04cd30dba --out data/drop1
    # resume after a crash / skip ahead: same --session; finished steps are skipped
    python -m aaltonomous.data.drop1.capture --backend real ... --session venue_1113 --only obj_blue_a,obj_blue_b
    # twinflow test against a sim instance (env isaac; AALTO_DDS_DOMAIN / AALTO_CAM_PORT_BASE / ... as for evaluate)
    python -m aaltonomous.data.drop1.capture --backend sim --yes --out runs/drop1
    python -m aaltonomous.data.drop1.capture --backend mock --yes --out /tmp/d1      # synthetic frames, no sim

At each step: Enter = capture, s = skip, q = quit, or type a note and Enter (= capture, note saved).
After a capture with warnings: Enter = keep, r = redo.

On disk: <out>/<session>/session.json (+ copies of checklist.yaml, calibration.yaml, scene.yaml)
         <out>/<session>/<step>/meta.json, frames.npz (t, wall, head_frame_id, joints, stream ages, image hashes),
         head_rgb/ left_wrist_rgb/ right_wrist_rgb/ NNNNNN.png (RGB8), head_depth/ NNNNNN.png (uint16 mm, 0 = invalid)
Next: stats.py, then package.py. Venue procedure: CHECKLIST.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl  # noqa: F401  before DDS (see record_demos)
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
IMGS = CAMS + ("head_depth",)
FORMAT = "aalto-drop1-v1"
CHECKLIST = HERE / "checklist.yaml"
CAL = REPO / "aaltonomous/interface/calibration.yaml"
SCENE = REPO / "aaltonomous/scene/scene.yaml"


def load_checklist(path=CHECKLIST):
    c = yaml.safe_load(Path(path).read_text())
    ids = [s["id"] for s in c["steps"]]
    assert len(ids) == len(set(ids)), "duplicate step ids"
    return c


def step_objects(step, scene=None):
    """Objects on the mat for a step (mat coords, yaw rad). layout steps: the layout card of (task, seed)."""
    if step["kind"] == "objects":
        return [dict(yaw=0.0, **o) for o in step["objects"]]
    if step["kind"] == "layout":
        from aaltonomous.scene.cards import make_card
        from aaltonomous.scene.scene import Scene
        return make_card(scene or Scene.load(), step["task"], step["seed"])["cubes"]
    return []


def step_prompt(step, scene=None):
    if step.get("prompt"):
        p = step["prompt"]
    elif step["kind"] == "objects":
        p = "Place on the mat (mat grid, cm; robot's left = larger x), upright, faces square to the mat edge:\n" + \
            "\n".join(f"      {o['color']:6s} {o['shape']:8s} x={o['x'] * 100:5.1f}  y={o['y'] * 100:5.1f}"
                      for o in step["objects"]) + "\n    Other objects off the counter. Nobody in view."
    elif step["kind"] == "layout":
        from aaltonomous.interface.tasks import layout_card
        from aaltonomous.scene.cards import make_card
        from aaltonomous.scene.scene import Scene
        c = make_card(scene or Scene.load(), step["task"], step["seed"])
        p = f"Layout card {step['task']} seed {step['seed']} (printed card set):\n" + layout_card(c["cubes"], c["instruction"])
    else:
        p = ""
    if step.get("lights"):
        p += f"\n    LIGHTS: {step['lights']}"
    return p


# ------------------------------------------------------------------ sources
class LiveSource:
    """Real robot or sim: record_demos' read-only DDS tap + camera clients. Never publishes."""

    def __init__(self, backend, host, network_interface=None, rgbd_port=55560, dds=True, head_rgbd=True):
        from aaltonomous.data.record_demos import CameraTap, DDSTap
        self.backend = backend
        domain = 0 if backend == "real" else int(os.environ.get("AALTO_DDS_DOMAIN", "1"))
        self.dds = DDSTap(domain, network_interface) if dds else None
        self.cams = CameraTap(host, head_rgbd=head_rgbd, rgbd_port=rgbd_port)
        self.scene = None
        if backend == "sim":
            try:
                from aaltonomous.sim.scene_client import SceneClient
                self.scene = SceneClient(host, timeout_ms=5000)
                self.scene.ping()
            except Exception as e:
                print(f"[capture] no sim scene service ({e}); objects will not be placed")
                self.scene = None
        self.info = {"backend": backend, "dds_domain": domain if dds else None, "host": host,
                     "rgbd_port": rgbd_port, "head_rgbd": head_rgbd,
                     "cam_port_base": os.environ.get("AALTO_CAM_PORT_BASE")}

    def place(self, objects):
        """sim only: put the step's objects on the mat (empty list parks all). Real: the human does it."""
        if self.scene is None:
            return False
        self.scene.reset([{k: o[k] for k in ("color", "shape", "x", "y", "yaw")} for o in objects])
        time.sleep(1.5)  # let them settle and the renderer catch up
        return True

    def grab(self):
        imgs = self.cams.images()
        fid = imgs.pop("head_frame_id", -1)
        row = self.dds.sample() if self.dds else {}
        return imgs, fid, row

    def close(self):
        self.cams.close()


class MockSource:
    """Synthetic static scene for offline tests: grey counter + coloured objects at their projected positions,
    Gaussian RGB noise, depth = counter plane / object tops with noise and random holes."""

    def __init__(self, seed=0, rgb_sigma=2.0, depth_sigma_mm=3.0, hole_p=0.02, obj_hole_p=0.2):
        from aaltonomous.interface.calibration import Calibration
        from aaltonomous.scene.scene import Scene
        self.cal, self.sc, self.rng = Calibration.load(), Scene.load(), np.random.default_rng(seed)
        self.rgb_sigma, self.dsig, self.hole_p, self.obj_hole_p = rgb_sigma, depth_sigma_mm, hole_p, obj_hole_p
        self.fid, self.objects = 0, []
        self.info = {"backend": "mock"}
        self._render()

    def place(self, objects):
        self.objects = list(objects)
        self._render()
        return True

    def _render(self):
        c = self.cal.cams["head"]
        H, W = c.height, c.width
        v, u = np.mgrid[0:H, 0:W].astype(float)
        # ray (optical) -> pelvis; intersect with z = counter
        d = np.stack([(u - c.K[0, 2]) / c.K[0, 0], (v - c.K[1, 2]) / c.K[1, 1], np.ones_like(u)], -1)
        T = c.T_pelvis_optical
        dp = d @ T[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.sc.counter_z - T[2, 3]) / dp[..., 2]
        depth = np.where(s > 0, s * 1000.0, 0.0)          # optical z = s (d has z = 1)
        rgb = np.zeros((H, W, 3)) + [150, 150, 140]
        self.obj_mask = np.zeros((H, W), bool)
        for o in self.objects:
            h = self.sc.object_height(o["shape"])
            ctr = self.sc.mat_to_pelvis(np.array([o["x"], o["y"], h]))
            uvz = self.cal.project("head", ctr)
            r = 0.5 * self.sc.size * c.K[0, 0] / uvz[2]
            m = (np.abs(u - uvz[0]) < r) & (np.abs(v - uvz[1]) < r)
            rgb[m] = self.sc.colors[o["color"]]
            depth[m] = uvz[2] * 1000.0
            self.obj_mask |= m
        self.rgb0, self.depth0 = rgb, depth

    def grab(self):
        self.fid += 1
        g = self.rng
        rgb = np.clip(self.rgb0 + g.normal(0, self.rgb_sigma, self.rgb0.shape), 0, 255).astype(np.uint8)
        dep = self.depth0 + g.normal(0, self.dsig, self.depth0.shape)
        holes = g.random(dep.shape) < np.where(self.obj_mask, self.obj_hole_p, self.hole_p)
        dep = np.where(holes | (self.depth0 <= 0), 0, np.clip(dep, 1, 65535)).astype(np.uint16)
        wr = lambda: np.clip(110 + g.normal(0, self.rgb_sigma, self.rgb0.shape), 0, 255).astype(np.uint8)  # noqa: E731
        imgs = {"head_rgb": rgb, "head_depth": dep, "left_wrist_rgb": wr(), "right_wrist_rgb": wr()}
        row = {"arm_q": np.zeros(14), "arm_dq": np.zeros(14), "hand_q": np.zeros(14), "body_q": np.zeros(29)}
        return imgs, self.fid, row

    def close(self):
        pass


# ------------------------------------------------------------------ capture one step
def h8(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).hexdigest()


def capture_frames(src, n, fps=10.0, new_head_timeout=1.0):
    """n frame sets, at most `fps`, each with a NEW head frame (frame_id changed) when the source has ids."""
    frames, rows, prev = [], [], None
    t0 = time.monotonic()
    for i in range(n):
        target = t0 + i / fps
        time.sleep(max(0.0, target - time.monotonic()))
        tw = time.monotonic()
        while True:
            imgs, fid, row = src.grab()
            if fid < 0 or fid != prev or time.monotonic() - tw > new_head_timeout:
                break
            time.sleep(0.005)
        prev = fid
        row = dict(row)
        row.update(t=time.monotonic() - t0, wall=time.time(), head_frame_id=fid,
                   **{f"hash_{k}": h8(imgs[k]) for k in IMGS})
        frames.append(imgs)
        rows.append(row)
    return frames, rows


def quick_check(frames, rows, kind):
    """One-line sanity check shown right after a capture, plus warnings worth a redo."""
    warn, info = [], {}
    for k in CAMS:
        m = float(np.mean([f[k].mean() for f in frames]))
        info[k + "_mean"] = round(m, 1)
        if m < 1.0:
            warn.append(f"{k} blank")
        u = len({r[f'hash_{k}'] for r in rows})
        info[k + "_unique"] = u
        if len(rows) > 2 and u < len(rows) // 2:
            warn.append(f"{k}: only {u}/{len(rows)} distinct frames (stale stream? noise stats biased)")
    valid = float(np.mean([(f["head_depth"] > 0).mean() for f in frames]))
    info["head_depth_valid"] = round(valid, 3)
    if valid < 0.3:
        warn.append(f"head depth only {valid:.0%} valid")
    ids = [r["head_frame_id"] for r in rows]
    if ids and ids[0] >= 0 and len(set(ids)) < len(ids):
        warn.append(f"head frame_id repeated ({len(set(ids))}/{len(ids)} distinct)")
    q = np.array([r.get("arm_q", np.full(14, np.nan)) for r in rows], float)
    if np.isfinite(q).all():
        span = float((q.max(0) - q.min(0)).max())       # joint drift during the capture (dq is too noisy)
        info["arm_q_span"] = round(span, 4)
        if kind == "arm_pose" and span > 0.03:
            warn.append(f"arms moved {span:.3f} rad during the capture: hold still and redo")
    return info, warn


def save_step(sdir: Path, step, frames, rows, meta):
    import cv2
    tmp = sdir.with_name(sdir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for k in IMGS:
        (tmp / k).mkdir(parents=True)
    for i, f in enumerate(frames):
        for k in CAMS:
            cv2.imwrite(str(tmp / k / f"{i:06d}.png"), f[k][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(tmp / "head_depth" / f"{i:06d}.png"), f["head_depth"].astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cols = {}
    for k in rows[0]:
        v = [r.get(k) for r in rows]
        cols[k] = np.array(v) if not isinstance(v[0], str) else np.array(v, dtype="U16")
    np.savez_compressed(tmp / "frames.npz", **cols)
    meta["camera_meta"] = {k: {"shape": list(frames[0][k].shape), "dtype": str(frames[0][k].dtype)} for k in IMGS}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    shutil.rmtree(sdir, ignore_errors=True)
    tmp.rename(sdir)


def git_rev():
    from aaltonomous.data.record_demos import git_rev as g
    return g()


# ------------------------------------------------------------------ session
def run(src, out, session=None, checklist=CHECKLIST, yes=False, only=None, skip_optional=False,
        frames_scale=1.0, fps=10.0, operator="", ask=input, log=print):
    cl = load_checklist(checklist)
    session = session or "drop1_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    sdir = Path(out) / session
    sdir.mkdir(parents=True, exist_ok=True)
    for p in (checklist, CAL, SCENE):
        shutil.copy(p, sdir / Path(p).name)
    sj = sdir / "session.json"
    sess = json.loads(sj.read_text()) if sj.exists() else {
        "format": FORMAT, "session": session, "start": datetime.now().isoformat(timespec="seconds"),
        "git": git_rev(), "source": src.info, "operator": operator, "log": []}
    sess["source"] = src.info

    def save_sess():
        sj.write_text(json.dumps(sess, indent=1))
    save_sess()

    steps = [s for s in cl["steps"] if (not only or s["id"] in only)]
    budget = sum(s.get("minutes", 0) for s in steps)
    log(f"[capture] session {sdir}  ({len(steps)} steps, ~{budget} min budget)")
    t_start = time.monotonic()
    done = []
    for k, st in enumerate(steps, 1):
        sid, kind = st["id"], st["kind"]
        if (sdir / sid / "meta.json").exists() or (kind == "manual" and any(e.get("step") == sid and e.get("done")
                                                                             for e in sess["log"])):
            log(f"-- [{k}/{len(steps)}] {sid}: already done, skipped")
            continue
        if st.get("optional") and skip_optional:
            continue
        objs = step_objects(st)
        el = (time.monotonic() - t_start) / 60
        log(f"\n== [{k}/{len(steps)}] {sid} ({kind}, ~{st.get('minutes', '?')} min; {el:.0f} min elapsed)"
            + ("  [optional]" if st.get("optional") else "") + f"\n    {step_prompt(st)}")
        if kind == "arm_pose":
            log("    !! This tool sends NOTHING to the robot. Only the teleop operator moves the arms; spotter at e-stop.")
        placed = src.place(objs) if kind in ("empty", "objects", "layout") else False
        if placed:
            log("    (sim: objects placed through the scene service)")
        note = ""
        if not yes:
            a = ask("    Enter = " + ("done" if kind == "manual" else "capture") + ", s = skip, q = quit, or a note: ").strip()
            if a.lower() == "q":
                break
            if a.lower() == "s":
                sess["log"].append({"step": sid, "skipped": True, "wall": time.time()})
                save_sess()
                continue
            note = a
        if kind == "manual":
            sess["log"].append({"step": sid, "done": not yes, "auto_skipped": yes, "note": note,
                                "wall": time.time(), "prompt": st.get("prompt", "")})
            save_sess()
            continue
        n = max(2, int(round(st.get("frames", cl.get("default_frames", 20)) * frames_scale)))
        while True:
            t0 = datetime.now().isoformat(timespec="milliseconds")
            frames, rows = capture_frames(src, n, fps)
            info, warn = quick_check(frames, rows, kind)
            log(f"    captured {n} frames: " + ", ".join(f"{a}={b}" for a, b in info.items())
                + (f"\n    WARN: {warn}" if warn else ""))
            if warn and not yes:
                if ask("    Enter = keep, r = redo: ").strip().lower() == "r":
                    continue
            break
        meta = {"format": FORMAT, "step": st, "kind": kind, "lights": st.get("lights", "venue_on"),
                "objects": objs, "start": t0, "end": datetime.now().isoformat(timespec="milliseconds"),
                "n_frames": n, "fps_target": fps, "note": note, "quick_check": info, "warnings": warn,
                "source": src.info, "sim_placed": placed, "calibration": "calibration.yaml (session copy)"}
        save_step(sdir / sid, st, frames, rows, meta)
        sess["log"].append({"step": sid, "done": True, "n_frames": n, "warnings": warn, "wall": time.time()})
        save_sess()
        done.append(sid)
    sess["end"] = datetime.now().isoformat(timespec="seconds")
    save_sess()
    log(f"\n[capture] {len(done)} step(s) captured this run in {(time.monotonic() - t_start) / 60:.1f} min -> {sdir}"
        f"\n  next: python -m aaltonomous.data.drop1.stats {sdir}")
    return sdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("real", "sim", "mock"), default="real")
    ap.add_argument("--network-interface", default=None, help="real: enx00e04cd30dba on the laptop")
    ap.add_argument("--img-server-ip", default=None, help="default 192.168.123.164 (real) / 127.0.0.1 (sim)")
    ap.add_argument("--rgbd-port", type=int, default=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    ap.add_argument("--no-dds", action="store_true", help="no joint states (cameras only)")
    ap.add_argument("--out", default=str(REPO / "data/drop1"))
    ap.add_argument("--session", default=None, help="reuse to resume (finished steps are skipped)")
    ap.add_argument("--checklist", default=str(CHECKLIST))
    ap.add_argument("--only", default=None, help="comma-separated step ids")
    ap.add_argument("--skip-optional", action="store_true", help="skip optional steps (arm poses, own-lamps-off)")
    ap.add_argument("--frames-scale", type=float, default=1.0, help="scale every step's frame count (hurry: 0.5)")
    ap.add_argument("--fps", type=float, default=10.0, help="max capture rate; frames spaced for temporal noise")
    ap.add_argument("--operator", default="")
    ap.add_argument("--yes", action="store_true", help="no prompts (tests): capture every step, log manual ones")
    a = ap.parse_args(argv)
    if a.backend == "mock":
        src = MockSource()
    else:
        host = a.img_server_ip or ("192.168.123.164" if a.backend == "real" else "127.0.0.1")
        src = LiveSource(a.backend, host, a.network_interface, a.rgbd_port, dds=not a.no_dds)
        time.sleep(1.5)
    try:
        run(src, a.out, a.session, a.checklist, a.yes, a.only.split(",") if a.only else None, a.skip_optional,
            a.frames_scale, a.fps, a.operator)
    except KeyboardInterrupt:
        print("\n[capture] interrupted; finished steps are saved, rerun with the same --session to resume")
    finally:
        src.close()
    sys.stdout.flush()
    os._exit(0)  # DDS threads


if __name__ == "__main__":
    main()
