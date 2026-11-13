"""M9 demo recorder: logs VR-teleop episodes in the challenge interface's obs/action space (spec §4, §7).

Runs NEXT TO teleop (scripts/teleop_safe.py) as a separate, READ-ONLY process: it only subscribes to DDS and
the camera servers and never creates a DDS publisher. At 30 Hz it samples
  action  arm_q_target (14)  <- rt/lowcmd motor_cmd[15..28].q   (what teleop commands; debug mode, no --motion)
          hand_q_target (14) <- rt/dex3/{left,right}/cmd motor_cmd[0..6].q
  obs     arm_q, arm_dq (14) <- rt/lowstate;  hand_q (14) <- rt/dex3/{left,right}/state
          head_rgb + head_depth (head_rgbd_server :55560), left/right_wrist_rgb (teleimager), instruction, t
in joint_spec.yaml order (arm = DDS index from the yaml, hands = each hand's own DDS order).
action[t] = the command at the tick where obs[t] is sampled (as in AaltoEnv/runner logs), computed after the episode as
the mean of all command messages within +-1/60 s of the tick (anti-aliased resample to 30 Hz, spec §7).

    # laptop, env uni3, while teleop_safe.py runs in another terminal (real robot: DDS domain 0)
    python -m aaltonomous.data.record_demos --backend real --network-interface enx00e04cd30dba \
        --task sort_fixed --card-seed 1000 --out data/demos --operator erkka
    python -m aaltonomous.data.record_demos --backend real --network-interface enx00e04cd30dba --check   # rates only
    # twinflow sim test (2nd sim instance: AALTO_DDS_DOMAIN / AALTO_CAM_PORT_BASE / AALTO_RGBD_PORT as for evaluate)
    python -m aaltonomous.data.record_demos --backend sim --img-server-ip 127.0.0.1 --task sort_fixed --out runs/demos

Keys (single key, no Enter): n = start episode, s = stop (then y = success / f = failure / d = discard),
d = discard while recording, ] / [ = next / previous layout card, c = show card, q = quit.
Layout card k uses seed --card-seed + k (same as `python -m aaltonomous.scene.cards --seed`), which fixes the
instruction (rung 2), mapping and object layout; print the cards with that command.

On disk (one dir per episode, written as <name>.partial and renamed when labelled, so a crash only loses the
episode in progress):  <out>/<session>/ep<NNN>_<task>_s<seed>/
  meta.json       task, instruction, mapping, layout, card seed, success, operator, rates, warnings, filter check
  lowdim.npz      t, tm, wall, action_arm, action_hand (box-resampled, see box_resample), action_*_latest, arm_q, arm_dq, hand_q, body_q (29), cmd_arm_kp, cmd_hand_kp,
                  age_* (s since each stream's last message; age_head_rgbd = head RGB-D frame), head_frame_id,
                  depth_valid (fraction of non-zero depth pixels in the frame written)
  head_rgb/ left_wrist_rgb/ right_wrist_rgb/  NNNNNN.jpg (q95);  head_depth/ NNNNNN.png (uint16 mm, 0 = invalid)
Head depth comes only from head_rgbd_server.py on PC2 (bash scripts/deploy_rgbd_pc2.sh). With that server down the
image client falls back to teleimager colour and ALL-ZERO depth, so the recorder refuses to start an episode while
no RGB-D frame is arriving, counts stale depth ticks per episode, and marks meta.json `head_depth`. --no-head-rgbd
records without depth on purpose (head_depth: false); the converter skips such episodes unless --allow-no-depth.
Convert with aaltonomous/data/to_lerobot.py (env `lerobot`): LeRobot v3.0, and/or a GR00T copy (--format groot|both).
"""
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import ssl  # noqa: F401  before DDS (cyclonedds loads system libcrypto; see sim/evaluate.py)
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HZ = 30.0
SPEC = yaml.safe_load((REPO / "aaltonomous/interface/joint_spec.yaml").read_text())
ARM_NAMES = [j["name"] for j in SPEC["arm"]]
HAND_NAMES = [j["name"] for j in SPEC["hand"]]
ARM_IDX = [j["dds_index"] for j in SPEC["arm"]]          # rt/lowcmd / rt/lowstate motor indices 15..28
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
FORMAT = "aalto-demo-v1"
STALE_S = 0.2           # a stream older than this at a tick counts as stale
CMD_TOPICS = {"rt/lowcmd": ARM_IDX, "rt/dex3/left/cmd": range(7), "rt/dex3/right/cmd": range(7)}
TOPICS = ("rt/lowcmd", "rt/lowstate", "rt/dex3/left/cmd", "rt/dex3/right/cmd",
          "rt/dex3/left/state", "rt/dex3/right/state")


# ------------------------------------------------------------------ DDS (read-only)
def extract(latest, now):
    """Latest DDS messages {topic: (msg, t_recv)} -> one 30 Hz row in joint_spec order (NaN where missing)."""
    nan14 = np.full(14, np.nan)
    row = {"action_arm": nan14.copy(), "action_hand": nan14.copy(), "arm_q": nan14.copy(),
           "arm_dq": nan14.copy(), "hand_q": nan14.copy(), "body_q": np.full(29, np.nan),
           "cmd_arm_kp": np.nan, "cmd_hand_kp": np.nan}
    if "rt/lowcmd" in latest:
        m = latest["rt/lowcmd"][0].motor_cmd
        row["action_arm"] = np.array([m[i].q for i in ARM_IDX])
        row["cmd_arm_kp"] = min(m[i].kp for i in ARM_IDX)
    if "rt/lowstate" in latest:
        m = latest["rt/lowstate"][0].motor_state
        row["arm_q"] = np.array([m[i].q for i in ARM_IDX])
        row["arm_dq"] = np.array([m[i].dq for i in ARM_IDX])
        row["body_q"] = np.array([m[i].q for i in range(29)])
    kps = []
    for k, side in enumerate(("left", "right")):
        if f"rt/dex3/{side}/cmd" in latest:
            m = latest[f"rt/dex3/{side}/cmd"][0].motor_cmd
            row["action_hand"][7 * k:7 * k + 7] = [m[i].q for i in range(7)]
            kps += [m[i].kp for i in range(7)]
        if f"rt/dex3/{side}/state" in latest:
            m = latest[f"rt/dex3/{side}/state"][0].motor_state
            row["hand_q"][7 * k:7 * k + 7] = [m[i].q for i in range(7)]
    if kps:
        row["cmd_hand_kp"] = min(kps)
    for t in TOPICS:
        row["age_" + t.replace("rt/", "").replace("/", "_")] = now - latest[t][1] if t in latest else np.inf
    return row


def box_resample(log, tick_t, dt=1.0 / HZ):
    """Command messages [(t, q)] -> one value per tick: the time-average of the (zero-order-hold) command signal
    over [t_i - dt/2, t_i + dt/2].

    Teleop updates its target at 60 Hz and the controller republishes it at 250 Hz (Dex3 100 Hz); picking the
    newest message at each 30 Hz tick aliases (consecutive samples 0 or 2 updates apart = up to 2x the true
    command velocity, which our safety filter then clips on replay). The exact box average is an anti-aliased
    resample whose tick-to-tick change never exceeds the command's own peak velocity. NaN before the first message."""
    if not log:
        return np.full((len(tick_t), 7), np.nan)
    t = np.array([m[0] for m in log])
    q = np.array([m[1] for m in log], float)
    C = np.vstack([np.zeros(q.shape[1]), np.cumsum(q[:-1] * np.diff(t)[:, None], 0)])  # integral at t[k]

    def integral(x):
        k = np.clip(np.searchsorted(t, x, side="right") - 1, 0, None)
        return C[k] + q[k] * (x - t[k])[:, None]

    out = (integral(tick_t + dt / 2) - integral(tick_t - dt / 2)) / dt
    out[tick_t - dt / 2 < t[0]] = np.nan
    return out


class DDSTap:
    """Subscribes to the six topics and keeps only the newest message of each. Never publishes."""

    def __init__(self, domain, network_interface=None):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
        ChannelFactoryInitialize(domain, networkInterface=network_interface)
        types_ = dict(zip(TOPICS, (LowCmd_, LowState_, HandCmd_, HandCmd_, HandState_, HandState_)))
        self.latest, self.count, self.subs = {}, {t: 0 for t in TOPICS}, []
        self.cmd_log = None   # while recording: every command message, for box resampling (resample_cmds)
        # (set before Init: callbacks start firing immediately)
        for topic, typ in types_.items():
            sub = ChannelSubscriber(topic, typ)
            sub.Init(lambda m, k=topic: self._on(k, m), 0)
            self.subs.append(sub)


    def _on(self, topic, msg):
        t = time.monotonic()
        self.latest[topic] = (msg, t)
        self.count[topic] += 1
        log = self.cmd_log
        if log is not None and topic in CMD_TOPICS:
            log[topic].append((t, [msg.motor_cmd[i].q for i in CMD_TOPICS[topic]]))

    def start_log(self):
        """Seeded with the newest message of each command topic, so the first tick's window is covered."""
        latest = dict(self.latest)
        self.cmd_log = {k: ([(latest[k][1], [latest[k][0].motor_cmd[i].q for i in idx])] if k in latest else [])
                        for k, idx in CMD_TOPICS.items()}

    def stop_log(self):
        log, self.cmd_log = self.cmd_log, None
        return log

    def sample(self):
        return extract(dict(self.latest), time.monotonic())

    def counts(self):
        return dict(self.count)


class CameraTap:
    """Head RGB-D + wrist RGB through the interface's own clients (backends.DDSBackend.images, no DDS)."""

    def __init__(self, host, head_rgbd=True, rgbd_port=55560, wrist=True):
        from aaltonomous.interface import backends
        base = os.environ.get("AALTO_CAM_PORT_BASE")   # 2nd sim instance (same env vars as sim/evaluate.py)
        if base:
            b = int(base)
            backends._V2ImageClient.PORTS = (("head_camera", b), ("left_wrist_camera", b + 1),
                                             ("right_wrist_camera", b + 2))
        self._images = backends.DDSBackend.images
        self.rgbd = self.img = None
        if head_rgbd:
            from aaltonomous.interface.head_rgbd import HeadRGBDClient
            self.rgbd = HeadRGBDClient(host, rgbd_port)
        if wrist:
            self.img = backends.make_image_client(host)

    def images(self):
        out = self._images(self)
        f = self.rgbd.get() if self.rgbd is not None else None
        out["head_frame_id"] = -1 if f is None else int(f[3])
        if self.rgbd is not None:
            out["age_head_rgbd"] = self.rgbd.age()
        out["depth_valid"] = float(np.count_nonzero(out["head_depth"])) / out["head_depth"].size
        return out

    def depth_age(self):
        """s since the newest head RGB-D frame arrived; None when recording without depth (--no-head-rgbd)."""
        return None if self.rgbd is None else self.rgbd.age()

    def close(self):
        for c in (self.rgbd, self.img):
            if c is not None:
                c.close()


# ------------------------------------------------------------------ episodes on disk
class EpisodeWriter:
    """Images go to disk from a thread pool as they arrive; low-dim rows are written on close()."""

    def __init__(self, path: Path, meta: dict, workers=4):
        import cv2
        self.cv2, self.path, self.meta = cv2, path.with_name(path.name + ".partial"), meta
        self.final = path
        for k in CAMS + ("head_depth",):
            (self.path / k).mkdir(parents=True, exist_ok=True)
        self.pool, self.futs, self.rows = ThreadPoolExecutor(workers), [], []
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1))

    def _write(self, i, imgs):
        for k in CAMS:
            self.cv2.imwrite(str(self.path / k / f"{i:06d}.jpg"), imgs[k][..., ::-1], [self.cv2.IMWRITE_JPEG_QUALITY, 95])
        self.cv2.imwrite(str(self.path / "head_depth" / f"{i:06d}.png"), imgs["head_depth"],
                         [self.cv2.IMWRITE_PNG_COMPRESSION, 1])

    def add(self, row, imgs):
        i = len(self.rows)
        self.rows.append(row)
        self.futs.append(self.pool.submit(self._write, i, imgs))
        self.futs = [f for f in self.futs if not f.done() or f.result() is not None]  # re-raise write errors

    def backlog(self):
        return sum(not f.done() for f in self.futs)

    def close(self, success, extra=None):
        self.pool.shutdown(wait=True)
        for f in self.futs:
            f.result()
        cols = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}
        np.savez(self.path / "lowdim.npz", **cols)
        self.meta.update(success=success, n_frames=len(self.rows), **(extra or {}))
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=1))
        self.path.rename(self.final)
        return self.final

    def discard(self):
        self.pool.shutdown(wait=True)
        shutil.rmtree(self.path, ignore_errors=True)


def filter_check(rows):
    """Would these actions pass our safety filter unchanged? Replays them through SafetyFilter offline."""
    try:
        import logging
        from aaltonomous.interface.safety import SafetyFilter
        logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)  # per-clip lines would flood the console
        f = SafetyFilter.from_files()
        ok = [i for i, r in enumerate(rows) if all(np.isfinite(r[k]).all()
                                                    for k in ("action_arm", "action_hand", "arm_q", "hand_q"))]
        rows = rows[ok[0]:] if ok else []
        if not rows:
            return {"error": "no finite rows"}
        A = np.array([r["action_arm"] for r in rows])
        Hd = np.array([r["action_hand"] for r in rows])
        Q, HQ = np.array([r["arm_q"] for r in rows]), np.array([r["hand_q"] for r in rows])
        f.reset(A[0], Hd[0], 0.0)
        f.cfg.max_clipped_steps = 10 ** 9          # count, never abort
        reasons, clipped = {}, 0
        for i in range(len(rows)):
            res = f.filter({"arm_q_target": A[i], "hand_q_target": Hd[i]}, i / HZ, arm_q=Q[i], hand_q=HQ[i])
            clipped += bool(res.reasons)
            for r in res.reasons:
                reasons[r] = reasons.get(r, 0) + 1
        dA = np.abs(np.diff(A, axis=0)).max(0) * HZ if len(A) > 1 else np.zeros(14)
        return {"clipped_steps": clipped, "reasons": reasons, "arm_peak_cmd_vel": round(float(np.nanmax(dA)), 3),
                "arm_ticks_over_cap": int((np.abs(np.diff(A, axis=0)).max(1) * HZ > f.cfg.arm_max_vel + 1e-6).sum())}
    except Exception as e:  # pinocchio missing etc.: the check is advisory
        return {"error": repr(e)}


# ------------------------------------------------------------------ recorder
def git_rev():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


class Recorder:
    """State machine idle -> recording -> labelling -> idle. tick() once per 30 Hz period, handle_key() on input."""

    def __init__(self, dds, cams, out, task="sort_fixed", card_seed=1000, operator="", session=None,
                 instruction=None, max_s=180.0, notes="", log=print):
        from aaltonomous.scene.cards import make_card
        from aaltonomous.scene.scene import Scene
        self.dds, self.cams, self.task, self.log = dds, cams, task, log
        self.scene, self.make_card = Scene.load(), make_card
        self.card_seed, self.card_k, self.operator, self.instruction = card_seed, 0, operator, instruction
        self.max_s, self.notes = max_s, notes
        self.session = session or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out) / self.session
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state, self.writer, self.quit, self.saved = "idle", None, False, []
        self.ep = len([p for p in self.dir.glob("ep*") if not p.name.endswith(".partial")])
        self.card = self._card()

    def _card(self):
        c = self.make_card(self.scene, self.task, self.card_seed + self.card_k)
        if self.instruction:
            c["instruction"] = self.instruction
        return c

    def show_card(self):
        from aaltonomous.interface.tasks import layout_card
        self.log(f"\n--- card {self.card_k} (seed {self.card['seed']}, {self.task}) ---\n"
                 f"{layout_card(self.card['cubes'], self.card['instruction'])}\n"
                 "place the objects, engage teleop (r, h), then press n")

    def start(self):
        s = self.dds.sample()
        stale = [k for k in s if k.startswith("age_") and s[k] > STALE_S]
        if s["age_lowcmd"] > STALE_S or s["age_dex3_left_cmd"] > STALE_S or s["age_dex3_right_cmd"] > STALE_S:
            self.log(f"!! not starting: teleop commands not arriving ({', '.join(stale)}); is teleop running?")
            return
        depth_age = self.cams.depth_age() if hasattr(self.cams, "depth_age") else None
        if depth_age is not None and depth_age > STALE_S:
            self.log("!! not starting: no head depth (last RGB-D frame "
                     + ("never" if depth_age == float("inf") else f"{depth_age:.1f} s ago")
                     + "); run scripts/deploy_rgbd_pc2.sh, or record without depth with --no-head-rgbd")
            return
        name = f"ep{self.ep:03d}_{self.task}_s{self.card['seed']}"
        c = self.card
        meta = {"format": FORMAT, "fps": HZ, "session": self.session, "episode": self.ep, "task": c["task"],
                "instruction": c["instruction"], "mapping": c["mapping"], "card_seed": c["seed"],
                "layout_id": f"{c['task']}/{c['seed']}", "layout": c["cubes"], "operator": self.operator,
                "notes": self.notes, "start": datetime.now().isoformat(timespec="seconds"), "git": git_rev(),
                "arm_names": ARM_NAMES, "hand_names": HAND_NAMES, "source": "rt/lowcmd + rt/dex3/*/cmd",
                "head_depth": depth_age is not None, "success": None}
        self.writer = EpisodeWriter(self.dir / name, meta)
        self.t0, self.next, self.counts0, self.stale_ticks = time.monotonic(), 0.0, self.dds.counts(), 0
        self.late_ticks, self.cmd_log, self.depth_stale_ticks = 0, None, 0
        if hasattr(self.dds, "start_log"):
            self.dds.start_log()
        self.state = "recording"
        self.log(f">> recording {name}: s = stop, d = discard")

    def tick(self):
        """Sample one row (if recording). Returns the time to sleep until the next 30 Hz tick."""
        if self.state != "recording":
            return 1.0 / HZ
        tm = time.monotonic()
        now = tm - self.t0
        if now < self.next:          # woken early by a key: not time for the next frame yet
            return self.next - now
        row = self.dds.sample()
        row["tm"] = tm
        imgs = self.cams.images()
        row.update(t=now, wall=time.time(), head_frame_id=imgs.pop("head_frame_id", -1))
        for k in ("age_head_rgbd", "depth_valid"):
            if k in imgs:
                row[k] = imgs.pop(k)
        if row.get("age_head_rgbd", 0.0) > STALE_S:
            self.depth_stale_ticks += 1
        if any(row[k] > STALE_S for k in row if k.startswith("age_")):
            self.stale_ticks += 1
        self.writer.add(row, imgs)
        n = len(self.writer.rows)
        if n % int(5 * HZ) == 0:
            self.log(f"   t={now:5.1f}s frames={n} stale={self.stale_ticks} write backlog={self.writer.backlog()}")
        if now >= self.max_s:
            self.log(f"!! max length {self.max_s:.0f} s reached")
            self.stop()
        self.next += 1.0 / HZ
        late = time.monotonic() - self.t0 - self.next
        if late > 0:                 # fell behind (slow camera/disk): skip ahead instead of bursting frames
            self.late_ticks += 1
            self.next += np.ceil(late * HZ) / HZ
        return max(0.0, self.next - (time.monotonic() - self.t0))

    def _stop_log(self):
        if hasattr(self.dds, "stop_log"):
            self.cmd_log = self.dds.stop_log()

    def stop(self):
        self._stop_log()
        self.state = "labelling"
        self.log("<< stopped. outcome?  y = success   f = failure   d = discard")

    def label(self, success):
        rows, dur = self.writer.rows, time.monotonic() - self.t0
        if self.cmd_log and rows:   # actions = anti-aliased resample of the full-rate command streams
            tt = np.array([r["tm"] for r in rows])
            arm = box_resample(self.cmd_log["rt/lowcmd"], tt)
            hand = np.c_[box_resample(self.cmd_log["rt/dex3/left/cmd"], tt),
                         box_resample(self.cmd_log["rt/dex3/right/cmd"], tt)]
            for r, a, h in zip(rows, arm, hand):
                r["action_arm_latest"], r["action_hand_latest"] = r["action_arm"], r["action_hand"]
                r["action_arm"], r["action_hand"] = a, h
        c1 = self.dds.counts()
        rates = {k.replace("rt/", ""): round((c1[k] - self.counts0[k]) / max(dur, 1e-6), 1) for k in c1}
        kp = np.array([r["cmd_arm_kp"] for r in rows]) if rows else np.zeros(0)
        warn = []
        if self.stale_ticks:
            warn.append(f"{self.stale_ticks} ticks with a stale DDS stream (> {STALE_S} s)")
        if (kp < 1.0).any():
            warn.append(f"{int((kp < 1.0).sum())} ticks with arm kp ~0 (limp / SPACE): actions not meaningful")
        if self.late_ticks:
            warn.append(f"{self.late_ticks} late ticks (loop could not keep 30 Hz)")
        depth = {}
        if self.writer.meta.get("head_depth"):
            dv = np.array([r.get("depth_valid", np.nan) for r in rows], float)
            depth = {"stale_ticks": self.depth_stale_ticks,
                     "blank_frames": int((dv == 0).sum()),
                     "valid_frac_mean": round(float(np.nanmean(dv)), 3) if rows and np.isfinite(dv).any() else None}
            if self.depth_stale_ticks:
                warn.append(f"{self.depth_stale_ticks} ticks with stale head depth (> {STALE_S} s since last RGB-D frame)")
            if depth["blank_frames"]:
                warn.append(f"{depth['blank_frames']} frames with all-zero head depth")
        fc = filter_check(rows) if rows else {}
        extra = {"duration_s": round(dur, 2), "dds_rates_hz": rates, "warnings": warn, "filter_check": fc,
                 "depth": depth, "end": datetime.now().isoformat(timespec="seconds")}
        p = self.writer.close(success, extra)
        self.saved.append(p)
        self.log(f"== saved {p.name}: {len(rows)} frames, success={success}, filter clips={fc.get('clipped_steps')}"
                 + (f"; WARN {warn}" if warn else ""))
        self.ep += 1
        self._advance(1)

    def discard(self):
        self._stop_log()
        self.writer.discard()
        self.writer, self.state = None, "idle"
        self.log("xx discarded")
        self.show_card()

    def _advance(self, d):
        self.card_k = max(0, self.card_k + d)
        self.card = self._card()
        self.state = "idle"
        self.show_card()

    def handle_key(self, k):
        if self.state == "idle":
            if k == "n":
                self.start()
            elif k in "]":
                self._advance(1)
            elif k in "[":
                self._advance(-1)
            elif k == "c":
                self.show_card()
            elif k == "q":
                self.quit = True
        elif self.state == "recording":
            if k == "s":
                self.stop()
            elif k == "d":
                self.state = "labelling"
                self.discard()
        elif self.state == "labelling":
            if k in "yf":
                self.label(k == "y")
            elif k == "d":
                self.discard()

    def close(self):
        if self.writer is not None and self.state in ("recording", "labelling"):
            self.log("!! quitting with an unlabelled episode: left as .partial (not converted)")
            self.writer.pool.shutdown(wait=True)


# ------------------------------------------------------------------ keyboard
class Keys:
    """Single keys from a TTY (raw mode, restored on exit) or characters from a pipe (tests, scripts)."""

    def __init__(self):
        self.tty = sys.stdin.isatty()
        if self.tty:
            import termios
            import tty
            self.fd, self.old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(self.fd)

    def get(self, timeout):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        c = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        return "q" if c == "" and not self.tty else c   # EOF on a pipe = quit

    def close(self):
        if self.tty:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


DEPTH_MIN_HZ = 15.0      # head RGB-D frames/s below which --check fails (D435 streams 30)
DEPTH_MIN_VALID = 0.3    # fraction of non-zero depth pixels below which --check fails (lens covered, too close)


def check(dds, cams, secs=3.0):
    """--check: stream rates and freshness, then exit. Use it on Thursday before the first episode."""
    c0 = dds.counts()
    f0 = cams.images().get("head_frame_id", -1) if cams else -1
    time.sleep(secs)
    c1, s = dds.counts(), dds.sample()
    ok = True
    for t in TOPICS:
        hz = (c1[t] - c0[t]) / secs
        good = hz > 20
        ok &= good
        print(f"  {t:22s} {hz:7.1f} Hz  {'ok' if good else 'MISSING'}")
    imgs = cams.images() if cams else {}
    for k in CAMS + ("head_depth",):
        if k in imgs:
            live = bool(imgs[k].any())
            ok &= live
            print(f"  {k:22s} {'live' if live else 'BLANK'}  mean={imgs[k].mean():.1f}")
    if getattr(cams, "depth_age", lambda: None)() is not None:
        f1 = imgs.get("head_frame_id", -1)
        hz = (f1 - f0) / secs if f0 >= 0 and f1 >= 0 else 0.0
        valid = imgs.get("depth_valid", 0.0)
        good = hz >= DEPTH_MIN_HZ and valid >= DEPTH_MIN_VALID
        ok &= good
        print(f"  {'head RGB-D (depth)':22s} {hz:7.1f} Hz  valid {100 * valid:.0f}%  "
              + ("ok" if good else f"FAIL (need >= {DEPTH_MIN_HZ:.0f} Hz and >= {100 * DEPTH_MIN_VALID:.0f}% valid)"))
    else:
        print(f"  {'head depth':22s} OFF (--no-head-rgbd): episodes are recorded WITHOUT depth")
    print("  action arm  ", np.round(s["action_arm"], 3), " kp", s["cmd_arm_kp"])
    print("  arm_q       ", np.round(s["arm_q"], 3))
    print("  action hand ", np.round(s["action_hand"], 3), " kp", s["cmd_hand_kp"])
    print("  hand_q      ", np.round(s["hand_q"], 3))
    print("CHECK", "PASS" if ok else "FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("real", "sim"), default="real")
    ap.add_argument("--network-interface", default=None, help="real: enx00e04cd30dba on the laptop")
    ap.add_argument("--img-server-ip", default=None, help="default 192.168.123.164 (real) / 127.0.0.1 (sim)")
    ap.add_argument("--rgbd-port", type=int, default=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    ap.add_argument("--no-head-rgbd", action="store_true",
                    help="record WITHOUT depth: head colour from teleimager, depth zeros, meta head_depth=false "
                         "(the converter skips these unless --allow-no-depth)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--task", default="sort_fixed", choices=("sort_fixed", "sort_language", "insert_shapes"))
    ap.add_argument("--card-seed", type=int, default=1000, help="layout card k = seed + k (scene/cards.py)")
    ap.add_argument("--instruction", default=None, help="override the card's instruction text")
    ap.add_argument("--out", default=str(REPO / "data/demos"))
    ap.add_argument("--session", default=None, help="sub-dir name; reuse one to append (default: timestamp)")
    ap.add_argument("--operator", default="")
    ap.add_argument("--notes", default="", help="free text stored in every episode (e.g. teleop args)")
    ap.add_argument("--max-s", type=float, default=180.0)
    ap.add_argument("--check", action="store_true", help="print stream rates and exit")
    a = ap.parse_args(argv)
    domain = 0 if a.backend == "real" else int(os.environ.get("AALTO_DDS_DOMAIN", "1"))
    host = a.img_server_ip or ("192.168.123.164" if a.backend == "real" else "127.0.0.1")

    dds = DDSTap(domain, a.network_interface)
    cams = None if a.no_cameras else CameraTap(host, head_rgbd=not a.no_head_rgbd, rgbd_port=a.rgbd_port)
    if cams is None:
        from aaltonomous.interface.backends import blank_images
        cams = type("NoCams", (), {"images": lambda self: {**blank_images(), "head_frame_id": -1},
                                   "close": lambda self: None})()
    time.sleep(1.0)
    if a.check:
        ok = check(dds, cams)
        cams.close()
        sys.stdout.flush()
        os._exit(0 if ok else 1)

    rec = Recorder(dds, cams, a.out, a.task, a.card_seed, a.operator, a.session, a.instruction, a.max_s, a.notes)
    print(f"[record_demos] {a.backend} DDS domain {domain}, cameras {host}; session dir {rec.dir}\n"
          "keys: n start | s stop -> y/f/d | d discard | ] [ card | c card | q quit")
    rec.show_card()
    keys = Keys()
    try:
        while not rec.quit:
            k = keys.get(rec.tick())
            if k:
                rec.handle_key(k.lower())
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        rec.close()
        cams.close()
        print(f"[record_demos] {len(rec.saved)} episode(s) saved in {rec.dir}")
    sys.stdout.flush()
    os._exit(0)  # DDS threads


if __name__ == "__main__":
    main()
